# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
"""Benchmarks for take-by-row-id under stable row ids.

This bench targets ``LanceDataset._take_rows`` (the by-stable-row-id path,
which goes through ``RowIdIndex::get`` per id) rather than the public
``take(indices)`` (which is by row offset and skips the index entirely).

The new dimension here is **update history**. Every update commit appends
a small "patch" fragment whose ``RowIdSequence`` contains only the
updated row ids — typically as an ``Array`` / ``SortedArray`` segment —
and adds one chunk to the in-memory ``RowIdIndex``. Many small scattered
updates can therefore inflate the chunk count by orders of magnitude,
blowing up both index build time (cold cache) and manifest size on disk.

We measure cold-cache per-round (re-open the dataset per iteration via
``benchmark.pedantic``) so the index build cost shows up alongside
lookup. The fragment count is logged once per variant as a proxy for
chunk count.

Knobs are env-var overridable so the bench can be scaled up without a
Rust rebuild. Cached datasets are namespaced by NUM_ROWS, N_DIMS and
NUM_FRAGMENTS; deletion of the cache directory or the sidecar
``_bench_version_map.json`` forces a rebuild.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Dict, Tuple

import lance
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

# --- Tuning knobs -----------------------------------------------------------


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_int_tuple(name: str, default: Tuple[int, ...]) -> Tuple[int, ...]:
    raw = os.environ.get(name)
    if not raw:
        return default
    return tuple(int(x) for x in raw.split(","))


def _env_str_tuple(name: str, default: Tuple[str, ...]) -> Tuple[str, ...]:
    raw = os.environ.get(name)
    if not raw:
        return default
    return tuple(s.strip() for s in raw.split(","))


NUM_ROWS = _env_int("BENCH_TAKE_NUM_ROWS", 1_000_000)
N_DIMS = _env_int("BENCH_TAKE_N_DIMS", 1024)
NUM_FRAGMENTS = _env_int("BENCH_TAKE_NUM_FRAGMENTS", 100)
UPDATE_COUNTS: Tuple[int, ...] = _env_int_tuple(
    "BENCH_TAKE_UPDATE_COUNTS", (0, 100, 1000)
)
TAKE_BATCH_SIZES: Tuple[int, ...] = _env_int_tuple(
    "BENCH_TAKE_BATCH_SIZES", (100, 10_000)
)
ACCESS_PATTERNS: Tuple[str, ...] = _env_str_tuple(
    "BENCH_TAKE_PATTERNS", ("random", "sorted", "sequential")
)
WRITE_BATCH_ROWS = _env_int("BENCH_TAKE_WRITE_BATCH_ROWS", 32_000)

# ----------------------------------------------------------------------------


def _dataset_dir(data_dir: Path, stable: bool) -> Path:
    return (
        data_dir
        / f"take_row_id_{NUM_FRAGMENTS}frags_stable{int(stable)}"
        f"_{NUM_ROWS}rows_{N_DIMS}dims"
    )


def _version_map_path(uri: Path) -> Path:
    return uri / "_bench_version_map.json"


def _build_reader(rows_per_frag: int) -> pa.RecordBatchReader:
    schema = pa.schema(
        [
            pa.field("vector", pa.list_(pa.float32(), N_DIMS), nullable=False),
            pa.field("row_idx", pa.int64(), nullable=False),
            pa.field("payload", pa.int64(), nullable=False),
        ]
    )

    def gen():
        for f in range(NUM_FRAGMENTS):
            base = f * rows_per_frag
            emitted = 0
            while emitted < rows_per_frag:
                this_batch = min(WRITE_BATCH_ROWS, rows_per_frag - emitted)
                values = pc.random(this_batch * N_DIMS).cast(pa.float32())
                vectors = pa.FixedSizeListArray.from_arrays(values, N_DIMS)
                row_idx = pa.array(
                    np.arange(
                        base + emitted,
                        base + emitted + this_batch,
                        dtype=np.int64,
                    )
                )
                payload = pa.array(np.zeros(this_batch, dtype=np.int64))
                yield pa.RecordBatch.from_arrays(
                    [vectors, row_idx, payload], schema=schema
                )
                emitted += this_batch

    return pa.RecordBatchReader.from_batches(schema, gen())


def _find_or_build_dataset(
    data_dir: Path, stable: bool
) -> Tuple[Path, Dict[int, int]]:
    """Build base + btree index + incremental update commits.

    Returns ``(uri, {update_count: version})``.
    """
    uri = _dataset_dir(data_dir, stable)
    map_path = _version_map_path(uri)
    if uri.exists() and map_path.exists():
        return uri, {
            int(k): int(v) for k, v in json.loads(map_path.read_text()).items()
        }
    if uri.exists():
        shutil.rmtree(uri)

    if NUM_ROWS % NUM_FRAGMENTS != 0:
        raise ValueError(
            f"NUM_ROWS={NUM_ROWS} not divisible by NUM_FRAGMENTS={NUM_FRAGMENTS}"
        )
    rows_per_frag = NUM_ROWS // NUM_FRAGMENTS

    lance.write_dataset(
        _build_reader(rows_per_frag),
        uri,
        max_rows_per_file=rows_per_frag,
        enable_stable_row_ids=stable,
    )

    # Btree on row_idx so update predicates of the form `row_idx = X`
    # don't full-scan the dataset on every commit.
    ds = lance.dataset(uri)
    ds.create_scalar_index("row_idx", "BTREE")

    sorted_counts = sorted(set(UPDATE_COUNTS))
    if sorted_counts[0] != 0:
        raise ValueError("UPDATE_COUNTS must include 0")

    version_map: Dict[int, int] = {}
    ds = lance.dataset(uri)
    version_map[0] = ds.version

    # Pre-pick all rows we will ever update so the targets are disjoint
    # across update_count tiers.
    rng = np.random.default_rng(42)
    max_updates = max(sorted_counts)
    update_targets = (
        rng.choice(NUM_ROWS, size=max_updates, replace=False)
        if max_updates > 0
        else np.array([], dtype=np.int64)
    )

    applied = 0
    for target_count in sorted_counts[1:]:
        for idx in update_targets[applied:target_count]:
            ds = lance.dataset(uri)
            ds.update({"payload": "payload + 1"}, where=f"row_idx = {int(idx)}")
            applied += 1
        version_map[target_count] = lance.dataset(uri).version

    map_path.write_text(json.dumps({str(k): v for k, v in version_map.items()}))
    return uri, version_map


@pytest.fixture(scope="module")
def take_datasets(
    data_dir: Path,
) -> Dict[bool, Tuple[Path, Dict[int, int]]]:
    out: Dict[bool, Tuple[Path, Dict[int, int]]] = {}
    for stable in (False, True):
        out[stable] = _find_or_build_dataset(data_dir, stable)
    # Print fragment count per variant as a proxy for RowIdIndex chunk count.
    for stable, (uri, vmap) in out.items():
        for upd_count, version in sorted(vmap.items()):
            ds = lance.dataset(uri, version=version)
            print(
                f"[take-bench] stable={stable} updates={upd_count} "
                f"fragments={len(ds.get_fragments())} version={version}"
            )
    return out


def _build_take_ids(
    rng: np.random.Generator,
    batch_size: int,
    pattern: str,
    stable: bool,
    rows_per_frag: int,
) -> pa.Array:
    """Generate ids to feed to ``_take_rows``.

    With stable row ids the id space is the dense sequence assigned at
    write time (``[0, NUM_ROWS)`` here). Without stable row ids the
    "row id" is literally the row address (``frag_id << 32 | offset``),
    so we convert positions accordingly to keep the bench valid for
    both branches.
    """
    if pattern == "sequential":
        start = int(rng.integers(0, NUM_ROWS - batch_size + 1))
        positions = np.arange(start, start + batch_size, dtype=np.int64)
    elif pattern == "sorted":
        positions = np.sort(
            rng.choice(NUM_ROWS, size=batch_size, replace=False)
        )
    elif pattern == "random":
        positions = rng.choice(NUM_ROWS, size=batch_size, replace=False)
    else:
        raise ValueError(f"unknown access pattern: {pattern!r}")

    if stable:
        ids = positions.astype(np.uint64)
    else:
        frag_id = (positions // rows_per_frag).astype(np.uint64)
        offset = (positions % rows_per_frag).astype(np.uint64)
        ids = (frag_id << 32) | offset
    return pa.array(ids, type=pa.uint64())


@pytest.mark.benchmark(group="take_by_row_id")
@pytest.mark.parametrize("stable", [False, True], ids=["stable_off", "stable_on"])
@pytest.mark.parametrize("pattern", list(ACCESS_PATTERNS))
@pytest.mark.parametrize(
    "batch_size",
    list(TAKE_BATCH_SIZES),
    ids=[f"batch_{n}" for n in TAKE_BATCH_SIZES],
)
@pytest.mark.parametrize(
    "num_updates",
    list(UPDATE_COUNTS),
    ids=[f"upd_{n}" for n in UPDATE_COUNTS],
)
def test_take_by_row_id(
    take_datasets, benchmark, num_updates, batch_size, pattern, stable
):
    uri, version_map = take_datasets[stable]
    version = version_map[num_updates]
    rows_per_frag = NUM_ROWS // NUM_FRAGMENTS
    rng = np.random.default_rng(0)

    def setup():
        # Re-open per round so the per-Dataset metadata cache (where the
        # RowIdIndex is memoized) is cold. We are measuring index build
        # plus lookup; the build cost is what scales with index
        # fragmentation.
        ds = lance.dataset(uri, version=version)
        ids = _build_take_ids(rng, batch_size, pattern, stable, rows_per_frag)
        return (ds, ids), {}

    def target(ds, ids):
        return ds._take_rows(ids, columns=["row_idx"])

    benchmark.pedantic(target, setup=setup, rounds=10, iterations=1)
