# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
"""Benchmarks for the vector-search prefilter cost.

The prefilter loads each fragment's deletion file (and, when stable row ids
are enabled, each fragment's row id sequence) before issuing a vector
search. The result is memoized in the per-Dataset metadata cache, so we
re-open the dataset on every iteration to measure cold-cache cost.

Knobs at the top of the file are intentionally Python-only — bump them and
re-run; no Rust rebuild required. Cached datasets are namespaced by
NUM_ROWS / N_DIMS / num_frags / stable_row_ids, so changes to those
constants invalidate the cache automatically. Changes to
DELETION_FRACTIONS do not — delete the cached directories or the sidecar
``_bench_version_map.json`` to force a rebuild.
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
# Each can be overridden via an environment variable so scaling up does not
# require editing the file.

def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float_tuple(name: str, default: Tuple[float, ...]) -> Tuple[float, ...]:
    raw = os.environ.get(name)
    if not raw:
        return default
    return tuple(float(x) for x in raw.split(","))


def _env_int_tuple(name: str, default: Tuple[int, ...]) -> Tuple[int, ...]:
    raw = os.environ.get(name)
    if not raw:
        return default
    return tuple(int(x) for x in raw.split(","))


NUM_ROWS = _env_int("BENCH_PREFILTER_NUM_ROWS", 1_000_000)
N_DIMS = _env_int("BENCH_PREFILTER_N_DIMS", 1024)
FRAG_COUNTS: Tuple[int, ...] = _env_int_tuple(
    "BENCH_PREFILTER_FRAG_COUNTS", (10, 1000)
)
DELETION_FRACTIONS: Tuple[float, ...] = _env_float_tuple(
    "BENCH_PREFILTER_DELETION_FRACTIONS", (0.0, 0.10, 0.50)
)

# Vector index params; bump together with NUM_ROWS for larger runs.
NUM_PARTITIONS = _env_int("BENCH_PREFILTER_NUM_PARTITIONS", 256)
NUM_SUB_VECTORS = _env_int("BENCH_PREFILTER_NUM_SUB_VECTORS", 64)

# Search params.
K = _env_int("BENCH_PREFILTER_K", 100)
NPROBES = _env_int("BENCH_PREFILTER_NPROBES", 10)

# Cap per-batch row count during dataset construction so peak memory stays
# bounded even when rows-per-fragment is large.
WRITE_BATCH_ROWS = _env_int("BENCH_PREFILTER_WRITE_BATCH_ROWS", 32_000)

# ----------------------------------------------------------------------------


def _dataset_dir(data_dir: Path, num_frags: int, stable: bool) -> Path:
    return (
        data_dir
        / f"prefilter_{num_frags}frags_stable{int(stable)}"
        f"_{NUM_ROWS}rows_{N_DIMS}dims"
    )


def _version_map_path(uri: Path) -> Path:
    return uri / "_bench_version_map.json"


def _build_reader(num_frags: int, rows_per_frag: int) -> pa.RecordBatchReader:
    schema = pa.schema(
        [
            pa.field("vector", pa.list_(pa.float32(), N_DIMS), nullable=False),
            pa.field("row_idx", pa.int64(), nullable=False),
        ]
    )

    def gen():
        for f in range(num_frags):
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
                yield pa.RecordBatch.from_arrays(
                    [vectors, row_idx], schema=schema
                )
                emitted += this_batch

    return pa.RecordBatchReader.from_batches(schema, gen())


def _find_or_build_dataset(
    data_dir: Path, num_frags: int, stable: bool
) -> Tuple[Path, Dict[float, int]]:
    """Build (or reuse) a dataset and apply incremental deletion variants
    as separate Lance versions. Returns ``(uri, {fraction: version})``.
    """
    uri = _dataset_dir(data_dir, num_frags, stable)
    map_path = _version_map_path(uri)
    if uri.exists() and map_path.exists():
        return uri, {
            float(k): int(v) for k, v in json.loads(map_path.read_text()).items()
        }

    if uri.exists():
        shutil.rmtree(uri)

    if NUM_ROWS % num_frags != 0:
        raise ValueError(
            f"NUM_ROWS={NUM_ROWS} not divisible by num_frags={num_frags}"
        )
    rows_per_frag = NUM_ROWS // num_frags

    reader = _build_reader(num_frags, rows_per_frag)
    lance.write_dataset(
        reader,
        uri,
        max_rows_per_file=rows_per_frag,
        enable_stable_row_ids=stable,
    )

    dataset = lance.dataset(uri)
    dataset.create_index(
        column="vector",
        index_type="IVF_PQ",
        metric_type="L2",
        num_partitions=NUM_PARTITIONS,
        num_sub_vectors=NUM_SUB_VECTORS,
    )

    sorted_fracs = sorted(DELETION_FRACTIONS)
    if sorted_fracs[0] != 0.0:
        raise ValueError("DELETION_FRACTIONS must include 0.0")

    version_map: Dict[float, int] = {}
    dataset = lance.dataset(uri)
    version_map[0.0] = dataset.version

    prev_count = 0
    for frac in sorted_fracs[1:]:
        target_count = max(1, round(num_frags * frac))
        new_targets = list(range(prev_count, target_count))
        if not new_targets:
            # Same effective fraction as the previous step — share the version.
            version_map[frac] = version_map[sorted_fracs[sorted_fracs.index(frac) - 1]]
            continue
        in_list = ", ".join(str(i * rows_per_frag) for i in new_targets)
        ds = lance.dataset(uri)
        ds.delete(f"row_idx IN ({in_list})")
        ds = lance.dataset(uri)
        version_map[frac] = ds.version
        prev_count = target_count

    map_path.write_text(json.dumps({str(k): v for k, v in version_map.items()}))
    return uri, version_map


@pytest.fixture(scope="module")
def prefilter_datasets(
    data_dir: Path,
) -> Dict[Tuple[int, bool], Tuple[Path, Dict[float, int]]]:
    out: Dict[Tuple[int, bool], Tuple[Path, Dict[float, int]]] = {}
    for num_frags in FRAG_COUNTS:
        for stable in (False, True):
            out[(num_frags, stable)] = _find_or_build_dataset(
                data_dir, num_frags, stable
            )
    return out


@pytest.mark.benchmark(group="prefilter")
@pytest.mark.parametrize("stable", [False, True], ids=["stable_off", "stable_on"])
@pytest.mark.parametrize(
    "deletion_frac",
    list(DELETION_FRACTIONS),
    ids=[f"del_{int(f * 100)}pct" for f in DELETION_FRACTIONS],
)
@pytest.mark.parametrize("num_frags", list(FRAG_COUNTS))
def test_ann_prefilter(
    prefilter_datasets, benchmark, num_frags, deletion_frac, stable
):
    uri, version_map = prefilter_datasets[(num_frags, stable)]
    version = version_map[deletion_frac]
    rng = np.random.default_rng(0)

    def setup():
        # Re-open the dataset each round so the per-Dataset metadata
        # cache (which memoizes the prefilter mask) is cold.
        ds = lance.dataset(uri, version=version)
        q = pa.array(rng.standard_normal(N_DIMS).astype(np.float32))
        return (ds, q), {}

    def target(ds, q):
        return ds.to_table(
            columns=[],
            with_row_id=True,
            nearest=dict(column="vector", q=q, k=K, nprobes=NPROBES),
        )

    benchmark.pedantic(target, setup=setup, rounds=10, iterations=1)
