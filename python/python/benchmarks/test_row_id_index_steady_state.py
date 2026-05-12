# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
"""Point-in-time size of the stable row id index under continuous updates.

Models the steady-state question:
    "We have ~1B rows across 1000 fragments and we are constantly updating.
     How big is the index right now?"

Unlike ``test_row_id_index_size.py`` (which sums every historical manifest
under ``_versions/``), this bench measures only the **current** version's
footprint:

- ``manifest_now``    ─ the single ``_versions/{latest}.manifest`` file
- ``data_now``        ─ sum of data files referenced by the current
                        manifest (older versions' orphan data files are
                        excluded)
- ``deletion_now``    ─ deletion files referenced by current fragments
- ``in_memory``       ─ ``deep_size_of`` of the assembled ``RowIdIndex``

Sizing is row-count-independent at the per-fragment level (a ``Range``
sequence is the same handful of bytes for 1K rows or 1M rows), so we
build the dataset at a workable row count and project the result to the
1B/1000-fragment scenario in the docstring.

The dataset is built once and updates are applied incrementally;
``UPDATE_CHECKPOINTS`` controls when we stop and snapshot.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict, List, NamedTuple, Tuple

import lance
import numpy as np
import pyarrow as pa
import pytest

# --- Tuning knobs -----------------------------------------------------------


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_int_tuple(name: str, default: Tuple[int, ...]) -> Tuple[int, ...]:
    raw = os.environ.get(name)
    if not raw:
        return default
    return tuple(int(x) for x in raw.split(","))


NUM_FRAGMENTS = _env_int("BENCH_SS_NUM_FRAGMENTS", 1000)
ROWS_PER_FRAG = _env_int("BENCH_SS_ROWS_PER_FRAG", 1000)
UPDATE_CHECKPOINTS: Tuple[int, ...] = _env_int_tuple(
    "BENCH_SS_UPDATE_CHECKPOINTS", (0, 100, 1000, 10_000)
)
# When set, compaction runs once after the last update checkpoint with this
# target rows-per-fragment, and an extra "post-compact" snapshot is recorded.
COMPACT_TARGET_ROWS = _env_int("BENCH_SS_COMPACT_TARGET_ROWS", 100_000)

# ----------------------------------------------------------------------------


class Snapshot(NamedTuple):
    label: str
    updates: int
    fragment_count: int
    manifest_now: int
    data_now: int
    deletion_now: int
    in_memory: int

    @property
    def total_now(self) -> int:
        return self.manifest_now + self.data_now + self.deletion_now


def _dataset_dir(data_dir: Path) -> Path:
    return (
        data_dir
        / f"steady_state_f{NUM_FRAGMENTS}_r{ROWS_PER_FRAG}"
    )


def _file_size(uri: Path, rel: str) -> int:
    p = uri / rel
    try:
        return p.stat().st_size
    except FileNotFoundError:
        return 0


def _measure(uri: Path, ds: lance.LanceDataset) -> Tuple[int, int, int]:
    """Return ``(manifest_now, data_now, deletion_now)`` referenced by the
    current manifest version only.
    """
    # V2 manifest paths name each manifest as `(u64::MAX - version).manifest`
    # for reverse-chronological sort order.
    v2_name = f"{(1 << 64) - 1 - ds.version}.manifest"
    v1_name = f"{ds.version}.manifest"
    versions_dir = uri / "_versions"
    candidate = versions_dir / v2_name
    if not candidate.exists():
        candidate = versions_dir / v1_name
    manifest_size = candidate.stat().st_size

    data_size = 0
    deletion_size = 0
    for frag in ds.get_fragments():
        meta = frag.metadata
        for df in meta.files:
            # `file_size_bytes` is populated when the writer recorded it; fall
            # back to a stat if not.
            if df.file_size_bytes is not None:
                data_size += df.file_size_bytes
            else:
                data_size += _file_size(uri, df._path)
        if meta.deletion_file is not None:
            # Lance stores deletion files at _deletions/{frag_id}-{read_version}-{...}.{ext}
            # but the exact path is not exposed; stat the directory and credit
            # any file matching the fragment id. This is approximate.
            ext = "arrow" if meta.deletion_file.file_type == "Array" else "bin"
            # The DeletionFile dataclass exposes id/read_version/num_deleted_rows
            # but not the file name pattern; fall back to globbing.
            for p in (uri / "_deletions").glob(f"{frag.fragment_id}-*"):
                deletion_size += p.stat().st_size

    return manifest_size, data_size, deletion_size


def _build_initial(data_dir: Path) -> Path:
    uri = _dataset_dir(data_dir)
    # Cache hit: trust the existing dataset and skip the (slow) rebuild.
    # Delete the cache directory manually to force a rebuild.
    if uri.exists() and (uri / "_versions").exists():
        return uri
    if uri.exists():
        shutil.rmtree(uri)

    schema = pa.schema(
        [
            pa.field("row_idx", pa.int64(), nullable=False),
            pa.field("payload", pa.int64(), nullable=False),
        ]
    )

    def gen():
        for f in range(NUM_FRAGMENTS):
            base = f * ROWS_PER_FRAG
            row_idx = pa.array(
                np.arange(base, base + ROWS_PER_FRAG, dtype=np.int64)
            )
            payload = pa.array(np.zeros(ROWS_PER_FRAG, dtype=np.int64))
            yield pa.RecordBatch.from_arrays([row_idx, payload], schema=schema)

    reader = pa.RecordBatchReader.from_batches(schema, gen())
    lance.write_dataset(
        reader,
        uri,
        max_rows_per_file=ROWS_PER_FRAG,
        enable_stable_row_ids=True,
    )

    ds = lance.dataset(uri)
    ds.create_scalar_index("row_idx", "BTREE")
    return uri


@pytest.fixture(scope="module")
def steady_state_snapshots(data_dir: Path) -> List[Snapshot]:
    uri = _build_initial(data_dir)

    rng = np.random.default_rng(42)
    max_updates = max(UPDATE_CHECKPOINTS)
    num_rows = NUM_FRAGMENTS * ROWS_PER_FRAG
    update_targets = rng.choice(num_rows, size=max_updates, replace=False)

    out: List[Snapshot] = []
    applied = 0

    def snapshot(label: str) -> Snapshot:
        ds = lance.dataset(uri)
        manifest_now, data_now, deletion_now = _measure(uri, ds)
        in_memory = ds.row_id_index_size_bytes() or 0
        return Snapshot(
            label=label,
            updates=applied,
            fragment_count=len(ds.get_fragments()),
            manifest_now=manifest_now,
            data_now=data_now,
            deletion_now=deletion_now,
            in_memory=in_memory,
        )

    # If the cache already has max(UPDATE_CHECKPOINTS) updates applied,
    # skip the (slow) per-checkpoint replay and snapshot only the final
    # state. Versions before the first update are write(1) + btree(2);
    # each applied update bumps the version.
    cached_version = lance.dataset(uri).version
    cached_updates = max(0, cached_version - 2)
    max_target = max(UPDATE_CHECKPOINTS)

    if cached_updates >= max_target:
        applied = cached_updates
        out.append(snapshot(f"updates={applied:,}"))
    else:
        for checkpoint in sorted(UPDATE_CHECKPOINTS):
            while applied < checkpoint:
                ds = lance.dataset(uri)
                idx = int(update_targets[applied])
                ds.update(
                    {"payload": "payload + 1"},
                    where=f"row_idx = {idx}",
                )
                applied += 1
            out.append(snapshot(f"updates={applied:,}"))

    if COMPACT_TARGET_ROWS > 0:
        ds = lance.dataset(uri)
        ds.optimize.compact_files(target_rows_per_fragment=COMPACT_TARGET_ROWS)
        out.append(snapshot(f"after compact ({COMPACT_TARGET_ROWS:,} rows/frag)"))

    return out


def _fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / 1024 / 1024:.1f} MB"


def test_steady_state_size_report(
    steady_state_snapshots: List[Snapshot], data_dir: Path
):
    """Print a Markdown table of point-in-time sizes; not a timing bench."""
    base_total = NUM_FRAGMENTS * ROWS_PER_FRAG

    lines: List[str] = []
    lines.append("")
    lines.append("## Row ID Index Steady-State Size")
    lines.append("")
    lines.append(
        f"`num_fragments={NUM_FRAGMENTS}`, `rows_per_frag={ROWS_PER_FRAG}` "
        f"(total base rows = {base_total:,}). Each row holds an `int64 "
        f"row_idx` and `int64 payload`. **Point-in-time** sizes reference "
        f"only the current manifest; older `_versions/` entries are "
        f"excluded. Sizing is row-count-independent at the per-fragment "
        f"level, so these numbers project to a 1B-row dataset by leaving "
        f"`num_fragments` at its target."
    )
    lines.append("")
    lines.append(
        "| state | fragments "
        "| manifest_now | data_now | deletion_now "
        "| in-memory index | total point-in-time |"
    )
    lines.append(
        "|-------|-----------"
        "|--------------|----------|---------------"
        "|-----------------|----------------------|"
    )
    for s in steady_state_snapshots:
        lines.append(
            f"| {s.label} | {s.fragment_count:,} "
            f"| {_fmt_bytes(s.manifest_now)} | {_fmt_bytes(s.data_now)} "
            f"| {_fmt_bytes(s.deletion_now)} "
            f"| {_fmt_bytes(s.in_memory)} "
            f"| {_fmt_bytes(s.total_now)} |"
        )

    table = "\n".join(lines)
    print(table)
    (data_dir / "_row_id_index_steady_state_report.md").write_text(table + "\n")
