# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
"""On-disk size measurements for the stable row id index.

Builds minimal datasets parametrized by ``num_fragments`` x
``update_count`` x ``deletion_fraction`` x ``stable on/off`` and reports
the on-disk size breakdown. The delta between the stable_on and
stable_off variant at matched shape is the cost of the feature on disk.

What this measures
------------------

- ``manifest``  ─ ``_versions/N.manifest`` files. The per-fragment
  ``RowIdMeta::Inline`` blobs live here, so this is where the row id
  index pays its on-disk cost.
- ``transaction`` ─ ``_transactions/*.txn`` files. One per commit; each
  ``update`` and ``delete`` creates one. Tracked separately so we don't
  conflate per-commit overhead with the per-fragment row id metadata.
- ``data`` ─ ``data/*.lance`` files. Same for stable on/off at matched
  shape (only the fragment metadata differs, not the data).
- ``deletion`` / ``index`` / ``other`` ─ for completeness.

In-memory ``RowIdIndex`` size is reported via the
``row_id_index_size_bytes`` binding (``DeepSizeOf`` on the assembled
index). For ``stable_off`` datasets that returns ``None``; we record 0
bytes there so the table compares cleanly.

This file is a report, not a timing benchmark. Run with ``-s`` to see
the table::

    pytest python/benchmarks/test_row_id_index_size.py -s
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


def _env_float_tuple(name: str, default: Tuple[float, ...]) -> Tuple[float, ...]:
    raw = os.environ.get(name)
    if not raw:
        return default
    return tuple(float(x) for x in raw.split(","))


ROWS_PER_FRAG = _env_int("BENCH_SIZE_ROWS_PER_FRAG", 100)
FRAG_COUNTS: Tuple[int, ...] = _env_int_tuple(
    "BENCH_SIZE_FRAG_COUNTS", (10, 100, 1000)
)
UPDATE_COUNTS: Tuple[int, ...] = _env_int_tuple(
    "BENCH_SIZE_UPDATE_COUNTS", (0, 1000)
)
DELETION_FRACTIONS: Tuple[float, ...] = _env_float_tuple(
    "BENCH_SIZE_DELETION_FRACTIONS", (0.0, 0.5)
)

# ----------------------------------------------------------------------------


class SizeReport(NamedTuple):
    stable: bool
    num_fragments: int
    updates: int
    deletion_fraction: float
    total: int
    manifest: int
    transaction: int
    data: int
    deletion: int
    index: int
    other: int
    in_memory_index: int


def _dataset_dir(
    data_dir: Path,
    stable: bool,
    num_frags: int,
    updates: int,
    deletion_frac: float,
) -> Path:
    return data_dir / (
        f"size_f{num_frags}_u{updates}_d{int(deletion_frac * 100)}_s{int(stable)}"
    )


def _classify(path: Path) -> str:
    s = str(path)
    name = path.name
    if "_versions" in s or name.endswith(".manifest"):
        return "manifest"
    if "_transactions" in s or name.endswith(".txn"):
        return "transaction"
    if "_indices" in s:
        return "index"
    if "_deletions" in s:
        return "deletion"
    if name.endswith(".lance"):
        return "data"
    return "other"


def _measure(uri: Path) -> Dict[str, int]:
    sizes = {
        "manifest": 0,
        "transaction": 0,
        "data": 0,
        "deletion": 0,
        "index": 0,
        "other": 0,
    }
    for p in Path(uri).rglob("*"):
        if not p.is_file():
            continue
        sizes[_classify(p)] += p.stat().st_size
    sizes["total"] = sum(sizes.values())
    return sizes


def _build_dataset(
    data_dir: Path,
    stable: bool,
    num_frags: int,
    updates: int,
    deletion_frac: float,
) -> Path:
    uri = _dataset_dir(data_dir, stable, num_frags, updates, deletion_frac)
    if uri.exists():
        return uri

    num_rows = num_frags * ROWS_PER_FRAG
    schema = pa.schema(
        [
            pa.field("row_idx", pa.int64(), nullable=False),
            pa.field("payload", pa.int64(), nullable=False),
        ]
    )

    def gen():
        for f in range(num_frags):
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
        enable_stable_row_ids=stable,
    )

    if deletion_frac > 0.0:
        # Delete a contiguous prefix of every fragment: row_idx % rows_per_frag
        # is the offset within the fragment, so this targets every fragment
        # uniformly without per-fragment commits.
        threshold = max(1, int(ROWS_PER_FRAG * deletion_frac))
        ds = lance.dataset(uri)
        ds.delete(f"row_idx % {ROWS_PER_FRAG} < {threshold}")

    if updates > 0:
        rng = np.random.default_rng(42)
        targets = rng.choice(num_rows, size=updates, replace=False)
        for idx in targets:
            ds = lance.dataset(uri)
            ds.update(
                {"payload": "payload + 1"},
                where=f"row_idx = {int(idx)}",
            )

    return uri


@pytest.fixture(scope="module")
def size_reports(data_dir: Path) -> List[SizeReport]:
    out: List[SizeReport] = []
    for num_frags in FRAG_COUNTS:
        for updates in UPDATE_COUNTS:
            for deletion_frac in DELETION_FRACTIONS:
                for stable in (False, True):
                    uri = _build_dataset(
                        data_dir, stable, num_frags, updates, deletion_frac
                    )
                    sizes = _measure(uri)
                    ds = lance.dataset(uri)
                    in_memory = ds.row_id_index_size_bytes() if stable else None
                    out.append(
                        SizeReport(
                            stable=stable,
                            num_fragments=num_frags,
                            updates=updates,
                            deletion_fraction=deletion_frac,
                            total=sizes["total"],
                            manifest=sizes["manifest"],
                            transaction=sizes["transaction"],
                            data=sizes["data"],
                            deletion=sizes["deletion"],
                            index=sizes["index"],
                            other=sizes["other"],
                            in_memory_index=in_memory or 0,
                        )
                    )
    return out


def _fmt_bytes(n: int) -> str:
    sign = "-" if n < 0 else ""
    n = abs(n)
    if n < 1024:
        return f"{sign}{n} B"
    if n < 1024 * 1024:
        return f"{sign}{n / 1024:.1f} KB"
    return f"{sign}{n / 1024 / 1024:.1f} MB"


def test_row_id_index_size_report(
    size_reports: List[SizeReport], data_dir: Path
):
    """Print a table of on-disk sizes; not a real benchmark, just a report.

    Also writes the table to ``<data_dir>/_row_id_index_size_report.md`` so
    the output is preserved regardless of pytest stdout capture.
    """
    grouped: Dict[Tuple[int, int, float], Dict[bool, SizeReport]] = {}
    for r in size_reports:
        key = (r.num_fragments, r.updates, r.deletion_fraction)
        grouped.setdefault(key, {})[r.stable] = r

    lines: List[str] = []
    lines.append("")
    lines.append("## Row ID Index On-Disk Size")
    lines.append("")
    lines.append(
        f"`rows_per_frag={ROWS_PER_FRAG}`. "
        "Each row holds an `int64 row_idx` and `int64 payload` column. "
        "Sizes are measured on the dataset directory after a fresh build "
        "(no compaction)."
    )
    lines.append("")
    lines.append(
        "| frags | upd | del% "
        "| manifest_off | manifest_on | manifest_Δ "
        "| txn_off | txn_on | txn_Δ "
        "| in-mem index "
        "| total_off | total_on | total_Δ |"
    )
    lines.append(
        "|-------|-----|------"
        "|--------------|-------------|------------"
        "|---------|--------|--------"
        "|--------------"
        "|-----------|----------|---------|"
    )
    for key in sorted(grouped):
        num_frags, updates, deletion_frac = key
        bucket = grouped[key]
        if False not in bucket or True not in bucket:
            continue
        off = bucket[False]
        on = bucket[True]
        lines.append(
            f"| {num_frags} | {updates} | {int(deletion_frac * 100)}% "
            f"| {_fmt_bytes(off.manifest)} | {_fmt_bytes(on.manifest)} "
            f"| {_fmt_bytes(on.manifest - off.manifest)} "
            f"| {_fmt_bytes(off.transaction)} | {_fmt_bytes(on.transaction)} "
            f"| {_fmt_bytes(on.transaction - off.transaction)} "
            f"| {_fmt_bytes(on.in_memory_index)} "
            f"| {_fmt_bytes(off.total)} | {_fmt_bytes(on.total)} "
            f"| {_fmt_bytes(on.total - off.total)} |"
        )

    table = "\n".join(lines)
    print(table)

    out_path = data_dir / "_row_id_index_size_report.md"
    out_path.write_text(table + "\n")
