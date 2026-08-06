#!/usr/bin/env python3
"""Source-aware syllable coverage inventory helpers.

The lookup vocabulary can contain valid syllables that do not occur in a
particular source corpus.  These helpers establish the attainable target set
directly from the annotated candidate pool before recovery or final curation.
"""

from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .syllable_stats import _SKIP_TOKENS


def _inventory_worker(file_paths: list[str]) -> tuple[int, Counter]:
    """Return record count and source syllable sentence frequencies."""
    records_scanned = 0
    frequency: Counter = Counter()
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                records_scanned += 1
                for syllable in set(record.get("unique_syllables", [])):
                    if syllable not in _SKIP_TOKENS:
                        frequency[syllable] += 1
    return records_scanned, frequency


def build_source_syllable_inventory(
    pool_dir: str | Path,
    output_path: str | Path,
    *,
    max_workers: int | None = None,
) -> dict:
    """Scan the pool in parallel and write the attainable syllable targets."""
    import os

    pool_files = sorted(Path(pool_dir).glob("pool_chunk_*.jsonl"))
    if not pool_files:
        raise FileNotFoundError(f"No pool_chunk_*.jsonl files in {pool_dir}")

    if max_workers is None:
        max_workers = os.cpu_count() or 4
    workers = min(max(1, max_workers), len(pool_files))
    tasks = [
        [str(path) for path in pool_files[worker_id::workers]]
        for worker_id in range(workers)
    ]

    print(
        f"▸ Building source coverage inventory from {len(pool_files)} chunks "
        f"with {workers} worker(s)..."
    )
    records_scanned = 0
    sentence_frequency: Counter = Counter()
    if workers == 1:
        worker_records, worker_frequency = _inventory_worker(tasks[0])
        records_scanned += worker_records
        sentence_frequency.update(worker_frequency)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for worker_records, worker_frequency in executor.map(_inventory_worker, tasks):
                records_scanned += worker_records
                sentence_frequency.update(worker_frequency)

    syllables = sorted(sentence_frequency)
    inventory = {
        "records_scanned": records_scanned,
        "unique_syllables": len(syllables),
        "syllables": syllables,
        "syllable_sentence_frequency": dict(sorted(sentence_frequency.items())),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(inventory, f, ensure_ascii=False, indent=2)

    print(f"✓ Source inventory: {len(syllables):,} attainable syllables")
    print(f"  Records scanned: {records_scanned:,}")
    print(f"  Output: {output_path}")
    return inventory


def load_coverage_targets(path: str | Path) -> set[str]:
    """Read a source inventory JSON, JSON syllable list, or one-per-line text."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        targets = {line.strip() for line in text.splitlines() if line.strip()}
    else:
        if isinstance(payload, dict):
            values = payload.get("syllables")
            if values is None:
                values = payload.get("syllable_sentence_frequency", {}).keys()
        elif isinstance(payload, list):
            values = payload
        else:
            raise ValueError("Coverage targets must be a JSON list or object with 'syllables'")
        targets = {str(value) for value in values if str(value)}

    targets -= _SKIP_TOKENS
    if not targets:
        raise ValueError(f"No coverage targets found in {path}")
    return targets
