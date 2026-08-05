#!/usr/bin/env python3
"""Stage 4: Balanced selection algorithm.

Two-objective greedy selection:
  4a. Metadata stratification (hard constraint) — round-robin fill from
      underfilled cells across tense × polarity × gender × sector.
  4b. Syllable-type frequency scoring (soft objective) — within each cell,
      pick sentences that most improve syllable balance.
  4c. Incremental rebalancing — score against cumulative deficits.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

from .syllable_stats import _SKIP_TOKENS

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent

# Metadata axis labels (must match annotate.py)
TENSE_LABELS = ("past", "present", "future", "mixed")
POLARITY_LABELS = ("positive", "negative", "neutral")
GENDER_LABELS = ("masculine", "feminine", "neutral")
SECTOR_LABELS = ("news", "literature", "formal", "conversational",
                  "sports", "technology", "health", "business",
                  "entertainment", "education")


def _cell_key(record: dict) -> tuple:
    """Create a metadata cell key from a record."""
    return (
        record.get("tense", "present"),
        record.get("polarity", "neutral"),
        record.get("gender", "neutral"),
        record.get("sector", "news"),
    )


def _syllable_deficit_score(
    record: dict,
    current_freq: Counter,
    target_per_syl: float,
) -> float:
    """Score a sentence by how much it reduces syllable deficit.

    score(s) = Σ max(0, target - current_count[syl])
               for syl in unique_syllables(s)
    """
    score = 0.0
    for syl in record.get("unique_syllables", []):
        if syl in _SKIP_TOKENS:
            continue
        deficit = target_per_syl - current_freq.get(syl, 0)
        if deficit > 0:
            score += deficit
    return score


def _pick_best_from_cell_worker(args_tuple: tuple) -> tuple:
    """Score a sample of candidates for one metadata cell (ProcessPool worker)."""
    cell, sample_records, freq_dict, target_per_syl = args_tuple
    if not sample_records:
        return cell, None

    freq = Counter(freq_dict)
    best_rec = None
    best_score = -1.0
    for rec in sample_records:
        score = _syllable_deficit_score(rec, freq, target_per_syl)
        if score > best_score:
            best_score = score
            best_rec = rec
    return cell, best_rec


def _reservoir_add(
    pool: list[dict],
    record: dict,
    seen: int,
    max_size: int,
    rng: random.Random,
) -> None:
    """Reservoir-sample a record into a fixed-size pool."""
    if max_size <= 0:
        return
    if len(pool) < max_size:
        pool.append(record)
    else:
        j = rng.randint(0, seen - 1)
        if j < max_size:
            pool[j] = record


def _merge_cell_pools(
    target: dict[tuple, list[dict]],
    source: dict[tuple, list[dict]],
    max_per_cell: int,
    rng: random.Random,
) -> None:
    """Merge worker cell pools and cap each cell with reservoir subsampling."""
    for cell, recs in source.items():
        if cell not in target:
            target[cell] = []
        pool = target[cell]
        for rec in recs:
            _reservoir_add(pool, rec, len(pool) + 1, max_per_cell, rng)
        if len(pool) > max_per_cell:
            target[cell] = rng.sample(pool, max_per_cell)


def _stream_pool_files_worker(args_tuple: tuple) -> dict[tuple, list[dict]]:
    """Stream a subset of pool chunk files; reservoir-sample per metadata cell."""
    file_paths, selected_ids, max_per_cell, seed = args_tuple
    rng = random.Random(seed)
    selected_set = frozenset(selected_ids)
    cell_pools: dict[tuple, list[dict]] = defaultdict(list)
    cell_counts: dict[tuple, int] = defaultdict(int)

    for pf in file_paths:
        with open(pf, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("pool_id") in selected_set:
                    continue
                cell = _cell_key(rec)
                cell_counts[cell] += 1
                _reservoir_add(cell_pools[cell], rec, cell_counts[cell], max_per_cell, rng)

    return dict(cell_pools)


def build_cell_pools_streaming(
    pool_dir: str | Path,
    selected_ids: set[str] | frozenset[str],
    *,
    max_per_cell: int = 2000,
    seed: int | None = 42,
    max_workers: int | None = None,
    show_progress: bool = True,
) -> dict[tuple, list[dict]]:
    """Build metadata cell pools by streaming chunk files (memory-safe).

    Uses reservoir sampling so RAM stays bounded (~max_per_cell × num_cells)
    even when the on-disk pool has 100M+ sentences.
    """
    import os
    from concurrent.futures import ProcessPoolExecutor

    pool_dir = Path(pool_dir)
    pool_files = sorted(pool_dir.glob("pool_chunk_*.jsonl"))
    if not pool_files:
        return {}

    if max_workers is None:
        max_workers = os.cpu_count() or 4
    max_workers = max(1, max_workers)

    print(
        f"▸ Streaming {len(pool_files)} pool chunks "
        f"(max {max_per_cell} candidates/cell, RAM-safe)..."
    )

    try:
        from tqdm import tqdm
        use_tqdm = show_progress
    except ImportError:
        use_tqdm = False

    rng = random.Random(seed)
    selected_list = list(selected_ids)

    # Sequential path for small pools or single worker
    if len(pool_files) <= 4 or max_workers <= 1:
        cell_pools: dict[tuple, list[dict]] = defaultdict(list)
        cell_counts: dict[tuple, int] = defaultdict(int)
        file_iter = pool_files
        if use_tqdm:
            file_iter = tqdm(pool_files, desc="Streaming pool", unit=" file")
        for pf in file_iter:
            with open(pf, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    if rec.get("pool_id") in selected_ids:
                        continue
                    cell = _cell_key(rec)
                    cell_counts[cell] += 1
                    _reservoir_add(cell_pools[cell], rec, cell_counts[cell], max_per_cell, rng)
        return dict(cell_pools)

    # Parallel: split files across workers, merge with capped reservoirs
    workers = min(max_workers, len(pool_files))
    chunk_size = (len(pool_files) + workers - 1) // workers
    tasks = []
    for worker_id in range(workers):
        batch = pool_files[worker_id * chunk_size:(worker_id + 1) * chunk_size]
        if not batch:
            continue
        tasks.append((
            [str(p) for p in batch],
            selected_list,
            max_per_cell,
            (seed or 0) + worker_id,
        ))

    merged: dict[tuple, list[dict]] = {}
    with ProcessPoolExecutor(max_workers=len(tasks)) as executor:
        if use_tqdm:
            partials = list(tqdm(
                executor.map(_stream_pool_files_worker, tasks),
                total=len(tasks),
                desc="Streaming pool",
                unit=" worker",
            ))
        else:
            partials = list(executor.map(_stream_pool_files_worker, tasks))

    for partial in partials:
        _merge_cell_pools(merged, partial, max_per_cell, rng)

    total_candidates = sum(len(v) for v in merged.values())
    print(f"  Reservoir pools ready: {len(merged)} cells, {total_candidates:,} candidates in RAM")
    return merged


def _remove_from_cell_pool(cell_pools: dict[tuple, list[dict]], cell: tuple, chosen: dict) -> None:
    """Remove a chosen record from a cell pool by pool_id."""
    pool = cell_pools.get(cell, [])
    pid = chosen.get("pool_id")
    for i, rec in enumerate(pool):
        if rec.get("pool_id") == pid:
            pool.pop(i)
            return


def _apply_selection(
    chosen: dict,
    cell: tuple,
    *,
    selected: list[dict],
    cell_pools: dict[tuple, list[dict]],
    cell_selected_counts: dict[tuple, int],
    cumulative_syl_freq: Counter,
) -> None:
    """Commit one chosen record and update cumulative syllable counts."""
    _remove_from_cell_pool(cell_pools, cell, chosen)
    selected.append(chosen)
    cell_selected_counts[cell] += 1
    for tok in chosen.get("syllables", []):
        if tok not in _SKIP_TOKENS:
            cumulative_syl_freq[tok] += 1


def _select_sequential(
    *,
    target_size: int,
    cells_queue: list[tuple],
    cell_pools: dict[tuple, list[dict]],
    cumulative_syl_freq: Counter,
    target_per_syl: float,
    cell_selected_counts: dict[tuple, int],
    show_progress: bool,
) -> list[dict]:
    """Original single-threaded round-robin greedy selection."""
    selected: list[dict] = []

    try:
        from tqdm import tqdm
        pbar = tqdm(total=target_size, desc="Selecting", unit=" sents") if show_progress else None
    except ImportError:
        pbar = None

    rounds = 0
    max_rounds = target_size * 2
    while len(selected) < target_size and rounds < max_rounds:
        rounds += 1
        progress_made = False

        for cell in cells_queue:
            if len(selected) >= target_size:
                break

            pool = cell_pools.get(cell, [])
            if not pool:
                continue

            best_idx = -1
            best_score = -1.0
            sample_size = min(len(pool), 200)
            sample_indices = random.sample(range(len(pool)), sample_size)

            for idx in sample_indices:
                rec = pool[idx]
                score = _syllable_deficit_score(rec, cumulative_syl_freq, target_per_syl)
                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx >= 0:
                chosen = pool[best_idx]
                _apply_selection(
                    chosen, cell,
                    selected=selected,
                    cell_pools=cell_pools,
                    cell_selected_counts=cell_selected_counts,
                    cumulative_syl_freq=cumulative_syl_freq,
                )
                progress_made = True
                if pbar:
                    pbar.update(1)

        if not progress_made:
            break

    if pbar:
        pbar.close()
    return selected


def _select_parallel_rounds(
    *,
    target_size: int,
    cells_queue: list[tuple],
    cell_pools: dict[tuple, list[dict]],
    cumulative_syl_freq: Counter,
    target_per_syl: float,
    cell_selected_counts: dict[tuple, int],
    max_workers: int,
    show_progress: bool,
) -> list[dict]:
    """Round-parallel greedy: all metadata cells pick concurrently each round.

    Each round snapshots cumulative syllable frequencies, scores cells in
    parallel, then merges picks and updates global state.  This preserves
    metadata round-robin stratification while parallelizing the scoring work.
    """
    import os
    from concurrent.futures import ProcessPoolExecutor

    selected: list[dict] = []
    workers = max(1, max_workers)

    try:
        from tqdm import tqdm
        pbar = tqdm(total=target_size, desc="Selecting (parallel)", unit=" sents") if show_progress else None
    except ImportError:
        pbar = None

    print(f"▸ Parallel selection: {workers} workers, round-robin across metadata cells")

    rounds = 0
    max_rounds = target_size * 2
    freq_snapshot = dict(cumulative_syl_freq)

    while len(selected) < target_size and rounds < max_rounds:
        rounds += 1
        tasks = []

        for cell in cells_queue:
            pool = cell_pools.get(cell, [])
            if not pool:
                continue
            sample_size = min(len(pool), 200)
            sample = random.sample(pool, sample_size)
            tasks.append((cell, sample, freq_snapshot, target_per_syl))

        if not tasks:
            break

        with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
            results = list(executor.map(_pick_best_from_cell_worker, tasks))

        progress_made = False
        for cell, chosen in results:
            if chosen is None or len(selected) >= target_size:
                continue
            _apply_selection(
                chosen, cell,
                selected=selected,
                cell_pools=cell_pools,
                cell_selected_counts=cell_selected_counts,
                cumulative_syl_freq=cumulative_syl_freq,
            )
            progress_made = True
            if pbar:
                pbar.update(1)

        if not progress_made:
            break

        freq_snapshot = dict(cumulative_syl_freq)

    if pbar:
        pbar.close()
    return selected


def select_balanced_batch(
    pool_records: list[dict] | None = None,
    target_size: int = 5000,
    corpus_state: dict | None = None,
    *,
    cell_pools: dict[tuple, list[dict]] | None = None,
    metadata_tolerance: float = 0.05,
    seed: int | None = 42,
    show_progress: bool = True,
    max_workers: int | None = None,
) -> tuple[list[dict], dict]:
    """Select a balanced batch from annotated pool records.

    Parameters
    ----------
    pool_records : full in-memory pool (optional if cell_pools provided)
    target_size : desired batch size
    corpus_state : existing corpus_state.json (cumulative state)
    cell_pools : pre-built metadata cell pools (from build_cell_pools_streaming)
    metadata_tolerance : ±fraction tolerance for metadata balance
    seed : random seed for reproducibility
    max_workers : parallel processes for round-robin scoring (defaults to CPU count)

    Returns
    -------
    (selected_records, updated_corpus_state)
    """
    if seed is not None:
        random.seed(seed)

    # Load cumulative state
    if corpus_state is None:
        corpus_state = {}

    selected_ids = set(corpus_state.get("selected_ids", []))
    cumulative_syl_freq = Counter(
        corpus_state.get("cumulative_syllable_freq", {})
    )
    cumulative_meta_counts = corpus_state.get("cumulative_meta_counts", {})

    # ---- Step 1: Group candidates by metadata cell ----
    if cell_pools is None:
        if pool_records is None:
            raise ValueError("Provide pool_records or cell_pools")
        candidates = [r for r in pool_records if r.get("pool_id") not in selected_ids]
        if not candidates:
            print("⚠ No candidates available (all already selected)")
            return [], corpus_state
        cell_pools = defaultdict(list)
        for rec in candidates:
            cell_pools[_cell_key(rec)].append(rec)
    else:
        if not any(cell_pools.values()):
            print("⚠ No candidates available (all already selected)")
            return [], corpus_state

    # ---- Step 2: Compute per-cell targets ----
    # All possible cells
    all_cells = list(product(TENSE_LABELS, POLARITY_LABELS, GENDER_LABELS, SECTOR_LABELS))
    # Only consider cells that have candidates
    active_cells = [c for c in all_cells if len(cell_pools.get(c, [])) > 0]

    if not active_cells:
        print("⚠ No active metadata cells with candidates")
        return [], corpus_state

    ideal_per_cell = target_size / len(active_cells) if active_cells else 0
    min_per_cell = max(1, int(ideal_per_cell * (1 - metadata_tolerance)))

    # ---- Step 3: Compute syllable target ----
    total_syl_tokens = sum(cumulative_syl_freq.values()) if cumulative_syl_freq else 1
    unique_syl_types = max(len(cumulative_syl_freq), 1)
    # After adding target_size sentences, target uniform distribution
    estimated_new_tokens = total_syl_tokens + target_size * 15  # ~15 tokens/sentence avg
    target_per_syl = estimated_new_tokens / max(unique_syl_types, 500)

    # ---- Step 4: Round-robin greedy selection ----
    selected: list[dict] = []
    cell_selected_counts: dict[tuple, int] = defaultdict(int)

    cells_queue = list(active_cells)
    random.shuffle(cells_queue)

    import os
    workers = max_workers if max_workers is not None else (os.cpu_count() or 4)
    use_parallel = workers > 1 and len(active_cells) > 1

    if use_parallel:
        selected = _select_parallel_rounds(
            target_size=target_size,
            cells_queue=cells_queue,
            cell_pools=cell_pools,
            cumulative_syl_freq=cumulative_syl_freq,
            target_per_syl=target_per_syl,
            cell_selected_counts=cell_selected_counts,
            max_workers=workers,
            show_progress=show_progress,
        )
    else:
        selected = _select_sequential(
            target_size=target_size,
            cells_queue=cells_queue,
            cell_pools=cell_pools,
            cumulative_syl_freq=cumulative_syl_freq,
            target_per_syl=target_per_syl,
            cell_selected_counts=cell_selected_counts,
            show_progress=show_progress,
        )

    # ---- Step 5: Build updated corpus state ----
    new_selected_ids = [r["pool_id"] for r in selected]
    all_selected_ids = list(selected_ids | set(new_selected_ids))

    # Update metadata counts
    meta_counts = defaultdict(lambda: defaultdict(int), 
                              {k: defaultdict(int, v) 
                               for k, v in cumulative_meta_counts.items()})
    for rec in selected:
        for axis in ("tense", "polarity", "gender", "sector"):
            meta_counts[axis][rec.get(axis, "unknown")] += 1

    total_syl_tokens = sum(cumulative_syl_freq.values())
    unique_count = len(cumulative_syl_freq)
    cv = _cv(list(cumulative_syl_freq.values()))

    updated_state = {
        "selected_ids": all_selected_ids,
        "total_selected": len(all_selected_ids),
        "cumulative_syllable_freq": dict(cumulative_syl_freq),
        "cumulative_total_tokens": total_syl_tokens,
        "cumulative_unique_syllables": unique_count,
        "cumulative_cv": round(cv, 4),
        "cumulative_meta_counts": {k: dict(v) for k, v in meta_counts.items()},
        "batches_completed": corpus_state.get("batches_completed", 0) + 1,
    }

    print(f"\n✓ Selected {len(selected)} sentences")
    print(f"  Active cells: {len(active_cells)} / {len(all_cells)}")
    print(f"  Cumulative CV: {cv:.4f}")

    return selected, updated_state


def _cv(values: list[int | float]) -> float:
    """Coefficient of variation."""
    if not values:
        return 0.0
    n = len(values)
    mean = sum(values) / n
    if mean == 0:
        return 0.0
    variance = sum((v - mean) ** 2 for v in values) / n
    return (variance ** 0.5) / mean


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Select balanced batch")
    parser.add_argument("--pool-dir", type=str, default=None)
    parser.add_argument("--target-size", type=int, default=5000)
    parser.add_argument("--state-file", type=str, default=None)
    parser.add_argument("--output", type=str, required=True,
                        help="Output batch JSONL path")
    parser.add_argument("--max-candidates-per-cell", type=int, default=2_000,
                        help="Bounded reservoir candidates retained per metadata cell")
    parser.add_argument("--workers", type=int, default=None,
                        help="Streaming/selection worker processes (default: CPU count)")

    args = parser.parse_args()

    # Build bounded pools instead of materializing every pool record in RAM.
    pool_dir = Path(args.pool_dir or (_PROJECT_ROOT / "dataset" / "asr_corpus" / "pool"))
    corpus_state = None
    if args.state_file and Path(args.state_file).exists():
        with open(args.state_file, "r", encoding="utf-8") as f:
            corpus_state = json.load(f)

    selected_ids = frozenset((corpus_state or {}).get("selected_ids", []))
    cell_pools = build_cell_pools_streaming(
        pool_dir,
        selected_ids,
        max_per_cell=args.max_candidates_per_cell,
        max_workers=args.workers,
    )

    selected, state = select_balanced_batch(
        target_size=args.target_size,
        corpus_state=corpus_state,
        cell_pools=cell_pools,
        max_workers=args.workers,
    )

    # Save batch
    with open(args.output, "w", encoding="utf-8") as f:
        for rec in selected:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"✓ Batch written: {args.output}")
