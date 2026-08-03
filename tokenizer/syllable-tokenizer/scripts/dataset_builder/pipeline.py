#!/usr/bin/env python3
"""Stage 5: Pipeline CLI — incremental batch runner.

Orchestrates the full pipeline:
  1. Extract candidate pool from source files
     (Nepali-Text-Corpus parquet, compiled.txt, Source_book.txt)
  2. Annotate pool with rule-based metadata
  3. Select balanced batch (5k sentences)
  4. Generate distribution report
  5. Update corpus_state.json
  6. Merge all batches into final corpus
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_SCRIPT_DIR.parent))

# Import pipeline modules
from dataset_builder.extract import extract_pool
from dataset_builder.annotate import annotate_pool
from dataset_builder.balance import select_balanced_batch
from dataset_builder.analyze import generate_batch_report


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_CORPUS_DIR = _PROJECT_ROOT / "dataset" / "asr_corpus"
_POOL_DIR = _CORPUS_DIR / "pool"
_BATCHES_DIR = _CORPUS_DIR / "batches"
_REPORTS_DIR = _CORPUS_DIR / "reports"
_STATE_FILE = _CORPUS_DIR / "corpus_state.json"

# Default path for the Nepali-Text-Corpus (sibling directory)
_DEFAULT_NEPALI_CORPUS = _PROJECT_ROOT.parent / "Nepali-Text-Corpus"


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

def _load_state() -> dict:
    """Load corpus state from JSON, or return empty state."""
    if _STATE_FILE.exists():
        with open(_STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_state(state: dict):
    """Save corpus state to JSON."""
    _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Don't serialize the huge syllable freq table for readability;
    # keep it but also save a compact version
    with open(_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    print(f"  State saved: {_STATE_FILE}")


def _load_pool_file_worker(pf_str: str) -> list[dict]:
    """Read one pool chunk JSONL file (top-level for ProcessPoolExecutor)."""
    records = []
    with open(pf_str, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_pool_sequential(
    pool_files: list[Path],
    max_records: int | None,
) -> list[dict]:
    """Load pool chunks one file at a time, stopping early when max_records is reached."""
    records = []
    for pf in pool_files:
        with open(pf, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
                    if max_records and len(records) >= max_records:
                        return records
    return records


def _load_pool(
    max_records: int | None = 1_000_000,
    *,
    max_workers: int | None = None,
    show_progress: bool = True,
) -> list[dict]:
    """Load pool records from JSONL chunks, using parallel I/O when beneficial."""
    import os
    from concurrent.futures import ProcessPoolExecutor

    pool_files = sorted(_POOL_DIR.glob("pool_chunk_*.jsonl"))
    if not pool_files:
        return []

    if max_workers is None:
        max_workers = os.cpu_count() or 4
    max_workers = max(1, max_workers)

    # Single file or single worker — sequential is simpler
    if len(pool_files) == 1 or max_workers <= 1:
        return _load_pool_sequential(pool_files, max_records)

    # Small cap — sequential early-stop avoids reading the entire corpus
    if max_records and max_records <= 50_000:
        return _load_pool_sequential(pool_files, max_records)

    workers = min(max_workers, len(pool_files))
    print(f"▸ Loading {len(pool_files)} pool chunks using {workers} parallel processes...")

    try:
        from tqdm import tqdm
        use_tqdm = show_progress
    except ImportError:
        use_tqdm = False

    tasks = [str(pf) for pf in pool_files]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        if use_tqdm:
            chunks = list(tqdm(
                executor.map(_load_pool_file_worker, tasks),
                total=len(tasks),
                desc="Loading pool chunks",
                unit=" file",
            ))
        else:
            chunks = list(executor.map(_load_pool_file_worker, tasks))

    records: list[dict] = []
    for chunk_records in chunks:
        records.extend(chunk_records)
        if max_records and len(records) >= max_records:
            return records[:max_records]
    return records


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_run_batch(args):
    """Run a single batch: extract (if needed) → annotate → select → report."""
    batch_id = args.batch_id
    target_size = args.target_size

    print(f"\n{'='*60}")
    print(f"  RUNNING BATCH {batch_id}  (target: {target_size} sentences)")
    print(f"{'='*60}")

    # ---- Step 1: Extract pool (only if pool is empty or --force-extract) ----
    pool_files = list(_POOL_DIR.glob("pool_chunk_*.jsonl"))
    if not pool_files or args.force_extract:
        print("\n▸ Stage 1: Extracting candidate pool...")

        # Resolve source paths
        nepali_corpus_path = None
        if args.nepali_corpus:
            nepali_corpus_path = Path(args.nepali_corpus)
        else:
            candidates = [
                _PROJECT_ROOT / "Nepali-Text-Corpus",
                _PROJECT_ROOT.parent / "Nepali-Text-Corpus",
                _PROJECT_ROOT.parent.parent / "Nepali-Text-Corpus",
                Path.cwd() / "Nepali-Text-Corpus",
                Path.cwd().parent / "Nepali-Text-Corpus",
            ]
            for cand in candidates:
                if cand.exists():
                    nepali_corpus_path = cand
                    print(f"  Auto-detected Nepali-Text-Corpus: {nepali_corpus_path}")
                    break

        compiled_path = Path(args.pool_source) if args.pool_source else _PROJECT_ROOT / "compiled.txt"
        if not compiled_path.exists():
            compiled_path = None

        book_path = Path(args.supplement) if args.supplement else _PROJECT_ROOT / "Source_book.txt"
        if not book_path.exists():
            book_path = None

        extract_pool(
            compiled_path,
            book_path,
            _POOL_DIR,
            nepali_corpus_path=nepali_corpus_path,
            max_compiled=args.max_compiled,
            max_corpus=args.max_corpus,
            max_shards=args.max_shards,
            min_syllables=args.min_syllables,
            max_syllables=args.max_syllables,
            max_workers=getattr(args, "workers", None),
        )
    else:
        print(f"\n▸ Stage 1: Pool already exists ({len(pool_files)} chunks). "
              f"Use --force-extract to re-extract.")

    # ---- Step 2: Annotate ----
    max_chunks = None if (args.max_chunks is not None and args.max_chunks <= 0) else args.max_chunks
    max_pool_records = None if (args.max_pool_records is not None and args.max_pool_records <= 0) else args.max_pool_records

    pool_records = _load_pool(
        max_records=max_pool_records,
        max_workers=getattr(args, "workers", None),
    )
    needs_annotation = any("tense" not in r for r in pool_records[:10])

    if needs_annotation:
        print("\n▸ Stage 2: Annotating pool with metadata...")
        rules_path = Path(args.rules) if args.rules else None
        annotate_pool(_POOL_DIR, rules_path=rules_path, max_chunks=max_chunks,
                      max_workers=getattr(args, "workers", None))
        # Reload after annotation
        pool_records = _load_pool(
            max_records=max_pool_records,
            max_workers=getattr(args, "workers", None),
        )
    else:
        print("\n▸ Stage 2: Pool already annotated. Skipping.")

    # ---- Step 3: Select balanced batch ----
    print(f"\n▸ Stage 3: Selecting {target_size} balanced sentences...")
    state = _load_state()

    selected, updated_state = select_balanced_batch(
        pool_records,
        target_size=target_size,
        corpus_state=state,
        seed=args.seed,
        max_workers=getattr(args, "workers", None),
    )

    if not selected:
        print("⚠ No sentences selected. Check pool size and state.")
        return

    # ---- Step 4: Assign final IDs and batch_id ----
    start_id = updated_state.get("total_selected", 0) - len(selected)
    for i, rec in enumerate(selected):
        rec["id"] = f"asr_{start_id + i:08d}"
        rec["batch_id"] = batch_id

    # ---- Step 5: Write batch file ----
    _BATCHES_DIR.mkdir(parents=True, exist_ok=True)
    batch_path = _BATCHES_DIR / f"batch_{batch_id:03d}.jsonl"
    with open(batch_path, "w", encoding="utf-8") as f:
        for rec in selected:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\n✓ Batch file: {batch_path}")

    # ---- Step 6: Generate report ----
    print("\n▸ Stage 4: Generating distribution report...")
    generate_batch_report(
        selected, batch_id,
        corpus_state=updated_state,
        reports_dir=_REPORTS_DIR,
    )

    # ---- Step 7: Save state ----
    _save_state(updated_state)

    print(f"\n{'='*60}")
    print(f"  BATCH {batch_id} COMPLETE")
    print(f"  Selected: {len(selected)} sentences")
    print(f"  Cumulative: {updated_state.get('total_selected', 0)} sentences")
    print(f"{'='*60}\n")


def cmd_analyze(args):
    """Analyze an existing batch without selecting new sentences."""
    batch_id = args.batch_id
    batch_path = _BATCHES_DIR / f"batch_{batch_id:03d}.jsonl"

    if not batch_path.exists():
        print(f"⚠ Batch file not found: {batch_path}")
        return

    records = []
    with open(batch_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    state = _load_state()
    generate_batch_report(records, batch_id, corpus_state=state, reports_dir=_REPORTS_DIR)


def cmd_merge(args):
    """Merge all batch files into a single corpus JSONL."""
    output_path = Path(args.output)
    batch_files = sorted(_BATCHES_DIR.glob("batch_*.jsonl"))

    if not batch_files:
        print("⚠ No batch files found to merge.")
        return

    total = 0
    seen_ids = set()

    with open(output_path, "w", encoding="utf-8") as out_f:
        for bf in batch_files:
            with open(bf, "r", encoding="utf-8") as in_f:
                for line in in_f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    rec_id = rec.get("id", rec.get("pool_id"))
                    if rec_id in seen_ids:
                        continue
                    seen_ids.add(rec_id)
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    total += 1

    print(f"\n✓ Merged {total} sentences from {len(batch_files)} batches")
    print(f"  Output: {output_path}")


def cmd_status(args):
    """Print current corpus status."""
    state = _load_state()
    if not state:
        print("No corpus state found. Run a batch first.")
        return

    print(f"\n{'='*60}")
    print(f"  CORPUS STATUS")
    print(f"{'='*60}")
    print(f"  Total selected     : {state.get('total_selected', 0)}")
    print(f"  Batches completed  : {state.get('batches_completed', 0)}")
    print(f"  Cumulative CV      : {state.get('cumulative_cv', 'N/A')}")
    print(f"  Unique syllables   : {state.get('cumulative_unique_syllables', 'N/A')}")

    meta = state.get("cumulative_meta_counts", {})
    for axis, counts in meta.items():
        print(f"\n  {axis.upper()}:")
        total = sum(counts.values())
        for label, count in sorted(counts.items()):
            pct = count / total * 100 if total > 0 else 0
            print(f"    {label:18s} {count:>6d} ({pct:5.1f}%)")


def cmd_reset(args):
    """Reset corpus state and clear previous batches/pool for a fresh start."""
    import shutil

    print("\n⚠ Resetting corpus state...")
    for d in [_POOL_DIR, _BATCHES_DIR, _REPORTS_DIR]:
        if d.exists():
            shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)
            print(f"  Cleared: {d}")
    if _STATE_FILE.exists():
        _STATE_FILE.unlink()
        print(f"  Removed: {_STATE_FILE}")

    merged = _CORPUS_DIR / "corpus_50k.jsonl"
    if merged.exists():
        merged.unlink()
        print(f"  Removed: {merged}")

    print("✓ Reset complete. Ready for fresh extraction.\n")


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Balanced ASR Dataset Builder Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Batch 1 with Nepali-Text-Corpus (auto-detected or explicit)
  python -m dataset_builder.pipeline run-batch --batch-id 1 --target-size 5000

  # Batch 1 with explicit corpus path + compiled.txt
  python -m dataset_builder.pipeline run-batch --batch-id 1 --target-size 5000 \\
    --nepali-corpus /path/to/Nepali-Text-Corpus \\
    --pool-source compiled.txt --supplement Source_book.txt

  # Batch 2: reuse pool, exclude prior selections
  python -m dataset_builder.pipeline run-batch --batch-id 2 --target-size 5000

  # Analyze batch 1
  python -m dataset_builder.pipeline analyze --batch-id 1

  # Merge all batches into final corpus
  python -m dataset_builder.pipeline merge --output dataset/asr_corpus/corpus_50k.jsonl

  # Check status
  python -m dataset_builder.pipeline status

  # Reset everything for a fresh start
  python -m dataset_builder.pipeline reset
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="Pipeline command")

    # ---- run-batch ----
    p_batch = subparsers.add_parser("run-batch", help="Run extraction + selection for one batch")
    p_batch.add_argument("--batch-id", type=int, required=True, help="Batch number (1-10)")
    p_batch.add_argument("--target-size", type=int, default=5000, help="Sentences per batch")
    p_batch.add_argument("--nepali-corpus", type=str, default=None,
                         help="Path to Nepali-Text-Corpus directory (parquet). "
                              "Auto-detected if sibling directory exists.")
    p_batch.add_argument("--pool-source", type=str, default=None, help="Path to compiled.txt")
    p_batch.add_argument("--supplement", type=str, default=None, help="Path to Source_book.txt")
    p_batch.add_argument("--max-compiled", type=int, default=100_000,
                         help="Max sentences from compiled.txt")
    p_batch.add_argument("--max-corpus", type=int, default=200_000,
                         help="Max sentences from Nepali-Text-Corpus")
    p_batch.add_argument("--max-shards", type=int, default=None,
                         help="Max parquet shards to read (for testing; default: all)")
    p_batch.add_argument("--max-chunks", type=int, default=30,
                         help="Max pool chunks to annotate (0 = all)")
    p_batch.add_argument("--max-pool-records", type=int, default=1_000_000,
                         help="Max candidate sentences to load for selection (0 = all)")
    p_batch.add_argument("--min-syllables", type=int, default=5)
    p_batch.add_argument("--max-syllables", type=int, default=80)
    p_batch.add_argument("--rules", type=str, default=None, help="Path to rules.yaml")
    p_batch.add_argument("--seed", type=int, default=42, help="Random seed")
    p_batch.add_argument("--force-extract", action="store_true",
                         help="Force re-extraction even if pool exists")
    p_batch.add_argument("--workers", type=int, default=None,
                         help="Parallel worker processes (default: CPU count)")
    p_batch.set_defaults(func=cmd_run_batch)

    # ---- analyze ----
    p_analyze = subparsers.add_parser("analyze", help="Analyze an existing batch")
    p_analyze.add_argument("--batch-id", type=int, required=True)
    p_analyze.set_defaults(func=cmd_analyze)

    # ---- merge ----
    p_merge = subparsers.add_parser("merge", help="Merge all batches into final corpus")
    p_merge.add_argument("--output", type=str,
                         default=str(_CORPUS_DIR / "corpus_50k.jsonl"),
                         help="Output JSONL path")
    p_merge.set_defaults(func=cmd_merge)

    # ---- status ----
    p_status = subparsers.add_parser("status", help="Print corpus status")
    p_status.set_defaults(func=cmd_status)

    # ---- reset ----
    p_reset = subparsers.add_parser("reset", help="Reset corpus state for fresh start")
    p_reset.set_defaults(func=cmd_reset)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
