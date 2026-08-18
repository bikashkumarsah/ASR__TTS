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
from dataset_builder.balance import (
    build_cell_pools_streaming,
    build_syllable_candidate_pools_streaming,
    select_balanced_batch,
)
from dataset_builder.analyze import generate_batch_report
from dataset_builder.coverage import build_source_syllable_inventory, load_coverage_targets
from dataset_builder.diverse import (
    build_diverse_final,
    compare_prepared_pools,
    prepare_five_corpus_pool,
    update_progress_markdown,
)


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


def _pool_needs_annotation(pool_files: list[Path]) -> bool:
    """Return whether a pool contains an unannotated chunk.

    Annotation is done one file at a time, so checking the first JSONL record
    in every chunk avoids loading the corpus merely to determine its state.
    """
    required_fields = {"tense", "polarity", "gender", "sector"}
    for pool_file in pool_files:
        with open(pool_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    if not required_fields.issubset(record):
                        return True
                    break
    return False


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

    # Refresh after extraction so a newly-created pool is included below.
    pool_files = sorted(_POOL_DIR.glob("pool_chunk_*.jsonl"))

    # ---- Step 2: Annotate ----
    max_chunks = None if (args.max_chunks is not None and args.max_chunks <= 0) else args.max_chunks
    needs_annotation = _pool_needs_annotation(pool_files)

    if needs_annotation:
        print("\n▸ Stage 2: Annotating pool with metadata...")
        rules_path = Path(args.rules) if args.rules else None
        annotate_pool(_POOL_DIR, rules_path=rules_path, max_chunks=max_chunks,
                      max_workers=getattr(args, "workers", None))
        if _pool_needs_annotation(pool_files):
            raise RuntimeError(
                "Pool annotation is incomplete. Re-run with --max-chunks 0 "
                "before selecting a balanced batch."
            )
    else:
        print("\n▸ Stage 2: Pool already annotated. Skipping.")

    # ---- Step 3: Select balanced batch ----
    print(f"\n▸ Stage 3: Selecting {target_size} balanced sentences...")
    state = _load_state()
    selected_ids = frozenset(state.get("selected_ids", []))
    coverage_targets = None
    coverage_candidate_pools = None
    if args.coverage_targets:
        coverage_targets = load_coverage_targets(args.coverage_targets)
        uncovered_targets = coverage_targets.difference(
            state.get("cumulative_syllable_freq", {})
        )
        print(
            f"▸ Source coverage target: {len(coverage_targets):,} syllables; "
            f"{len(uncovered_targets):,} currently absent"
        )
        coverage_candidate_pools = build_syllable_candidate_pools_streaming(
            _POOL_DIR,
            selected_ids,
            uncovered_targets,
            max_per_syllable=args.coverage_candidates_per_syllable,
            seed=args.seed,
            max_workers=getattr(args, "workers", None),
        )
    cell_pools = build_cell_pools_streaming(
        _POOL_DIR,
        selected_ids,
        max_per_cell=args.max_candidates_per_cell,
        seed=args.seed,
        max_workers=getattr(args, "workers", None),
    )

    selected, updated_state = select_balanced_batch(
        target_size=target_size,
        corpus_state=state,
        cell_pools=cell_pools,
        seed=args.seed,
        coverage_priority=args.coverage_priority,
        coverage_targets=coverage_targets,
        coverage_candidate_pools=coverage_candidate_pools,
        require_full_coverage=args.require_full_coverage,
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


def cmd_coverage_inventory(args):
    """Write the syllables actually attainable from the existing pool."""
    build_source_syllable_inventory(
        _POOL_DIR,
        args.output,
        max_workers=getattr(args, "workers", None),
    )


def cmd_prepare_five_corpus_pool(args):
    prepare_five_corpus_pool(
        corpus_config=args.config,
        input_root=args.input_root,
        output_dir=args.output_dir,
        diverse_config=args.diverse_config,
        candidate_limit=args.candidate_limit,
        workers=args.workers,
        resume=args.resume,
        max_records_per_corpus=args.max_records_per_corpus,
    )


def cmd_build_diverse_final(args):
    baseline = args.baseline
    if baseline is None:
        default_baseline = _CORPUS_DIR / "final_50k_all_syllables.jsonl"
        baseline = str(default_baseline) if default_baseline.exists() else None
    build_diverse_final(
        pool_dir=args.pool_dir,
        output=args.output,
        target_size=args.target_size,
        diverse_config=args.diverse_config,
        workers=args.workers,
        resume=args.resume,
        baseline=baseline,
        seed=args.seed,
    )


def cmd_compare_prepared_pools(args):
    compare_prepared_pools(args.left, args.right)


def cmd_update_progress(args):
    update_progress_markdown(
        run_report=args.run_report,
        progress_file=args.progress_file,
    )


def _metadata_balance_summary(
    records: list[dict],
    expected_labels: dict[str, list[str]],
) -> dict:
    """Measure marginal metadata balance against labels present in the pool."""
    summary = {}
    total = len(records)
    for axis, labels in expected_labels.items():
        counts = {label: 0 for label in labels}
        for record in records:
            label = record.get(axis, "unknown")
            if label in counts:
                counts[label] += 1
        ideal_pct = 100 / len(labels) if labels else 0
        deviations = {
            label: round(abs((count / total * 100) - ideal_pct), 4) if total else 0.0
            for label, count in counts.items()
        }
        summary[axis] = {
            "counts": counts,
            "ideal_pct": round(ideal_pct, 4),
            "max_abs_deviation_pct_points": max(deviations.values(), default=0.0),
            "abs_deviation_pct_points": deviations,
        }
    return summary


def cmd_build_final(args):
    """Create a fresh 50k corpus with required source-syllable coverage.

    This command is intentionally non-destructive: it reads the annotated
    cloud pool but does not change incremental batches or corpus_state.json.
    """
    target_size = args.target_size
    target_syllables = load_coverage_targets(args.coverage_targets)
    pool_files = sorted(_POOL_DIR.glob("pool_chunk_*.jsonl"))
    if not pool_files:
        raise FileNotFoundError(f"No candidate pool found at {_POOL_DIR}")
    if _pool_needs_annotation(pool_files):
        raise RuntimeError("Final curation requires a fully annotated pool")

    print(f"\n{'='*60}")
    print(f"  BUILDING FINAL {target_size:,}-RECORD COVERAGE CORPUS")
    print(f"{'='*60}")
    print(f"▸ Required source syllables: {len(target_syllables):,}")

    cell_pools = build_cell_pools_streaming(
        _POOL_DIR,
        frozenset(),
        max_per_cell=args.max_candidates_per_cell,
        seed=args.seed,
        max_workers=getattr(args, "workers", None),
    )
    expected_labels = {
        "tense": sorted({cell[0] for cell, pool in cell_pools.items() if pool}),
        "polarity": sorted({cell[1] for cell, pool in cell_pools.items() if pool}),
        "gender": sorted({cell[2] for cell, pool in cell_pools.items() if pool}),
        "sector": sorted({cell[3] for cell, pool in cell_pools.items() if pool}),
    }
    coverage_candidate_pools = build_syllable_candidate_pools_streaming(
        _POOL_DIR,
        frozenset(),
        target_syllables,
        max_per_syllable=args.coverage_candidates_per_syllable,
        seed=args.seed,
        max_workers=getattr(args, "workers", None),
    )
    selected, final_state = select_balanced_batch(
        target_size=target_size,
        corpus_state={},
        cell_pools=cell_pools,
        seed=args.seed,
        coverage_priority=args.coverage_priority,
        coverage_targets=target_syllables,
        coverage_candidate_pools=coverage_candidate_pools,
        require_full_coverage=True,
        max_workers=getattr(args, "workers", None),
    )
    if len(selected) != target_size:
        raise RuntimeError(
            f"Final curation selected {len(selected):,}/{target_size:,} records; no output written"
        )

    covered = {
        syllable
        for record in selected
        for syllable in record.get("unique_syllables", [])
    }
    missing = target_syllables.difference(covered)
    if missing:
        raise RuntimeError(
            f"Final coverage verification failed: {len(missing)} syllable(s) missing"
        )

    metadata_balance = _metadata_balance_summary(selected, expected_labels)
    tolerance_pct_points = args.max_metadata_deviation * 100
    violations = [
        f"{axis}={info['max_abs_deviation_pct_points']:.2f}pp"
        for axis, info in metadata_balance.items()
        if info["max_abs_deviation_pct_points"] > tolerance_pct_points
    ]
    if violations:
        raise RuntimeError(
            "Final metadata balance verification failed "
            f"(limit {tolerance_pct_points:.2f} percentage points): "
            + ", ".join(violations)
        )

    for index, record in enumerate(selected):
        record["id"] = f"final_{index:08d}"
        record["batch_id"] = "final_50k"

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in selected:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    report_path = Path(args.report_output)
    report = generate_batch_report(
        selected,
        batch_id=0,
        corpus_state=final_state,
        reports_dir=report_path.parent,
    )
    report["source_coverage"] = {
        "target_syllables": len(target_syllables),
        "covered_syllables": len(target_syllables),
        "missing_syllables": [],
        "coverage_pct": 100.0,
    }
    report["final_curation"] = {
        "target_size": target_size,
        "coverage_required": True,
        "max_candidates_per_cell": args.max_candidates_per_cell,
        "coverage_candidates_per_syllable": args.coverage_candidates_per_syllable,
        "max_metadata_deviation_pct_points": tolerance_pct_points,
    }
    report["final_metadata_balance"] = metadata_balance
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"✓ Final corpus written: {output_path}")
    print(f"✓ Final coverage report: {report_path}")
    print(f"  Source-syllable coverage: {len(target_syllables):,}/{len(target_syllables):,} (100.0%)")
    print(f"  Metadata balance: passed within {tolerance_pct_points:.2f} percentage points")


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
    print(f"  Normalized entropy : {state.get('cumulative_normalized_entropy', 'N/A')}")
    print(f"  Gini               : {state.get('cumulative_gini', 'N/A')}")
    print(f"  CV (diagnostic)    : {state.get('cumulative_cv', 'N/A')}")
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
    p_batch.add_argument("--batch-id", type=int, required=True,
                         help="New batch number (for example, 31+ after a 150k corpus)")
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
    p_batch.add_argument("--max-chunks", type=int, default=0,
                         help="Max pool chunks to annotate (0 = all; recommended)")
    p_batch.add_argument("--max-candidates-per-cell", type=int, default=2_000,
                         help="Bounded reservoir candidates retained per metadata cell during selection")
    p_batch.add_argument("--min-syllables", type=int, default=5)
    p_batch.add_argument("--max-syllables", type=int, default=80)
    p_batch.add_argument("--rules", type=str, default=None, help="Path to rules.yaml")
    p_batch.add_argument("--seed", type=int, default=42, help="Random seed")
    p_batch.add_argument("--coverage-priority", type=float, default=0.0,
                         help="Extra score multiplier for syllables not yet in corpus state")
    p_batch.add_argument("--coverage-targets", type=str, default=None,
                         help="Source inventory JSON or syllable list for guaranteed recovery")
    p_batch.add_argument("--coverage-candidates-per-syllable", type=int, default=256,
                         help="Bounded candidate examples retained per requested syllable")
    p_batch.add_argument("--require-full-coverage", action="store_true",
                         help="Fail the batch if any coverage target remains absent")
    p_batch.add_argument("--force-extract", action="store_true",
                         help="Force re-extraction even if pool exists")
    p_batch.add_argument("--workers", type=int, default=None,
                         help="Parallel worker processes (default: CPU count)")
    p_batch.set_defaults(func=cmd_run_batch)

    # ---- analyze ----
    p_analyze = subparsers.add_parser("analyze", help="Analyze an existing batch")
    p_analyze.add_argument("--batch-id", type=int, required=True)
    p_analyze.set_defaults(func=cmd_analyze)

    # ---- coverage-inventory ----
    p_inventory = subparsers.add_parser(
        "coverage-inventory",
        help="Scan the pool and write the source-supported syllable target set",
    )
    p_inventory.add_argument("--output", type=str, required=True,
                             help="Output JSON inventory path")
    p_inventory.add_argument("--workers", type=int, default=None,
                             help="Parallel worker processes (default: CPU count)")
    p_inventory.set_defaults(func=cmd_coverage_inventory)

    # ---- build-final ----
    p_final = subparsers.add_parser(
        "build-final",
        help="Non-destructively curate a fresh balanced final corpus with full source coverage",
    )
    p_final.add_argument("--coverage-targets", type=str, required=True,
                         help="Source inventory JSON or syllable list to cover")
    p_final.add_argument("--target-size", type=int, default=50_000,
                         help="Final corpus size")
    p_final.add_argument("--output", type=str,
                         default=str(_CORPUS_DIR / "final_50k_all_syllables.jsonl"),
                         help="Final corpus JSONL path")
    p_final.add_argument("--report-output", type=str,
                         default=str(_REPORTS_DIR / "final_50k_all_syllables_report.json"),
                         help="Final coverage and distribution report JSON path")
    p_final.add_argument("--max-candidates-per-cell", type=int, default=8_000,
                         help="Bounded reservoir candidates retained per metadata cell")
    p_final.add_argument("--coverage-candidates-per-syllable", type=int, default=256,
                         help="Bounded candidate examples retained per target syllable")
    p_final.add_argument("--coverage-priority", type=float, default=0.0,
                         help="Additional priority for uncovered syllables after coverage seed")
    p_final.add_argument("--max-metadata-deviation", type=float, default=0.02,
                         help="Maximum allowed marginal metadata deviation from uniform (fraction; default: 0.02)")
    p_final.add_argument("--seed", type=int, default=42, help="Random seed")
    p_final.add_argument("--workers", type=int, default=None,
                         help="Parallel worker processes (default: CPU count)")
    p_final.set_defaults(func=cmd_build_final)

    # ---- prepare-five-corpus-pool ----
    p_prepare_diverse = subparsers.add_parser(
        "prepare-five-corpus-pool",
        help="Prepare an exact-deduplicated, rare-aware shortlist from five corpora",
    )
    p_prepare_diverse.add_argument("--config", required=True,
                                   help="Five-corpus source configuration YAML")
    p_prepare_diverse.add_argument("--input-root", required=True,
                                   help="Root containing the downloaded raw corpora")
    p_prepare_diverse.add_argument("--output-dir", required=True,
                                   help="Checkpointed shortlist directory")
    p_prepare_diverse.add_argument(
        "--diverse-config",
        default=str(_PROJECT_ROOT / "configs" / "final_50k_diverse.yaml"),
        help="Pinned diverse-final configuration",
    )
    p_prepare_diverse.add_argument("--candidate-limit", type=int, default=None)
    p_prepare_diverse.add_argument(
        "--max-records-per-corpus",
        type=int,
        default=None,
        help="Bounded smoke-test limit for each source corpus",
    )
    p_prepare_diverse.add_argument("--workers", type=int, default=None,
                                   help="Preparation workers (default: all CPU cores)")
    p_prepare_diverse.add_argument("--resume", action="store_true")
    p_prepare_diverse.set_defaults(func=cmd_prepare_five_corpus_pool)

    p_compare_prepare = subparsers.add_parser(
        "compare-prepared-pools",
        help="Compare one-worker and all-core preparation artifacts",
    )
    p_compare_prepare.add_argument("--left", required=True)
    p_compare_prepare.add_argument("--right", required=True)
    p_compare_prepare.set_defaults(func=cmd_compare_prepared_pools)

    # ---- build-diverse-final ----
    p_diverse = subparsers.add_parser(
        "build-diverse-final",
        help="Build the calibrated semantic-diverse, rare-aware final corpus",
    )
    p_diverse.add_argument("--pool-dir", required=True)
    p_diverse.add_argument("--target-size", type=int, default=50_000)
    p_diverse.add_argument(
        "--output",
        default=str(_CORPUS_DIR / "final_50k_diverse_rare.jsonl"),
    )
    p_diverse.add_argument(
        "--diverse-config",
        default=str(_PROJECT_ROOT / "configs" / "final_50k_diverse.yaml"),
    )
    p_diverse.add_argument(
        "--baseline",
        default=None,
        help="Previous 50k JSONL for entropy, Gini, JSD and similarity acceptance",
    )
    p_diverse.add_argument("--workers", type=int, default=None,
                           help="Total CPU cores available (default: all)")
    p_diverse.add_argument("--seed", type=int, default=42)
    p_diverse.add_argument("--resume", action="store_true")
    p_diverse.set_defaults(func=cmd_build_diverse_final)

    # ---- update-progress ----
    p_progress = subparsers.add_parser(
        "update-progress",
        help="Idempotently update the authoritative Markdown progress report",
    )
    p_progress.add_argument("--run-report", required=True)
    p_progress.add_argument(
        "--progress-file",
        default=str(_PROJECT_ROOT.parent.parent / "docs" / "technical_progress_report.md"),
    )
    p_progress.set_defaults(func=cmd_update_progress)

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
