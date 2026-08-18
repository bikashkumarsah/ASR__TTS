#!/usr/bin/env python3
"""Stage 3b: Distribution analysis and reporting.

Produces per-batch JSON reports with syllable balance metrics
and metadata distribution crosstabs.
"""

from __future__ import annotations

import json
from collections import Counter
from itertools import product
from pathlib import Path

from syllable_metrics import distribution_statistics, rarity_counts

from .syllable_stats import compute_syllable_stats, _coefficient_of_variation, _SKIP_TOKENS

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent

# Metadata axis definitions
TENSE_LABELS = ("past", "present", "future", "mixed")
POLARITY_LABELS = ("positive", "negative", "neutral")
GENDER_LABELS = ("masculine", "feminine", "neutral")
SECTOR_LABELS = ("news", "literature", "formal", "conversational",
                  "sports", "technology", "health", "business",
                  "entertainment", "education")

METADATA_AXES = {
    "tense": TENSE_LABELS,
    "polarity": POLARITY_LABELS,
    "gender": GENDER_LABELS,
    "sector": SECTOR_LABELS,
}


def generate_batch_report(
    batch_records: list[dict],
    batch_id: int,
    corpus_state: dict | None = None,
    reports_dir: str | Path | None = None,
    pool_dir: str | Path | None = None,
) -> dict:
    """Generate a comprehensive distribution report for a batch.

    Parameters
    ----------
    batch_records : list of selected records for this batch
    batch_id : batch number
    corpus_state : cumulative state (for trend analysis)
    reports_dir : where to save the report JSON
    pool_dir : pool directory (for pool-level stats comparison)

    Returns
    -------
    dict with the full report
    """
    if reports_dir is None:
        reports_dir = _PROJECT_ROOT / "dataset" / "asr_corpus" / "reports"
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "batch_id": batch_id,
        "batch_size": len(batch_records),
    }

    # ---- Metadata distribution ----
    meta_counts = {}
    for axis, labels in METADATA_AXES.items():
        counts = Counter(rec.get(axis, "unknown") for rec in batch_records)
        total = sum(counts.values())
        ideal = total / len(labels) if labels else 0

        dist = {}
        for label in labels:
            c = counts.get(label, 0)
            pct = (c / total * 100) if total > 0 else 0
            deviation = ((c - ideal) / ideal * 100) if ideal > 0 else 0
            dist[label] = {
                "count": c,
                "pct": round(pct, 2),
                "deviation_from_uniform_pct": round(deviation, 2),
            }
        meta_counts[axis] = dist

    report["metadata_distribution"] = meta_counts

    # ---- Metadata crosstab (tense × gender) ----
    crosstab = {}
    for t, g in product(TENSE_LABELS, GENDER_LABELS):
        key = f"{t}×{g}"
        crosstab[key] = sum(
            1 for r in batch_records
            if r.get("tense") == t and r.get("gender") == g
        )
    report["crosstab_tense_gender"] = crosstab

    # ---- Syllable balance metrics ----
    syll_freq = Counter()
    for rec in batch_records:
        for tok in rec.get("syllables", []):
            if tok not in _SKIP_TOKENS:
                syll_freq[tok] += 1

    total_tokens = sum(syll_freq.values())
    unique_syls = len(syll_freq)
    distribution = distribution_statistics(syll_freq)

    # Coverage against lookup vocab
    try:
        import sys
        sys.path.insert(0, str(_SCRIPT_DIR.parent))
        from syllabic_tokenizer import get_lookup_tokens
        lookup = frozenset(
            token for token in get_lookup_tokens(
                str(_PROJECT_ROOT / "dataset" / "nepali_syllables_lookup.vocab")
            )
            if token.strip()
        )
        covered = sum(1 for s in lookup if s in syll_freq)
        coverage = covered / len(lookup) if lookup else 0
    except Exception:
        coverage = 0
        covered = 0

    # Sentence length stats
    lengths = [rec.get("syllable_count", 0) for rec in batch_records]
    lengths.sort()
    median_idx = len(lengths) // 2

    report["syllable_metrics"] = {
        "total_tokens": total_tokens,
        "unique_syllables": unique_syls,
        "cv": distribution["coefficient_of_variation"],
        "entropy_bits": distribution["entropy_bits"],
        "normalized_entropy": distribution["normalized_entropy"],
        "gini": distribution["gini"],
        "rarity_counts": rarity_counts(syll_freq),
        "coverage": round(coverage, 4),
        "coverage_pct": round(coverage * 100, 2),
        "covered_count": covered,
    }

    report["sentence_length"] = {
        "min": min(lengths) if lengths else 0,
        "max": max(lengths) if lengths else 0,
        "median": lengths[median_idx] if lengths else 0,
        "mean": round(sum(lengths) / len(lengths), 1) if lengths else 0,
    }

    # ---- Top 20 most / least frequent syllables ----
    most_common = syll_freq.most_common(20)
    least_common = syll_freq.most_common()[:-21:-1] if len(syll_freq) >= 20 else []
    report["top_syllables"] = [{"syllable": s, "count": c} for s, c in most_common]
    report["rare_syllables"] = [{"syllable": s, "count": c} for s, c in least_common]

    # ---- Cumulative trend (if corpus_state available) ----
    if corpus_state:
        report["cumulative"] = {
            "total_selected": corpus_state.get("total_selected", 0),
            "cumulative_cv": corpus_state.get("cumulative_cv", None),
            "batches_completed": corpus_state.get("batches_completed", 0),
        }

    # ---- Save ----
    report_path = reports_dir / f"batch_{batch_id:03d}_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Report saved: {report_path}")
    _print_report_summary(report)

    return report


def _print_report_summary(report: dict):
    """Pretty-print key report metrics."""
    batch_id = report.get("batch_id", "?")
    size = report.get("batch_size", 0)
    print(f"\n{'='*60}")
    print(f"  BATCH {batch_id} REPORT  ({size} sentences)")
    print(f"{'='*60}")

    # Metadata distribution
    meta = report.get("metadata_distribution", {})
    for axis, dist in meta.items():
        print(f"\n  {axis.upper()}:")
        for label, info in dist.items():
            bar = "█" * int(info["pct"] / 3)
            print(f"    {label:18s} {info['count']:>6d} ({info['pct']:5.1f}%) "
                  f"[{info['deviation_from_uniform_pct']:+6.1f}%] {bar}")

    # Syllable metrics
    sm = report.get("syllable_metrics", {})
    print(f"\n  SYLLABLE METRICS:")
    print(f"    Unique syllables : {sm.get('unique_syllables', 0)}")
    print(f"    Normalized entropy: {sm.get('normalized_entropy', 0):.6f}")
    print(f"    Gini              : {sm.get('gini', 0):.6f}")
    print(f"    CV (diagnostic)   : {sm.get('cv', 0):.4f}")
    print(f"    Coverage         : {sm.get('coverage_pct', 0):.1f}%")

    # Sentence length
    sl = report.get("sentence_length", {})
    print(f"\n  SENTENCE LENGTH:")
    print(f"    Min / Median / Max : {sl.get('min',0)} / {sl.get('median',0)} / {sl.get('max',0)}")
    print(f"    Mean               : {sl.get('mean', 0)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate distribution report")
    parser.add_argument("--batch-file", type=str, required=True,
                        help="Path to a batch JSONL file")
    parser.add_argument("--batch-id", type=int, required=True)
    parser.add_argument("--reports-dir", type=str, default=None)

    args = parser.parse_args()

    records = []
    with open(args.batch_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    generate_batch_report(records, args.batch_id, reports_dir=args.reports_dir)
