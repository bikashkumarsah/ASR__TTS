#!/usr/bin/env python3
"""Stage 3a: Syllable frequency statistics.

Computes per-sentence and corpus-level syllable frequency metrics,
including coefficient of variation and vocab coverage.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_SCRIPT_DIR.parent))

from syllabic_tokenizer import get_lookup_tokens  # noqa: E402
from syllable_metrics import distribution_statistics, rarity_counts  # noqa: E402

# Punctuation/whitespace tokens to exclude from syllable frequency counts
_SKIP_TOKENS = frozenset({" ", "।", "?", "!", "\t", "\n", ""})


def compute_syllable_stats(
    pool_dir: str | Path | None = None,
    vocab_path: str | Path | None = None,
) -> dict:
    """Compute corpus-level syllable frequency statistics from pool files.

    Returns a dict with:
        - syllable_freq: {syllable: count}
        - total_tokens: int
        - unique_count: int
        - coverage: float (% of lookup vocab represented)
        - cv: float (coefficient of variation)
        - rare_syllables: list of syllables below threshold
    """
    if pool_dir is None:
        pool_dir = _PROJECT_ROOT / "dataset" / "asr_corpus" / "pool"
    pool_dir = Path(pool_dir)

    if vocab_path is None:
        vocab_path = _PROJECT_ROOT / "dataset" / "nepali_syllables_lookup.vocab"

    lookup_vocab = frozenset(
        token for token in get_lookup_tokens(str(vocab_path))
        if token.strip()
    )

    freq = Counter()
    total_sentences = 0

    pool_files = sorted(pool_dir.glob("pool_chunk_*.jsonl"))
    for pf in pool_files:
        with open(pf, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                total_sentences += 1
                for tok in rec.get("syllables", []):
                    if tok not in _SKIP_TOKENS:
                        freq[tok] += 1

    total_tokens = sum(freq.values())
    unique_count = len(freq)
    vocab_size = len(lookup_vocab)

    # Coverage: what fraction of the lookup vocab appears at least once
    covered = sum(1 for s in lookup_vocab if s in freq)
    coverage = covered / vocab_size if vocab_size > 0 else 0.0

    distribution = distribution_statistics(freq)

    # Rare syllables (below average / 10)
    mean_freq = total_tokens / unique_count if unique_count > 0 else 0
    rare_threshold = max(1, mean_freq / 10)
    rare_syllables = [s for s, c in freq.items() if c < rare_threshold]

    return {
        "syllable_freq": dict(freq.most_common()),
        "total_tokens": total_tokens,
        "unique_count": unique_count,
        "vocab_size": vocab_size,
        "coverage": round(coverage, 4),
        "coverage_pct": round(coverage * 100, 2),
        "cv": distribution["coefficient_of_variation"],
        "entropy_bits": distribution["entropy_bits"],
        "normalized_entropy": distribution["normalized_entropy"],
        "gini": distribution["gini"],
        "rarity_counts": rarity_counts(freq),
        "rare_count": len(rare_syllables),
        "rare_threshold": round(rare_threshold, 1),
        "total_sentences": total_sentences,
    }


def compute_cumulative_stats(
    corpus_state: dict,
    new_batch_records: list[dict],
) -> dict:
    """Update cumulative syllable frequency with a new batch.

    Parameters
    ----------
    corpus_state : existing corpus_state.json content (or empty dict)
    new_batch_records : list of annotated records from the new batch

    Returns updated stats dict.
    """
    freq = Counter(corpus_state.get("cumulative_syllable_freq", {}))

    for rec in new_batch_records:
        for tok in rec.get("syllables", []):
            if tok not in _SKIP_TOKENS:
                freq[tok] += 1

    total = sum(freq.values())
    unique = len(freq)
    distribution = distribution_statistics(freq)

    return {
        "cumulative_syllable_freq": dict(freq),
        "cumulative_total_tokens": total,
        "cumulative_unique_syllables": unique,
        "cumulative_cv": distribution["coefficient_of_variation"],
        "cumulative_entropy_bits": distribution["entropy_bits"],
        "cumulative_normalized_entropy": distribution["normalized_entropy"],
        "cumulative_gini": distribution["gini"],
    }


def _coefficient_of_variation(values: list[int | float]) -> float:
    """Compute CV = std_dev / mean."""
    if not values:
        return 0.0
    n = len(values)
    mean = sum(values) / n
    if mean == 0:
        return 0.0
    variance = sum((v - mean) ** 2 for v in values) / n
    std_dev = variance ** 0.5
    return std_dev / mean


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute syllable frequency stats")
    parser.add_argument("--pool-dir", type=str, default=None)
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path")

    args = parser.parse_args()
    stats = compute_syllable_stats(args.pool_dir)

    # Don't write full freq table to stdout
    display = {k: v for k, v in stats.items() if k != "syllable_freq"}
    print(json.dumps(display, indent=2, ensure_ascii=False))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Full stats written to {args.output}")
