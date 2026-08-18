"""Shared syllable-distribution metrics used by analysis and corpus building."""

from __future__ import annotations

import math
from collections.abc import Mapping


def distribution_statistics(
    frequencies: Mapping[str, int | float],
    *,
    inventory: set[str] | frozenset[str] | None = None,
) -> dict[str, int | float]:
    """Return comparable balance statistics over a fixed inventory.

    Missing entries are represented by zero when ``inventory`` is supplied.
    That makes baseline/new-corpus comparisons valid even when coverage differs.
    """
    keys = sorted(inventory if inventory is not None else frequencies)
    counts = [float(frequencies.get(key, 0)) for key in keys]
    total = sum(counts)
    size = len(counts)
    if not counts or total <= 0:
        return {
            "mean": 0,
            "median": 0,
            "std_dev": 0,
            "coefficient_of_variation": 0,
            "entropy_bits": 0,
            "normalized_entropy": 0,
            "gini": 0,
        }

    ordered = sorted(counts)
    mean = total / size
    variance = sum((value - mean) ** 2 for value in counts) / size
    median = ordered[size // 2] if size % 2 else (
        ordered[size // 2 - 1] + ordered[size // 2]
    ) / 2
    entropy = -sum(
        (value / total) * math.log2(value / total)
        for value in counts
        if value > 0
    )
    max_entropy = math.log2(size) if size > 1 else 0.0
    weighted_sum = sum((index + 1) * value for index, value in enumerate(ordered))
    gini = (2 * weighted_sum) / (size * total) - (size + 1) / size
    return {
        "mean": round(mean, 4),
        "median": median,
        "std_dev": round(math.sqrt(variance), 4),
        "coefficient_of_variation": round(math.sqrt(variance) / mean, 4) if mean else 0,
        "entropy_bits": round(entropy, 6),
        "normalized_entropy": round(entropy / max_entropy, 8) if max_entropy else 0,
        "gini": round(max(0.0, gini), 8),
    }


def jensen_shannon_divergence(
    frequencies: Mapping[str, int | float],
    target: Mapping[str, int | float],
    *,
    inventory: set[str] | frozenset[str] | None = None,
) -> float:
    """Return base-2 Jensen-Shannon divergence in the range [0, 1]."""
    keys = sorted(inventory if inventory is not None else set(frequencies) | set(target))
    observed_total = sum(max(0.0, float(frequencies.get(key, 0))) for key in keys)
    target_total = sum(max(0.0, float(target.get(key, 0))) for key in keys)
    if not keys or observed_total <= 0 or target_total <= 0:
        return 0.0

    observed = [max(0.0, float(frequencies.get(key, 0))) / observed_total for key in keys]
    expected = [max(0.0, float(target.get(key, 0))) / target_total for key in keys]
    midpoint = [(left + right) / 2 for left, right in zip(observed, expected)]

    def _kl(left: list[float], right: list[float]) -> float:
        return sum(
            value * math.log2(value / reference)
            for value, reference in zip(left, right)
            if value > 0 and reference > 0
        )

    return round((_kl(observed, midpoint) + _kl(expected, midpoint)) / 2, 8)


def rarity_counts(frequencies: Mapping[str, int | float]) -> dict[str, int]:
    """Count observed types at the reporting thresholds."""
    counts = [float(value) for value in frequencies.values()]
    return {
        "hapax": sum(value == 1 for value in counts),
        "below_2": sum(value < 2 for value in counts),
        "below_5": sum(value < 5 for value in counts),
        "below_10": sum(value < 10 for value in counts),
        "below_20": sum(value < 20 for value in counts),
    }
