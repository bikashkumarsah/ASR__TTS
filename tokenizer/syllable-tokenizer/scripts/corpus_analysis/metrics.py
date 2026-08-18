"""Tokenization workers, mergeable metrics, and deterministic report writers."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from syllabic_tokenizer import clean_text, tokenize
from syllable_metrics import distribution_statistics


def normalize_text(value: str) -> str:
    """Apply the analysis-wide canonical normalization before hashing."""
    return unicodedata.normalize("NFC", clean_text(unicodedata.normalize("NFC", value)))


def _devanagari_count(value: str) -> int:
    return sum("\u0900" <= char <= "\u097f" for char in value)


def _metadata_value(value: object) -> str:
    if value is None or value == "":
        return "unknown"
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


@dataclass
class RecordMetrics:
    digest: bytes
    frequencies: dict[str, int]
    normalized_characters: int
    devanagari_characters: int
    token_count: int
    unmatched_devanagari_characters: int


@dataclass
class Accumulator:
    input_records: int = 0
    usable_records: int = 0
    empty_records: int = 0
    rejected_records: int = 0
    total_tokens: int = 0
    devanagari_characters: int = 0
    unmatched_devanagari_characters: int = 0
    frequencies: Counter = field(default_factory=Counter)
    normalized_length_histogram: Counter = field(default_factory=Counter)
    devanagari_length_histogram: Counter = field(default_factory=Counter)
    token_length_histogram: Counter = field(default_factory=Counter)
    metadata_counts: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))

    def merge(self, other: "Accumulator") -> None:
        for field_name in (
            "input_records",
            "usable_records",
            "empty_records",
            "rejected_records",
            "total_tokens",
            "devanagari_characters",
            "unmatched_devanagari_characters",
        ):
            setattr(self, field_name, getattr(self, field_name) + getattr(other, field_name))
        self.frequencies.update(other.frequencies)
        self.normalized_length_histogram.update(other.normalized_length_histogram)
        self.devanagari_length_histogram.update(other.devanagari_length_histogram)
        self.token_length_histogram.update(other.token_length_histogram)
        for axis, counts in other.metadata_counts.items():
            self.metadata_counts[axis].update(counts)

    def add_unique_record(self, record: RecordMetrics) -> None:
        """Add a usable already-tokenized record to the deduplicated union."""
        self.input_records += 1
        self.usable_records += 1
        self.total_tokens += record.token_count
        self.devanagari_characters += record.devanagari_characters
        self.unmatched_devanagari_characters += record.unmatched_devanagari_characters
        self.frequencies.update(record.frequencies)
        self.normalized_length_histogram[record.normalized_characters] += 1
        self.devanagari_length_histogram[record.devanagari_characters] += 1
        self.token_length_histogram[record.token_count] += 1


def analyze_batch(records: list[dict], lookup_vocab: frozenset[str]) -> tuple[Accumulator, list[RecordMetrics]]:
    """Worker entry point: normalize and tokenize a bounded record batch."""
    accumulator = Accumulator()
    details: list[RecordMetrics] = []
    for record in records:
        accumulator.input_records += 1
        raw_text = record.get("text")
        if not isinstance(raw_text, str):
            accumulator.rejected_records += 1
            continue
        normalized = normalize_text(raw_text)
        if not normalized:
            accumulator.empty_records += 1
            continue

        all_tokens = tokenize(normalized, lookup_vocab, debug=False)
        syllables = [token for token in all_tokens if token.strip() and _devanagari_count(token)]
        frequencies = Counter(syllables)
        deva_characters = _devanagari_count(normalized)
        recognized_characters = sum(_devanagari_count(token) for token in syllables)
        unmatched = max(0, deva_characters - recognized_characters)

        accumulator.usable_records += 1
        accumulator.total_tokens += len(syllables)
        accumulator.devanagari_characters += deva_characters
        accumulator.unmatched_devanagari_characters += unmatched
        accumulator.frequencies.update(frequencies)
        accumulator.normalized_length_histogram[len(normalized)] += 1
        accumulator.devanagari_length_histogram[deva_characters] += 1
        accumulator.token_length_histogram[len(syllables)] += 1
        for axis, value in record.get("metadata", {}).items():
            accumulator.metadata_counts[axis][_metadata_value(value)] += 1

        details.append(
            RecordMetrics(
                digest=hashlib.sha256(normalized.encode("utf-8")).digest(),
                frequencies=dict(frequencies),
                normalized_characters=len(normalized),
                devanagari_characters=deva_characters,
                token_count=len(syllables),
                unmatched_devanagari_characters=unmatched,
            )
        )
    return accumulator, details


def _histogram_statistics(histogram: Counter) -> dict:
    count = sum(histogram.values())
    if not count:
        return {key: 0 for key in ("count", "min", "max", "mean", "median", "p95", "std_dev")}
    total = sum(value * frequency for value, frequency in histogram.items())
    mean = total / count
    variance = sum(((value - mean) ** 2) * frequency for value, frequency in histogram.items()) / count

    def percentile(fraction: float) -> int:
        target = max(1, math.ceil(count * fraction))
        cumulative = 0
        for value, frequency in sorted(histogram.items()):
            cumulative += frequency
            if cumulative >= target:
                return value
        return max(histogram)

    return {
        "count": count,
        "min": min(histogram),
        "max": max(histogram),
        "mean": round(mean, 4),
        "median": percentile(0.5),
        "p95": percentile(0.95),
        "std_dev": round(math.sqrt(variance), 4),
    }


def _frequency_statistics(frequencies: Counter) -> dict:
    return distribution_statistics(frequencies)


def _cumulative_coverage(frequencies: Counter) -> dict[str, int]:
    total = sum(frequencies.values())
    thresholds = (50, 80, 90, 95, 99)
    result = {str(threshold): 0 for threshold in thresholds}
    if not total:
        return result
    cumulative = 0
    threshold_index = 0
    for rank, count in enumerate(sorted(frequencies.values(), reverse=True), 1):
        cumulative += count
        while threshold_index < len(thresholds) and cumulative / total * 100 >= thresholds[threshold_index]:
            result[str(thresholds[threshold_index])] = rank
            threshold_index += 1
    return result


def build_summary(name: str, accumulator: Accumulator, lookup_vocab: frozenset[str]) -> tuple[dict, dict]:
    frequencies = accumulator.frequencies
    observed = set(frequencies)
    sorted_frequency = sorted(frequencies.items(), key=lambda item: (-item[1], item[0]))
    least = sorted(frequencies.items(), key=lambda item: (item[1], item[0]))
    vocabulary_size = len(lookup_vocab)
    summary = {
        "name": name,
        "total_syllable_tokens": accumulator.total_tokens,
        "unique_observed_syllables": len(observed),
        "lookup_vocabulary_size": vocabulary_size,
        "lookup_vocabulary_coverage": round(len(observed) / vocabulary_size, 6) if vocabulary_size else 0,
        "lookup_vocabulary_coverage_pct": round(len(observed) / vocabulary_size * 100, 4) if vocabulary_size else 0,
        "observed_syllables": sorted(observed),
        "most_frequent": [{"syllable": token, "count": count} for token, count in sorted_frequency[:20]],
        "least_frequent": [{"syllable": token, "count": count} for token, count in least[:20]],
        "hapax_syllables": sorted(token for token, count in frequencies.items() if count == 1),
        "unobserved_syllable_count": len(lookup_vocab.difference(observed)),
        "frequency_statistics": _frequency_statistics(frequencies),
        "cumulative_token_coverage": _cumulative_coverage(frequencies),
    }
    quality = {
        "input_records": accumulator.input_records,
        "usable_records": accumulator.usable_records,
        "empty_after_normalization": accumulator.empty_records,
        "rejected_non_string_records": accumulator.rejected_records,
        "normalized_character_length": _histogram_statistics(accumulator.normalized_length_histogram),
        "devanagari_character_length": _histogram_statistics(accumulator.devanagari_length_histogram),
        "syllable_token_length": _histogram_statistics(accumulator.token_length_histogram),
        "total_devanagari_characters": accumulator.devanagari_characters,
        "unmatched_devanagari_characters": accumulator.unmatched_devanagari_characters,
        "unmatched_devanagari_ratio": round(
            accumulator.unmatched_devanagari_characters / accumulator.devanagari_characters, 8
        ) if accumulator.devanagari_characters else 0,
    }
    return summary, quality


def write_result(
    name: str,
    accumulator: Accumulator,
    lookup_vocab: frozenset[str],
    output_dir: str | Path,
    figures_dir: str | Path,
    figure_slug: str,
) -> tuple[dict, dict]:
    """Write the complete JSON/CSV/Parquet/text report for one result view."""
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    summary, quality = build_summary(name, accumulator, lookup_vocab)
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "quality_metrics.json", quality)

    total = accumulator.total_tokens
    rows = [
        {
            "syllable": syllable,
            "count": accumulator.frequencies.get(syllable, 0),
            "relative_frequency": accumulator.frequencies.get(syllable, 0) / total if total else 0,
            "observed": syllable in accumulator.frequencies,
        }
        for syllable in sorted(lookup_vocab, key=lambda token: (-accumulator.frequencies.get(token, 0), token))
    ]
    _write_csv(output_dir / "syllable_frequency.csv", rows)
    _write_parquet(output_dir / "syllable_frequency.parquet", rows)
    with open(output_dir / "unobserved_syllables.txt", "w", encoding="utf-8") as handle:
        for syllable in sorted(lookup_vocab.difference(accumulator.frequencies)):
            handle.write(syllable + "\n")

    metadata_rows = []
    metadata_json = {}
    for axis in sorted(accumulator.metadata_counts):
        counts = accumulator.metadata_counts[axis]
        metadata_json[axis] = dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))
        axis_total = sum(counts.values())
        for value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            metadata_rows.append({
                "axis": axis,
                "value": value,
                "records": count,
                "relative_frequency": count / axis_total if axis_total else 0,
            })
    _write_json(output_dir / "metadata_breakdown.json", metadata_json)
    _write_csv(
        output_dir / "metadata_breakdown.csv",
        metadata_rows,
        fieldnames=["axis", "value", "records", "relative_frequency"],
    )
    _write_frequency_svg(
        figures_dir / f"{figure_slug}_frequency_rank.svg",
        name,
        [count for _, count in sorted(accumulator.frequencies.items(), key=lambda item: -item[1])],
    )
    _write_coverage_svg(
        figures_dir / f"{figure_slug}_cumulative_coverage.svg",
        name,
        [count for _, count in sorted(accumulator.frequencies.items(), key=lambda item: -item[1])],
    )
    return summary, quality


def _write_json(path: Path, value: object) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with open(path, "w", encoding="utf-8", newline="") as handle:
        if not fieldnames:
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_parquet(path: Path, rows: list[dict]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema([
        ("syllable", pa.string()),
        ("count", pa.int64()),
        ("relative_frequency", pa.float64()),
        ("observed", pa.bool_()),
    ])
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path, compression="zstd")


def _svg_polyline(points: Iterable[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def _write_frequency_svg(path: Path, title: str, counts: list[int]) -> None:
    width, height, margin = 900, 500, 60
    if counts:
        max_log = math.log10(max(counts) + 1)
        points = [
            (
                margin + index / max(1, len(counts) - 1) * (width - 2 * margin),
                height - margin - math.log10(count + 1) / max(1e-12, max_log) * (height - 2 * margin),
            )
            for index, count in enumerate(counts)
        ]
    else:
        points = []
    svg = _base_svg(
        title + " — frequency by rank (log scale)",
        width,
        height,
        _svg_polyline(points),
        "Syllable rank",
        "log10(count + 1)",
    )
    path.write_text(svg, encoding="utf-8")


def _write_coverage_svg(path: Path, title: str, counts: list[int]) -> None:
    width, height, margin = 900, 500, 60
    total = sum(counts)
    cumulative = 0
    points = []
    for index, count in enumerate(counts):
        cumulative += count
        points.append((
            margin + index / max(1, len(counts) - 1) * (width - 2 * margin),
            height - margin - (cumulative / total if total else 0) * (height - 2 * margin),
        ))
    svg = _base_svg(
        title + " — cumulative token coverage",
        width,
        height,
        _svg_polyline(points),
        "Syllables ordered by frequency",
        "Cumulative fraction",
    )
    path.write_text(svg, encoding="utf-8")


def _base_svg(title: str, width: int, height: int, points: str, x_label: str, y_label: str) -> str:
    escaped = (
        title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <text x="{width / 2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="16">{escaped}</text>
  <line x1="60" y1="440" x2="840" y2="440" stroke="#555"/>
  <line x1="60" y1="60" x2="60" y2="440" stroke="#555"/>
  <polyline points="{points}" fill="none" stroke="#2864dc" stroke-width="2"/>
  <text x="450" y="485" text-anchor="middle" font-family="sans-serif" font-size="13">{x_label}</text>
  <text x="18" y="250" transform="rotate(-90 18 250)" text-anchor="middle" font-family="sans-serif" font-size="13">{y_label}</text>
</svg>
'''
