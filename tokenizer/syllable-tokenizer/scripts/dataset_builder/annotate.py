#!/usr/bin/env python3
"""Stage 2: Rule-based metadata annotation for Nepali sentences.

Reads pool JSONL, applies pattern-matching rules from rules.yaml to assign
tense, polarity, gender, and sector labels, then writes annotated JSONL.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
_DEFAULT_RULES = _SCRIPT_DIR / "rules.yaml"


# ---------------------------------------------------------------------------
# Rule loader
# ---------------------------------------------------------------------------

def load_rules(rules_path: str | Path | None = None) -> dict:
    """Load annotation rules from YAML file."""
    if rules_path is None:
        rules_path = _DEFAULT_RULES
    rules_path = Path(rules_path)
    with open(rules_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # Pre-compile all regex patterns
    compiled: dict[str, dict[str, list[re.Pattern]]] = {}
    for field, categories in raw.items():
        compiled[field] = {}
        for category, patterns in categories.items():
            compiled[field][category] = [
                re.compile(p) for p in patterns
            ]
    return compiled


# ---------------------------------------------------------------------------
# Annotation logic
# ---------------------------------------------------------------------------

def annotate_tense(text: str, rules: dict) -> str:
    """Determine tense: past, present, future, or mixed."""
    tense_rules = rules.get("tense", {})
    matches = set()

    for tense_label, patterns in tense_rules.items():
        for pat in patterns:
            if pat.search(text):
                matches.add(tense_label)
                break

    if len(matches) == 0:
        return "present"  # default
    if len(matches) == 1:
        return matches.pop()
    return "mixed"


def annotate_polarity(text: str, rules: dict) -> str:
    """Determine polarity: positive, negative, or neutral."""
    polarity_rules = rules.get("polarity", {})

    neg_count = 0
    pos_count = 0

    for pat in polarity_rules.get("negative", []):
        neg_count += len(pat.findall(text))

    for pat in polarity_rules.get("positive", []):
        pos_count += len(pat.findall(text))

    if neg_count > 0 and neg_count >= pos_count:
        return "negative"
    if pos_count > 0 and pos_count > neg_count:
        return "positive"
    return "neutral"


def annotate_gender(text: str, rules: dict) -> str:
    """Determine grammatical gender: masculine, feminine, or neutral."""
    gender_rules = rules.get("gender", {})

    fem_count = 0
    masc_count = 0

    for pat in gender_rules.get("feminine", []):
        fem_count += len(pat.findall(text))

    for pat in gender_rules.get("masculine", []):
        masc_count += len(pat.findall(text))

    if fem_count > 0 and fem_count > masc_count:
        return "feminine"
    if masc_count > 0 and masc_count > fem_count:
        return "masculine"
    return "neutral"


def annotate_sector(text: str, source_sector: str, rules: dict) -> str:
    """Determine sector with possible override from source default.

    Priority order:
      1. conversational (direct speech markers — highest override)
      2. formal (legal/governance register)
      3. Topic-specific sectors from rules (sports, technology, health,
         business, entertainment, education) — only override if the
         source_sector is generic ("news")
      4. Source-based sector from extract (news, literature, sports, etc.)
    """
    sector_rules = rules.get("sector", {})

    # Check conversational markers first (highest priority override)
    for pat in sector_rules.get("conversational", []):
        if pat.search(text):
            return "conversational"

    # Then formal markers
    for pat in sector_rules.get("formal", []):
        if pat.search(text):
            return "formal"

    # Topic-specific sectors — only override generic "news" source
    # (don't override source-assigned sports/tech/health etc.)
    if source_sector == "news":
        topic_sectors = ["sports", "technology", "health", "business",
                         "entertainment", "education"]
        best_sector = None
        best_count = 0
        for sector_name in topic_sectors:
            count = 0
            for pat in sector_rules.get(sector_name, []):
                count += len(pat.findall(text))
            if count > best_count:
                best_count = count
                best_sector = sector_name
        # Require at least 2 keyword matches to override source sector
        if best_sector and best_count >= 2:
            return best_sector

    # Fall back to source-based sector
    return source_sector


def annotate_record(record: dict, rules: dict) -> dict:
    """Apply all annotations to a single pool record.

    Adds tense, polarity, gender fields and may override sector.
    """
    text = record["text"]
    source_sector = record.get("sector", "news")

    record["tense"] = annotate_tense(text, rules)
    record["polarity"] = annotate_polarity(text, rules)
    record["gender"] = annotate_gender(text, rules)
    record["sector"] = annotate_sector(text, source_sector, rules)

    return record


# ---------------------------------------------------------------------------
# Batch annotation of pool files
# ---------------------------------------------------------------------------

def annotate_pool(
    pool_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    rules_path: str | Path | None = None,
    *,
    show_progress: bool = True,
) -> dict:
    """Annotate all pool chunk files with metadata labels.

    Parameters
    ----------
    pool_dir : directory containing pool JSONL chunk files
    output_dir : directory for annotated JSONL (defaults to same as pool_dir)
    rules_path : path to rules.yaml

    Returns
    -------
    dict with annotation statistics
    """
def _annotate_file_worker(args_tuple):
    pf_str, rules_path_str, output_dir_str = args_tuple
    pf = Path(pf_str)
    rules_path = Path(rules_path_str) if rules_path_str else None
    output_dir = Path(output_dir_str)

    rules = load_rules(rules_path)
    records = []
    file_stats = {
        "total": 0,
        "tense": {"past": 0, "present": 0, "future": 0, "mixed": 0},
        "polarity": {"positive": 0, "negative": 0, "neutral": 0},
        "gender": {"masculine": 0, "feminine": 0, "neutral": 0},
        "sector": {},
    }

    with open(pf, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rec = annotate_record(rec, rules)
            records.append(rec)
            file_stats["total"] += 1
            file_stats["tense"][rec["tense"]] += 1
            file_stats["polarity"][rec["polarity"]] += 1
            file_stats["gender"][rec["gender"]] += 1
            sec = rec["sector"]
            file_stats["sector"][sec] = file_stats["sector"].get(sec, 0) + 1

    out_path = pf if output_dir == pf.parent else output_dir / pf.name
    with open(out_path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return file_stats


def annotate_pool(
    pool_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    rules_path: str | Path | None = None,
    *,
    show_progress: bool = True,
    max_workers: int | None = None,
    max_chunks: int | None = 30,
) -> dict:
    """Annotate all pool chunk files with metadata labels in parallel.

    Parameters
    ----------
    pool_dir : directory containing pool JSONL files
    output_dir : directory for annotated files (defaults to pool_dir)
    rules_path : path to rules.yaml
    max_workers : parallel worker processes (defaults to CPU count)

    Returns
    -------
    dict with annotation statistics
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import os

    if pool_dir is None:
        pool_dir = _PROJECT_ROOT / "dataset" / "asr_corpus" / "pool"
    pool_dir = Path(pool_dir)

    if output_dir is None:
        output_dir = pool_dir  # annotate in-place
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pool_files = sorted(pool_dir.glob("pool_chunk_*.jsonl"))
    if not pool_files:
        print(f"⚠ No pool chunk files found in {pool_dir}")
        return {}

    if max_chunks is not None:
        pool_files = pool_files[:max_chunks]

    rules_path_str = str(rules_path) if rules_path else ""
    output_dir_str = str(output_dir)
    tasks = [(str(pf), rules_path_str, output_dir_str) for pf in pool_files]

    workers = max_workers or min(os.cpu_count() or 4, len(pool_files))
    print(f"▸ Annotating {len(pool_files)} pool chunks using {workers} parallel processes...")

    stats = {
        "total": 0,
        "tense": {"past": 0, "present": 0, "future": 0, "mixed": 0},
        "polarity": {"positive": 0, "negative": 0, "neutral": 0},
        "gender": {"masculine": 0, "feminine": 0, "neutral": 0},
        "sector": {},
    }

    try:
        from tqdm import tqdm
        use_tqdm = show_progress
    except ImportError:
        use_tqdm = False

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_annotate_file_worker, t) for t in tasks]
        iterable = as_completed(futures)
        if use_tqdm:
            iterable = tqdm(iterable, total=len(tasks), desc="Annotating pool chunks", unit=" file")

        for fut in iterable:
            fstats = fut.result()
            stats["total"] += fstats["total"]
            for k in ("tense", "polarity", "gender"):
                for label, count in fstats[k].items():
                    stats[k][label] = stats[k].get(label, 0) + count
            for label, count in fstats["sector"].items():
                stats["sector"][label] = stats["sector"].get(label, 0) + count

    # Summary
    stats_path = output_dir / "annotation_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Annotation complete: {stats['total']} records annotated")
    print(f"  TENSE: {stats['tense']}")
    print(f"  POLARITY: {stats['polarity']}")
    print(f"  GENDER: {stats['gender']}")
    print(f"  SECTOR: {stats['sector']}")

    return stats


def _print_distribution(stats: dict):
    """Print a concise summary of metadata distributions."""
    for field in ("tense", "polarity", "gender", "sector"):
        total = stats["total"]
        if total == 0:
            continue
        print(f"\n  {field.upper()}:")
        for label, count in sorted(stats[field].items()):
            pct = count / total * 100
            print(f"    {label:20s} {count:>8d}  ({pct:5.1f}%)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Annotate pool with metadata labels")
    parser.add_argument("--pool-dir", type=str, default=None,
                        help="Directory with pool JSONL chunks")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: annotate in-place)")
    parser.add_argument("--rules", type=str, default=None,
                        help="Path to rules.yaml")

    args = parser.parse_args()
    annotate_pool(args.pool_dir, args.output_dir, args.rules)
