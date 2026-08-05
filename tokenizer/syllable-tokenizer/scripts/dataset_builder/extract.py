#!/usr/bin/env python3
"""Stage 1: Sentence Extraction from Nepali text sources.

Supports three source types:
  1. compiled.txt   — HTML news (<p> tags), streamed in chunks
  2. Source_book.txt — Literary prose, line-by-line
  3. Nepali-Text-Corpus parquet files — 6.4M articles with Source metadata,
     read shard-by-shard for memory efficiency.

Applies quality filters, deduplicates, and writes candidate sentences
to a JSONL pool for downstream annotation and selection.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Generator, TextIO

# ---------------------------------------------------------------------------
# Resolve project root so we can import the existing syllabic_tokenizer.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent          # tokenizer/syllable-tokenizer
sys.path.insert(0, str(_SCRIPT_DIR.parent))         # scripts/

from syllabic_tokenizer import clean_text, get_lookup_tokens, tokenize  # noqa: E402

# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------
_P_TAG_RE = re.compile(r"<p[^>]*>(.*?)</p>", re.DOTALL | re.IGNORECASE)
_SENTENCE_SPLIT_RE = re.compile(r"[।!?]")
_PAGE_MARKER_RE = re.compile(r"/[०-९]+")            # /१, /२ etc.
_TITLE_LINE_RE = re.compile(r"^[^\u0900-\u097F]*$")  # lines with no Devanagari

# Quality filter defaults
MIN_SYLLABLES = 5
MAX_SYLLABLES = 80
MIN_CONTENT_SYLLABLES = 3

# ---------------------------------------------------------------------------
# Source domain → sector mapping for Nepali-Text-Corpus
# Used as a *hint* that can be overridden by rule-based sector annotation.
# ---------------------------------------------------------------------------
_SOURCE_SECTOR_MAP = {
    # Sports
    "hamrokhelkud.com":     "sports",

    # Technology
    "technologykhabar.com": "technology",
    "techpana.com":         "technology",
    "ictsamachar.com":      "technology",

    # Health
    "nepalihealth.com":     "health",
    "healthpati.com":       "health",

    # Business / Economy
    "karobardaily.com":     "business",
    "arthasarokar.com":     "business",
    "bizshala.com":         "business",
    "newskarobar.com":      "business",

    # Entertainment / Lifestyle
    "narimag.com.np":       "entertainment",

    # Everything else defaults to "news" in _get_sector_from_source()
}


def _get_sector_from_source(source: str) -> str:
    """Map a corpus Source domain to a sector label."""
    if not source:
        return "news"
    source = source.strip().lower()
    return _SOURCE_SECTOR_MAP.get(source, "news")


# ---------------------------------------------------------------------------
# Compiled.txt — HTML news extraction (streaming)
# ---------------------------------------------------------------------------

def _extract_compiled_sentences(
    filepath: str | Path,
    *,
    chunk_size: int = 1024 * 1024,     # 1 MB chunks
    max_sentences: int | None = None,
) -> Generator[dict, None, None]:
    """Stream sentences from compiled.txt without loading the full file.

    The file is a series of ``<p>…</p>`` blocks.  We read overlapping
    chunks to avoid splitting tags, extract <p> content, clean it and
    split on Nepali sentence delimiters.
    """
    filepath = Path(filepath)
    count = 0
    leftover = ""

    with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            text = leftover + chunk

            # Keep a tail to handle tags split across chunk boundaries
            last_close = text.rfind("</p>")
            if last_close == -1:
                leftover = text
                continue
            boundary = last_close + len("</p>")
            leftover = text[boundary:]
            text = text[:boundary]

            for m in _P_TAG_RE.finditer(text):
                raw = m.group(1)
                cleaned = clean_text(raw)
                if not cleaned:
                    continue
                for sent in _split_sentences(cleaned):
                    if sent:
                        count += 1
                        yield {
                            "text": sent,
                            "sector": "news",
                            "source_file": "compiled.txt",
                        }
                        if max_sentences and count >= max_sentences:
                            return


# ---------------------------------------------------------------------------
# Source_book.txt — literary prose extraction
# ---------------------------------------------------------------------------

def _extract_book_sentences(filepath: str | Path) -> Generator[dict, None, None]:
    """Read Source_book.txt, skip header/footer noise, yield sentences."""
    filepath = Path(filepath)
    with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()

    # Skip line 1 (TOC) and lines 782–786 (OCR noise) — 1-indexed
    skip_indices = {0}  # line 1
    for i in range(781, 786):
        skip_indices.add(i)

    for idx, raw_line in enumerate(lines):
        if idx in skip_indices:
            continue
        line = raw_line.strip()
        if not line:
            continue
        # Remove page markers like /१
        line = _PAGE_MARKER_RE.sub("", line)
        cleaned = clean_text(line)
        if not cleaned:
            continue
        for sent in _split_sentences(cleaned):
            if sent:
                yield {
                    "text": sent,
                    "sector": "literature",
                    "source_file": "Source_book.txt",
                }


# ---------------------------------------------------------------------------
# Nepali-Text-Corpus — parquet article extraction (shard-by-shard)
# ---------------------------------------------------------------------------

def _extract_parquet_sentences(
    corpus_dir: str | Path,
    *,
    max_sentences: int | None = None,
    max_shards: int | None = None,
    split: str = "train",
) -> Generator[dict, None, None]:
    """Stream sentences from Nepali-Text-Corpus parquet files.

    Reads one shard at a time to keep memory usage bounded.
    Each article is split into sentences, cleaned, and yielded
    with sector inferred from the Source column.

    Parameters
    ----------
    corpus_dir : path to the Nepali-Text-Corpus root (contains data/ subdir)
    max_sentences : stop after this many sentences total
    max_shards : only read this many parquet shard files (for testing)
    split : parquet split prefix ("train" or "test")
    """
    import pyarrow.parquet as pq

    corpus_dir = Path(corpus_dir)
    data_dir = corpus_dir / "data"
    if not data_dir.exists():
        data_dir = corpus_dir  # maybe they pointed directly at data/

    if split in ("all", None, "*"):
        shard_files = sorted(data_dir.glob("*.parquet"))
    else:
        shard_files = sorted(data_dir.glob(f"{split}-*.parquet"))
    if not shard_files:
        print(f"⚠ No parquet files found in {data_dir}")
        return

    if max_shards is not None:
        shard_files = shard_files[:max_shards]

    count = 0
    for shard_idx, shard_path in enumerate(shard_files):
        try:
            table = pq.read_table(shard_path, columns=["Article", "Source"])
        except Exception as e:
            print(f"⚠ Error reading {shard_path.name}: {e}")
            continue

        articles = table.column("Article")
        sources = table.column("Source")

        for row_idx in range(len(table)):
            article = articles[row_idx].as_py()
            source = sources[row_idx].as_py() if sources[row_idx].is_valid else ""

            if not article or not isinstance(article, str):
                continue

            # Clean and split the article into sentences
            cleaned = clean_text(article)
            if not cleaned:
                continue

            sector = _get_sector_from_source(source)
            source_label = source if source else "unknown"

            for sent in _split_sentences(cleaned):
                if sent:
                    count += 1
                    yield {
                        "text": sent,
                        "sector": sector,
                        "source_file": f"nepali_corpus:{source_label}",
                    }
                    if max_sentences and count >= max_sentences:
                        return

        # Free memory after each shard
        del table, articles, sources


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    """Split on Nepali/punctuation delimiters, strip, drop empty."""
    parts = _SENTENCE_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Quality filtering
# ---------------------------------------------------------------------------

def passes_quality(
    text: str,
    lookup_vocab: frozenset,
    *,
    min_syllables: int = MIN_SYLLABLES,
    max_syllables: int = MAX_SYLLABLES,
    min_content: int = MIN_CONTENT_SYLLABLES,
    max_unknown_ratio: float = 0.3,
) -> tuple[bool, list[str], int]:
    """Check if a sentence passes quality gates.

    Returns (passes, tokens, syllable_count).
    """
    tokens = tokenize(text, lookup_vocab)
    # Filter out whitespace tokens for syllable counting
    content_tokens = [t for t in tokens if t.strip()]
    syllable_count = len(content_tokens)

    if syllable_count < min_syllables or syllable_count > max_syllables:
        return False, tokens, syllable_count

    if syllable_count < min_content:
        return False, tokens, syllable_count

    # Check for excessive unmatched characters
    # We compare the cleaned text length vs. the length covered by tokens
    token_coverage = sum(len(t) for t in tokens)
    cleaned_len = len(text.replace(" ", ""))
    if cleaned_len > 0:
        unknown_ratio = max(0, cleaned_len - token_coverage) / cleaned_len
        if unknown_ratio > max_unknown_ratio:
            return False, tokens, syllable_count

    return True, tokens, syllable_count


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _text_hash(text: str) -> str:
    """Produce a dedup hash from normalized text."""
    # Normalize spaces and strip for consistent hashing
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Parallel parquet extraction (multi-core)
# ---------------------------------------------------------------------------

def _parquet_shard_worker(args_tuple: tuple) -> tuple[dict, str]:
    """Process assigned parquet shards; write accepted records to a temp JSONL.

    Top-level function for ProcessPoolExecutor pickling.
    """
    shard_paths, lookup_vocab_path, min_syllables, max_syllables, temp_path = args_tuple

    lookup_vocab = get_lookup_tokens(lookup_vocab_path)
    stats = {
        "raw": 0,
        "accepted": 0,
        "filtered_short": 0,
        "filtered_long": 0,
        "filtered_quality": 0,
        "duplicates": 0,
    }
    seen_local: set[str] = set()

    import pyarrow.parquet as pq

    with open(temp_path, "w", encoding="utf-8") as out_f:
        for shard_path in shard_paths:
            try:
                table = pq.read_table(shard_path, columns=["Article", "Source"])
            except Exception as e:
                print(f"⚠ Worker error reading {Path(shard_path).name}: {e}")
                continue

            articles = table.column("Article")
            sources = table.column("Source")

            for row_idx in range(len(table)):
                article = articles[row_idx].as_py()
                source = sources[row_idx].as_py() if sources[row_idx].is_valid else ""

                if not article or not isinstance(article, str):
                    continue

                cleaned = clean_text(article)
                if not cleaned:
                    continue

                sector = _get_sector_from_source(source)
                source_label = source if source else "unknown"

                for sent in _split_sentences(cleaned):
                    if not sent:
                        continue
                    stats["raw"] += 1

                    passes, tokens, syll_count = passes_quality(
                        sent, lookup_vocab,
                        min_syllables=min_syllables,
                        max_syllables=max_syllables,
                    )
                    if not passes:
                        if syll_count < min_syllables:
                            stats["filtered_short"] += 1
                        elif syll_count > max_syllables:
                            stats["filtered_long"] += 1
                        else:
                            stats["filtered_quality"] += 1
                        continue

                    h = _text_hash(sent)
                    if h in seen_local:
                        stats["duplicates"] += 1
                        continue
                    seen_local.add(h)

                    content_tokens = [t for t in tokens if t.strip()]
                    record = {
                        "text": sent,
                        "syllables": tokens,
                        "syllable_count": len(content_tokens),
                        "unique_syllables": list(set(content_tokens)),
                        "sector": sector,
                        "source_file": f"nepali_corpus:{source_label}",
                        "_dedup_hash": h,
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stats["accepted"] += 1

            del table, articles, sources

    return stats, temp_path


def _hash_partition(h: str, num_partitions: int) -> int:
    return int(h[:8], 16) % num_partitions


def _route_shard_temp_file(args_tuple: tuple) -> int:
    """Route one shard temp file into hash-partition buckets for parallel dedup."""
    temp_path, partition_dir, num_partitions = args_tuple
    temp_path = Path(temp_path)
    partition_dir = Path(partition_dir)
    handles: dict[int, TextIO] = {}
    routed = 0

    try:
        with open(temp_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                h = rec.get("_dedup_hash") or _text_hash(rec["text"])
                p = _hash_partition(h, num_partitions)
                if p not in handles:
                    part_dir = partition_dir / f"part_{p:03d}"
                    part_dir.mkdir(parents=True, exist_ok=True)
                    handles[p] = open(part_dir / f"{temp_path.stem}.jsonl", "w", encoding="utf-8")
                handles[p].write(line if line.endswith("\n") else line + "\n")
                routed += 1
    finally:
        for fh in handles.values():
            fh.close()

    return routed


def _dedup_partition(args_tuple: tuple) -> dict:
    """Dedup routed records for one hash partition (runs in parallel)."""
    partition_id, partition_dir, output_path = args_tuple
    part_subdir = Path(partition_dir) / f"part_{partition_id:03d}"
    seen: set[str] = set()
    stats = {"accepted": 0, "duplicates": 0}

    with open(output_path, "w", encoding="utf-8") as out_f:
        if not part_subdir.exists():
            return stats
        for part_file in sorted(part_subdir.glob("*.jsonl")):
            with open(part_file, "r", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    h = rec.pop("_dedup_hash", None) or _text_hash(rec["text"])
                    if h in seen:
                        stats["duplicates"] += 1
                        continue
                    seen.add(h)
                    rec["_dedup_hash"] = h
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    stats["accepted"] += 1

    return stats


def _write_pool_partition_worker(args_tuple: tuple) -> dict:
    """Write one deduped partition file into pool chunks (parallel merge phase 3)."""
    (
        dedup_path,
        output_dir,
        start_global_id,
        start_chunk_idx,
        chunk_size,
        max_records,
        hash_sidecar,
    ) = args_tuple

    output_dir = Path(output_dir)
    chunk_idx = start_chunk_idx
    global_id = start_global_id
    accepted = 0
    buffer_lines: list[str] = []

    def _flush() -> None:
        nonlocal chunk_idx, buffer_lines
        if not buffer_lines:
            return
        chunk_file = output_dir / f"pool_chunk_{chunk_idx:04d}.jsonl"
        with open(chunk_file, "w", encoding="utf-8") as out_f:
            out_f.writelines(buffer_lines)
        chunk_idx += 1
        buffer_lines = []

    with open(dedup_path, "r", encoding="utf-8") as in_f, \
         open(hash_sidecar, "w", encoding="utf-8") as hash_f:
        for line in in_f:
            if max_records is not None and accepted >= max_records:
                break
            rec = json.loads(line)
            h = rec.pop("_dedup_hash", None) or _text_hash(rec["text"])
            hash_f.write(h + "\n")
            rec["pool_id"] = f"pool_{global_id:08d}"
            global_id += 1
            accepted += 1
            buffer_lines.append(json.dumps(rec, ensure_ascii=False) + "\n")
            if len(buffer_lines) >= chunk_size:
                _flush()

    _flush()
    return {
        "accepted": accepted,
        "end_chunk_idx": chunk_idx,
        "end_global_id": global_id,
    }


def _load_hash_sidecar(hash_path: str, seen_hashes: set[str]) -> None:
    """Load dedup hashes from a sidecar file into seen_hashes."""
    with open(hash_path, "r", encoding="utf-8") as f:
        for line in f:
            h = line.strip()
            if h:
                seen_hashes.add(h)


def _parallel_merge_temp_files(
    temp_files: list[Path],
    *,
    output_dir: Path,
    seen_hashes: set[str],
    stats: dict,
    source_stats: dict,
    max_sentences: int | None,
    max_workers: int,
    chunk_size: int,
    start_chunk_idx: int,
    start_global_id: int,
    show_progress: bool,
) -> tuple[int, int]:
    """Merge shard temp files with parallel hash-partitioned dedup."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import shutil

    try:
        from tqdm import tqdm
        use_tqdm = show_progress
    except ImportError:
        use_tqdm = False

    # Match the requested worker count so all cloud-allocated CPU cores can
    # participate in hash-partitioned deduplication and chunk writing.
    num_partitions = max(1, max_workers)
    partition_dir = output_dir / "_merge_partitions"
    dedup_dir = output_dir / "_merge_deduped"
    partition_dir.mkdir(parents=True, exist_ok=True)
    dedup_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: route each shard file into hash partitions (parallel)
    print(f"▸ Merge phase 1/3: routing {len(temp_files)} shard files into {num_partitions} partitions...")
    route_tasks = [(str(tf), str(partition_dir), num_partitions) for tf in temp_files]
    with ProcessPoolExecutor(max_workers=min(max_workers, len(route_tasks))) as executor:
        futures = [executor.submit(_route_shard_temp_file, t) for t in route_tasks]
        iterable = as_completed(futures)
        if use_tqdm:
            iterable = tqdm(iterable, total=len(futures), desc="Routing shards", unit=" file")
        for fut in iterable:
            fut.result()

    # Phase 2: dedup each partition independently (parallel)
    print(f"▸ Merge phase 2/3: deduplicating {num_partitions} partitions in parallel...")
    dedup_tasks = [
        (p, str(partition_dir), str(dedup_dir / f"partition_{p:03d}.jsonl"))
        for p in range(num_partitions)
    ]
    partition_counts: list[int] = []
    with ProcessPoolExecutor(max_workers=num_partitions) as executor:
        if use_tqdm:
            pstats_list = list(tqdm(
                executor.map(_dedup_partition, dedup_tasks),
                total=len(dedup_tasks),
                desc="Dedup partitions",
                unit=" part",
            ))
        else:
            pstats_list = list(executor.map(_dedup_partition, dedup_tasks))

    for pstats in pstats_list:
        partition_counts.append(pstats["accepted"])
        stats["duplicates"] += pstats["duplicates"]

    dedup_files = [dedup_dir / f"partition_{p:03d}.jsonl" for p in range(num_partitions)]

    # Phase 3: write pool chunks in parallel (one worker per partition)
    total_accepted = sum(partition_counts)
    print(f"▸ Merge phase 3/3: writing {total_accepted:,} records to pool chunks in parallel...")

    prefix_global_id = [start_global_id]
    prefix_chunk_idx = [start_chunk_idx]
    for count in partition_counts:
        prefix_global_id.append(prefix_global_id[-1] + count)
        prefix_chunk_idx.append(
            prefix_chunk_idx[-1] + ((count + chunk_size - 1) // chunk_size if count else 0)
        )

    hash_dir = output_dir / "_merge_hashes"
    hash_dir.mkdir(parents=True, exist_ok=True)

    write_tasks = []
    remaining = max_sentences
    for p, dedup_file in enumerate(dedup_files):
        if not partition_counts[p]:
            continue
        take = partition_counts[p]
        if max_sentences is not None:
            if remaining is not None and remaining <= 0:
                break
            take = min(take, remaining)
            remaining -= take
        write_tasks.append((
            str(dedup_file),
            str(output_dir),
            prefix_global_id[p],
            prefix_chunk_idx[p],
            chunk_size,
            take if max_sentences is not None else None,
            str(hash_dir / f"partition_{p:03d}.hashes"),
        ))

    end_chunk_idx = start_chunk_idx
    end_global_id = start_global_id
    if write_tasks:
        with ProcessPoolExecutor(max_workers=min(max_workers, len(write_tasks))) as executor:
            if use_tqdm:
                write_results = list(tqdm(
                    executor.map(_write_pool_partition_worker, write_tasks),
                    total=len(write_tasks),
                    desc="Writing pool chunks",
                    unit=" part",
                ))
            else:
                write_results = list(executor.map(_write_pool_partition_worker, write_tasks))

        for wr in write_results:
            stats["accepted"] += wr["accepted"]
            source_stats["accepted"] += wr["accepted"]
            end_chunk_idx = max(end_chunk_idx, wr["end_chunk_idx"])
            end_global_id = max(end_global_id, wr["end_global_id"])

    # Load hash sidecars for downstream compiled/book dedup
    for hash_file in sorted(hash_dir.glob("*.hashes")):
        _load_hash_sidecar(str(hash_file), seen_hashes)

    shutil.rmtree(partition_dir, ignore_errors=True)
    shutil.rmtree(dedup_dir, ignore_errors=True)
    shutil.rmtree(hash_dir, ignore_errors=True)

    return end_chunk_idx, end_global_id


def _extract_parquet_parallel(
    corpus_dir: Path,
    *,
    lookup_vocab_path: str,
    output_dir: Path,
    seen_hashes: set[str],
    stats: dict,
    min_syllables: int,
    max_syllables: int,
    max_sentences: int | None,
    max_shards: int | None,
    max_workers: int,
    chunk_size: int,
    show_progress: bool,
    start_chunk_idx: int,
    start_global_id: int,
) -> tuple[int, int]:
    """Extract parquet corpus using one process per shard group."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import tempfile

    import pyarrow.parquet as pq  # noqa: F401 — verify dependency early

    data_dir = corpus_dir / "data"
    if not data_dir.exists():
        data_dir = corpus_dir

    shard_files = sorted(data_dir.glob("*.parquet"))
    if not shard_files:
        print(f"⚠ No parquet files found in {data_dir}")
        return start_chunk_idx, start_global_id

    if max_shards is not None:
        shard_files = shard_files[:max_shards]

    workers = min(max_workers, len(shard_files))
    temp_dir = Path(tempfile.mkdtemp(prefix="extract_workers_", dir=output_dir))

    # One future per shard so tqdm updates as each shard finishes (not per worker batch)
    tasks = []
    for shard_idx, shard_path in enumerate(shard_files):
        temp_path = str(temp_dir / f"shard_{shard_idx:04d}.jsonl")
        tasks.append((
            [str(shard_path)],
            lookup_vocab_path,
            min_syllables,
            max_syllables,
            temp_path,
        ))

    print(f"▸ Parallel extraction: {len(shard_files)} shards, {workers} concurrent workers")
    print(f"  (Spawning workers — first progress tick when a shard finishes, typically 2–5 min)")

    worker_stats: list[dict] = []
    try:
        from tqdm import tqdm
        use_tqdm = show_progress
    except ImportError:
        use_tqdm = False

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_parquet_shard_worker, t) for t in tasks]
        iterable = as_completed(futures)
        if use_tqdm:
            iterable = tqdm(iterable, total=len(futures), desc="Extracting nepali_corpus", unit=" shard")

        for fut in iterable:
            wstats, _ = fut.result()
            worker_stats.append(wstats)

    # Aggregate worker stats
    source_stats = stats["by_source"].setdefault("nepali_corpus", {"raw": 0, "accepted": 0})
    for ws in worker_stats:
        stats["total_raw"] += ws["raw"]
        source_stats["raw"] += ws["raw"]
        stats["filtered_short"] += ws["filtered_short"]
        stats["filtered_long"] += ws["filtered_long"]
        stats["filtered_quality"] += ws["filtered_quality"]
        stats["duplicates"] += ws["duplicates"]

    # Parallel hash-partitioned merge (replaces slow single-threaded global dedup)
    temp_files = sorted(temp_dir.glob("shard_*.jsonl"))
    chunk_idx, global_id = _parallel_merge_temp_files(
        temp_files,
        output_dir=output_dir,
        seen_hashes=seen_hashes,
        stats=stats,
        source_stats=source_stats,
        max_sentences=max_sentences,
        max_workers=max_workers,
        chunk_size=chunk_size,
        start_chunk_idx=start_chunk_idx,
        start_global_id=start_global_id,
        show_progress=show_progress,
    )

    # Clean up temp shard files
    for temp_file in temp_files:
        temp_file.unlink(missing_ok=True)
    temp_dir.rmdir()

    return chunk_idx, global_id


# ---------------------------------------------------------------------------
# Main extraction pipeline
# ---------------------------------------------------------------------------

def extract_pool(
    compiled_path: str | Path | None = None,
    book_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    *,
    nepali_corpus_path: str | Path | None = None,
    max_compiled: int | None = 100_000,
    max_corpus: int | None = 200_000,
    max_shards: int | None = None,
    min_syllables: int = MIN_SYLLABLES,
    max_syllables: int = MAX_SYLLABLES,
    chunk_size: int = 50_000,
    show_progress: bool = True,
    max_workers: int | None = None,
) -> dict:
    """Extract, filter, dedup and write candidate pool to JSONL chunks.

    Parameters
    ----------
    compiled_path : path to compiled.txt (optional, can be None)
    book_path : path to Source_book.txt (optional)
    output_dir : directory for pool JSONL chunks
    nepali_corpus_path : path to Nepali-Text-Corpus directory (parquet)
    max_compiled : maximum sentences to extract from compiled.txt
    max_corpus : maximum sentences to extract from parquet corpus
    max_shards : max parquet shard files to read (None = all)
    chunk_size : sentences per JSONL chunk file
    max_workers : parallel processes for parquet extraction (defaults to CPU count)

    Returns
    -------
    dict with extraction stats
    """
    if output_dir is None:
        output_dir = _PROJECT_ROOT / "dataset" / "asr_corpus" / "pool"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lookup_vocab_path = str(_PROJECT_ROOT / "dataset" / "nepali_syllables_lookup.vocab")
    lookup_vocab = get_lookup_tokens(lookup_vocab_path)

    if max_workers is None:
        max_workers = os.cpu_count() or 4
    max_workers = max(1, max_workers)

    seen_hashes: set[str] = set()
    stats = {
        "total_raw": 0,
        "filtered_short": 0,
        "filtered_long": 0,
        "filtered_quality": 0,
        "duplicates": 0,
        "accepted": 0,
        "by_source": {},
    }

    try:
        from tqdm import tqdm
        use_tqdm = show_progress
    except ImportError:
        use_tqdm = False

    def _progress(iterable, desc):
        if use_tqdm:
            return tqdm(iterable, desc=desc, unit=" sents")
        return iterable

    # Collect sentence generators (compiled/book — always sequential)
    generators = []
    nepali_corpus_resolved: Path | None = None
    corpus_limit: int | None = None

    # 1. Nepali-Text-Corpus (parquet) — primary source if provided
    if nepali_corpus_path:
        nepali_corpus_path = Path(nepali_corpus_path)
        if nepali_corpus_path.exists():
            nepali_corpus_resolved = nepali_corpus_path
            corpus_limit = None if (max_corpus is not None and max_corpus <= 0) else max_corpus
            if max_workers <= 1:
                generators.append(("nepali_corpus", _extract_parquet_sentences(
                    nepali_corpus_path,
                    max_sentences=corpus_limit,
                    max_shards=max_shards,
                    split="all",
                )))
        else:
            print(f"⚠ Nepali-Text-Corpus path not found: {nepali_corpus_path}")

    # 2. compiled.txt (HTML)
    if compiled_path:
        compiled_path = Path(compiled_path)
        if compiled_path.exists():
            generators.append(("compiled.txt", _extract_compiled_sentences(
                compiled_path, max_sentences=max_compiled
            )))

    # 3. Source_book.txt (literary)
    if book_path:
        book_path = Path(book_path)
        if book_path.exists():
            generators.append(("Source_book.txt", _extract_book_sentences(book_path)))

    if not generators and nepali_corpus_resolved is None:
        print("⚠ No source files found. Provide at least one source.")
        return stats

    chunk_idx = 0
    buffer: list[dict] = []

    # Parallel parquet extraction + parallel hash-partitioned merge
    if nepali_corpus_resolved is not None and max_workers > 1:
        stats["by_source"]["nepali_corpus"] = {"raw": 0, "accepted": 0}
        chunk_idx, global_id = _extract_parquet_parallel(
            nepali_corpus_resolved,
            lookup_vocab_path=lookup_vocab_path,
            output_dir=output_dir,
            seen_hashes=seen_hashes,
            stats=stats,
            min_syllables=min_syllables,
            max_syllables=max_syllables,
            max_sentences=corpus_limit,
            max_shards=max_shards,
            max_workers=max_workers,
            chunk_size=chunk_size,
            show_progress=show_progress,
            start_chunk_idx=0,
            start_global_id=0,
        )
    else:
        global_id = 0

    def _flush():
        nonlocal chunk_idx, buffer
        if not buffer:
            return
        chunk_file = output_dir / f"pool_chunk_{chunk_idx:04d}.jsonl"
        with open(chunk_file, "w", encoding="utf-8") as f:
            for rec in buffer:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        chunk_idx += 1
        buffer = []

    for source_name, gen in generators:
        stats["by_source"][source_name] = {"raw": 0, "accepted": 0}
        for raw_rec in _progress(gen, desc=f"Extracting {source_name}"):
            stats["total_raw"] += 1
            stats["by_source"][source_name]["raw"] += 1

            text = raw_rec["text"]
            # Quality check
            passes, tokens, syll_count = passes_quality(
                text, lookup_vocab,
                min_syllables=min_syllables, max_syllables=max_syllables,
            )
            if not passes:
                if syll_count < min_syllables:
                    stats["filtered_short"] += 1
                elif syll_count > max_syllables:
                    stats["filtered_long"] += 1
                else:
                    stats["filtered_quality"] += 1
                continue

            # Dedup
            h = _text_hash(text)
            if h in seen_hashes:
                stats["duplicates"] += 1
                continue
            seen_hashes.add(h)

            # Build pool record
            content_tokens = [t for t in tokens if t.strip()]
            record = {
                "pool_id": f"pool_{global_id:08d}",
                "text": text,
                "syllables": tokens,
                "syllable_count": len(content_tokens),
                "unique_syllables": list(set(content_tokens)),
                "sector": raw_rec["sector"],
                "source_file": raw_rec["source_file"],
            }
            buffer.append(record)
            global_id += 1
            stats["accepted"] += 1
            stats["by_source"][source_name]["accepted"] += 1

            if len(buffer) >= chunk_size:
                _flush()

    _flush()

    # Write extraction stats
    stats_path = output_dir / "extraction_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Extraction complete: {stats['accepted']} sentences "
          f"({stats['duplicates']} duplicates removed)")
    print(f"  Pool chunks written to: {output_dir}")
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract sentence pool from Nepali sources")
    parser.add_argument("--compiled", type=str, default=None,
                        help="Path to compiled.txt")
    parser.add_argument("--book", type=str, default=None,
                        help="Path to Source_book.txt")
    parser.add_argument("--nepali-corpus", type=str, default=None,
                        help="Path to Nepali-Text-Corpus directory (parquet)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for pool JSONL chunks")
    parser.add_argument("--max-compiled", type=int, default=100_000,
                        help="Max sentences from compiled.txt")
    parser.add_argument("--max-corpus", type=int, default=200_000,
                        help="Max sentences from Nepali-Text-Corpus")
    parser.add_argument("--max-shards", type=int, default=None,
                        help="Max parquet shards to read (for testing)")
    parser.add_argument("--min-syllables", type=int, default=MIN_SYLLABLES)
    parser.add_argument("--max-syllables", type=int, default=MAX_SYLLABLES)
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel worker processes for parquet extraction (default: CPU count)")

    args = parser.parse_args()
    extract_pool(
        args.compiled,
        args.book,
        args.output_dir,
        nepali_corpus_path=args.nepali_corpus,
        max_compiled=args.max_compiled,
        max_corpus=args.max_corpus,
        max_shards=args.max_shards,
        min_syllables=args.min_syllables,
        max_syllables=args.max_syllables,
        max_workers=args.workers,
    )
