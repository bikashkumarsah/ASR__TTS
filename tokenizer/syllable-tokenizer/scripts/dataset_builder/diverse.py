"""Five-corpus, rare-aware and semantic-diversity final corpus workflow.

The expensive stages are checkpointed independently.  Optional ML imports are
kept inside their stages so the original tokenizer and analysis commands remain
usable with the lightweight requirements file.
"""

from __future__ import annotations

import csv
import hashlib
import heapq
import itertools
import json
import math
import multiprocessing
import os
import platform
import re
import resource
import sqlite3
import sys
import tempfile
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

import yaml

from corpus_analysis.metrics import normalize_text
from corpus_analysis.sources import iter_records, load_config, validate_source
from dataset_builder.annotate import annotate_record, load_rules
from dataset_builder.extract import passes_quality
from dataset_builder.syllable_stats import _SKIP_TOKENS
from syllabic_tokenizer import get_lookup_tokens, tokenize
from syllable_metrics import (
    distribution_statistics,
    jensen_shannon_divergence,
    rarity_counts,
)


_SENTENCE_SPLIT_RE = re.compile(r"[।!?]+")
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_TOKENIZER_SOURCE = Path(__file__).resolve().parents[1] / "syllabic_tokenizer.py"
_DEFAULT_DIVERSE_CONFIG = _PROJECT_ROOT / "configs" / "final_50k_diverse.yaml"
_DEFAULT_RULES = Path(__file__).with_name("rules.yaml")
_WORKER_VOCAB: frozenset[str] = frozenset()
_WORKER_RULES: dict = {}
_WORKER_MIN_SYLLABLES = 5
_WORKER_MAX_SYLLABLES = 80


def _utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _peak_rss_gb() -> float:
    """Peak resident set size of this process in GiB.

    ``ru_maxrss`` is bytes on Darwin and kibibytes on Linux, so the cloud runner
    and a local Mac would otherwise disagree by a factor of 1024.
    """
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024**3 if sys.platform == "darwin" else 1024**2
    return peak / divisor


def _log_rss(stage: str) -> None:
    """Print the peak RSS reached by the end of ``stage``.

    An out-of-memory kill leaves no traceback, so without a per-stage watermark
    the only evidence of where a run died is the last line of stdout.  Printing
    the watermark makes the growth curve visible before the kill instead of
    after it.
    """
    print(f"[rss] {stage}: peak {_peak_rss_gb():.2f} GiB", flush=True)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: str | Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _config_fingerprint(config: dict) -> str:
    payload = json.dumps(
        {
            "configuration": config,
            "tokenizer_source_sha256": _sha256(_TOKENIZER_SOURCE),
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _resolve_lookup(config: dict) -> tuple[Path, frozenset[str], str]:
    entry = config.get("lookup_vocabulary", {})
    configured = Path(entry.get("path", "dataset/nepali_syllables_lookup.vocab"))
    path = configured if configured.is_absolute() else _PROJECT_ROOT / configured
    if not path.is_file():
        raise FileNotFoundError(f"Pinned lookup vocabulary not found: {path}")
    actual = _sha256(path)
    expected = entry.get("sha256")
    if not expected:
        raise ValueError("Diverse configuration must pin lookup_vocabulary.sha256")
    if actual != expected:
        raise RuntimeError(
            f"Pinned vocabulary checksum mismatch for {path}: expected {expected}, got {actual}"
        )
    raw = get_lookup_tokens(str(path))
    analytical = frozenset(token for token in raw if token.strip())
    expected_raw = entry.get("raw_entries")
    expected_analytical = entry.get("analytical_entries")
    if expected_raw is not None and len(raw) != int(expected_raw):
        raise RuntimeError(f"Pinned vocabulary entry count changed: {len(raw)} != {expected_raw}")
    if expected_analytical is not None and len(analytical) != int(expected_analytical):
        raise RuntimeError(
            f"Pinned analytical vocabulary count changed: {len(analytical)} != {expected_analytical}"
        )
    return path, analytical, actual


def _vocabulary_layers(analytical_vocab: frozenset[str]) -> dict:
    """Separate tokenizer structure, source support and final selection coverage."""
    selection_vocab = frozenset(analytical_vocab.difference(_SKIP_TOKENS))
    emittable = frozenset(
        token for token in selection_vocab
        if tokenize(token, analytical_vocab) == [token]
    )
    return {
        "selection_vocabulary": selection_vocab,
        "tokenizer_emittable": emittable,
        "tokenizer_unemittable": selection_vocab.difference(emittable),
    }


def _validate_tokenizer_inventory_checkpoint(
    output_dir: Path,
    vocabulary_layers: dict,
    tokenizer_sha256: str,
) -> None:
    """Reject a prepared inventory created with different tokenizer behaviour."""
    path = output_dir / "frequency_inventory.json"
    if not path.is_file():
        raise RuntimeError("Prepared pool is missing frequency_inventory.json")
    inventory = json.loads(path.read_text(encoding="utf-8"))
    expected_emittable = sorted(vocabulary_layers["tokenizer_emittable"])
    expected_unemittable = sorted(vocabulary_layers["tokenizer_unemittable"])
    checks = {
        "tokenizer_source_sha256": tokenizer_sha256,
        "tokenizer_emittable_count": len(expected_emittable),
        "tokenizer_emittable_syllables": expected_emittable,
        "tokenizer_unemittable_count": len(expected_unemittable),
        "tokenizer_unemittable_syllables": expected_unemittable,
    }
    mismatched = [key for key, expected in checks.items() if inventory.get(key) != expected]
    if mismatched:
        raise RuntimeError(
            "Prepared tokenizer inventory does not match the current tokenizer: "
            + ", ".join(mismatched)
        )


def _init_prepare_worker(
    vocab_path: str,
    rules_path: str,
    min_syllables: int,
    max_syllables: int,
) -> None:
    global _WORKER_VOCAB, _WORKER_RULES, _WORKER_MIN_SYLLABLES, _WORKER_MAX_SYLLABLES
    _WORKER_VOCAB = get_lookup_tokens(vocab_path)
    _WORKER_RULES = load_rules(rules_path)
    _WORKER_MIN_SYLLABLES = min_syllables
    _WORKER_MAX_SYLLABLES = max_syllables


def _sector_hint(metadata: dict) -> str:
    domain = str(metadata.get("domain") or metadata.get("source") or "").lower()
    keyword_map = {
        "sports": ("sport", "khel", "खेल"),
        "technology": ("tech", "ict", "प्रविध"),
        "health": ("health", "स्वास्थ्य"),
        "business": ("business", "econom", "artha", "अर्थ"),
        "entertainment": ("entertain", "film", "मनोरञ्ज"),
        "education": ("education", "school", "शिक्ष"),
    }
    for sector, fragments in keyword_map.items():
        if any(fragment in domain for fragment in fragments):
            return sector
    return "news"


def _prepare_record_worker(payload: tuple[str, dict, str]) -> list[dict]:
    raw_text, metadata, corpus = payload
    if not isinstance(raw_text, str) or not raw_text.strip():
        return []
    prepared: list[dict] = []
    for raw_sentence in _SENTENCE_SPLIT_RE.split(raw_text):
        text = normalize_text(raw_sentence)
        if not text:
            continue
        passed, tokens, count = passes_quality(
            text,
            _WORKER_VOCAB,
            min_syllables=_WORKER_MIN_SYLLABLES,
            max_syllables=_WORKER_MAX_SYLLABLES,
        )
        if not passed:
            continue
        content = [token for token in tokens if token.strip() and token not in _SKIP_TOKENS]
        if not content:
            continue
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        record = {
            "pool_id": f"five_{digest[:24]}",
            "normalized_sha256": digest,
            "text": text,
            "syllables": content,
            "syllable_count": len(content),
            "unique_syllables": sorted(set(content)),
            "source_corpora": [corpus],
            "source_metadata": metadata,
            "source_file": corpus,
            "sector": _sector_hint(metadata),
        }
        prepared.append(annotate_record(record, _WORKER_RULES))
    return prepared


def _source_payloads(
    config_path: Path,
    input_root: Path,
    max_records_per_corpus: int | None = None,
) -> Iterator[tuple[str, dict, str]]:
    for spec in load_config(config_path):
        validate_source(spec, input_root)
        for row in iter_records(spec, input_root, max_records=max_records_per_corpus):
            yield row.get("text"), row.get("metadata", {}), spec.slug


def _prepare_record_batch(payloads: list[tuple[str, dict, str]]) -> list[dict]:
    """Tokenize and annotate one bounded payload batch inside a worker process."""
    prepared: list[dict] = []
    for payload in payloads:
        prepared.extend(_prepare_record_worker(payload))
    return prepared


def _processed_records(
    config_path: Path,
    input_root: Path,
    *,
    vocab_path: Path,
    rules_path: Path,
    min_syllables: int,
    max_syllables: int,
    workers: int,
    max_records_per_corpus: int | None = None,
    chunksize: int = 64,
    max_pending_batches: int | None = None,
) -> Iterator[dict]:
    payloads = _source_payloads(config_path, input_root, max_records_per_corpus)
    initializer = (str(vocab_path), str(rules_path), min_syllables, max_syllables)
    if workers <= 1:
        _init_prepare_worker(*initializer)
        for payload in payloads:
            yield from _prepare_record_worker(payload)
        return
    # ``Executor.map`` builds its whole future list with a comprehension before
    # yielding the first result, so mapping it across a five-corpus generator
    # makes the parent hold every source sentence at once.  Submit a bounded
    # window and refill it in FIFO order instead: the yielded sequence stays
    # identical to ``map`` while resident payloads are capped at
    # ``max_pending_batches * chunksize``.
    batches = iter(lambda: list(itertools.islice(payloads, chunksize)), [])
    in_flight = max(1, int(max_pending_batches or workers * 4))
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_prepare_worker,
        initargs=initializer,
    ) as executor:
        pending: deque = deque(
            executor.submit(_prepare_record_batch, batch)
            for batch in itertools.islice(batches, in_flight)
        )
        while pending:
            records = pending.popleft().result()
            for batch in itertools.islice(batches, 1):
                pending.append(executor.submit(_prepare_record_batch, batch))
            yield from records


class ExactSentenceIndex:
    """Disk-backed exact-dedup and source-membership index."""

    def __init__(self, path: Path, *, reset: bool = False):
        path.parent.mkdir(parents=True, exist_ok=True)
        if reset:
            for candidate in (path, Path(str(path) + "-wal"), Path(str(path) + "-shm")):
                if candidate.exists():
                    candidate.unlink()
        self.path = path
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute("PRAGMA temp_store=FILE")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS texts ("
            "hash TEXT PRIMARY KEY, first_corpus TEXT NOT NULL) WITHOUT ROWID"
        )
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS membership ("
            "hash TEXT NOT NULL, corpus TEXT NOT NULL, "
            "PRIMARY KEY(hash, corpus)) WITHOUT ROWID"
        )
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS second_pass_seen ("
            "hash TEXT PRIMARY KEY) WITHOUT ROWID"
        )

    def add(self, record: dict) -> bool:
        digest = record["normalized_sha256"]
        corpus = record["source_corpora"][0]
        cursor = self.connection.execute(
            "INSERT OR IGNORE INTO texts(hash, first_corpus) VALUES (?, ?)",
            (digest, corpus),
        )
        self.connection.execute(
            "INSERT OR IGNORE INTO membership(hash, corpus) VALUES (?, ?)",
            (digest, corpus),
        )
        return cursor.rowcount == 1

    def claim_first_occurrence(self, record: dict) -> bool:
        row = self.connection.execute(
            "SELECT first_corpus FROM texts WHERE hash = ?",
            (record["normalized_sha256"],),
        ).fetchone()
        if not row or row[0] != record["source_corpora"][0]:
            return False
        cursor = self.connection.execute(
            "INSERT OR IGNORE INTO second_pass_seen(hash) VALUES (?)",
            (record["normalized_sha256"],),
        )
        return cursor.rowcount == 1

    def sources(self, digest: str) -> list[str]:
        return [
            row[0]
            for row in self.connection.execute(
                "SELECT corpus FROM membership WHERE hash = ? ORDER BY corpus", (digest,)
            )
        ]

    def sources_many(self, hashes: list[str]) -> dict[str, list[str]]:
        result: dict[str, list[str]] = defaultdict(list)
        for start in range(0, len(hashes), 500):
            group = hashes[start:start + 500]
            placeholders = ",".join("?" for _ in group)
            for digest, corpus in self.connection.execute(
                f"SELECT hash, corpus FROM membership WHERE hash IN ({placeholders}) "
                "ORDER BY hash, corpus",
                group,
            ):
                result[digest].append(corpus)
        return result

    def reset_pass_seen(self) -> None:
        self.connection.execute("DELETE FROM second_pass_seen")
        self.connection.commit()

    def commit(self) -> None:
        self.connection.commit()

    def count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM texts").fetchone()[0])

    def close(self) -> None:
        self.connection.commit()
        self.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self.connection.close()


def _stable_priority(record: dict, sentence_frequency: Counter) -> float:
    rarity = sum(
        1.0 / math.sqrt(max(1, sentence_frequency.get(syllable, 1)))
        for syllable in record["unique_syllables"]
    )
    jitter = int(record["normalized_sha256"][:12], 16) / float(16**12)
    return rarity / math.sqrt(max(1, record["syllable_count"])) + jitter * 1e-9


def _push_bounded(heap: list[tuple[float, str]], item: tuple[float, str], limit: int) -> bool:
    if limit <= 0:
        return False
    if len(heap) < limit:
        heapq.heappush(heap, item)
        return True
    if item > heap[0]:
        heapq.heapreplace(heap, item)
        return True
    return False


def _write_sharded_records(records: Iterable[dict], directory: Path, chunk_size: int) -> int:
    directory.mkdir(parents=True, exist_ok=True)
    count = 0
    handle = None
    try:
        for record in records:
            if count % chunk_size == 0:
                if handle:
                    handle.close()
                handle = open(directory / f"part-{count // chunk_size:05d}.jsonl", "w", encoding="utf-8")
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    finally:
        if handle:
            handle.close()
    return count


def _materialize_selected_records(
    records: Iterable[dict],
    *,
    selected_hashes: set[str],
    index: ExactSentenceIndex,
    batch_size: int = 500,
) -> Iterator[dict]:
    """Yield only final shortlist members; full evicted records never enter a cache."""
    pending: list[dict] = []

    def hydrate(group: list[dict]) -> Iterator[dict]:
        memberships = index.sources_many(
            [record["normalized_sha256"] for record in group]
        )
        for record in group:
            record["source_corpora"] = memberships.get(
                record["normalized_sha256"], record["source_corpora"]
            )
            yield record

    for record in records:
        digest = record["normalized_sha256"]
        if digest not in selected_hashes or not index.claim_first_occurrence(record):
            continue
        pending.append(record)
        if len(pending) >= batch_size:
            yield from hydrate(pending)
            pending = []
    if pending:
        yield from hydrate(pending)


def prepare_five_corpus_pool(
    *,
    corpus_config: str | Path,
    input_root: str | Path,
    output_dir: str | Path,
    diverse_config: str | Path = _DEFAULT_DIVERSE_CONFIG,
    candidate_limit: int | None = None,
    workers: int | None = None,
    resume: bool = False,
    max_records_per_corpus: int | None = None,
) -> Path:
    """Build the exact-deduplicated, rarity-retaining semantic shortlist."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    complete = output_dir / "prepare.complete.json"
    config = _load_yaml(diverse_config)
    config_fingerprint = _config_fingerprint(config)
    tokenizer_sha = _sha256(_TOKENIZER_SOURCE)
    vocab_path, analytical_vocab, vocab_sha = _resolve_lookup(config)
    vocabulary_layers = _vocabulary_layers(analytical_vocab)
    selection_vocab = vocabulary_layers["selection_vocabulary"]
    tokenizer_emittable = vocabulary_layers["tokenizer_emittable"]
    tokenizer_unemittable = vocabulary_layers["tokenizer_unemittable"]
    requested_candidate_limit = int(
        candidate_limit or config.get("preparation", {}).get("candidate_limit", 1_200_000)
    )
    if resume and complete.exists():
        state = json.loads(complete.read_text(encoding="utf-8"))
        if state.get("vocabulary_sha256") != vocab_sha:
            raise RuntimeError("Resume checkpoint uses a different vocabulary")
        if state.get("corpus_config_sha256") != _sha256(corpus_config):
            raise RuntimeError("Resume checkpoint uses a different corpus configuration")
        if state.get("max_records_per_corpus") != max_records_per_corpus:
            raise RuntimeError("Resume checkpoint uses a different per-corpus record limit")
        if state.get("diverse_config_sha256") != config_fingerprint:
            raise RuntimeError("Resume checkpoint uses a different diverse-build configuration")
        if state.get("tokenizer_source_sha256") != tokenizer_sha:
            raise RuntimeError("Resume checkpoint uses a different tokenizer implementation")
        if state.get("candidate_limit") != requested_candidate_limit:
            raise RuntimeError("Resume checkpoint uses a different candidate limit")
        _validate_tokenizer_inventory_checkpoint(
            output_dir, vocabulary_layers, tokenizer_sha
        )
        return output_dir

    prep = config.get("preparation", {})
    min_syllables = int(prep.get("min_syllables", 5))
    max_syllables = int(prep.get("max_syllables", 80))
    candidate_limit = requested_candidate_limit
    per_syllable = int(prep.get("candidates_per_syllable", 512))
    chunk_size = int(prep.get("chunk_size", 10_000))
    workers = max(1, int(workers or os.cpu_count() or 1))
    corpus_config = Path(corpus_config)
    input_root = Path(input_root)
    index = ExactSentenceIndex(output_dir / "exact_dedup.sqlite3", reset=True)
    occurrence = Counter()
    sentence_frequency = Counter()
    source_native = Counter()
    duplicates = 0
    eligible = 0

    print(f"▸ Pass 1/3: exact union and syllable frequencies with {workers} workers")
    try:
        for record in _processed_records(
            corpus_config,
            input_root,
            vocab_path=vocab_path,
            rules_path=_DEFAULT_RULES,
            min_syllables=min_syllables,
            max_syllables=max_syllables,
            workers=workers,
            max_records_per_corpus=max_records_per_corpus,
        ):
            source_native[record["source_corpora"][0]] += 1
            if not index.add(record):
                duplicates += 1
                continue
            eligible += 1
            occurrence.update(record["syllables"])
            sentence_frequency.update(record["unique_syllables"])
            if eligible % 100_000 == 0:
                index.commit()
                print(
                    f"  unique eligible sentences: {eligible:,}"
                    f"  (peak {_peak_rss_gb():.2f} GiB)"
                )
        index.commit()
        _log_rss("prepare pass 1/3")

        source_observed = frozenset(sentence_frequency).intersection(selection_vocab)
        attainable = source_observed.intersection(tokenizer_emittable)
        inventory = {
            "vocabulary_path": str(vocab_path),
            "vocabulary_sha256": vocab_sha,
            "tokenizer_source_path": str(_TOKENIZER_SOURCE),
            "tokenizer_source_sha256": tokenizer_sha,
            "diverse_config_sha256": config_fingerprint,
            "raw_vocabulary_size": int(config["lookup_vocabulary"]["raw_entries"]),
            "analytical_vocabulary_size": len(analytical_vocab),
            "selection_vocabulary_size": len(selection_vocab),
            "selection_vocabulary": sorted(selection_vocab),
            "tokenizer_emittable_count": len(tokenizer_emittable),
            "tokenizer_emittable_syllables": sorted(tokenizer_emittable),
            "tokenizer_unemittable_count": len(tokenizer_unemittable),
            "tokenizer_unemittable_syllables": sorted(tokenizer_unemittable),
            "source_observed_count": len(source_observed),
            "source_observed_syllables": sorted(source_observed),
            "attainable_syllables": sorted(attainable),
            "attainable_count": len(attainable),
            "unique_eligible_sentences": eligible,
            "exact_duplicate_sentences": duplicates,
            "source_native_eligible": dict(source_native),
            "syllable_occurrence_frequency": dict(sorted(occurrence.items())),
            "syllable_sentence_frequency": dict(sorted(sentence_frequency.items())),
        }
        _write_json(output_dir / "frequency_inventory.json", inventory)

        print(f"▸ Pass 2/3: deterministic bounded hash shortlist (limit {candidate_limit:,})")
        global_heap: list[tuple[float, str]] = []
        coverage_heaps: dict[str, list[tuple[float, str]]] = defaultdict(list)
        second_pass_seen = 0
        for record in _processed_records(
            corpus_config,
            input_root,
            vocab_path=vocab_path,
            rules_path=_DEFAULT_RULES,
            min_syllables=min_syllables,
            max_syllables=max_syllables,
            workers=workers,
            max_records_per_corpus=max_records_per_corpus,
        ):
            digest = record["normalized_sha256"]
            if not index.claim_first_occurrence(record):
                continue
            second_pass_seen += 1
            priority = _stable_priority(record, sentence_frequency)
            _push_bounded(global_heap, (priority, digest), candidate_limit)
            for syllable in record["unique_syllables"]:
                if syllable in attainable:
                    _push_bounded(coverage_heaps[syllable], (priority, digest), per_syllable)
            if second_pass_seen % 100_000 == 0:
                index.commit()
        index.commit()

        required_hashes = {
            digest for heap in coverage_heaps.values() for _, digest in heap
        }
        if len(required_hashes) > candidate_limit:
            raise RuntimeError(
                f"Coverage reservoirs require {len(required_hashes):,} records, "
                f"exceeding candidate limit {candidate_limit:,}"
            )
        selected_hashes = set(required_hashes)
        for _, digest in sorted(global_heap, reverse=True):
            if len(selected_hashes) >= candidate_limit:
                break
            selected_hashes.add(digest)
        print("▸ Pass 3/3: stream and materialize only the final bounded shortlist")
        _log_rss("prepare pass 2/3")
        index.reset_pass_seen()
        shortlist_dir = output_dir / "shortlist"
        for old in shortlist_dir.glob("part-*.jsonl") if shortlist_dir.exists() else []:
            old.unlink()
        materialized = _materialize_selected_records(
            _processed_records(
                corpus_config,
                input_root,
                vocab_path=vocab_path,
                rules_path=_DEFAULT_RULES,
                min_syllables=min_syllables,
                max_syllables=max_syllables,
                workers=workers,
                max_records_per_corpus=max_records_per_corpus,
            ),
            selected_hashes=selected_hashes,
            index=index,
        )
        written = _write_sharded_records(materialized, shortlist_dir, chunk_size)
        if written != len(selected_hashes):
            raise RuntimeError(
                f"Shortlist materialization is incomplete: {written:,}/{len(selected_hashes):,} records"
            )
        state = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "vocabulary_sha256": vocab_sha,
            "tokenizer_source_sha256": tokenizer_sha,
            "diverse_config_sha256": config_fingerprint,
            "corpus_config_sha256": _sha256(corpus_config),
            "candidate_limit": candidate_limit,
            "shortlist_records": written,
            "workers": workers,
            "max_records_per_corpus": max_records_per_corpus,
            "preparation_passes": 3,
            "preparation_reproducibility": (
                "deterministic for fixed normalized inputs, vocabulary, Python version, "
                "architecture and seed; verify different worker counts by artifact comparison"
            ),
            "dataset_revisions": (
                json.loads((input_root / "source_manifest.json").read_text(encoding="utf-8"))
                if (input_root / "source_manifest.json").exists()
                else None
            ),
            "source_manifest_sha256": (
                _sha256(input_root / "source_manifest.json")
                if (input_root / "source_manifest.json").exists()
                else None
            ),
            "shortlist_files": [
                {"path": path.name, "sha256": _sha256(path)}
                for path in sorted(shortlist_dir.glob("part-*.jsonl"))
            ],
        }
        _write_json(complete, state)
        print(f"✓ Prepared {written:,} exact-deduplicated candidates at {output_dir}")
        _log_rss("prepare pass 3/3")
        return output_dir
    finally:
        index.close()


def iter_shortlist(pool_dir: str | Path) -> Iterator[dict]:
    files = sorted((Path(pool_dir) / "shortlist").glob("part-*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No shortlist shards in {Path(pool_dir) / 'shortlist'}")
    for path in files:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def compare_prepared_pools(left: str | Path, right: str | Path) -> None:
    """Fail unless two fixed-environment preparation artifacts are identical."""
    left, right = Path(left), Path(right)
    left_inventory = json.loads((left / "frequency_inventory.json").read_text(encoding="utf-8"))
    right_inventory = json.loads((right / "frequency_inventory.json").read_text(encoding="utf-8"))
    if left_inventory != right_inventory:
        raise AssertionError("Prepared frequency inventories differ")
    left_state = json.loads((left / "prepare.complete.json").read_text(encoding="utf-8"))
    right_state = json.loads((right / "prepare.complete.json").read_text(encoding="utf-8"))
    left_files = [(entry["path"], entry["sha256"]) for entry in left_state["shortlist_files"]]
    right_files = [(entry["path"], entry["sha256"]) for entry in right_state["shortlist_files"]]
    if left_files != right_files:
        raise AssertionError("Prepared shortlist files differ")
    for key in (
        "vocabulary_sha256",
        "tokenizer_source_sha256",
        "diverse_config_sha256",
        "corpus_config_sha256",
        "candidate_limit",
        "shortlist_records",
    ):
        if left_state.get(key) != right_state.get(key):
            raise AssertionError(f"Prepared state differs for {key}")
    print("✓ Preparation artifacts are identical for the tested environments")


def _character_ngrams(text: str, size: int = 5) -> set[str]:
    compact = " ".join(text.split())
    if len(compact) <= size:
        return {compact} if compact else set()
    return {compact[index:index + size] for index in range(len(compact) - size + 1)}


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


class DiskMinHashLSH:
    """Compact SQLite-backed MinHash band index with exact verification."""

    def __init__(
        self,
        path: Path,
        *,
        num_perm: int = 128,
        bands: int = 16,
        threshold: float = 0.85,
        reset: bool = False,
    ):
        try:
            from datasketch import MinHash
        except ImportError as exc:
            raise RuntimeError(
                "MinHash-LSH requires requirements-diversity.txt (missing datasketch)"
            ) from exc
        if num_perm % bands:
            raise ValueError("MinHash permutations must be divisible by LSH bands")
        if reset and path.exists():
            path.unlink()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.MinHash = MinHash
        self.num_perm = num_perm
        self.bands = bands
        self.rows = num_perm // bands
        self.threshold = threshold
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute("PRAGMA temp_store=FILE")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS representatives ("
            "hash TEXT PRIMARY KEY, text TEXT NOT NULL) WITHOUT ROWID"
        )
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS bands ("
            "band INTEGER NOT NULL, bucket BLOB NOT NULL, hash TEXT NOT NULL, "
            "PRIMARY KEY(band, bucket, hash)) WITHOUT ROWID"
        )

    def _signature(self, ngrams: set[str]):
        value = self.MinHash(num_perm=self.num_perm)
        for ngram in sorted(ngrams):
            value.update(ngram.encode("utf-8"))
        return value.hashvalues

    def find_duplicate(self, text: str) -> tuple[str | None, float]:
        ngrams = _character_ngrams(text)
        signature = self._signature(ngrams)
        candidates: set[str] = set()
        buckets: list[tuple[int, bytes]] = []
        for band in range(self.bands):
            start = band * self.rows
            bucket = signature[start:start + self.rows].tobytes()
            buckets.append((band, bucket))
        clauses = " OR ".join("(band = ? AND bucket = ?)" for _ in buckets)
        parameters = [value for pair in buckets for value in pair]
        candidates.update(
            row[0]
            for row in self.connection.execute(
                f"SELECT hash FROM bands WHERE {clauses} LIMIT 500", parameters
            )
        )
        best_hash = None
        best_score = 0.0
        ordered = sorted(candidates)
        for start in range(0, len(ordered), 500):
            group = ordered[start:start + 500]
            placeholders = ",".join("?" for _ in group)
            for digest, candidate_text in self.connection.execute(
                f"SELECT hash, text FROM representatives WHERE hash IN ({placeholders})",
                group,
            ):
                score = _jaccard(ngrams, _character_ngrams(candidate_text))
                if score > best_score or (score == best_score and (best_hash is None or digest < best_hash)):
                    best_hash, best_score = digest, score
        return (best_hash, best_score) if best_score >= self.threshold else (None, best_score)

    def add(self, digest: str, text: str) -> None:
        signature = self._signature(_character_ngrams(text))
        self.connection.execute(
            "INSERT OR IGNORE INTO representatives(hash, text) VALUES (?, ?)",
            (digest, text),
        )
        for band in range(self.bands):
            start = band * self.rows
            bucket = signature[start:start + self.rows].tobytes()
            self.connection.execute(
                "INSERT OR IGNORE INTO bands(band, bucket, hash) VALUES (?, ?, ?)",
                (band, bucket, digest),
            )

    def commit(self) -> None:
        self.connection.commit()

    def close(self) -> None:
        self.connection.commit()
        self.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self.connection.close()


def build_lexical_annotations(
    pool_dir: str | Path,
    *,
    diverse_config: str | Path = _DEFAULT_DIVERSE_CONFIG,
    resume: bool = False,
) -> Path:
    pool_dir = Path(pool_dir)
    complete = pool_dir / "lexical.complete.json"
    config = _load_yaml(diverse_config)
    config_fingerprint = _config_fingerprint(config)
    tokenizer_sha = _sha256(_TOKENIZER_SOURCE)
    _, analytical_vocab, vocab_sha = _resolve_lookup(config)
    vocabulary_layers = _vocabulary_layers(analytical_vocab)
    prepare_state = json.loads((pool_dir / "prepare.complete.json").read_text(encoding="utf-8"))
    if prepare_state.get("vocabulary_sha256") != vocab_sha:
        raise RuntimeError("Prepared shortlist vocabulary differs from the pinned vocabulary")
    if prepare_state.get("diverse_config_sha256") != config_fingerprint:
        raise RuntimeError("Prepared shortlist uses a different build configuration")
    if prepare_state.get("tokenizer_source_sha256") != tokenizer_sha:
        raise RuntimeError("Prepared shortlist uses a different tokenizer implementation")
    _validate_tokenizer_inventory_checkpoint(pool_dir, vocabulary_layers, tokenizer_sha)
    if resume and complete.exists():
        state = json.loads(complete.read_text(encoding="utf-8"))
        if state.get("diverse_config_sha256") != config_fingerprint:
            raise RuntimeError("Lexical checkpoint uses a different build configuration")
        if state.get("tokenizer_source_sha256") != tokenizer_sha:
            raise RuntimeError("Lexical checkpoint uses a different tokenizer implementation")
        return pool_dir / "lexical"

    prep = config.get("preparation", {})
    permutations = int(prep.get("minhash_permutations", 128))
    threshold = float(prep.get("lexical_jaccard_threshold", 0.85))
    output_dir = pool_dir / "lexical"
    output_dir.mkdir(parents=True, exist_ok=True)
    for old in output_dir.glob("part-*.jsonl"):
        old.unlink()
    pairs_path = pool_dir / "lexical_near_duplicate_pairs.jsonl"
    index = DiskMinHashLSH(
        pool_dir / "minhash_lsh.sqlite3",
        num_perm=permutations,
        threshold=threshold,
        reset=True,
    )
    total = 0
    duplicates = 0
    try:
        with open(pairs_path, "w", encoding="utf-8") as pair_handle:
            for input_path in sorted((pool_dir / "shortlist").glob("part-*.jsonl")):
                output_path = output_dir / input_path.name
                with open(input_path, "r", encoding="utf-8") as source, open(
                    output_path, "w", encoding="utf-8"
                ) as target:
                    for line in source:
                        if not line.strip():
                            continue
                        record = json.loads(line)
                        duplicate_of, score = index.find_duplicate(record["text"])
                        record["lexical_near_duplicate_of"] = duplicate_of
                        record["lexical_jaccard"] = round(score, 6) if duplicate_of else None
                        if duplicate_of:
                            duplicates += 1
                            pair_handle.write(json.dumps({
                                "left": record["normalized_sha256"],
                                "right": duplicate_of,
                                "jaccard": round(score, 6),
                            }) + "\n")
                        else:
                            index.add(record["normalized_sha256"], record["text"])
                        target.write(json.dumps(record, ensure_ascii=False) + "\n")
                        total += 1
                        if total % 10_000 == 0:
                            index.commit()
        state = {
            "vocabulary_sha256": vocab_sha,
            "tokenizer_source_sha256": tokenizer_sha,
            "diverse_config_sha256": config_fingerprint,
            "records": total,
            "lexical_near_duplicates": duplicates,
            "minhash_permutations": permutations,
            "lsh_bands": index.bands,
            "jaccard_threshold": threshold,
            "files": [
                {"path": path.name, "sha256": _sha256(path)}
                for path in sorted(output_dir.glob("part-*.jsonl"))
            ],
        }
        _write_json(complete, state)
        print(f"✓ MinHash-LSH marked {duplicates:,}/{total:,} lexical near-duplicates")
        _log_rss("lexical annotations")
        return output_dir
    finally:
        index.close()


_EMBED_TOKENIZER = None
_EMBED_SESSION = None
_EMBED_BATCH_SIZE = 256
_EMBED_PREFIX = "query: "


def _require_ml_stack() -> None:
    missing = []
    for module in ("numpy", "onnxruntime", "transformers", "faiss"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        raise RuntimeError(
            "Semantic selection requires requirements-diversity.txt; missing: "
            + ", ".join(missing)
        )


def _find_onnx_model(model_dir: Path) -> Path:
    candidates = sorted(model_dir.rglob("*.onnx"))
    if not candidates:
        raise FileNotFoundError(f"No ONNX model found under {model_dir}")
    quantized = [path for path in candidates if "quant" in path.name.lower()]
    return quantized[0] if quantized else candidates[0]


def export_quantized_e5(
    model_name: str,
    output_dir: Path,
    *,
    resume: bool,
) -> dict:
    """Export a sentence embedding model to ONNX and apply dynamic int8 quantization."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "model_manifest.json"
    if resume and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        model_path = output_dir / manifest["onnx_relative_path"]
        if model_path.exists() and _sha256(model_path) == manifest["onnx_sha256"]:
            return manifest
    try:
        from huggingface_hub import model_info
        from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTQuantizer
        from optimum.onnxruntime.configuration import AutoQuantizationConfig
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "ONNX export requires optimum, onnxruntime, transformers and huggingface_hub"
        ) from exc

    revision = model_info(model_name).sha
    exported = output_dir / "fp32"
    quantized = output_dir / "int8"
    model = ORTModelForFeatureExtraction.from_pretrained(
        model_name, revision=revision, export=True
    )
    model.save_pretrained(exported)
    AutoTokenizer.from_pretrained(model_name, revision=revision).save_pretrained(exported)
    quantizer = ORTQuantizer.from_pretrained(exported)
    try:
        quantization_config = AutoQuantizationConfig.avx512_vnni(
            is_static=False, per_channel=False
        )
    except (AttributeError, ValueError):
        quantization_config = AutoQuantizationConfig.avx2(
            is_static=False, per_channel=False
        )
    quantizer.quantize(save_dir=quantized, quantization_config=quantization_config)
    AutoTokenizer.from_pretrained(exported).save_pretrained(quantized)
    onnx_path = _find_onnx_model(quantized)
    manifest = {
        "model": model_name,
        "model_revision": revision,
        "quantization": "dynamic_int8",
        "onnx_relative_path": str(onnx_path.relative_to(output_dir)),
        "onnx_sha256": _sha256(onnx_path),
    }
    _write_json(manifest_path, manifest)
    return manifest


def _init_embedding_worker(
    model_path: str,
    tokenizer_path: str,
    threads: int,
    batch_size: int,
    prefix: str = "query: ",
) -> None:
    global _EMBED_TOKENIZER, _EMBED_SESSION, _EMBED_BATCH_SIZE, _EMBED_PREFIX
    import onnxruntime as ort
    from transformers import AutoTokenizer

    options = ort.SessionOptions()
    options.intra_op_num_threads = max(1, threads)
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    _EMBED_TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_path)
    _EMBED_SESSION = ort.InferenceSession(
        model_path,
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    _EMBED_BATCH_SIZE = batch_size
    _EMBED_PREFIX = prefix


def _mean_pool_numpy(hidden, attention_mask):
    import numpy as np

    mask = attention_mask[..., None].astype(np.float32)
    summed = (hidden.astype(np.float32) * mask).sum(axis=1)
    denominator = np.maximum(mask.sum(axis=1), 1e-9)
    pooled = summed / denominator
    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    return pooled / np.maximum(norms, 1e-12)


def _embed_shard_worker(task: tuple[str, str, str]) -> dict:
    import numpy as np

    input_path, array_path, ids_path = map(Path, task)
    records = []
    with open(input_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    batches = []
    input_names = {entry.name for entry in _EMBED_SESSION.get_inputs()}
    for start in range(0, len(records), _EMBED_BATCH_SIZE):
        texts = [_EMBED_PREFIX + row["text"] for row in records[start:start + _EMBED_BATCH_SIZE]]
        encoded = _EMBED_TOKENIZER(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np",
        )
        inputs = {
            key: value.astype(np.int64)
            for key, value in encoded.items()
            if key in input_names
        }
        hidden = _EMBED_SESSION.run(None, inputs)[0]
        batches.append(_mean_pool_numpy(hidden, encoded["attention_mask"]))
    embeddings = np.concatenate(batches, axis=0) if batches else np.empty((0, 0), np.float32)
    np.save(array_path, embeddings.astype(np.float16))
    with open(ids_path, "w", encoding="utf-8") as handle:
        for row in records:
            handle.write(row["normalized_sha256"] + "\n")
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_mb = rss / (1024 * 1024 if sys.platform == "darwin" else 1024)
    except (ImportError, AttributeError):
        rss_mb = 0.0
    return {
        "input": input_path.name,
        "records": len(records),
        "array": array_path.name,
        "ids": ids_path.name,
        "rss_mb": round(rss_mb, 2),
    }


def _validate_onnx_pytorch_tolerance(
    records: list[dict],
    *,
    model_name: str,
    revision: str,
    model_path: Path,
    tokenizer_path: Path,
    prefix: str,
) -> dict:
    """Verify quantized ONNX embeddings against the source PyTorch model."""
    import numpy as np
    from sentence_transformers import SentenceTransformer

    sample = records[:32]
    if not sample:
        raise RuntimeError("Cannot validate ONNX model without calibration text")
    texts = [prefix + record["text"] for record in sample]
    pytorch = SentenceTransformer(model_name, revision=revision, device="cpu")
    expected = pytorch.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)
    _init_embedding_worker(
        str(model_path), str(tokenizer_path),
        max(1, min(8, os.cpu_count() or 1)), 32, prefix,
    )
    encoded = _EMBED_TOKENIZER(
        texts, padding=True, truncation=True, max_length=512, return_tensors="np"
    )
    input_names = {entry.name for entry in _EMBED_SESSION.get_inputs()}
    inputs = {
        key: value.astype(np.int64)
        for key, value in encoded.items()
        if key in input_names
    }
    hidden = _EMBED_SESSION.run(None, inputs)[0]
    actual = _mean_pool_numpy(hidden, encoded["attention_mask"])
    pair_cosines = np.sum(expected * actual, axis=1)
    result = {
        "records": len(sample),
        "minimum_same_text_cosine": round(float(pair_cosines.min()), 8),
        "mean_same_text_cosine": round(float(pair_cosines.mean()), 8),
        "required_minimum_cosine": 0.98,
    }
    if result["minimum_same_text_cosine"] < result["required_minimum_cosine"]:
        raise RuntimeError(
            "Quantized ONNX/PyTorch embedding tolerance failed: "
            f"minimum cosine {result['minimum_same_text_cosine']:.6f}"
        )
    del pytorch
    return result


def _run_embedding_layout(
    tasks: list[tuple[str, str, str]],
    *,
    model_path: Path,
    tokenizer_path: Path,
    processes: int,
    threads: int,
    batch_size: int,
    prefix: str = "query: ",
) -> tuple[list[dict], float]:
    # The parent has already initialized Hugging Face tokenizers, PyTorch and
    # ONNX for tolerance validation.  Forking that native thread state can
    # terminate a child without a Python exception (BrokenProcessPool).  Spawn
    # gives every embedding worker a clean interpreter and native runtime.
    context = multiprocessing.get_context("spawn")
    started = time.monotonic()
    with ProcessPoolExecutor(
        max_workers=processes,
        initializer=_init_embedding_worker,
        initargs=(str(model_path), str(tokenizer_path), threads, batch_size, prefix),
        mp_context=context,
    ) as executor:
        results = list(executor.map(_embed_shard_worker, tasks))
    return results, time.monotonic() - started


def _write_benchmark_shards(records: list[dict], directory: Path, count: int) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = [directory / f"bench-{index:03d}.jsonl" for index in range(count)]
    handles = [open(path, "w", encoding="utf-8") for path in paths]
    try:
        for index, record in enumerate(records):
            handles[index % count].write(json.dumps(record, ensure_ascii=False) + "\n")
    finally:
        for handle in handles:
            handle.close()
    return [path for path in paths if path.stat().st_size]


def embed_shortlist(
    pool_dir: str | Path,
    *,
    diverse_config: str | Path = _DEFAULT_DIVERSE_CONFIG,
    workers: int | None = None,
    resume: bool = False,
) -> Path:
    """Export/quantize E5, benchmark CPU topology, and embed all candidates."""
    _require_ml_stack()
    # Set this before any tokenizer/model initialization.  Worker processes
    # inherit it, preventing Rust tokenizer thread pools from being created in
    # both the parent and child process.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    pool_dir = Path(pool_dir)
    config = _load_yaml(diverse_config)
    config_fingerprint = _config_fingerprint(config)
    tokenizer_sha = _sha256(_TOKENIZER_SOURCE)
    _, analytical_vocab, vocab_sha = _resolve_lookup(config)
    vocabulary_layers = _vocabulary_layers(analytical_vocab)
    prepare_state = json.loads((pool_dir / "prepare.complete.json").read_text(encoding="utf-8"))
    if prepare_state.get("tokenizer_source_sha256") != tokenizer_sha:
        raise RuntimeError("Prepared shortlist uses a different tokenizer implementation")
    _validate_tokenizer_inventory_checkpoint(pool_dir, vocabulary_layers, tokenizer_sha)
    state_path = pool_dir / "embeddings.complete.json"
    if resume and state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("vocabulary_sha256") != vocab_sha:
            raise RuntimeError("Embedding checkpoint uses a different vocabulary")
        if state.get("diverse_config_sha256") != config_fingerprint:
            raise RuntimeError("Embedding checkpoint uses a different build configuration")
        if state.get("tokenizer_source_sha256") != tokenizer_sha:
            raise RuntimeError("Embedding checkpoint uses a different tokenizer implementation")
        return pool_dir / "embeddings"
    if not (pool_dir / "lexical.complete.json").exists():
        build_lexical_annotations(pool_dir, diverse_config=diverse_config, resume=resume)

    embedding_config = config.get("embedding", {})
    model_name = embedding_config.get("model", "intfloat/multilingual-e5-small")
    prefix = str(embedding_config.get("prefix", "query: "))
    model_manifest = export_quantized_e5(
        model_name, pool_dir / "model", resume=resume
    )
    model_path = pool_dir / "model" / model_manifest["onnx_relative_path"]
    tokenizer_path = model_path.parent
    max_workers = max(1, int(workers or os.cpu_count() or 1))
    configured_layouts = [
        (int(item[0]), int(item[1]))
        for item in embedding_config.get("layouts", [[4, 8], [8, 4], [16, 2]])
    ]
    layouts = [
        (processes, threads)
        for processes, threads in configured_layouts
        if processes * threads <= max_workers
    ]
    if not layouts:
        # Falling back silently would run the whole corpus through a single
        # process while the report still claims a benchmarked topology, so say
        # so loudly instead of hiding a large throughput loss.
        print(
            f"! No configured embedding layout fits {max_workers} workers "
            f"(configured: {configured_layouts}); falling back to 1 process x "
            f"{max_workers} threads with no topology comparison"
        )
        layouts = [(1, max_workers)]
    benchmark_size = int(embedding_config.get("benchmark_size", 10_000))
    # Use lexically annotated records for both benchmark and final encoding.
    lexical_files = sorted((pool_dir / "lexical").glob("part-*.jsonl"))
    sample = []
    for path in lexical_files:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    sample.append(json.loads(line))
                    if len(sample) >= benchmark_size:
                        break
        if len(sample) >= benchmark_size:
            break
    onnx_validation = _validate_onnx_pytorch_tolerance(
        sample,
        model_name=model_name,
        revision=model_manifest["model_revision"],
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        prefix=prefix,
    )
    benchmark_results = []
    with tempfile.TemporaryDirectory(prefix="e5-benchmark-") as temporary:
        temp = Path(temporary)
        for processes, threads in layouts:
            input_paths = _write_benchmark_shards(sample, temp / f"{processes}x{threads}", processes)
            tasks = [
                (str(path), str(path.with_suffix(".npy")), str(path.with_suffix(".ids")))
                for path in input_paths
            ]
            results, elapsed = _run_embedding_layout(
                tasks,
                model_path=model_path,
                tokenizer_path=tokenizer_path,
                processes=processes,
                threads=threads,
                batch_size=256,
                prefix=prefix,
            )
            total = sum(result["records"] for result in results)
            peak = sum(result["rss_mb"] for result in results)
            benchmark_results.append({
                "processes": processes,
                "threads_per_process": threads,
                "records": total,
                "seconds": round(elapsed, 3),
                "sentences_per_second": round(total / elapsed, 3) if elapsed else 0,
                "estimated_peak_rss_mb": round(peak, 2),
            })
    memory_limit_mb = float(embedding_config.get("memory_limit_gb", 96)) * 1024
    eligible_layouts = [
        result for result in benchmark_results
        if not result["estimated_peak_rss_mb"] or result["estimated_peak_rss_mb"] <= memory_limit_mb
    ]
    if not eligible_layouts:
        raise RuntimeError("Every embedding topology exceeded the configured memory limit")
    chosen = max(eligible_layouts, key=lambda result: result["sentences_per_second"])

    output_dir = pool_dir / "embeddings"
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        (
            str(path),
            str(output_dir / f"{path.stem}.npy"),
            str(output_dir / f"{path.stem}.ids"),
        )
        for path in lexical_files
    ]
    results, elapsed = _run_embedding_layout(
        tasks,
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        processes=int(chosen["processes"]),
        threads=int(chosen["threads_per_process"]),
        batch_size=256,
        prefix=prefix,
    )
    state = {
        "vocabulary_sha256": vocab_sha,
        "tokenizer_source_sha256": tokenizer_sha,
        "diverse_config_sha256": config_fingerprint,
        "model": model_manifest,
        "prefix": prefix,
        "benchmark": benchmark_results,
        "onnx_pytorch_validation": onnx_validation,
        "chosen_layout": chosen,
        "worker_start_method": "spawn",
        "fixed_topology_reproducibility_only": True,
        "elapsed_seconds": round(elapsed, 3),
        "shards": results,
    }
    _write_json(state_path, state)
    print(
        f"✓ Embedded {sum(item['records'] for item in results):,} candidates "
        f"with {chosen['processes']}×{chosen['threads_per_process']} CPU topology"
    )
    return output_dir


def _load_candidates_and_embeddings(pool_dir: Path):
    import numpy as np

    state = json.loads((pool_dir / "embeddings.complete.json").read_text(encoding="utf-8"))
    records: list[dict] = []
    lexical_dir = pool_dir / "lexical"
    # Size the destination from the shard headers first.  Collecting per-shard
    # float32 copies and calling ``np.concatenate`` would hold the shards and the
    # joined array at the same time, doubling the resident embedding matrix.
    shard_plan = []
    total_rows = 0
    dimensions = None
    for shard in state["shards"]:
        array_path = pool_dir / "embeddings" / shard["array"]
        header = np.load(array_path, mmap_mode="r")
        if dimensions is None:
            dimensions = int(header.shape[1])
        elif int(header.shape[1]) != dimensions:
            raise RuntimeError(f"Embedding dimension mismatch for {array_path}")
        shard_plan.append((shard, array_path, int(header.shape[0])))
        total_rows += int(header.shape[0])
        del header
    if not shard_plan or dimensions is None:
        raise RuntimeError("No embedding shards were produced")
    embeddings = np.empty((total_rows, dimensions), dtype=np.float32)
    offset = 0
    for shard, array_path, rows in shard_plan:
        input_path = lexical_dir / shard["input"]
        ids_path = pool_dir / "embeddings" / shard["ids"]
        with open(input_path, "r", encoding="utf-8") as handle:
            shard_records = [json.loads(line) for line in handle if line.strip()]
        ids = ids_path.read_text(encoding="utf-8").splitlines()
        if [row["normalized_sha256"] for row in shard_records] != ids:
            raise RuntimeError(f"Embedding ID order mismatch for {input_path}")
        if rows != len(shard_records):
            raise RuntimeError(f"Embedding count mismatch for {input_path}")
        array = np.load(array_path, mmap_mode="r")
        # Assigning into the preallocated slice widens float16 to float32 in
        # place, so no whole-shard temporary is materialized.
        embeddings[offset:offset + rows] = array
        del array
        offset += rows
        records.extend(shard_records)
    return records, embeddings, state


def _normalize_rows(values):
    import numpy as np

    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def _roc_auc(positive, negative) -> float:
    """Mann-Whitney AUC without adding scikit-learn to the runtime."""
    import numpy as np

    positive = np.asarray(positive, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)
    if not len(positive) or not len(negative):
        return 0.5
    values = np.concatenate([positive, negative])
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1)
    rank_sum = ranks[:len(positive)].sum()
    return float(
        (rank_sum - len(positive) * (len(positive) + 1) / 2)
        / (len(positive) * len(negative))
    )


@dataclass
class EmbeddingTransform:
    name: str
    mean: object | None = None
    components: object | None = None
    scales: object | None = None

    def apply(self, values):
        import numpy as np

        output = np.asarray(values, dtype=np.float32)
        if self.mean is not None:
            output = output - self.mean
        if self.components is not None:
            output = output @ self.components
        if self.scales is not None:
            output = output * self.scales
        return _normalize_rows(output).astype(np.float32, copy=False)


def _paired_cosines(values, left, right, *, block: int = 65_536):
    """Row-wise dot products for index pairs without materializing gathered copies.

    ``values[left] * values[right]`` would allocate three arrays the size of the
    pair list times the embedding width; on a 500k-pair background sample that is
    over 2 GB of transient. Blocking keeps the transient proportional to ``block``.
    """
    import numpy as np

    left = np.asarray(left, dtype=np.int64)
    right = np.asarray(right, dtype=np.int64)
    output = np.empty(len(left), dtype=np.float32)
    for start in range(0, len(left), block):
        stop = start + block
        # ``np.sum(..., axis=1)`` reduces each row independently, so blocking
        # yields bit-identical results to the unblocked expression.
        output[start:stop] = np.sum(
            values[left[start:stop]] * values[right[start:stop]], axis=1
        )
    return output


def calibrate_embeddings(
    records: list[dict],
    embeddings,
    *,
    config: dict,
    output_dir: Path,
    seed: int = 42,
):
    """Calibrate empirical cutoffs in the model's native normalized space."""
    import numpy as np

    embedding_config = config.get("embedding", {})
    size = min(len(records), int(embedding_config.get("calibration_size", 100_000)))
    rng = np.random.default_rng(seed)
    sample_indices = np.sort(rng.choice(len(records), size=size, replace=False))
    index_by_hash = {record["normalized_sha256"]: index for index, record in enumerate(records)}
    positive_pairs = []
    for index, record in enumerate(records):
        other = record.get("lexical_near_duplicate_of")
        if other in index_by_hash:
            positive_pairs.append((index, index_by_hash[other]))
        if len(positive_pairs) >= 100_000:
            break
    background_size = max(100_000, min(500_000, size * 5))
    left = rng.choice(sample_indices, size=background_size, replace=True)
    right = rng.choice(sample_indices, size=background_size, replace=True)
    different = left != right
    left, right = left[different], right[different]

    # The embedding space is fixed before inspecting lexical-neighbour labels.
    # Those pairs are retained only as a diagnostic, not as semantic ground truth.
    chosen_transform = EmbeddingTransform("raw")
    transformed = chosen_transform.apply(embeddings)
    background = _paired_cosines(transformed, left, right)
    lexical_pair_cosines = _paired_cosines(
        transformed,
        np.asarray([first for first, _ in positive_pairs], dtype=np.int64),
        np.asarray([second for _, second in positive_pairs], dtype=np.int64),
    )
    rates = [float(value) for value in config.get("selection", {}).get(
        "background_upper_tail_rates", [0.02, 0.01, 0.005]
    )]
    if rates != sorted(rates, reverse=True):
        raise ValueError(
            "background_upper_tail_rates must be strict-to-permissive (descending rates)"
        )
    thresholds = [float(np.quantile(background, 1.0 - rate)) for rate in rates]
    sorted_background = np.sort(background.astype(np.float32))
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "background_cosines.npy", sorted_background)
    np.save(output_dir / "transformed_embeddings.npy", transformed.astype(np.float16))
    calibration = {
        "config_sha256": _config_fingerprint(config),
        "tokenizer_source_sha256": _sha256(_TOKENIZER_SOURCE),
        "sample_size": size,
        "prefix": str(embedding_config.get("prefix", "query: ")),
        "chosen_transform": chosen_transform.name,
        "transform_selection_policy": "raw model space fixed before diagnostic labels",
        "background_pair_sampling": "random distinct shortlist records; not treated as labelled negatives",
        "random_background_pairs": len(background),
        "lexical_neighbour_pairs": len(lexical_pair_cosines),
        "lexical_pair_separation_auc_diagnostic_only": round(
            _roc_auc(lexical_pair_cosines, background), 8
        ),
        "background_percentiles": {
            str(percentile): round(float(np.quantile(background, percentile / 100)), 8)
            for percentile in (50, 90, 95, 98, 99, 99.5, 99.9)
        },
        "background_upper_tail_rates": rates,
        "derived_cosine_thresholds": [round(value, 8) for value in thresholds],
        "thresholds_are_empirical_not_absolute": True,
        "threshold_direction": "ascending cosine cutoff; later levels are more permissive",
    }
    _write_json(output_dir / "similarity_calibration.json", calibration)
    return transformed, sorted_background, thresholds, calibration, chosen_transform


def load_calibration(output_dir: Path):
    import numpy as np

    calibration = json.loads(
        (output_dir / "similarity_calibration.json").read_text(encoding="utf-8")
    )
    name = calibration["chosen_transform"]
    mean_path = output_dir / "transform_mean.npy"
    components_path = output_dir / "transform_components.npy"
    scales_path = output_dir / "transform_scales.npy"
    transform = EmbeddingTransform(
        name,
        mean=np.load(mean_path) if mean_path.exists() else None,
        components=np.load(components_path) if components_path.exists() else None,
        scales=np.load(scales_path) if scales_path.exists() else None,
    )
    transformed = np.asarray(
        np.load(output_dir / "transformed_embeddings.npy", mmap_mode="r"),
        dtype=np.float32,
    )
    background = np.load(output_dir / "background_cosines.npy", mmap_mode="r")
    thresholds = [float(value) for value in calibration["derived_cosine_thresholds"]]
    return transformed, background, thresholds, calibration, transform


def _empirical_percentile(sorted_values, value: float) -> float:
    import numpy as np

    return float(np.searchsorted(sorted_values, value, side="right") / max(1, len(sorted_values)))


def _tempered_target(inventory: dict, target_tokens: float, exponent: float) -> dict[str, float]:
    source = inventory["syllable_occurrence_frequency"]
    weights = {
        syllable: math.pow(max(1.0, float(source.get(syllable, 0))), exponent)
        for syllable in inventory["attainable_syllables"]
    }
    denominator = sum(weights.values()) or 1.0
    return {syllable: target_tokens * weight / denominator for syllable, weight in weights.items()}


def _metadata_novelty(record: dict, counts: dict[str, Counter]) -> float:
    axes = ("tense", "polarity", "gender", "sector")
    return sum(1.0 / (1 + counts[axis][record.get(axis, "unknown")]) for axis in axes) / len(axes)


def _candidate_balance_gain(record: dict, frequencies: Counter, target: dict[str, float]) -> float:
    counts = Counter(record["syllables"])
    gain = 0.0
    for syllable, amount in counts.items():
        deficit = max(0.0, target.get(syllable, 0.0) - frequencies[syllable])
        if deficit:
            gain += min(float(amount), deficit) / max(1.0, target.get(syllable, 1.0))
    return gain / math.sqrt(max(1, record["syllable_count"]))


def _nearest(index, vector, selected_indices: list[int]) -> tuple[float, int | None]:
    if not selected_indices:
        return -1.0, None
    scores, locations = index.search(vector.reshape(1, -1), 1)
    location = int(locations[0, 0])
    if location < 0:
        return -1.0, None
    return float(scores[0, 0]), selected_indices[location]


def _build_faiss_clusters(
    embeddings,
    *,
    clusters: int,
    seed: int,
    cache_path: Path | None = None,
    resume: bool = False,
    faiss_threads: int = 1,
):
    import faiss
    import numpy as np

    faiss.omp_set_num_threads(max(1, int(faiss_threads)))
    count, dimensions = embeddings.shape
    if resume and cache_path is not None and cache_path.exists():
        cached = np.load(cache_path)
        if len(cached) == count:
            return cached.astype(np.int32, copy=False), int(cached.max()) + 1
    clusters = max(1, min(clusters, count))
    rng = np.random.default_rng(seed)
    train_size = min(count, 300_000)
    training = embeddings[
        np.sort(rng.choice(count, size=train_size, replace=False))
    ].astype(np.float32)
    model = faiss.Kmeans(
        dimensions,
        clusters,
        niter=20,
        spherical=True,
        seed=seed,
        verbose=True,
    )
    model.train(training)
    assignments = np.empty(count, dtype=np.int32)
    for start in range(0, count, 50_000):
        _, labels = model.index.search(embeddings[start:start + 50_000].astype(np.float32), 1)
        assignments[start:start + len(labels)] = labels[:, 0]
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, assignments)
    return assignments, clusters


def select_diverse_records(
    records: list[dict],
    embeddings,
    background_cosines,
    thresholds: list[float],
    *,
    inventory: dict,
    config: dict,
    target_size: int,
    seed: int,
    checkpoint_dir: Path | None = None,
    resume: bool = False,
    faiss_threads: int = 1,
):
    import faiss
    import numpy as np

    selection_config = config.get("selection", {})
    exponent = float(selection_config.get("tempering_exponent", 0.5))
    rare_floor = int(selection_config.get("rare_floor", 5))
    estimated_tokens = target_size * (
        sum(record["syllable_count"] for record in records) / max(1, len(records))
    )
    target = _tempered_target(inventory, estimated_tokens, exponent)
    assignments, cluster_count = _build_faiss_clusters(
        embeddings,
        clusters=int(selection_config.get("faiss_clusters", 8192)),
        seed=seed,
        cache_path=checkpoint_dir / "cluster_assignments.npy" if checkpoint_dir else None,
        resume=resume,
        faiss_threads=faiss_threads,
    )
    weights = list(map(float, selection_config.get("score_weights", [0.60, 0.35, 0.05])))
    if len(weights) != 3 or not math.isclose(sum(weights), 1.0, abs_tol=1e-9):
        raise ValueError("selection.score_weights must contain three values summing to 1")
    by_syllable: dict[str, list[int]] = defaultdict(list)
    static_priority = []
    source_sentence_frequency = inventory["syllable_sentence_frequency"]
    cluster_population = Counter(map(int, assignments))
    for index, record in enumerate(records):
        rarity = sum(
            1 / math.sqrt(max(1, source_sentence_frequency.get(syllable, 1)))
            for syllable in record["unique_syllables"]
        ) / math.sqrt(max(1, record["syllable_count"]))
        novelty = 1 / math.sqrt(cluster_population[int(assignments[index])])
        static_priority.append(weights[0] * rarity + weights[1] * novelty)
        for syllable in record["unique_syllables"]:
            if len(by_syllable[syllable]) < 2048:
                by_syllable[syllable].append(index)

    dimensions = embeddings.shape[1]
    faiss.omp_set_num_threads(max(1, int(faiss_threads)))
    semantic_index = faiss.IndexFlatIP(dimensions)
    selected: list[int] = []
    selected_set: set[int] = set()
    selected_hashes: set[str] = set()
    selected_lexical_roots: set[str] = set()
    frequencies = Counter()
    selected_sentence_frequency = Counter()
    metadata_counts: dict[str, Counter] = defaultdict(Counter)
    cluster_selected = Counter()
    exceptions: list[dict] = []

    def commit(index: int, reason: str, *, allow_exception: bool) -> None:
        record = records[index]
        nearest_score, nearest_index = _nearest(semantic_index, embeddings[index], selected)
        root = record.get("lexical_near_duplicate_of") or record["normalized_sha256"]
        lexical_conflict = root in selected_lexical_roots
        semantic_conflict = nearest_score >= thresholds[0] if thresholds else False
        if allow_exception and (lexical_conflict or semantic_conflict):
            exceptions.append({
                "record_sha256": record["normalized_sha256"],
                "reason": reason,
                "required_syllable": (
                    reason.split(":", 1)[1] if reason.startswith("rare_floor:") else None
                ),
                "lexical_conflict": lexical_conflict,
                "nearest_sha256": records[nearest_index]["normalized_sha256"] if nearest_index is not None else None,
                "nearest_cosine": round(nearest_score, 8) if nearest_index is not None else None,
                "threshold": round(thresholds[0], 8) if thresholds else None,
            })
        selected.append(index)
        selected_set.add(index)
        selected_hashes.add(record["normalized_sha256"])
        selected_lexical_roots.add(root)
        semantic_index.add(embeddings[index:index + 1].astype(np.float32))
        frequencies.update(record["syllables"])
        selected_sentence_frequency.update(record["unique_syllables"])
        cluster_selected[int(assignments[index])] += 1
        for axis in ("tense", "polarity", "gender", "sector"):
            metadata_counts[axis][record.get(axis, "unknown")] += 1
        record["selection_reason"] = reason
        record["nearest_cosine_at_selection"] = round(nearest_score, 8) if nearest_index is not None else None
        record["semantic_cluster"] = int(assignments[index])
        record["similarity_exception"] = bool(allow_exception and (lexical_conflict or semantic_conflict))

    # Least-supported syllables are satisfied first; coverage/rare-floor wins.
    required = {
        syllable: min(rare_floor, int(inventory["syllable_sentence_frequency"].get(syllable, 0)))
        for syllable in inventory["attainable_syllables"]
    }
    while any(selected_sentence_frequency[syllable] < floor for syllable, floor in required.items()):
        syllable = min(
            (key for key, floor in required.items() if selected_sentence_frequency[key] < floor),
            key=lambda key: (len(by_syllable.get(key, [])), key),
        )
        candidates = [index for index in by_syllable.get(syllable, []) if index not in selected_set]
        if not candidates:
            raise RuntimeError(f"Rare floor cannot be satisfied for attainable syllable {syllable!r}")
        def seed_score(index: int):
            gain = sum(
                selected_sentence_frequency[token] < required.get(token, 0)
                for token in records[index]["unique_syllables"]
            )
            duplicate_penalty = bool(records[index].get("lexical_near_duplicate_of"))
            return gain, -duplicate_penalty, static_priority[index], -index
        commit(max(candidates, key=seed_score), f"rare_floor:{syllable}", allow_exception=True)
        if len(selected) > target_size:
            raise RuntimeError("Rare-floor seed exceeds requested corpus size")

    cluster_candidates: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        if index not in selected_set:
            cluster_candidates[int(assignments[index])].append(index)
    for indices in cluster_candidates.values():
        indices.sort(key=lambda index: (-static_priority[index], records[index]["normalized_sha256"]))
    pointers = defaultdict(int)
    deferred: dict[int, list[int]] = defaultdict(list)
    initial_cap = int(selection_config.get("initial_cluster_cap", 7))

    for threshold_level, threshold in enumerate(thresholds):
        cluster_cap = initial_cap + threshold_level
        progress = True
        while len(selected) < target_size and progress:
            progress = False
            for cluster in range(cluster_count):
                if len(selected) >= target_size:
                    break
                if cluster_selected[cluster] >= cluster_cap:
                    continue
                indices = cluster_candidates.get(cluster, [])
                sample = []
                while pointers[cluster] < len(indices) and len(sample) < 20:
                    index = indices[pointers[cluster]]
                    pointers[cluster] += 1
                    if index not in selected_set:
                        sample.append(index)
                best = None
                best_score = -math.inf
                for index in sample:
                    record = records[index]
                    root = record.get("lexical_near_duplicate_of") or record["normalized_sha256"]
                    if root in selected_lexical_roots:
                        continue
                    nearest_score, _ = _nearest(semantic_index, embeddings[index], selected)
                    if nearest_score >= threshold:
                        deferred[threshold_level].append(index)
                        continue
                    balance = _candidate_balance_gain(record, frequencies, target)
                    novelty = 1.0 - _empirical_percentile(background_cosines, nearest_score)
                    metadata = _metadata_novelty(record, metadata_counts)
                    score = weights[0] * balance + weights[1] * novelty + weights[2] * metadata
                    if score > best_score or (score == best_score and (best is None or index < best)):
                        best, best_score = index, score
                if best is not None:
                    tail_rates = selection_config.get(
                        "background_upper_tail_rates", [0.02, 0.01, 0.005]
                    )
                    commit(
                        best,
                        f"tempered_semantic:background_upper_tail={tail_rates[threshold_level]}",
                        allow_exception=False,
                    )
                    cluster_candidates[cluster].extend(
                        index for index in sample if index != best and index not in selected_set
                    )
                    progress = True
        if len(selected) >= target_size:
            break
        # Reconsider deferred candidates under the relaxed empirical cutoff.
        for earlier in range(threshold_level + 1):
            for index in deferred[earlier]:
                cluster_candidates[int(assignments[index])].append(index)
            deferred[earlier].clear()
        for cluster, indices in cluster_candidates.items():
            indices[pointers[cluster]:] = sorted(
                indices[pointers[cluster]:],
                key=lambda index: (-static_priority[index], records[index]["normalized_sha256"]),
            )

    # At the final calibrated threshold, increase only the cluster cap if needed.
    final_threshold = thresholds[-1]
    cluster_cap = initial_cap + len(thresholds)
    while len(selected) < target_size and cluster_cap <= max(initial_cap + 64, target_size):
        progress = False
        for cluster in range(cluster_count):
            if len(selected) >= target_size:
                break
            if cluster_selected[cluster] >= cluster_cap:
                continue
            for index in cluster_candidates.get(cluster, []):
                if index in selected_set:
                    continue
                record = records[index]
                root = record.get("lexical_near_duplicate_of") or record["normalized_sha256"]
                if root in selected_lexical_roots:
                    continue
                nearest_score, _ = _nearest(semantic_index, embeddings[index], selected)
                if nearest_score >= final_threshold:
                    continue
                commit(index, "tempered_semantic:final_cluster_relaxation", allow_exception=False)
                progress = True
                break
        if not progress:
            break
        cluster_cap += 1
    if len(selected) != target_size:
        raise RuntimeError(
            f"Calibrated semantic selection produced {len(selected):,}/{target_size:,} records"
        )
    return [records[index] for index in selected], {
        "target": target,
        "frequencies": frequencies,
        "selected_sentence_frequency": selected_sentence_frequency,
        "exceptions": exceptions,
        "weights": weights,
        "weight_policy": "fixed_checked_configuration",
        "cluster_count": cluster_count,
        "cluster_selected": dict(cluster_selected),
        "faiss_threads": max(1, int(faiss_threads)),
        "semantic_threshold_index": "exact_IndexFlatIP",
    }


def _nearest_neighbor_statistics(embeddings, *, faiss_threads: int = 1) -> dict:
    import faiss
    import numpy as np

    if len(embeddings) < 2:
        return {"count": len(embeddings), "median": 0, "p95": 0, "p99": 0, "max": 0}
    faiss.omp_set_num_threads(max(1, int(faiss_threads)))
    index = faiss.IndexFlatIP(embeddings.shape[1])
    values = embeddings.astype(np.float32)
    index.add(values)
    nearest = []
    for start in range(0, len(values), 2048):
        scores, locations = index.search(values[start:start + 2048], 2)
        for offset, candidates in enumerate(locations):
            row = start + offset
            score = -1.0
            for column, location in enumerate(candidates):
                if location >= 0 and int(location) != row:
                    score = float(scores[offset, column])
                    break
            nearest.append(score)
    nearest = np.asarray(nearest, dtype=np.float32)
    return {
        "count": len(nearest),
        "median": round(float(np.quantile(nearest, 0.5)), 8),
        "p95": round(float(np.quantile(nearest, 0.95)), 8),
        "p99": round(float(np.quantile(nearest, 0.99)), 8),
        "max": round(float(nearest.max()), 8),
        "search": "exact_IndexFlatIP",
        "faiss_threads": max(1, int(faiss_threads)),
    }


def _read_corpus_records(path: Path, vocab: frozenset[str]) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            text = normalize_text(record.get("text", ""))
            syllables = [
                token for token in tokenize(text, vocab)
                if token.strip() and token not in _SKIP_TOKENS
            ]
            record["text"] = text
            record["syllables"] = syllables
            record["unique_syllables"] = sorted(set(syllables))
            record["normalized_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
            records.append(record)
    return records


def _holdout_semantic_comparison(
    selected: list[dict],
    baseline: list[dict],
    *,
    model_name: str,
    prefix: str,
    faiss_threads: int,
) -> dict:
    """Evaluate both corpora with a model never used during candidate selection."""
    import numpy as np
    import torch
    from huggingface_hub import model_info
    from sentence_transformers import SentenceTransformer

    revision = model_info(model_name).sha
    torch.set_num_threads(max(1, int(faiss_threads)))
    model = SentenceTransformer(model_name, revision=revision, device="cpu")

    def encode(records: list[dict]):
        values = model.encode(
            [prefix + record["text"] for record in records],
            batch_size=128,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        return np.asarray(values, dtype=np.float32)

    selected_vectors = encode(selected)
    baseline_vectors = encode(baseline)
    result = {
        "model": model_name,
        "model_revision": revision,
        "prefix": prefix,
        "role": "held-out evaluation only; never used for shortlist or selection",
        "selected_nearest_neighbor_cosine": _nearest_neighbor_statistics(
            selected_vectors, faiss_threads=faiss_threads
        ),
        "baseline_nearest_neighbor_cosine": _nearest_neighbor_statistics(
            baseline_vectors, faiss_threads=faiss_threads
        ),
    }
    return result


def _write_distribution_table(
    path: Path,
    frequencies: Counter,
    *,
    analytical_vocab: frozenset[str],
    attainable: frozenset[str],
) -> list[dict]:
    total = sum(frequencies.values())
    rows = [
        {
            "syllable": syllable,
            "count": frequencies.get(syllable, 0),
            "relative_frequency": frequencies.get(syllable, 0) / total if total else 0,
            "attainable": syllable in attainable,
            "observed": frequencies.get(syllable, 0) > 0,
        }
        for syllable in sorted(analytical_vocab, key=lambda item: (-frequencies.get(item, 0), item))
    ]
    with open(path.with_suffix(".csv"), "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["syllable", "count"])
        writer.writeheader()
        writer.writerows(rows)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        pq.write_table(pa.Table.from_pylist(rows), path.with_suffix(".parquet"))
    except ImportError:
        pass
    return rows


def _write_figures(report_dir: Path, frequencies: Counter, nearest: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    figure_dir = report_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(frequencies.values(), reverse=True)
    if ordered:
        fig, axis = plt.subplots(figsize=(9, 5))
        axis.loglog(range(1, len(ordered) + 1), ordered)
        axis.set(title="Syllable frequency-rank distribution", xlabel="Rank", ylabel="Occurrences")
        axis.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(figure_dir / "syllable_frequency_rank.png", dpi=160)
        plt.close(fig)

        cumulative = []
        running = 0
        total = sum(ordered)
        for count in ordered:
            running += count
            cumulative.append(running / total)
        fig, axis = plt.subplots(figsize=(9, 5))
        axis.plot(range(1, len(cumulative) + 1), cumulative)
        axis.set(title="Cumulative syllable-token coverage", xlabel="Syllable types", ylabel="Token fraction")
        axis.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(figure_dir / "cumulative_syllable_coverage.png", dpi=160)
        plt.close(fig)


def _target_size_label(target_size: int) -> str:
    return f"{target_size // 1000}k" if target_size % 1000 == 0 else str(target_size)


def _validate_baseline_size(baseline: str | Path, target_size: int) -> int:
    """Require a size-matched baseline before enabling comparative gates."""
    path = Path(baseline)
    if not path.is_file():
        raise FileNotFoundError(f"Baseline corpus not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        records = sum(bool(line.strip()) for line in handle)
    if records != target_size:
        raise ValueError(
            f"Comparative acceptance requires a size-matched baseline: "
            f"candidate target is {target_size:,}, baseline contains {records:,}. "
            "Omit --baseline for an exploratory build or provide an equal-size baseline."
        )
    return records


def build_diverse_final(
    *,
    pool_dir: str | Path,
    output: str | Path,
    target_size: int = 50_000,
    diverse_config: str | Path = _DEFAULT_DIVERSE_CONFIG,
    workers: int | None = None,
    resume: bool = False,
    baseline: str | Path | None = None,
    seed: int = 42,
) -> Path:
    """Run lexical, embedding, calibration, selection and verified reporting."""
    if baseline:
        _validate_baseline_size(baseline, target_size)
    _require_ml_stack()
    import numpy as np

    pool_dir = Path(pool_dir)
    output = Path(output)
    report_dir = output.parent / "reports" / output.stem
    completed_manifest = report_dir / "manifest.json"
    config = _load_yaml(diverse_config)
    config_fingerprint = _config_fingerprint(config)
    if resume and output.exists() and completed_manifest.exists():
        completed = json.loads(completed_manifest.read_text(encoding="utf-8"))
        expected_baseline_sha = _sha256(baseline) if baseline else None
        if (
            _sha256(output) == completed.get("output_sha256")
            and completed.get("diverse_config_sha256") == config_fingerprint
            and completed.get("target_size") == target_size
            and completed.get("baseline_sha256") == expected_baseline_sha
        ):
            print(f"✓ Final corpus already complete: {output}")
            return output
    vocab_path, analytical_vocab, vocab_sha = _resolve_lookup(config)
    tokenizer_sha = _sha256(_TOKENIZER_SOURCE)
    vocabulary_layers = _vocabulary_layers(analytical_vocab)
    selection_vocab = vocabulary_layers["selection_vocabulary"]
    tokenizer_emittable = vocabulary_layers["tokenizer_emittable"]
    tokenizer_unemittable = vocabulary_layers["tokenizer_unemittable"]
    faiss_threads = max(1, int(workers or os.cpu_count() or 1))
    prepare_state = json.loads((pool_dir / "prepare.complete.json").read_text(encoding="utf-8"))
    if prepare_state.get("vocabulary_sha256") != vocab_sha:
        raise RuntimeError("Preparation used a different pinned vocabulary")
    if prepare_state.get("diverse_config_sha256") != config_fingerprint:
        raise RuntimeError("Preparation used a different diverse-build configuration")
    if prepare_state.get("tokenizer_source_sha256") != tokenizer_sha:
        raise RuntimeError("Preparation used a different tokenizer implementation")
    _validate_tokenizer_inventory_checkpoint(pool_dir, vocabulary_layers, tokenizer_sha)
    build_lexical_annotations(pool_dir, diverse_config=diverse_config, resume=resume)
    embed_shortlist(
        pool_dir,
        diverse_config=diverse_config,
        workers=workers,
        resume=resume,
    )
    records, raw_embeddings, embedding_state = _load_candidates_and_embeddings(pool_dir)
    _log_rss("candidate and embedding load")
    calibration_dir = report_dir / "calibration"
    if resume and (calibration_dir / "similarity_calibration.json").exists():
        transformed, background, thresholds, calibration, _transform = load_calibration(
            calibration_dir
        )
        if calibration.get("config_sha256") != config_fingerprint:
            raise RuntimeError("Calibration checkpoint uses a different build configuration")
        if calibration.get("tokenizer_source_sha256") != tokenizer_sha:
            raise RuntimeError("Calibration checkpoint uses a different tokenizer implementation")
        if len(transformed) != len(records):
            raise RuntimeError("Calibration checkpoint candidate count changed")
    else:
        transformed, background, thresholds, calibration, _transform = calibrate_embeddings(
            records,
            raw_embeddings,
            config=config,
            output_dir=calibration_dir,
            seed=seed,
        )
    # ``transformed`` is a freshly allocated matrix in both branches above, so the
    # raw float32 copy is dead from here on.  Holding both for the remainder of the
    # build would keep a second candidates x dimensions matrix resident for nothing.
    del raw_embeddings
    _log_rss("calibration")
    inventory = json.loads((pool_dir / "frequency_inventory.json").read_text(encoding="utf-8"))
    attainable = frozenset(inventory["attainable_syllables"])
    checkpoint_dir = report_dir / "checkpoints"
    selection_records_path = checkpoint_dir / "selected_records.jsonl"
    selection_state_path = checkpoint_dir / "selection_state.json"
    if resume and selection_records_path.exists() and selection_state_path.exists():
        selected = [
            json.loads(line)
            for line in selection_records_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        selection = json.loads(selection_state_path.read_text(encoding="utf-8"))
        checkpoint = selection.get("checkpoint", {})
        if checkpoint != {
            "config_sha256": config_fingerprint,
            "vocabulary_sha256": vocab_sha,
            "tokenizer_source_sha256": tokenizer_sha,
            "target_size": target_size,
            "seed": seed,
            "faiss_threads": faiss_threads,
        }:
            raise RuntimeError("Selection checkpoint does not match this fixed build topology")
        selection["frequencies"] = Counter(selection["frequencies"])
        selection["selected_sentence_frequency"] = Counter(
            selection["selected_sentence_frequency"]
        )
        if len(selected) != target_size:
            raise RuntimeError("Selection checkpoint target size changed")
    else:
        selected, selection = select_diverse_records(
            records,
            transformed,
            background,
            thresholds,
            inventory=inventory,
            config=config,
            target_size=target_size,
            seed=seed,
            checkpoint_dir=checkpoint_dir,
            resume=resume,
            faiss_threads=faiss_threads,
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        with open(selection_records_path, "w", encoding="utf-8") as handle:
            for record in selected:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        serializable_selection = dict(selection)
        serializable_selection["frequencies"] = dict(selection["frequencies"])
        serializable_selection["selected_sentence_frequency"] = dict(
            selection["selected_sentence_frequency"]
        )
        serializable_selection["checkpoint"] = {
            "config_sha256": config_fingerprint,
            "vocabulary_sha256": vocab_sha,
            "tokenizer_source_sha256": tokenizer_sha,
            "target_size": target_size,
            "seed": seed,
            "faiss_threads": faiss_threads,
        }
        _write_json(selection_state_path, serializable_selection)
    _log_rss("selection")
    frequencies = Counter(selection["frequencies"])
    distribution = distribution_statistics(frequencies, inventory=attainable)
    target = selection["target"]
    jsd = jensen_shannon_divergence(frequencies, target, inventory=attainable)
    selected_indices = {record["normalized_sha256"]: index for index, record in enumerate(records)}
    selected_vectors = np.asarray(
        [transformed[selected_indices[record["normalized_sha256"]]] for record in selected],
        dtype=np.float32,
    )
    nearest = _nearest_neighbor_statistics(selected_vectors, faiss_threads=faiss_threads)
    baseline_report = None
    holdout_semantic = None
    comparative_acceptance: dict[str, bool] = {}
    construction_invariants = {
        "exact_size": len(selected) == target_size,
        "unique_hashes": len({record["normalized_sha256"] for record in selected}) == target_size,
        "complete_attainable_coverage": attainable.issubset(frequencies),
        "rare_floor_met": all(
            selection["selected_sentence_frequency"].get(syllable, 0)
            >= min(
                int(config.get("selection", {}).get("rare_floor", 5)),
                int(inventory["syllable_sentence_frequency"].get(syllable, 0)),
            )
            for syllable in attainable
        ),
        "all_tokens_in_pinned_selection_vocabulary": set(frequencies).issubset(selection_vocab),
        "frequency_total_matches": sum(frequencies.values())
        == sum(len(record["syllables"]) for record in selected),
    }
    if baseline:
        baseline_path = Path(baseline)
        if not baseline_path.is_file():
            raise FileNotFoundError(f"Baseline corpus not found: {baseline_path}")
        # Stored legacy token arrays are deliberately ignored; both corpora are
        # retokenized from normalized text with the same pinned tokenizer.
        baseline_records = _read_corpus_records(baseline_path, analytical_vocab)
        baseline_frequency = Counter(token for record in baseline_records for token in record["syllables"])
        shared_support = attainable.intersection(baseline_frequency)
        if not shared_support:
            raise RuntimeError("Baseline and new corpus have no shared observed syllable support")
        selected_shared_distribution = distribution_statistics(
            frequencies, inventory=shared_support
        )
        baseline_shared_distribution = distribution_statistics(
            baseline_frequency, inventory=shared_support
        )
        selected_shared_jsd = jensen_shannon_divergence(
            frequencies, target, inventory=shared_support
        )
        baseline_shared_jsd = jensen_shannon_divergence(
            baseline_frequency, target, inventory=shared_support
        )
        embedding_config = config.get("embedding", {})
        holdout_semantic = _holdout_semantic_comparison(
            selected,
            baseline_records,
            model_name=str(
                embedding_config.get("evaluation_model", "sentence-transformers/LaBSE")
            ),
            prefix=str(embedding_config.get("evaluation_prefix", "")),
            faiss_threads=faiss_threads,
        )
        selected_holdout = holdout_semantic["selected_nearest_neighbor_cosine"]
        baseline_holdout = holdout_semantic["baseline_nearest_neighbor_cosine"]
        baseline_report = {
            "path": str(baseline_path),
            "sha256": _sha256(baseline_path),
            "records": len(baseline_records),
            "tokenization": "retokenized normalized text with pinned lookup vocabulary",
            "observed_syllables": len(baseline_frequency),
            "shared_comparison_support_count": len(shared_support),
            "shared_comparison_support": sorted(shared_support),
            "selected_distribution_on_shared_support": selected_shared_distribution,
            "baseline_distribution_on_shared_support": baseline_shared_distribution,
            "selected_jsd_on_shared_support": selected_shared_jsd,
            "baseline_jsd_on_shared_support": baseline_shared_jsd,
            "held_out_semantic_evaluation": holdout_semantic,
        }
        comparative_acceptance = {
            "normalized_entropy_improved_on_shared_support": (
                selected_shared_distribution["normalized_entropy"]
                > baseline_shared_distribution["normalized_entropy"]
            ),
            "gini_improved_on_shared_support": (
                selected_shared_distribution["gini"]
                < baseline_shared_distribution["gini"]
            ),
            "target_jsd_improved_on_shared_support": selected_shared_jsd < baseline_shared_jsd,
            "heldout_median_similarity_improved": (
                selected_holdout["median"] < baseline_holdout["median"]
            ),
            "heldout_p95_similarity_improved": (
                selected_holdout["p95"] < baseline_holdout["p95"]
            ),
        }

    source_counts = Counter(
        source for record in selected for source in record.get("source_corpora", [])
    )
    metadata_counts = {
        axis: dict(Counter(record.get(axis, "unknown") for record in selected))
        for axis in ("tense", "polarity", "gender", "sector")
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    _write_distribution_table(
        report_dir / "syllable_frequency",
        frequencies,
        analytical_vocab=analytical_vocab,
        attainable=attainable,
    )
    with open(report_dir / "coverage_exceptions.csv", "w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "record_sha256", "reason", "required_syllable", "lexical_conflict", "nearest_sha256",
            "nearest_cosine", "threshold",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selection["exceptions"])
    _write_json(report_dir / "source_distribution.json", dict(source_counts))
    _write_json(report_dir / "metadata_distribution.json", metadata_counts)
    with open(report_dir / "source_distribution.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source", "record_memberships"])
        writer.writeheader()
        writer.writerows(
            {"source": source, "record_memberships": count}
            for source, count in sorted(source_counts.items())
        )
    with open(report_dir / "metadata_distribution.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["axis", "label", "count"])
        writer.writeheader()
        writer.writerows(
            {"axis": axis, "label": label, "count": count}
            for axis, values in sorted(metadata_counts.items())
            for label, count in sorted(values.items())
        )
    (report_dir / "unobserved_syllables.txt").write_text(
        "".join(syllable + "\n" for syllable in sorted(selection_vocab.difference(frequencies))),
        encoding="utf-8",
    )
    lexical_state = json.loads((pool_dir / "lexical.complete.json").read_text(encoding="utf-8"))
    _write_json(report_dir / "similarity_metrics.json", {
        "calibration": calibration,
        "selector_model_final_nearest_neighbor_cosine_diagnostic": nearest,
        "held_out_baseline_comparison": holdout_semantic,
        "coverage_similarity_exceptions": len(selection["exceptions"]),
    })
    _write_figures(report_dir, frequencies, nearest)
    report = {
        "run_id": _utc_run_id(),
        "target_size": target_size,
        "vocabulary": {
            "sha256": vocab_sha,
            "tokenizer_source_sha256": tokenizer_sha,
            "raw_entries": int(config["lookup_vocabulary"]["raw_entries"]),
            "analytical_entries": len(analytical_vocab),
            "selection_entries": len(selection_vocab),
            "tokenizer_emittable_entries": len(tokenizer_emittable),
            "tokenizer_unemittable_entries": len(tokenizer_unemittable),
            "tokenizer_unemittable_syllables": sorted(tokenizer_unemittable),
            "source_observed_entries": int(inventory["source_observed_count"]),
            "attainable_entries": len(attainable),
        },
        "coverage_layers": {
            "tokenizer_structural_coverage": {
                "numerator": len(tokenizer_emittable),
                "denominator": len(selection_vocab),
                "ratio": len(tokenizer_emittable) / max(1, len(selection_vocab)),
            },
            "five_corpus_source_support": {
                "numerator": len(attainable),
                "denominator": len(tokenizer_emittable),
                "ratio": len(attainable) / max(1, len(tokenizer_emittable)),
            },
            "final_selection_coverage": {
                "numerator": len(attainable.intersection(frequencies)),
                "denominator": len(attainable),
                "ratio": len(attainable.intersection(frequencies)) / max(1, len(attainable)),
            },
        },
        "syllable_tokens": sum(frequencies.values()),
        "observed_syllables": len(frequencies),
        "distribution": distribution,
        "rarity_counts": rarity_counts(frequencies),
        "jensen_shannon_divergence_to_tempered_target": jsd,
        "nearest_neighbor_cosine": nearest,
        "selected_score_weights": selection["weights"],
        "score_weight_policy": selection["weight_policy"],
        "source_distribution": dict(source_counts),
        "metadata_distribution": metadata_counts,
        "similarity_exceptions": len(selection["exceptions"]),
        "lexical_near_duplicates_identified": lexical_state["lexical_near_duplicates"],
        "baseline": baseline_report,
        "construction_invariants": construction_invariants,
        "comparative_acceptance": comparative_acceptance,
        "acceptance": {**construction_invariants, **comparative_acceptance},
    }
    _write_json(report_dir / "report.json", report)
    required_checks = {**construction_invariants, **comparative_acceptance}
    required_acceptance = all(required_checks.values())
    if not required_acceptance:
        failed = [name for name, passed in required_checks.items() if not passed]
        raise RuntimeError(
            "Final corpus acceptance failed; no JSONL written: " + ", ".join(failed)
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        for index, record in enumerate(selected):
            record["id"] = f"diverse_final_{index:08d}"
            record["batch_id"] = f"final_{_target_size_label(target_size)}_diverse_rare"
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest = {
        "run_id": report["run_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "workers_requested": int(workers or os.cpu_count() or 1),
        "fixed_embedding_topology": embedding_state["chosen_layout"],
        "seed": seed,
        "target_size": target_size,
        "diverse_config_sha256": config_fingerprint,
        "baseline_sha256": _sha256(baseline) if baseline else None,
        "vocabulary_sha256": vocab_sha,
        "tokenizer_source_sha256": tokenizer_sha,
        "model": embedding_state["model"],
        "dataset_revisions": prepare_state.get("dataset_revisions"),
        "source_manifest_sha256": prepare_state.get("source_manifest_sha256"),
        "output": str(output),
        "output_sha256": _sha256(output),
        "report_sha256": _sha256(report_dir / "report.json"),
        "reproducibility": {
            "preparation": (
                "artifact equality must be verified for the same Python version, "
                "architecture, seed and normalized inputs"
            ),
            "embedding_and_faiss": (
                "reproducible only with fixed seed, software, model revisions, "
                "quantization, machine and thread topology"
            ),
        },
        "faiss_threads": faiss_threads,
        "semantic_threshold_index": "exact_IndexFlatIP",
    }
    manifest["artifact_checksums"] = {
        str(path.relative_to(report_dir)): _sha256(path)
        for path in sorted(report_dir.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    _write_json(report_dir / "manifest.json", manifest)
    print(f"✓ Verified diverse final corpus: {output}")
    print(f"✓ Report directory: {report_dir}")
    return output


def update_progress_markdown(
    *,
    run_report: str | Path,
    progress_file: str | Path,
) -> Path:
    """Idempotently add one verified run section to the Markdown progress log."""
    report_path = Path(run_report)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    run_id = str(report["run_id"])
    progress_file = Path(progress_file)
    if progress_file.exists():
        content = progress_file.read_text(encoding="utf-8")
    else:
        content = (
            "# Technical Progress Report: Nepali ASR Corpus\n\n"
            "This Markdown file is the authoritative daily progress record. "
            "The earlier LaTeX report is retained as historical material.\n\n"
            "## Daily verified runs\n"
        )
    start = f"<!-- run:{run_id}:start -->"
    end = f"<!-- run:{run_id}:end -->"
    distribution = report["distribution"]
    nearest = report["nearest_neighbor_cosine"]
    coverage = report.get("coverage_layers", {})
    structural = coverage.get("tokenizer_structural_coverage", {})
    source_support = coverage.get("five_corpus_source_support", {})
    final_coverage = coverage.get("final_selection_coverage", {})
    comparison = report.get("comparative_acceptance", {})
    comparison_status = (
        "passed" if comparison and all(comparison.values())
        else "not run" if not comparison
        else "failed"
    )
    block = f"""{start}
### {run_id}: Diverse rare-aware {_target_size_label(report['target_size'])} corpus

- Records: {report['target_size']:,}
- Attainable syllables: {report['vocabulary']['attainable_entries']:,}
- Observed syllables: {report['observed_syllables']:,}
- Tokenizer structural coverage: {structural.get('numerator', 0):,} / {structural.get('denominator', 0):,}
- Five-corpus source support: {source_support.get('numerator', 0):,} / {source_support.get('denominator', 0):,}
- Final source-supported coverage: {final_coverage.get('numerator', report['observed_syllables']):,} / {final_coverage.get('denominator', report['vocabulary']['attainable_entries']):,}
- Recognized syllable tokens: {report['syllable_tokens']:,}
- Normalized Shannon entropy: {distribution['normalized_entropy']:.8f}
- Gini coefficient: {distribution['gini']:.8f}
- Jensen-Shannon divergence from tempered target: {report['jensen_shannon_divergence_to_tempered_target']:.8f}
- CV (diagnostic only): {distribution['coefficient_of_variation']:.4f}
- Selector-model nearest-neighbor cosine median / p95 (diagnostic): {nearest['median']:.8f} / {nearest['p95']:.8f}
- Similarity exceptions required for rare coverage: {report['similarity_exceptions']:,}
- Construction invariants: {'passed' if all(report.get('construction_invariants', report['acceptance']).values()) else 'failed'}
- Held-out baseline comparison: {comparison_status}
- Machine-readable report: `{report_path}`
{end}
"""
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.DOTALL)
    if pattern.search(content):
        content = pattern.sub(block.strip(), content)
    else:
        content = content.rstrip() + "\n\n" + block
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    progress_file.write_text(content.rstrip() + "\n", encoding="utf-8")
    print(f"✓ Updated Markdown progress report: {progress_file}")
    return progress_file
