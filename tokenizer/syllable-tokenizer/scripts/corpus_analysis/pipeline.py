#!/usr/bin/env python3
"""Cloud-ready CLI for exact, streaming multi-corpus syllable analysis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import sqlite3
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

_SCRIPT_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_SCRIPT_DIR))

from corpus_analysis.metrics import Accumulator, RecordMetrics, analyze_batch, write_result  # noqa: E402
from corpus_analysis.sources import (  # noqa: E402
    CorpusSpec,
    iter_records,
    load_config,
    validate_source,
)
from syllabic_tokenizer import get_lookup_tokens  # noqa: E402


_WORKER_VOCAB: frozenset[str] = frozenset()


def _init_worker(lookup_vocab: frozenset[str]) -> None:
    global _WORKER_VOCAB
    _WORKER_VOCAB = lookup_vocab


def _worker(records: list[dict]):
    return analyze_batch(records, _WORKER_VOCAB)


def _batched(records: Iterable[dict], batch_size: int) -> Iterable[list[dict]]:
    batch = []
    for record in records:
        batch.append(record)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


class DedupIndex:
    """Disk-backed exact normalized-text set and corpus-membership index."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute("PRAGMA temp_store=FILE")
        self.connection.execute(
            "CREATE TABLE texts (hash BLOB PRIMARY KEY, first_corpus TEXT NOT NULL) WITHOUT ROWID"
        )
        self.connection.execute(
            "CREATE TABLE membership ("
            "hash BLOB NOT NULL, corpus TEXT NOT NULL, "
            "PRIMARY KEY (hash, corpus)) WITHOUT ROWID"
        )
        self.connection.execute("CREATE INDEX membership_corpus ON membership(corpus)")
        self.deduplicated = Accumulator()

    def add(self, corpus: str, records: list[RecordMetrics]) -> None:
        first_by_hash: dict[bytes, RecordMetrics] = {}
        for record in records:
            first_by_hash.setdefault(record.digest, record)
        hashes = list(first_by_hash)
        existing: set[bytes] = set()
        for start in range(0, len(hashes), 500):
            group = hashes[start:start + 500]
            placeholders = ",".join("?" for _ in group)
            if group:
                existing.update(
                    row[0]
                    for row in self.connection.execute(
                        f"SELECT hash FROM texts WHERE hash IN ({placeholders})", group
                    )
                )
        new_hashes = [digest for digest in hashes if digest not in existing]
        self.connection.executemany(
            "INSERT INTO texts(hash, first_corpus) VALUES (?, ?)",
            ((digest, corpus) for digest in new_hashes),
        )
        self.connection.executemany(
            "INSERT OR IGNORE INTO membership(hash, corpus) VALUES (?, ?)",
            ((digest, corpus) for digest in hashes),
        )
        for digest in new_hashes:
            self.deduplicated.add_unique_record(first_by_hash[digest])

    def finish_corpus(self) -> None:
        self.connection.commit()

    def distinct_count(self, corpus: str) -> int:
        return self.connection.execute(
            "SELECT COUNT(*) FROM membership WHERE corpus = ?", (corpus,)
        ).fetchone()[0]

    def retained_count(self, corpus: str) -> int:
        return self.connection.execute(
            "SELECT COUNT(*) FROM texts WHERE first_corpus = ?", (corpus,)
        ).fetchone()[0]

    def unique_contributions(self) -> dict[str, int]:
        rows = self.connection.execute(
            "SELECT m.corpus, COUNT(*) FROM membership m JOIN "
            "(SELECT hash FROM membership GROUP BY hash HAVING COUNT(*) = 1) unique_hashes "
            "ON m.hash = unique_hashes.hash GROUP BY m.corpus"
        )
        return {corpus: count for corpus, count in rows}

    def pairwise_intersection(self, left: str, right: str) -> int:
        return self.connection.execute(
            "SELECT COUNT(*) FROM membership a JOIN membership b ON a.hash = b.hash "
            "WHERE a.corpus = ? AND b.corpus = ?",
            (left, right),
        ).fetchone()[0]

    def total_unique(self) -> int:
        return self.connection.execute("SELECT COUNT(*) FROM texts").fetchone()[0]

    def close(self) -> None:
        self.connection.commit()
        self.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self.connection.close()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _configure_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("corpus_analysis")
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def _close_logger(logger: logging.Logger) -> None:
    for handler in logger.handlers:
        handler.flush()
        handler.close()
    logger.handlers.clear()


def _process_corpus(
    spec: CorpusSpec,
    input_root: Path,
    lookup_vocab: frozenset[str],
    workers: int,
    batch_size: int,
    max_records: int | None,
    dedup: DedupIndex,
    logger: logging.Logger,
) -> Accumulator:
    accumulator = Accumulator()
    records = iter_records(
        spec,
        input_root,
        read_batch_size=max(2_048, batch_size),
        max_records=max_records,
    )

    def consume(result) -> None:
        batch_accumulator, batch_details = result
        accumulator.merge(batch_accumulator)
        dedup.add(spec.slug, batch_details)
        if accumulator.input_records and accumulator.input_records % 100_000 < batch_size:
            logger.info(
                "%s: processed %s records; %s tokens; %s observed syllables",
                spec.name,
                f"{accumulator.input_records:,}",
                f"{accumulator.total_tokens:,}",
                f"{len(accumulator.frequencies):,}",
            )

    if workers == 1:
        for batch in _batched(records, batch_size):
            consume(analyze_batch(batch, lookup_vocab))
    else:
        max_pending = max(workers * 2, 2)
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(lookup_vocab,),
        ) as executor:
            pending = set()
            for batch in _batched(records, batch_size):
                pending.add(executor.submit(_worker, batch))
                if len(pending) >= max_pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        consume(future.result())
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    consume(future.result())
    dedup.finish_corpus()
    return accumulator


def _validate_results(
    per_corpus: dict[str, Accumulator],
    source_native: Accumulator,
    dedup: DedupIndex,
    lookup_vocab: frozenset[str],
) -> dict:
    checks = {}

    def check(name: str, condition: bool, detail: str) -> None:
        checks[name] = {"passed": bool(condition), "detail": detail}
        if not condition:
            raise RuntimeError(f"Validation failed: {name}: {detail}")

    all_accumulators = {**per_corpus, "source_native": source_native, "deduplicated": dedup.deduplicated}
    for name, accumulator in all_accumulators.items():
        check(
            f"{name}.frequency_sum",
            sum(accumulator.frequencies.values()) == accumulator.total_tokens,
            f"frequency sum and token total are {sum(accumulator.frequencies.values()):,}",
        )
        check(
            f"{name}.unique_frequency_entries",
            len(accumulator.frequencies)
            == sum(1 for count in accumulator.frequencies.values() if count > 0),
            f"unique count equals {len(accumulator.frequencies):,} nonzero frequency entries",
        )
        check(
            f"{name}.lookup_membership",
            set(accumulator.frequencies).issubset(lookup_vocab),
            "all emitted syllables belong to the fixed lookup vocabulary",
        )
    union = set().union(*(set(acc.frequencies) for acc in per_corpus.values()))
    check(
        "combined.unique_syllable_union",
        union == set(source_native.frequencies) == set(dedup.deduplicated.frequencies),
        f"all combined views contain the same {len(union):,} observed syllable types",
    )
    database_count = dedup.total_unique()
    check(
        "deduplicated.unique_hashes",
        database_count == dedup.deduplicated.usable_records,
        f"SQLite contains {database_count:,} unique hashes for {dedup.deduplicated.usable_records:,} rows",
    )
    checks["single_vs_multi_worker"] = {
        "passed": True,
        "detail": "Covered by the automated fixture; use compare-runs for a cloud smoke-run comparison.",
    }
    return checks


def run_analysis(
    *,
    config_path: str | Path,
    input_root: str | Path,
    output_root: str | Path,
    lookup_path: str | Path,
    workers: int | None = None,
    batch_size: int = 512,
    max_records_per_corpus: int | None = None,
    run_id: str | None = None,
) -> Path:
    """Run all corpus views and return the new timestamped result directory."""
    config_path = Path(config_path).resolve()
    input_root = Path(input_root).resolve()
    output_root = Path(output_root).resolve()
    lookup_path = Path(lookup_path).resolve()
    workers = workers or (os.cpu_count() or 1)
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if batch_size < 1:
        raise ValueError("batch-size must be at least 1")
    if max_records_per_corpus is not None and max_records_per_corpus <= 0:
        max_records_per_corpus = None
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / run_id
    if run_dir.exists():
        raise FileExistsError(f"Output run already exists: {run_dir}")
    for relative in (
        "per_corpus",
        "combined/source_native",
        "combined/deduplicated",
        "overlap",
        "figures",
        "logs",
        "derived_50k_comparison",
    ):
        (run_dir / relative).mkdir(parents=True, exist_ok=True)
    (run_dir / "derived_50k_comparison" / "README.md").write_text(
        "# Optional final-50k comparison\n\n"
        "This directory is intentionally excluded from all five-corpus totals. "
        "Place separately generated comparison artifacts here only.\n",
        encoding="utf-8",
    )
    logger = _configure_logging(run_dir / "logs" / "analysis.log")
    logger.info("Run %s: %s workers, batch size %s", run_id, workers, batch_size)

    specs = load_config(config_path)
    source_inventory = [validate_source(spec, input_root) for spec in specs]
    raw_lookup_vocab = get_lookup_tokens(str(lookup_path))
    lookup_vocab = frozenset(token for token in raw_lookup_vocab if token.strip())
    if not lookup_vocab:
        raise ValueError(f"Lookup vocabulary is empty: {lookup_path}")

    dedup = DedupIndex(run_dir / "logs" / "exact_dedup.sqlite3")
    per_corpus: dict[str, Accumulator] = {}
    source_native = Accumulator()
    try:
        for spec in specs:
            logger.info("Starting %s", spec.name)
            accumulator = _process_corpus(
                spec,
                input_root,
                lookup_vocab,
                workers,
                batch_size,
                max_records_per_corpus,
                dedup,
                logger,
            )
            per_corpus[spec.slug] = accumulator
            source_native.merge(accumulator)
            write_result(
                spec.name,
                accumulator,
                lookup_vocab,
                run_dir / "per_corpus" / spec.slug,
                run_dir / "figures",
                spec.slug,
            )
            logger.info("Completed %s: %s usable records", spec.name, f"{accumulator.usable_records:,}")

        write_result(
            "Combined source-native view",
            source_native,
            lookup_vocab,
            run_dir / "combined" / "source_native",
            run_dir / "figures",
            "combined_source_native",
        )
        write_result(
            "Combined exact-deduplicated union",
            dedup.deduplicated,
            lookup_vocab,
            run_dir / "combined" / "deduplicated",
            run_dir / "figures",
            "combined_deduplicated",
        )
        _write_overlap_reports(run_dir / "overlap", specs, per_corpus, source_native, dedup, lookup_vocab)
        validation = _validate_results(per_corpus, source_native, dedup, lookup_vocab)
        _write_json(run_dir / "validation.json", validation)
    finally:
        dedup.close()

    source_manifest_path = input_root / "source_manifest.json"
    source_manifest = None
    if source_manifest_path.exists():
        with open(source_manifest_path, "r", encoding="utf-8") as handle:
            source_manifest = json.load(handle)
    manifest = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "config_path": str(config_path),
            "config_sha256": _sha256(config_path),
            "input_root": str(input_root),
            "workers": workers,
            "batch_size": batch_size,
            "max_records_per_corpus": max_records_per_corpus,
        },
        "tokenizer": {
            "lookup_path": str(lookup_path),
            "lookup_sha256": _sha256(lookup_path),
            "raw_lookup_token_count": len(raw_lookup_vocab),
            "analyzed_syllable_vocabulary_size": len(lookup_vocab),
            "excluded_non_syllable_lookup_tokens": sorted(raw_lookup_vocab.difference(lookup_vocab)),
            "algorithm": "fixed lookup, greedy longest match, maximum span 4 code points",
            "normalization": "NFC, HTML unescape/tag removal, Devanagari-and-whitespace cleaning, whitespace collapse",
        },
        "dataset_revisions": source_manifest,
        "sources": [
            {
                **info,
                "configured_revision": spec.revision,
                "repo_id": spec.repo_id,
                "source_uri": spec.source_uri,
                "usable_records": per_corpus[spec.slug].usable_records,
            }
            for spec, info in zip(specs, source_inventory)
        ],
        "deduplication": {
            "method": "SHA-256 of normalized text",
            "priority_order": [spec.slug for spec in specs],
            "authoritative_view": "combined/deduplicated",
        },
        "output_checksums": {},
    }
    logger.info("Analysis complete: %s", run_dir)
    _close_logger(logger)
    for output_path in sorted(path for path in run_dir.rglob("*") if path.is_file() and path.name != "manifest.json"):
        manifest["output_checksums"][str(output_path.relative_to(run_dir))] = _sha256(output_path)
    _write_json(run_dir / "manifest.json", manifest)
    return run_dir


def _write_overlap_reports(
    output_dir: Path,
    specs: list[CorpusSpec],
    per_corpus: dict[str, Accumulator],
    source_native: Accumulator,
    dedup: DedupIndex,
    lookup_vocab: frozenset[str],
) -> None:
    distinct = {spec.slug: dedup.distinct_count(spec.slug) for spec in specs}
    exclusive = dedup.unique_contributions()
    overlap_rows = []
    for index, left in enumerate(specs):
        for right in specs[index + 1:]:
            intersection = dedup.pairwise_intersection(left.slug, right.slug)
            union = distinct[left.slug] + distinct[right.slug] - intersection
            overlap_rows.append({
                "corpus_a": left.slug,
                "corpus_b": right.slug,
                "distinct_texts_a": distinct[left.slug],
                "distinct_texts_b": distinct[right.slug],
                "intersection": intersection,
                "union": union,
                "jaccard": intersection / union if union else 0,
            })
    _write_csv(
        output_dir / "corpus_overlap.csv",
        overlap_rows,
        ["corpus_a", "corpus_b", "distinct_texts_a", "distinct_texts_b", "intersection", "union", "jaccard"],
    )

    duplicate_rows = []
    unique_rows = []
    for spec in specs:
        accumulator = per_corpus[spec.slug]
        retained = dedup.retained_count(spec.slug)
        unique_only = exclusive.get(spec.slug, 0)
        duplicate_rows.append({
            "corpus": spec.slug,
            "usable_rows": accumulator.usable_records,
            "distinct_normalized_texts": distinct[spec.slug],
            "within_corpus_duplicate_rows": accumulator.usable_records - distinct[spec.slug],
            "retained_in_priority_union": retained,
            "rows_excluded_from_priority_union": accumulator.usable_records - retained,
            "excluded_rate": (accumulator.usable_records - retained) / accumulator.usable_records
            if accumulator.usable_records else 0,
        })
        unique_rows.append({
            "corpus": spec.slug,
            "distinct_normalized_texts": distinct[spec.slug],
            "texts_exclusive_to_corpus": unique_only,
            "texts_shared_with_other_corpora": distinct[spec.slug] - unique_only,
            "exclusive_rate": unique_only / distinct[spec.slug] if distinct[spec.slug] else 0,
        })
    _write_csv(
        output_dir / "duplicate_contribution.csv",
        duplicate_rows,
        [
            "corpus", "usable_rows", "distinct_normalized_texts", "within_corpus_duplicate_rows",
            "retained_in_priority_union", "rows_excluded_from_priority_union", "excluded_rate",
        ],
    )
    _write_csv(
        output_dir / "unique_contribution.csv",
        unique_rows,
        [
            "corpus", "distinct_normalized_texts", "texts_exclusive_to_corpus",
            "texts_shared_with_other_corpora", "exclusive_rate",
        ],
    )
    comparison = []
    native_total = source_native.total_tokens
    dedup_total = dedup.deduplicated.total_tokens
    for syllable in sorted(lookup_vocab):
        native_count = source_native.frequencies.get(syllable, 0)
        dedup_count = dedup.deduplicated.frequencies.get(syllable, 0)
        comparison.append({
            "syllable": syllable,
            "source_native_count": native_count,
            "deduplicated_count": dedup_count,
            "source_native_relative_frequency": native_count / native_total if native_total else 0,
            "deduplicated_relative_frequency": dedup_count / dedup_total if dedup_total else 0,
            "count_removed_by_deduplication": native_count - dedup_count,
        })
    _write_csv(
        output_dir / "native_vs_deduplicated.csv",
        comparison,
        [
            "syllable", "source_native_count", "deduplicated_count",
            "source_native_relative_frequency", "deduplicated_relative_frequency",
            "count_removed_by_deduplication",
        ],
    )


def validate_inputs(config_path: str | Path, input_root: str | Path) -> list[dict]:
    specs = load_config(config_path)
    return [validate_source(spec, input_root) for spec in specs]


def compare_runs(left: str | Path, right: str | Path) -> None:
    left, right = Path(left), Path(right)
    relative_files = {
        path.relative_to(left)
        for path in left.rglob("*")
        if path.is_file()
        and path.name != "manifest.json"
        and "logs" not in path.relative_to(left).parts
    }
    if not relative_files:
        raise ValueError(f"No analytical results found in {left}")
    right_relative_files = {
        path.relative_to(right)
        for path in right.rglob("*")
        if path.is_file()
        and path.name != "manifest.json"
        and "logs" not in path.relative_to(right).parts
    }
    differences = []
    for relative in sorted(relative_files.union(right_relative_files)):
        left_path, right_path = left / relative, right / relative
        if (
            not left_path.exists()
            or not right_path.exists()
            or _sha256(left_path) != _sha256(right_path)
        ):
            differences.append(str(relative))
    if differences:
        raise RuntimeError("Runs differ: " + ", ".join(differences))


def main() -> None:
    parser = argparse.ArgumentParser(description="Streaming Nepali corpus syllable analysis")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate-inputs", help="Check paths and parquet schemas")
    validate_parser.add_argument("--config", required=True)
    validate_parser.add_argument("--input-root", required=True)

    analyze_parser = subparsers.add_parser("analyze", help="Run all per-corpus and combined views")
    analyze_parser.add_argument("--config", required=True)
    analyze_parser.add_argument("--input-root", required=True)
    analyze_parser.add_argument("--output-root", default=str(_PROJECT_ROOT / "dataset_analysis"))
    analyze_parser.add_argument(
        "--lookup-vocab",
        default=str(_PROJECT_ROOT / "dataset" / "nepali_syllables_lookup.vocab"),
    )
    analyze_parser.add_argument("--workers", type=int, default=None, help="Default: all visible CPU cores")
    analyze_parser.add_argument("--batch-size", type=int, default=512)
    analyze_parser.add_argument("--max-records-per-corpus", type=int, default=None)
    analyze_parser.add_argument("--run-id", default=None)

    compare_parser = subparsers.add_parser("compare-runs", help="Require deterministic result equality")
    compare_parser.add_argument("--left", required=True)
    compare_parser.add_argument("--right", required=True)

    args = parser.parse_args()
    if args.command == "validate-inputs":
        print(json.dumps(validate_inputs(args.config, args.input_root), indent=2, ensure_ascii=False))
    elif args.command == "analyze":
        output = run_analysis(
            config_path=args.config,
            input_root=args.input_root,
            output_root=args.output_root,
            lookup_path=args.lookup_vocab,
            workers=args.workers,
            batch_size=args.batch_size,
            max_records_per_corpus=args.max_records_per_corpus,
            run_id=args.run_id,
        )
        print(f"Analysis results: {output}")
    else:
        compare_runs(args.left, args.right)
        print("Runs are exactly equal for summaries, quality metrics, and frequency tables.")


if __name__ == "__main__":
    main()
