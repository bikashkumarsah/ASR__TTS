from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

_SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SCRIPTS))

from corpus_analysis.metrics import normalize_text  # noqa: E402
from corpus_analysis.pipeline import compare_runs, run_analysis, validate_inputs  # noqa: E402
from corpus_analysis.sources import CorpusSpec, iter_records  # noqa: E402


class CorpusAnalysisTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        self._write_jsonl(
            self.inputs / "a.jsonl",
            [
                {"text": "नेपाल", "source": "alpha"},
                {"text": "नेपाल", "source": "alpha"},
                {"text": "क", "source": "beta"},
                {"text": ""},
                {"text": None},
            ],
        )
        self._write_jsonl(
            self.inputs / "b.jsonl",
            [
                {"text": "<p>नेपाल</p>", "domain": "shared"},
                {"text": "ख", "domain": "unique"},
                {"text": "क", "domain": "shared"},
            ],
        )
        self.config = self.root / "config.yaml"
        self.config.write_text(
            yaml.safe_dump({
                "corpora": [
                    {
                        "name": "Corpus A",
                        "slug": "a",
                        "kind": "jsonl",
                        "path": "a.jsonl",
                        "text_column": "text",
                        "metadata_columns": {"source": "source"},
                    },
                    {
                        "name": "Corpus B",
                        "slug": "b",
                        "kind": "jsonl",
                        "path": "b.jsonl",
                        "text_column": "text",
                        "metadata_columns": {"domain": "domain"},
                    },
                ]
            }, sort_keys=False),
            encoding="utf-8",
        )
        self.vocab = self.root / "lookup.vocab"
        self.vocab.write_text("न\nे\nप\nा\nल\nक\nख\nग\n", encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict]) -> None:
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )

    def _run(self, run_id: str, workers: int) -> Path:
        return run_analysis(
            config_path=self.config,
            input_root=self.inputs,
            output_root=self.root / "results",
            lookup_path=self.vocab,
            workers=workers,
            batch_size=2,
            run_id=run_id,
        )

    def test_normalization_is_hash_stable(self):
        self.assertEqual(normalize_text("<p> नेपाल! </p>"), "नेपाल")
        self.assertEqual(normalize_text("ने\u092a\u093eल"), "नेपाल")

    def test_exact_dedup_and_reports(self):
        run_dir = self._run("single", workers=1)
        dedup_summary = json.loads(
            (run_dir / "combined/deduplicated/summary.json").read_text(encoding="utf-8")
        )
        native_quality = json.loads(
            (run_dir / "combined/source_native/quality_metrics.json").read_text(encoding="utf-8")
        )
        validation = json.loads((run_dir / "validation.json").read_text(encoding="utf-8"))
        self.assertEqual(dedup_summary["unique_observed_syllables"], 7)
        self.assertEqual(native_quality["input_records"], 8)
        self.assertEqual(native_quality["usable_records"], 6)
        self.assertEqual(
            json.loads((run_dir / "combined/deduplicated/quality_metrics.json").read_text())["usable_records"],
            3,
        )
        self.assertTrue(all(check["passed"] for check in validation.values()))

        with open(run_dir / "overlap/duplicate_contribution.csv", encoding="utf-8") as handle:
            duplicate_rows = {row["corpus"]: row for row in csv.DictReader(handle)}
        self.assertEqual(int(duplicate_rows["a"]["within_corpus_duplicate_rows"]), 1)
        self.assertEqual(int(duplicate_rows["b"]["retained_in_priority_union"]), 1)

        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        for relative, expected in manifest["output_checksums"].items():
            digest = hashlib.sha256((run_dir / relative).read_bytes()).hexdigest()
            self.assertEqual(digest, expected, relative)

    def test_single_and_multi_worker_results_match(self):
        single = self._run("workers_1", workers=1)
        multi = self._run("workers_2", workers=2)
        compare_runs(single, multi)

    def test_input_validation(self):
        info = validate_inputs(self.config, self.inputs)
        self.assertEqual([entry["files"] for entry in info], [1, 1])

    def test_parquet_and_streaming_html_readers(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        parquet_dir = self.inputs / "parquet"
        parquet_dir.mkdir()
        pq.write_table(
            pa.table({"Article": ["नेपाल", "क"], "Source": ["one", "two"]}),
            parquet_dir / "train.parquet",
        )
        parquet_spec = CorpusSpec(
            name="Parquet",
            slug="parquet",
            kind="parquet",
            path="parquet",
            file_glob="*.parquet",
            text_column="Article",
            metadata_columns={"source": "Source"},
        )
        self.assertEqual(
            list(iter_records(parquet_spec, self.inputs, read_batch_size=1)),
            [
                {"text": "नेपाल", "metadata": {"source": "one"}},
                {"text": "क", "metadata": {"source": "two"}},
            ],
        )

        html_path = self.inputs / "compiled.txt"
        html_path.write_text(
            "header<p>पहिलो <b>अनुच्छेद</b></p><p>दोस्रो अनुच्छेद</p>",
            encoding="utf-8",
        )
        html_spec = CorpusSpec(
            name="HTML", slug="html", kind="html_paragraphs", path="compiled.txt"
        )
        self.assertEqual(
            [row["text"] for row in iter_records(html_spec, self.inputs)],
            ["पहिलो  अनुच्छेद", "दोस्रो अनुच्छेद"],
        )


if __name__ == "__main__":
    unittest.main()
