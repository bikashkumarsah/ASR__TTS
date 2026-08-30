from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

_SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SCRIPTS))

import dataset_builder.diverse as diverse_module  # noqa: E402

from dataset_builder.diverse import (  # noqa: E402
    DiskMinHashLSH,
    calibrate_embeddings,
    compare_prepared_pools,
    _read_corpus_records,
    _resolve_lookup,
    prepare_five_corpus_pool,
    select_diverse_records,
    update_progress_markdown,
)
from dataset_builder.syllable_stats import _SKIP_TOKENS  # noqa: E402
from syllable_metrics import (  # noqa: E402
    distribution_statistics,
    jensen_shannon_divergence,
    rarity_counts,
)
from syllabic_tokenizer import (  # noqa: E402
    DEFAULT_LOOKUP_WINDOW_SIZE,
    get_lookup_tokens,
    tokenize,
)


class SharedMetricTest(unittest.TestCase):
    def test_comparative_baseline_must_match_target_size(self):
        with tempfile.TemporaryDirectory() as temporary:
            baseline = Path(temporary) / "baseline.jsonl"
            baseline.write_text("{}\n{}\n", encoding="utf-8")
            self.assertEqual(diverse_module._validate_baseline_size(baseline, 2), 2)
            with self.assertRaisesRegex(ValueError, "size-matched baseline"):
                diverse_module._validate_baseline_size(baseline, 1)

    def test_embedding_pool_uses_spawn_context(self):
        from unittest.mock import MagicMock, patch

        executor = MagicMock()
        executor.__enter__.return_value.map.return_value = []
        with patch.object(diverse_module, "ProcessPoolExecutor", return_value=executor) as factory:
            diverse_module._run_embedding_layout(
                [],
                model_path=Path("model.onnx"),
                tokenizer_path=Path("tokenizer"),
                processes=3,
                threads=4,
                batch_size=32,
            )
        context = factory.call_args.kwargs["mp_context"]
        self.assertEqual(context.get_start_method(), "spawn")

    def test_config_fingerprint_changes_with_tokenizer_source(self):
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "tokenizer_a.py"
            second = Path(temporary) / "tokenizer_b.py"
            first.write_text("window = 4\n", encoding="utf-8")
            second.write_text("window = 7\n", encoding="utf-8")
            with patch.object(diverse_module, "_TOKENIZER_SOURCE", first):
                first_fingerprint = diverse_module._config_fingerprint({"fixture": True})
            with patch.object(diverse_module, "_TOKENIZER_SOURCE", second):
                second_fingerprint = diverse_module._config_fingerprint({"fixture": True})
        self.assertNotEqual(first_fingerprint, second_fingerprint)

    def test_default_window_matches_paper_and_legacy_full_window_is_explicit(self):
        project = _SCRIPTS.parent
        vocabulary = get_lookup_tokens(str(project / "dataset/nepali_syllables_lookup.vocab"))
        default_unemittable = [
            token for token in vocabulary
            if token.strip() and token not in _SKIP_TOKENS and tokenize(token, vocabulary) != [token]
        ]
        legacy_unemittable = [
            token for token in vocabulary
            if token.strip() and token not in _SKIP_TOKENS
            and tokenize(token, vocabulary, max_token_length=None) != [token]
        ]
        self.assertEqual(DEFAULT_LOOKUP_WINDOW_SIZE, 4)
        self.assertEqual(len(default_unemittable), 534)
        self.assertTrue(all(len(token) > DEFAULT_LOOKUP_WINDOW_SIZE for token in default_unemittable))
        self.assertEqual(legacy_unemittable, [])

    def test_entropy_gini_and_jsd(self):
        inventory = frozenset({"a", "b", "c"})
        uniform = distribution_statistics({"a": 10, "b": 10, "c": 10}, inventory=inventory)
        skewed = distribution_statistics({"a": 28, "b": 1, "c": 1}, inventory=inventory)
        self.assertAlmostEqual(uniform["normalized_entropy"], 1.0)
        self.assertAlmostEqual(uniform["gini"], 0.0)
        self.assertGreater(uniform["normalized_entropy"], skewed["normalized_entropy"])
        self.assertLess(uniform["gini"], skewed["gini"])
        self.assertEqual(
            jensen_shannon_divergence(
                {"a": 10, "b": 10, "c": 10},
                {"a": 1, "b": 1, "c": 1},
                inventory=inventory,
            ),
            0.0,
        )
        self.assertEqual(rarity_counts({"a": 1, "b": 4, "c": 12})["below_5"], 2)

    def test_baseline_is_retokenized_from_text_not_stored_tokens(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "baseline.jsonl"
            path.write_text(
                json.dumps({"text": "क ख", "syllables": ["stale"]}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            records = _read_corpus_records(path, frozenset({"क", "ख", " "}))
        self.assertEqual(records[0]["syllables"], ["क", "ख"])

    def test_disk_minhash_lsh_uses_exact_jaccard_verification(self):
        with tempfile.TemporaryDirectory() as temporary:
            index = DiskMinHashLSH(Path(temporary) / "lsh.sqlite3", reset=True)
            try:
                original = "नेपालको सुन्दर प्राकृतिक वातावरण सबैका लागि महत्वपूर्ण छ"
                near = "नेपालको सुन्दर प्राकृतिक वातावरण सबैका लागि महत्वपूर्ण छन्"
                index.add("original", original)
                duplicate, score = index.find_duplicate(near)
                self.assertEqual(duplicate, "original")
                self.assertGreaterEqual(score, 0.85)
                unrelated, _ = index.find_duplicate("कृषि बजार र प्रविधि समाचार")
                self.assertIsNone(unrelated)
            finally:
                index.close()

    def test_bounded_semantic_selector_meets_size_and_rare_floor(self):
        import numpy as np

        rng = np.random.default_rng(7)
        records = []
        syllables = ("क", "ख", "ग")
        for index in range(36):
            token = syllables[index % len(syllables)]
            records.append({
                "normalized_sha256": f"{index:064x}",
                "text": f"fixture {index}",
                "syllables": ["क", token],
                "unique_syllables": sorted({"क", token}),
                "syllable_count": 2,
                "tense": ("past", "present", "future")[index % 3],
                "polarity": ("positive", "negative", "neutral")[index % 3],
                "gender": ("masculine", "feminine", "neutral")[index % 3],
                "sector": "news",
            })
        embeddings = rng.normal(size=(len(records), 16)).astype(np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        inventory = {
            "attainable_syllables": list(syllables),
            "syllable_occurrence_frequency": {"क": 48, "ख": 12, "ग": 12},
            "syllable_sentence_frequency": {"क": 36, "ख": 12, "ग": 12},
        }
        config = {"selection": {
            "tempering_exponent": 0.5,
            "rare_floor": 2,
            "faiss_clusters": 4,
            "initial_cluster_cap": 3,
            "background_upper_tail_rates": [0.005],
            "score_weights": [0.60, 0.35, 0.05],
        }}
        selected, state = select_diverse_records(
            records,
            embeddings,
            np.linspace(-1, 1, 1000, dtype=np.float32),
            [0.9999],
            inventory=inventory,
            config=config,
            target_size=10,
            seed=42,
        )
        self.assertEqual(len(selected), 10)
        self.assertTrue(all(
            state["selected_sentence_frequency"][syllable] >= 2
            for syllable in syllables
        ))
        self.assertEqual(len({row["normalized_sha256"] for row in selected}), 10)

    def test_similarity_thresholds_are_empirically_calibrated(self):
        import numpy as np

        rng = np.random.default_rng(11)
        embeddings = rng.normal(size=(120, 12)).astype(np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings[1] = embeddings[0] * 0.999 + embeddings[1] * 0.001
        embeddings[1] /= np.linalg.norm(embeddings[1])
        records = [
            {"normalized_sha256": f"{index:064x}"}
            for index in range(len(embeddings))
        ]
        records[1]["lexical_near_duplicate_of"] = records[0]["normalized_sha256"]
        with tempfile.TemporaryDirectory() as temporary:
            _, _, thresholds, report, _ = calibrate_embeddings(
                records,
                embeddings,
                config={
                    "embedding": {"calibration_size": 100},
                    "selection": {"background_upper_tail_rates": [0.02, 0.005]},
                },
                output_dir=Path(temporary),
                seed=42,
            )
        self.assertEqual(len(thresholds), 2)
        self.assertLessEqual(thresholds[0], thresholds[1])
        self.assertTrue(report["thresholds_are_empirical_not_absolute"])
        self.assertEqual(report["prefix"], "query: ")
        self.assertEqual(
            report["transform_selection_policy"],
            "raw model space fixed before diagnostic labels",
        )


class DiversePreparationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        self._write_jsonl(
            self.inputs / "a.jsonl",
            [
                {"text": "क क क क क। ख ख ख ख ख", "source": "alpha"},
                {"text": "ग ग ग ग ग", "source": "alpha"},
            ],
        )
        self._write_jsonl(
            self.inputs / "b.jsonl",
            [
                {"text": "क क क क क", "source": "duplicate"},
                {"text": "घ घ घ घ घ", "source": "beta"},
            ],
        )
        self.corpus_config = self.root / "corpora.yaml"
        self.corpus_config.write_text(
            yaml.safe_dump({"corpora": [
                {
                    "name": "A", "slug": "a", "kind": "jsonl", "path": "a.jsonl",
                    "text_column": "text", "metadata_columns": {"source": "source"},
                },
                {
                    "name": "B", "slug": "b", "kind": "jsonl", "path": "b.jsonl",
                    "text_column": "text", "metadata_columns": {"source": "source"},
                },
            ]}, sort_keys=False),
            encoding="utf-8",
        )
        self.vocab = self.root / "lookup.vocab"
        self.vocab.write_text("क\nख\nग\nघ\n", encoding="utf-8")
        digest = hashlib.sha256(self.vocab.read_bytes()).hexdigest()
        self.diverse_config = self.root / "diverse.yaml"
        self.diverse_config.write_text(
            yaml.safe_dump({
                "lookup_vocabulary": {
                    "path": str(self.vocab), "sha256": digest,
                    "raw_entries": 4, "analytical_entries": 4,
                },
                "preparation": {
                    "min_syllables": 5, "max_syllables": 80,
                    "candidate_limit": 10, "candidates_per_syllable": 2,
                    "chunk_size": 2,
                },
            }, sort_keys=False),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict]) -> None:
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )

    def _run(self, name: str, workers: int) -> Path:
        return prepare_five_corpus_pool(
            corpus_config=self.corpus_config,
            input_root=self.inputs,
            output_dir=self.root / name,
            diverse_config=self.diverse_config,
            workers=workers,
        )

    def test_preparation_is_worker_invariant_and_preserves_provenance(self):
        single = self._run("single", 1)
        multi = self._run("multi", 2)
        self.assertEqual(
            json.loads((single / "frequency_inventory.json").read_text()),
            json.loads((multi / "frequency_inventory.json").read_text()),
        )
        single_rows = [json.loads(line) for path in sorted((single / "shortlist").glob("*.jsonl")) for line in path.read_text().splitlines()]
        multi_rows = [json.loads(line) for path in sorted((multi / "shortlist").glob("*.jsonl")) for line in path.read_text().splitlines()]
        self.assertEqual(single_rows, multi_rows)
        compare_prepared_pools(single, multi)
        duplicate = next(row for row in single_rows if row["text"] == "क क क क क")
        self.assertEqual(duplicate["source_corpora"], ["a", "b"])
        inventory = json.loads((single / "frequency_inventory.json").read_text())
        self.assertEqual(inventory["unique_eligible_sentences"], 4)
        self.assertEqual(inventory["exact_duplicate_sentences"], 1)
        self.assertEqual(inventory["tokenizer_unemittable_count"], 0)
        with sqlite3.connect(single / "exact_dedup.sqlite3") as connection:
            tables = {
                row[0] for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
        self.assertNotIn("shortlist_cache", tables)

    def test_vocabulary_checksum_is_enforced(self):
        config = yaml.safe_load(self.diverse_config.read_text())
        config["lookup_vocabulary"]["sha256"] = "0" * 64
        broken = self.root / "broken.yaml"
        broken.write_text(yaml.safe_dump(config), encoding="utf-8")
        with self.assertRaises(RuntimeError):
            _resolve_lookup(yaml.safe_load(broken.read_text()))

    def test_lookup_window_is_pinned(self):
        config = yaml.safe_load(self.diverse_config.read_text())
        config["lookup_vocabulary"]["lookup_window_size"] = 7
        with self.assertRaisesRegex(RuntimeError, "lookup-window mismatch"):
            _resolve_lookup(config)

    def test_resume_rejects_changed_tokenizer_inventory(self):
        prepared = self._run("tampered_inventory", 1)
        inventory_path = prepared / "frequency_inventory.json"
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        inventory["tokenizer_emittable_count"] += 1
        inventory_path.write_text(
            json.dumps(inventory, ensure_ascii=False), encoding="utf-8"
        )
        with self.assertRaisesRegex(RuntimeError, "tokenizer inventory"):
            prepare_five_corpus_pool(
                corpus_config=self.corpus_config,
                input_root=self.inputs,
                output_dir=prepared,
                diverse_config=self.diverse_config,
                workers=1,
                resume=True,
            )

    def test_markdown_progress_update_is_idempotent(self):
        report = self.root / "report.json"
        report.write_text(json.dumps({
            "run_id": "fixture",
            "target_size": 50_000,
            "vocabulary": {"attainable_entries": 4},
            "observed_syllables": 4,
            "syllable_tokens": 100,
            "distribution": {
                "normalized_entropy": 0.9, "gini": 0.1,
                "coefficient_of_variation": 0.2,
            },
            "jensen_shannon_divergence_to_tempered_target": 0.01,
            "nearest_neighbor_cosine": {"median": 0.4, "p95": 0.7},
            "similarity_exceptions": 1,
            "acceptance": {"fixture": True},
        }), encoding="utf-8")
        progress = self.root / "progress.md"
        update_progress_markdown(run_report=report, progress_file=progress)
        update_progress_markdown(run_report=report, progress_file=progress)
        self.assertEqual(progress.read_text().count("<!-- run:fixture:start -->"), 1)


if __name__ == "__main__":
    unittest.main()
