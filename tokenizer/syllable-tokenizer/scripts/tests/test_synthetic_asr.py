from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
import unittest
import wave
from pathlib import Path

import yaml

_SCRIPTS = Path(__file__).resolve().parents[1]
_PROJECT = _SCRIPTS.parent
sys.path.insert(0, str(_SCRIPTS))

from dataset_builder.synthetic_asr import (
    _effective_requests_per_minute,
    _fingerprint,
    _init_job_db,
    _load_run_config,
    _phase_jobs,
    _rebase_synthesis_paths,
    _voice_qualification,
    _write_jsonl,
    export_kaggle_qc,
    finalize_synthetic_asr,
    import_kaggle_qc,
    normalize_spoken_text,
    prepare_slr54_speechain,
    prepare_synthetic_asr,
    validate_synthetic_audio,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _wav(path: Path, seconds: float = 0.25, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(seconds * sample_rate)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x01" * frames)


class SyntheticAsrTest(unittest.TestCase):
    def _config(self, root: Path, target: int = 2) -> Path:
        vocab = _PROJECT / "dataset" / "nepali_syllables_lookup.vocab"
        tokenizer = _SCRIPTS / "syllabic_tokenizer.py"
        config = {
            "run": {"target_canonical": target, "seed": 42, "legal_review_file": "legal_review.json"},
            "tokenizer": {
                "vocabulary": str(vocab), "vocabulary_sha256": _sha256(vocab),
                "source": str(tokenizer), "attainable_inventory": 2,
            },
            "google_tts": {
                "max_input_bytes": 4000, "training_sample_rate": 16000, "master_sample_rate": 24000,
                "voices": [
                    {"name": "F1", "gender": "female"}, {"name": "F2", "gender": "female"},
                    {"name": "M1", "gender": "male"}, {"name": "M2", "gender": "male"},
                ],
                "selected_female": 1, "selected_male": 1,
            },
            "phases": {"audition_texts": 1, "pilot_texts": 1, "full_texts": target},
            "rare_tail": {"occurrence_threshold": 20, "required_voice_realizations": 3},
            "budget": {
                "audio_tokens_per_second": 25, "input_usd_per_million_tokens": 0.5,
                "audio_usd_per_million_tokens": 10, "tts_usd": 85, "total_usd": 100,
                "vm_runtime_storage_usd": 10,
            },
            "qc": {
                "speaker_embedding_model": "fixture", "speaker_embedding_revision": "speaker-rev",
                "whisper_model": "whisper-fixture", "whisper_revision": "whisper-rev",
                "mms_model": "mms-fixture", "mms_revision": "mms-rev", "mms_language_adapter": "npi",
                "max_cer": 0.35, "max_wer": 0.60, "min_duration_seconds": 0.2,
                "max_duration_seconds": 10, "clipping_amplitude": 0.999, "max_clipping_ratio": 0.001,
            },
        }
        path = root / "config.yaml"
        path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
        return path

    def test_spoken_normalization_expands_supported_forms(self):
        result = normalize_spoken_text("मूल्य रु १२.५ र छुट १०%")
        self.assertEqual(result["status"], "ok")
        self.assertIn("एक दुई दशमलव पाँच रुपैयाँ", result["asr_text"])
        self.assertIn("एक शून्य प्रतिशत", result["asr_text"])
        self.assertTrue(result["tts_text"].endswith("।"))
        self.assertNotIn("१२", result["asr_text"])

    def test_runtime_request_rate_can_only_lower_configured_ceiling(self):
        self.assertEqual(_effective_requests_per_minute(120, None), 120)
        self.assertEqual(_effective_requests_per_minute(120, 5), 5)
        with self.assertRaises(ValueError):
            _effective_requests_per_minute(120, 121)
        with self.assertRaises(ValueError):
            _effective_requests_per_minute(120, 0)

    def test_synthesis_checkpoint_paths_rebase_after_host_transfer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_id, phase = "aud_fixture", "audition"
            master = root / "audio" / "master_24k" / phase / f"{job_id}.wav"
            training = root / "audio" / "train_16k" / phase / f"{job_id}.wav"
            _wav(master)
            _wav(training)
            connection = _init_job_db(root / "state" / "synthesis.sqlite3")
            connection.execute(
                """INSERT INTO synthesis_jobs(
                job_id,phase,status,voice,master_path,training_path,updated_at
                ) VALUES(?,?,?,?,?,?,?)""",
                (job_id, phase, "succeeded", "F1", "/old/master.wav", "/old/training.wav", "fixture"),
            )
            connection.commit()
            self.assertEqual(_rebase_synthesis_paths(connection, root), 1)
            stored = connection.execute(
                "SELECT master_path,training_path FROM synthesis_jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            connection.close()
            self.assertEqual(stored, (str(master), str(training)))

    def test_prepared_run_fingerprint_survives_host_transfer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = self._config(root, target=1)
            input_path = root / "input.jsonl"
            input_path.write_text(
                json.dumps({"text": "नेपाल राम्रो छ", "normalized_sha256": "a" * 64}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            hashes = root / "eval_hashes.txt"
            hashes.write_text("", encoding="utf-8")
            slr = root / "slr.json"
            slr.write_text(json.dumps({
                "splits": {
                    split: {"text_hashes": str(hashes)}
                    for split in ("train", "dev", "test")
                }
            }), encoding="utf-8")
            original = root / "original" / "run"
            prepare_synthetic_asr(input_path, slr, config_path, original)
            transferred = root / "transferred" / "run"
            shutil.copytree(original, transferred)

            config, resolved = _load_run_config(transferred)
            self.assertEqual(resolved, (transferred / "synthetic_asr_config.yaml").resolve())
            self.assertEqual(config["tokenizer"]["vocabulary"], "pins/nepali_syllables_lookup.vocab")

            pinned_tokenizer = transferred / "pins" / "syllabic_tokenizer.py"
            pinned_tokenizer.write_text(
                pinned_tokenizer.read_text(encoding="utf-8") + "\n# incompatible transfer\n",
                encoding="utf-8",
            )
            with self.assertRaises(RuntimeError):
                _load_run_config(transferred)

    def test_ambiguous_numeric_form_is_quarantined(self):
        result = normalize_spoken_text("मिति ०१/०२/८१")
        self.assertEqual(result["status"], "quarantine")
        self.assertEqual(result["reason"], "ambiguous_numeric_form")

    def test_year_month_day_date_is_expanded(self):
        result = normalize_spoken_text("मिति २०८१/०२/०१")
        self.assertEqual(result["status"], "ok")
        self.assertIn("साल", result["asr_text"])
        self.assertIn("महिना", result["asr_text"])
        self.assertIn("गते", result["asr_text"])

    def test_preparation_separates_numeric_glyphs_from_spoken_inventory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = self._config(root, target=1)
            input_path = root / "input.jsonl"
            input_path.write_text(
                json.dumps({"text": "१ नेपाल", "normalized_sha256": "a" * 64}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            hashes = root / "eval_hashes.txt"
            hashes.write_text("", encoding="utf-8")
            slr = root / "slr.json"
            slr.write_text(json.dumps({
                "splits": {
                    split: {"text_hashes": str(hashes)}
                    for split in ("train", "dev", "test")
                }
            }), encoding="utf-8")
            summary = prepare_synthetic_asr(input_path, slr, config_path, root / "run")
            self.assertEqual(summary["normalization_removed_numeric_symbols"], {"१": 1})
            self.assertNotEqual(
                summary["source_observed_types"],
                summary["spoken_observed_syllables"],
            )

    def test_tokenizer_source_is_part_of_fingerprint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = self._config(root)
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            first = _fingerprint(config, config_path)
            source_copy = root / "tokenizer.py"
            source_copy.write_text((_SCRIPTS / "syllabic_tokenizer.py").read_text(encoding="utf-8") + "\n# fixture\n", encoding="utf-8")
            config["tokenizer"]["source"] = str(source_copy)
            second = _fingerprint(config, config_path)
            self.assertNotEqual(first["tokenizer_sha256"], second["tokenizer_sha256"])

    def test_voice_qualification_uses_accuracy_and_embedding_separation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = self._config(root)
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            records, whisper, mms, cpu_rows = {}, {}, {}, []
            embeddings = {"F1": [1.0, 0.0], "F2": [0.9, 0.1], "M1": [0.0, 1.0], "M2": [-1.0, 0.0]}
            for index, voice in enumerate(("F1", "F2", "M1", "M2")):
                job_id = f"aud_{voice}"
                records[job_id] = {"phase": "audition", "voice": voice}
                whisper[job_id] = {
                    "job_id": job_id, "cer": 0.05 + index * 0.01,
                    "rare_syllable_recall": 1.0, "speaker_embedding": embeddings[voice],
                }
                mms[job_id] = {"rare_syllable_recall": 1.0}
                cpu_rows.append({"job_id": job_id, "cpu_pass": True})
            _write_jsonl(root / "qc" / "cpu_metrics.jsonl", cpu_rows)
            result = _voice_qualification(root, config, records, whisper, mms)
            self.assertIsNotNone(result)
            self.assertEqual(len(result["selected_voices"]), 2)
            self.assertIn("wavlm", result["method"])

    def test_slr54_split_is_speaker_disjoint_and_reproducible(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            lines = []
            for speaker in range(12):
                for utterance in range(2):
                    file_id = f"u{speaker:02d}_{utterance}"
                    _wav(source / f"{file_id}.wav")
                    lines.append(f"{file_id}\tspk{speaker:02d}\tनेपाल राम्रो छ")
            (source / "utt_spk_text.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
            first = prepare_slr54_speechain(source, root / "out1", seed=42, workers=2)
            second = prepare_slr54_speechain(source, root / "out2", seed=42, workers=1)
            self.assertTrue(first["speaker_disjoint"])
            self.assertEqual(
                [first["splits"][split]["utterances"] for split in ("train", "dev", "test")],
                [second["splits"][split]["utterances"] for split in ("train", "dev", "test")],
            )

    def test_prepare_pins_inputs_and_constructs_rare_voice_jobs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = self._config(root)
            input_path = root / "input.jsonl"
            rows = [
                {"text": "नेपाल राम्रो छ", "normalized_sha256": "a" * 64, "rare_syllables": ["ने"]},
                {"text": "भाषा राम्रो छ", "normalized_sha256": "b" * 64, "rare_syllables": ["भा"]},
            ]
            input_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
            hashes = root / "eval_hashes.txt"
            hashes.write_text("", encoding="utf-8")
            slr = root / "slr.json"
            slr.write_text(json.dumps({
                "splits": {
                    "train": {"text_hashes": str(hashes)},
                    "dev": {"text_hashes": str(hashes)},
                    "test": {"text_hashes": str(hashes)},
                }
            }), encoding="utf-8")
            run = root / "run"
            summary = prepare_synthetic_asr(input_path, slr, config_path, run)
            self.assertEqual(summary["prepared_canonical"], 2)
            self.assertTrue((run / "pins" / "syllabic_tokenizer.py").exists())
            run_config = yaml.safe_load((run / "synthetic_asr_config.yaml").read_text(encoding="utf-8"))
            (run / "qc").mkdir()
            (run / "qc" / "voice_qualification.json").write_text(json.dumps({"selected_voices": ["F1", "M1"]}), encoding="utf-8")
            (run / "qc" / "pilot_qc_gate.json").write_text(json.dumps({"passed": True}), encoding="utf-8")
            voice_options = [
                {"name": "F1", "gender": "female"}, {"name": "M1", "gender": "male"},
                {"name": "F2", "gender": "female"}, {"name": "M2", "gender": "male"},
            ]
            jobs = _phase_jobs(run, run_config, "full", voice_options)
            canonical = [row for row in jobs if row["realization"] == "canonical"]
            extras = [row for row in jobs if row["realization"] == "rare_extra"]
            self.assertEqual(len(canonical), 2)
            self.assertGreaterEqual(len(extras), 2)
            for token in ("ने", "भा"):
                voice_set = {row["voice"] for row in canonical + extras if token in row["syllables"]}
                self.assertGreaterEqual(len(voice_set), 3)
            _write_jsonl(run / "manifests" / "regeneration_requests.jsonl", [
                {"utterance_id": canonical[0]["utterance_id"], "attempt": 1, "failed_job_id": canonical[0]["job_id"]},
                {"utterance_id": canonical[0]["utterance_id"], "attempt": 2, "failed_job_id": f"regen1_{canonical[0]['utterance_id']}"},
            ])
            regenerated = _phase_jobs(run, run_config, "full", voice_options)
            alternatives = [row["voice"] for row in regenerated if row["realization"].startswith("regeneration")]
            self.assertEqual(len(alternatives), 2)
            self.assertEqual(len(set(alternatives + [canonical[0]["voice"]])), 3)

    def test_bounded_end_to_end_fixture_accepts_and_exports_speechain(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = self._config(root)
            input_path = root / "input.jsonl"
            source_rows = [
                {"text": "नेपाल राम्रो छ", "normalized_sha256": "a" * 64},
                {"text": "भाषा राम्रो छ", "normalized_sha256": "b" * 64},
            ]
            input_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in source_rows), encoding="utf-8")
            empty_hashes = root / "eval_hashes.txt"
            empty_hashes.write_text("", encoding="utf-8")
            slr_manifest = root / "slr.json"
            slr_manifest.write_text(json.dumps({
                "splits": {
                    split: {
                        "text_hashes": str(empty_hashes),
                        "paths": {
                            "idx2wav": "real-wav", "idx2wav_len": "real-len",
                            "idx2no-punc_text": "real-text", "idx2spk": "real-spk",
                        },
                    }
                    for split in ("train", "dev", "test")
                }
            }), encoding="utf-8")
            run = root / "run"
            prepare_synthetic_asr(input_path, slr_manifest, config_path, run)
            (run / "legal_review.json").write_text(json.dumps({
                "approved": True, "usage": "academic_private", "google_terms_reviewed": True,
                "kaggle_private_transfer_approved": True,
            }), encoding="utf-8")
            run_config = yaml.safe_load((run / "synthetic_asr_config.yaml").read_text(encoding="utf-8"))
            voices = [
                {"name": "F1", "gender": "female"}, {"name": "M1", "gender": "male"},
                {"name": "F2", "gender": "female"}, {"name": "M2", "gender": "male"},
            ]
            jobs = _phase_jobs(run, run_config, "full", voices)
            _write_jsonl(run / "manifests" / "synthesis_full_jobs.jsonl", jobs)
            connection = _init_job_db(run / "state" / "synthesis.sqlite3")
            for ordinal, job in enumerate(jobs):
                master = run / "audio" / "master_24k" / "full" / f"{job['job_id']}.wav"
                training = run / "audio" / "train_16k" / "full" / f"{job['job_id']}.wav"
                fixture_duration = 0.25 + ordinal * 0.001
                _wav(master, seconds=fixture_duration)
                _wav(training, seconds=fixture_duration)
                audio_hash = _sha256(training)
                connection.execute(
                    """INSERT INTO synthesis_jobs(
                    job_id,phase,status,attempts,voice,master_path,training_path,audio_sha256,
                    duration_seconds,input_tokens_estimate,audio_tokens,estimated_cost_usd,updated_at
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (job["job_id"], "full", "succeeded", 1, job["voice"], str(master), str(training),
                     audio_hash, fixture_duration, 10, 10, 0.001, "fixture"),
                )
            connection.commit()
            connection.close()
            cpu = validate_synthetic_audio(run, workers=2)
            self.assertEqual(cpu["passed"], len(jobs))
            kaggle_input = root / "kaggle_input"
            export = export_kaggle_qc(run, kaggle_input, shard_size_gb=0.001)
            self.assertEqual(export["records"], len(jobs))
            qc_rows = [json.loads(line) for line in (kaggle_input / "qc_input_manifest.jsonl").read_text(encoding="utf-8").splitlines()]
            results = root / "results"
            results.mkdir()
            whisper_rows, mms_rows = [], []
            for row in qc_rows:
                common = {"job_id": row["job_id"], "audio_sha256": row["audio_sha256"], "hypothesis": row["reference_text"], "cer": 0.0, "wer": 0.0, "syllable_error_rate": 0.0, "rare_syllable_recall": 1.0}
                whisper_rows.append({**common, "model": "whisper-fixture", "revision": "whisper-rev"})
                mms_rows.append({**common, "model": "mms-fixture", "revision": "mms-rev", "language_adapter": "npi"})
            _write_jsonl(results / "whisper_results.jsonl", whisper_rows)
            _write_jsonl(results / "mms_results.jsonl", mms_rows)
            (results / "kaggle_qc_summary.json").write_text("{}\n", encoding="utf-8")
            output_hashes = {
                name: _sha256(results / name)
                for name in ("whisper_results.jsonl", "mms_results.jsonl", "kaggle_qc_summary.json")
            }
            (results / "kaggle_manifest.json").write_text(json.dumps({
                "input_manifest_sha256": export["checksums"]["qc_input_manifest.jsonl"],
                "outputs": output_hashes,
            }), encoding="utf-8")
            imported = import_kaggle_qc(run, results)
            self.assertTrue(imported["complete_required_mms"])
            final = finalize_synthetic_asr(run)
            self.assertEqual(final["status"], "accepted")
            self.assertEqual(final["canonical_accepted"], 2)
            self.assertTrue((run / "speechain" / "integration" / "install_adapter.py").exists())
            recipe = yaml.safe_load(
                (run / "speechain" / "recipes" / "real_plus_synthetic_seed42_train.yaml").read_text(encoding="utf-8")
            )
            data_recipe = yaml.safe_load(
                (run / "speechain" / "recipes" / "real_plus_synthetic_seed42_data.yaml").read_text(encoding="utf-8")
            )
            self.assertEqual(
                recipe["model"]["model_conf"]["customize_conf"]["token_type"], "syllable"
            )
            self.assertEqual(set(data_recipe), {"train", "valid", "test"})


if __name__ == "__main__":
    unittest.main()
