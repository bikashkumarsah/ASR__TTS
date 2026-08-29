"""Kaggle GPU runner for Whisper-all and targeted MMS Nepali QC.

This program consumes only the private Kaggle input package. It does not need or
accept Google credentials and writes only compact result tables and a manifest.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import re
import tarfile
import tempfile
import unicodedata
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[^\u0900-\u097F\s]", " ", text)
    text = text.replace("।", " ").replace("॥", " ")
    return re.sub(r"\s+", " ", text).strip()


def edit_distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for row_index, left_item in enumerate(left, 1):
        current = [row_index]
        for column_index, right_item in enumerate(right, 1):
            current.append(min(
                current[-1] + 1,
                previous[column_index] + 1,
                previous[column_index - 1] + (left_item != right_item),
            ))
        previous = current
    return previous[-1]


def error_rate(reference: list[str], hypothesis: list[str]) -> float:
    return edit_distance(reference, hypothesis) / max(1, len(reference))


def load_tokenizer(root: Path):
    source = root / "syllabic_tokenizer.py"
    vocab = root / "nepali_syllables_lookup.vocab"
    spec = importlib.util.spec_from_file_location("kaggle_pinned_tokenizer", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load tokenizer: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, module.get_lookup_tokens(str(vocab))


def syllable_metrics(reference: str, hypothesis: str, required: list[str], tokenizer: Any, lookup: list[str]) -> tuple[float, float]:
    ref_tokens = [item for item in tokenizer.tokenize(reference, lookup) if item.strip()]
    hyp_tokens = [item for item in tokenizer.tokenize(hypothesis, lookup) if item.strip()]
    ser = error_rate(ref_tokens, hyp_tokens)
    hyp_set = set(hyp_tokens)
    recall = sum(token in hyp_set for token in set(required)) / max(1, len(set(required))) if required else 1.0
    return ser, recall


def verify_package(root: Path) -> dict[str, Any]:
    manifest = json.loads((root / "kaggle_export_manifest.json").read_text(encoding="utf-8"))
    bad = []
    for relative, expected in manifest["checksums"].items():
        path = root / relative
        if not path.exists() or sha256(path) != expected:
            bad.append(relative)
    if bad:
        raise RuntimeError(f"Input checksum verification failed: {bad[:10]}")
    return manifest


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_table(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.Table.from_pylist(list(rows)), path, compression="zstd")


def whisper_pipeline(model_id: str, revision: str):
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, revision=revision, torch_dtype=dtype, low_cpu_mem_usage=True, use_safetensors=True,
    )
    if torch.cuda.is_available():
        model.to("cuda:0")
    processor = AutoProcessor.from_pretrained(model_id, revision=revision)
    return pipeline(
        "automatic-speech-recognition", model=model, tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor, torch_dtype=dtype,
        device=0 if torch.cuda.is_available() else -1,
        generate_kwargs={"language": "ne", "task": "transcribe"},
    )


def mms_components(model_id: str, revision: str, language: str):
    import torch
    from transformers import AutoProcessor, Wav2Vec2ForCTC

    processor = AutoProcessor.from_pretrained(model_id, revision=revision)
    model = Wav2Vec2ForCTC.from_pretrained(model_id, revision=revision)
    processor.tokenizer.set_target_lang(language)
    model.load_adapter(language)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return processor, model, device


def speaker_components(model_id: str, revision: str):
    import torch
    from transformers import AutoFeatureExtractor, WavLMForXVector

    extractor = AutoFeatureExtractor.from_pretrained(model_id, revision=revision)
    model = WavLMForXVector.from_pretrained(model_id, revision=revision)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return extractor, model, device


def speaker_embedding(path: Path, extractor: Any, model: Any, device: str) -> list[float]:
    import soundfile as sf
    import torch

    audio, sample_rate = sf.read(path, dtype="float32")
    values = extractor(audio, sampling_rate=sample_rate, return_tensors="pt")
    with torch.inference_mode():
        embedding = model(**{key: value.to(device) for key, value in values.items()}).embeddings[0]
        embedding = embedding / torch.clamp(torch.linalg.vector_norm(embedding), min=1e-12)
    return embedding.detach().cpu().float().tolist()


def transcribe_mms(path: Path, processor: Any, model: Any, device: str) -> str:
    import soundfile as sf
    import torch

    audio, sample_rate = sf.read(path, dtype="float32")
    if sample_rate != 16000:
        raise RuntimeError(f"MMS expects 16 kHz audio, got {sample_rate}: {path}")
    values = processor(audio, sampling_rate=sample_rate, return_tensors="pt")
    with torch.inference_mode():
        logits = model(values.input_values.to(device)).logits
    ids = torch.argmax(logits, dim=-1)[0]
    return processor.decode(ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default="/kaggle/input/nepali-synthetic-asr-qc")
    parser.add_argument("--output-dir", default="/kaggle/working/qc_results")
    parser.add_argument("--start-shard", type=int, default=int(os.environ.get("START_SHARD", "0")))
    parser.add_argument("--end-shard", type=int, default=int(os.environ.get("END_SHARD", "-1")))
    args = parser.parse_args()
    root, output = Path(args.input_root), Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    package = verify_package(root)
    config = yaml.safe_load((root / "synthetic_asr_config.yaml").read_text(encoding="utf-8"))
    tokenizer, lookup = load_tokenizer(root)
    all_rows = read_jsonl(root / "qc_input_manifest.jsonl")
    archives = sorted({row["archive"] for row in all_rows})
    end = len(archives) if args.end_shard < 0 else min(len(archives), args.end_shard)
    selected_archives = set(archives[args.start_shard:end])
    rows = [row for row in all_rows if row["archive"] in selected_archives]
    if not rows:
        raise RuntimeError("Selected shard range contains no records")

    qc = config["qc"]
    whisper = whisper_pipeline(qc["whisper_model"], qc.get("whisper_revision", "main"))
    audition_present = any(row.get("phase") == "audition" for row in rows)
    speaker_bundle = speaker_components(
        qc["speaker_embedding_model"], qc.get("speaker_embedding_revision", "main")
    ) if audition_present else None
    whisper_rows: list[dict[str, Any]] = []
    mms_targets: list[tuple[dict[str, Any], Path]] = []
    with tempfile.TemporaryDirectory(prefix="nepali-qc-", dir="/kaggle/temp" if Path("/kaggle/temp").exists() else None) as temporary:
        temp_root = Path(temporary)
        for archive_name in selected_archives:
            with tarfile.open(root / archive_name, "r:gz") as archive:
                archive.extractall(temp_root, filter="data")
        prepared = []
        for row in rows:
            audio_path = temp_root / row["archive_member"]
            if sha256(audio_path) != row["audio_sha256"]:
                raise RuntimeError(f"Extracted audio checksum mismatch: {row['job_id']}")
            prepared.append((row, audio_path))
        chunk_size = int(qc.get("whisper_chunk_size", 64))
        batch_size = int(qc.get("whisper_batch_size", 8))
        for start in range(0, len(prepared), chunk_size):
            chunk = prepared[start : start + chunk_size]
            outputs = whisper([str(audio_path) for _row, audio_path in chunk], batch_size=batch_size)
            for (row, audio_path), output_row in zip(chunk, outputs):
                hypothesis = normalize(output_row["text"])
                reference = normalize(row["reference_text"])
                ser, rare_recall = syllable_metrics(reference, hypothesis, row.get("rare_syllables", []), tokenizer, lookup)
                result = {
                    "job_id": row["job_id"], "audio_sha256": row["audio_sha256"],
                    "model": qc["whisper_model"], "revision": qc.get("whisper_revision", "main"),
                    "hypothesis": hypothesis,
                    "cer": error_rate(list(reference.replace(" ", "")), list(hypothesis.replace(" ", ""))),
                    "wer": error_rate(reference.split(), hypothesis.split()),
                    "syllable_error_rate": ser, "rare_syllable_recall": rare_recall,
                }
                if row.get("phase") == "audition" and speaker_bundle is not None:
                    result["speaker_embedding"] = speaker_embedding(audio_path, *speaker_bundle)
                    result["speaker_embedding_model"] = qc["speaker_embedding_model"]
                    result["speaker_embedding_revision"] = qc.get("speaker_embedding_revision", "main")
                whisper_rows.append(result)
                max_cer = float(qc.get("max_cer", 0.35))
                if row.get("rare_syllables") or result["cer"] > max_cer or abs(result["cer"] - max_cer) <= 0.05 or rare_recall < 1.0:
                    mms_targets.append((row, audio_path))

        write_table(output / "whisper_results.parquet", whisper_rows)
        del whisper, speaker_bundle
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        mms_rows: list[dict[str, Any]] = []
        if mms_targets:
            processor, model, device = mms_components(qc["mms_model"], qc.get("mms_revision", "main"), qc["mms_language_adapter"])
            for row, audio_path in mms_targets:
                hypothesis = normalize(transcribe_mms(audio_path, processor, model, device))
                reference = normalize(row["reference_text"])
                ser, rare_recall = syllable_metrics(reference, hypothesis, row.get("rare_syllables", []), tokenizer, lookup)
                mms_rows.append({
                    "job_id": row["job_id"], "audio_sha256": row["audio_sha256"],
                    "model": qc["mms_model"], "revision": qc.get("mms_revision", "main"),
                    "language_adapter": qc["mms_language_adapter"], "hypothesis": hypothesis,
                    "cer": error_rate(list(reference.replace(" ", "")), list(hypothesis.replace(" ", ""))),
                    "wer": error_rate(reference.split(), hypothesis.split()),
                    "syllable_error_rate": ser, "rare_syllable_recall": rare_recall,
                })
        write_table(output / "mms_results.parquet", mms_rows)

    summary = {
        "records": len(rows), "whisper_records": len(whisper_rows), "mms_records": len(mms_rows),
        "start_shard": args.start_shard, "end_shard": end,
        "mean_whisper_cer": sum(row["cer"] for row in whisper_rows) / len(whisper_rows),
    }
    (output / "kaggle_qc_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    result_manifest = {
        "input_manifest_sha256": package["checksums"]["qc_input_manifest.jsonl"],
        "models": package["models"],
        "outputs": {name: sha256(output / name) for name in ("whisper_results.parquet", "mms_results.parquet", "kaggle_qc_summary.json")},
    }
    (output / "kaggle_manifest.json").write_text(json.dumps(result_manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
