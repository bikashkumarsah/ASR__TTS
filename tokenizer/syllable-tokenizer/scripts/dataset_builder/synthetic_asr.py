"""Existing-VM and Kaggle synthetic Nepali ASR data pipeline.

The module deliberately contains no cloud-resource creation operations. Google Cloud
is used only through the Text-to-Speech data-plane API, and Kaggle interaction is an
explicit export/import boundary. Expensive stages are resumable and checksum-gated.
"""

from __future__ import annotations

import concurrent.futures
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import re
import shutil
import sqlite3
import subprocess
import tarfile
import threading
import time
import unicodedata
import wave
import zipfile
from collections import Counter, defaultdict, deque
from collections.abc import Iterable, Iterator, Mapping, Sequence
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import yaml

_SCRIPT_ROOT = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = _SCRIPT_ROOT.parent
_REPO_ROOT = _PROJECT_ROOT.parent.parent
_DEFAULT_VOCAB = _PROJECT_ROOT / "dataset" / "nepali_syllables_lookup.vocab"
_DEFAULT_TOKENIZER = _SCRIPT_ROOT / "syllabic_tokenizer.py"
_DEVANAGARI_DIGITS = "०१२३४५६७८९"
_DIGIT_WORDS = ("शून्य", "एक", "दुई", "तीन", "चार", "पाँच", "छ", "सात", "आठ", "नौ")
_PUNCT_RE = re.compile(r"[^\u0900-\u097F\s]")
_SPACE_RE = re.compile(r"\s+")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    os.replace(temporary, path)
    return count


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise TypeError(f"Expected a JSON object at {path}:{line_number}")
            yield value


def _load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError(f"Configuration must be a mapping: {config_path}")
    return config, config_path


def _resolve_config_path(config_path: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def _load_tokenizer(config: Mapping[str, Any], config_path: Path):
    section = config.get("tokenizer", {})
    vocab_path = _resolve_config_path(config_path, section.get("vocabulary", _DEFAULT_VOCAB))
    source_path = _resolve_config_path(config_path, section.get("source", _DEFAULT_TOKENIZER))
    expected = section.get("vocabulary_sha256")
    actual = _sha256(vocab_path)
    if expected and actual != expected:
        raise RuntimeError(
            f"Authoritative vocabulary checksum mismatch: expected {expected}, got {actual} ({vocab_path})"
        )
    spec = importlib.util.spec_from_file_location("synthetic_asr_pinned_tokenizer", source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load tokenizer source: {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    lookup = module.get_lookup_tokens(str(vocab_path))
    return module, lookup, vocab_path, source_path


def _fingerprint(config: Mapping[str, Any], config_path: Path) -> dict[str, Any]:
    _module, lookup, vocab_path, source_path = _load_tokenizer(config, config_path)
    stable_config = json.dumps(config, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return {
        "configuration_sha256": hashlib.sha256(stable_config).hexdigest(),
        "vocabulary_path": str(vocab_path),
        "vocabulary_sha256": _sha256(vocab_path),
        "tokenizer_source": str(source_path),
        "tokenizer_sha256": _sha256(source_path),
        "lookup_entries": len(lookup),
    }


def _run_quiet(command: Sequence[str], timeout: int = 30) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            list(command), capture_output=True, text=True, timeout=timeout, check=False,
        )
        detail = (result.stderr or result.stdout).strip().splitlines()
        return result.returncode == 0, (detail[-1][:300] if detail else "")
    except (OSError, subprocess.SubprocessError) as error:
        return False, str(error)[:300]


def _system_memory_bytes() -> int | None:
    try:
        if Path("/proc/meminfo").exists():
            match = re.search(r"MemTotal:\s+(\d+)\s+kB", Path("/proc/meminfo").read_text())
            if match:
                return int(match.group(1)) * 1024
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (OSError, ValueError, KeyError):
        return None


def _software_versions(packages: Sequence[str]) -> dict[str, str | None]:
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def synthetic_asr_preflight(work_dir: str | Path, config_path: str | Path) -> dict[str, Any]:
    """Perform read-only environment checks and write a reproducible report."""
    run_dir = Path(work_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    config, resolved_config = _load_config(config_path)
    disk = shutil.disk_usage(run_dir)
    checks: dict[str, Any] = {
        "run_id": _text_sha256(str(run_dir) + _fingerprint(config, resolved_config)["configuration_sha256"])[:16],
        "checked_at": _utc_now(),
        "resource_policy": {
            "existing_vm_only": True,
            "creates_cloud_resources": False,
            "gpu_work": "kaggle_qc_only",
            "training_execution": False,
        },
        "system": {
            "nproc": os.cpu_count() or 1,
            "ram_bytes": _system_memory_bytes(),
            "disk_total_bytes": disk.total,
            "disk_available_bytes": disk.free,
            "operating_system": platform.platform(),
            "python": platform.python_version(),
        },
        "tools": {},
        "inputs": _fingerprint(config, resolved_config),
        "issues": [],
    }
    for tool in ("ffmpeg", "ffprobe", "gcloud", "kaggle"):
        checks["tools"][tool] = shutil.which(tool)
    if not checks["tools"]["ffmpeg"]:
        checks["issues"].append("FFmpeg is missing; install it on the existing VM.")
    checks["python_libraries"] = {}
    for library in ("numpy", "soundfile", "pyarrow", "google.cloud.texttospeech"):
        try:
            spec = importlib.util.find_spec(library)
            checks["python_libraries"][library] = bool(spec)
        except (ImportError, ModuleNotFoundError, AttributeError):
            checks["python_libraries"][library] = False
    missing_libraries = [name for name, available in checks["python_libraries"].items() if not available]
    if missing_libraries:
        checks["issues"].append("Missing Python libraries: " + ", ".join(missing_libraries))
    minimum_disk = float(config.get("storage", {}).get("minimum_available_disk_gb", 100)) * 1024**3
    if disk.free < minimum_disk:
        checks["issues"].append(
            f"Available disk is below the configured {minimum_disk / 1024**3:.0f} GiB minimum; "
            "free space on the existing VM before synthesis."
        )
    checks["configured_inputs"] = {}
    for label, configured_path in config.get("inputs", {}).items():
        path = Path(str(configured_path)).expanduser()
        checks["configured_inputs"][label] = {
            "path": str(path), "exists": path.is_file(),
            "sha256": _sha256(path) if path.is_file() else None,
        }
        if not path.is_file():
            checks["issues"].append(f"Configured input is missing: {label}={path}")

    project_ok, project_detail = _run_quiet(["gcloud", "config", "get-value", "project"])
    adc_ok, _adc_detail = _run_quiet(["gcloud", "auth", "application-default", "print-access-token"])
    checks["google"] = {
        "project_configured": project_ok and project_detail not in {"", "(unset)"},
        "project": project_detail if project_ok else None,
        "application_default_credentials": adc_ok,
    }
    try:
        from google.cloud import texttospeech  # type: ignore

        client = texttospeech.TextToSpeechClient()
        client.list_voices(request={"language_code": "ne-NP"}, timeout=15)
        checks["google"]["text_to_speech_api_access"] = True
    except Exception as error:  # noqa: BLE001 - credentials/API failures vary by installed client
        checks["google"]["text_to_speech_api_access"] = False
        checks["google"]["text_to_speech_error"] = str(error)[:500]
        checks["issues"].append("Cloud Text-to-Speech API access could not be verified.")

    bucket = config.get("storage", {}).get("existing_gcs_bucket")
    if bucket:
        bucket_uri = str(bucket) if str(bucket).startswith("gs://") else f"gs://{bucket}"
        bucket_ok, bucket_detail = _run_quiet(["gcloud", "storage", "ls", bucket_uri])
        checks["google"]["existing_bucket"] = bucket_uri
        checks["google"]["existing_bucket_access"] = bucket_ok
        if not bucket_ok:
            checks["issues"].append(f"Configured existing bucket is not accessible: {bucket_detail}")
    else:
        checks["google"]["existing_bucket"] = None
        checks["google"]["existing_bucket_access"] = None

    kaggle_ok, kaggle_detail = _run_quiet(["kaggle", "datasets", "list", "--mine", "--page-size", "1"])
    checks["kaggle"] = {"cli_authenticated": kaggle_ok}
    if not kaggle_ok:
        checks["kaggle"]["error"] = kaggle_detail
        checks["issues"].append("Kaggle CLI authentication could not be verified.")

    legal_path = run_dir / str(config.get("run", {}).get("legal_review_file", "legal_review.json"))
    checks["legal_review"] = {"path": str(legal_path), "present": legal_path.exists()}
    checks["ready"] = not checks["issues"]
    _atomic_json(run_dir / "preflight.json", checks)
    return checks


def _normalize_spaces(text: str) -> str:
    return _SPACE_RE.sub(" ", unicodedata.normalize("NFC", text)).strip()


def _asr_punctuation_free(text: str) -> str:
    return _normalize_spaces(_PUNCT_RE.sub(" ", text).replace("।", " ").replace("॥", " "))


def _spell_digits(value: str) -> str:
    words = []
    for char in value:
        if char in _DEVANAGARI_DIGITS:
            words.append(_DIGIT_WORDS[_DEVANAGARI_DIGITS.index(char)])
        elif char.isdigit():
            words.append(_DIGIT_WORDS[int(char)])
    return " ".join(words)


def normalize_spoken_text(text: str) -> dict[str, Any]:
    """Produce aligned TTS and punctuation-free ASR forms.

    Numerals are intentionally spelled digit-by-digit. This is deterministic and
    avoids making an unverifiable semantic guess about dates, identifiers, or large
    numbers. Ambiguous separated numeric forms are quarantined for manual correction.
    """
    original = _normalize_spaces(text)
    actions: list[str] = []
    value = original

    def replace_date(match: re.Match[str]) -> str:
        actions.append("expanded_date_ymd")
        return (
            f"{_spell_digits(match.group(1))} साल "
            f"{_spell_digits(match.group(2))} महिना {_spell_digits(match.group(3))} गते"
        )

    value = re.sub(
        r"(?<![०-९0-9])([०-९0-9]{4})\s*[-/]\s*([०-९0-9]{1,2})\s*[-/]\s*([०-९0-9]{1,2})(?![०-९0-9])",
        replace_date,
        value,
    )
    ambiguous = bool(re.search(r"[०-९0-9]+\s*[/:-]\s*[०-९0-9]+", value))
    if ambiguous:
        return {"status": "quarantine", "reason": "ambiguous_numeric_form", "source_text": original}
    value, percent_count = re.subn(
        r"([०-९0-9]+(?:[.,][०-९0-9]+)?)\s*%",
        lambda match: match.group(1) + " प्रतिशत",
        value,
    )
    if percent_count:
        actions.append("expanded_percentage")
    value, currency_count = re.subn(
        r"(?:रु\.?|रू\.?|₨|NPR)\s*([०-९0-9]+(?:[.,][०-९0-9]+)?)",
        r"\1 रुपैयाँ",
        value,
    )
    if currency_count:
        actions.append("expanded_currency")

    def replace_number(match: re.Match[str]) -> str:
        raw = match.group(0)
        if "." in raw or "," in raw:
            left, right = re.split(r"[.,]", raw, maxsplit=1)
            actions.append("expanded_decimal")
            return f"{_spell_digits(left)} दशमलव {_spell_digits(right)}"
        actions.append("expanded_digits")
        return _spell_digits(raw)

    value = re.sub(r"[०-९0-9]+(?:[.,][०-९0-9]+)?", replace_number, value)
    asr_text = _asr_punctuation_free(value)
    if not asr_text:
        return {"status": "quarantine", "reason": "empty_after_normalization", "source_text": original}
    tts_text = asr_text if asr_text.endswith("।") else asr_text + "।"
    return {
        "status": "ok",
        "source_text": original,
        "tts_text": tts_text,
        "asr_text": asr_text,
        "actions": sorted(set(actions)),
    }


def _normalize_slr54_text(text: str) -> str:
    return _asr_punctuation_free(text)


def _audio_info(path: Path) -> tuple[int, int, float]:
    """Return sample count, sample rate, duration without decoding full audio."""
    try:
        import soundfile as sf  # type: ignore

        info = sf.info(str(path))
        return int(info.frames), int(info.samplerate), float(info.duration)
    except Exception:  # noqa: BLE001 - fall back from any soundfile decoder failure for WAV
        if path.suffix.lower() != ".wav":
            raise RuntimeError(f"soundfile is required to inspect non-WAV input: {path}")
        with wave.open(str(path), "rb") as handle:
            frames, rate = handle.getnframes(), handle.getframerate()
        return frames, rate, frames / rate


def _verify_archive(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            bad_member = archive.testzip()
            members = len(archive.infolist())
        if bad_member:
            raise RuntimeError(f"Corrupt archive member {bad_member} in {path}")
    else:
        members = 0
        with tarfile.open(path, "r:*") as archive:
            for member in archive:
                members += 1
                if member.isfile():
                    extracted = archive.extractfile(member)
                    if extracted is not None:
                        while extracted.read(1 << 20):
                            pass
    return {"sha256": _sha256(path), "members": members, "integrity": "passed"}


def _slr54_rows(tsv_path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with tsv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for line_number, values in enumerate(reader, 1):
            if not values or not any(value.strip() for value in values):
                continue
            if line_number == 1 and any("file" in value.lower() for value in values[:1]):
                continue
            if len(values) < 3:
                raise ValueError(f"Expected three TSV fields at {tsv_path}:{line_number}")
            file_id, speaker_id = values[0].strip(), values[1].strip()
            transcript = "\t".join(values[2:]).strip()
            if not file_id or not speaker_id or not transcript:
                continue
            rows.append({"file_id": file_id, "speaker_id": speaker_id, "transcript": transcript})
    if not rows:
        raise RuntimeError(f"No SLR54 records found in {tsv_path}")
    return rows


def _assign_speaker_splits(rows: Sequence[Mapping[str, str]], seed: int) -> dict[str, str]:
    speaker_sizes = Counter(row["speaker_id"] for row in rows)
    total = sum(speaker_sizes.values())
    targets = {"train": total * 0.8, "dev": total * 0.1, "test": total * 0.1}
    current = Counter()
    result: dict[str, str] = {}
    ordered = sorted(
        speaker_sizes,
        key=lambda speaker: (-speaker_sizes[speaker], _text_sha256(f"{seed}:{speaker}")),
    )
    for speaker in ordered:
        split = max(("train", "dev", "test"), key=lambda name: (targets[name] - current[name], name))
        result[speaker] = split
        current[split] += speaker_sizes[speaker]
    return result


def _write_idx(path: Path, pairs: Iterable[tuple[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for key, value in pairs:
            handle.write(f"{key} {value}\n")
    os.replace(temporary, path)


def prepare_slr54_speechain(
    input_root: str | Path,
    output_dir: str | Path,
    seed: int = 42,
    workers: int | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Prepare deterministic, speaker-disjoint SLR54 metadata for SpeeChain."""
    source_root = Path(input_root).expanduser().resolve()
    target_root = Path(output_dir).expanduser().resolve()
    manifest_path = target_root / "manifest.json"
    tsv_candidates = sorted(source_root.rglob("utt_spk_text.tsv"))
    if len(tsv_candidates) != 1:
        raise RuntimeError(f"Expected exactly one utt_spk_text.tsv under {source_root}, found {len(tsv_candidates)}")
    tsv_path = tsv_candidates[0]
    archive_paths = sorted(set(source_root.rglob("*.zip")) | set(source_root.rglob("*.tar.gz")) | set(source_root.rglob("*.tgz")))
    fingerprint = {
        "utt_spk_text_sha256": _sha256(tsv_path), "seed": seed,
        "archive_sha256": {str(path.relative_to(source_root)): _sha256(path) for path in archive_paths},
    }
    if resume and manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("input_fingerprint") == fingerprint:
            for split, details in existing.get("splits", {}).items():
                for name, expected in details.get("checksums", {}).items():
                    path = target_root / split / name
                    if not path.is_file() or _sha256(path) != expected:
                        raise RuntimeError(f"SLR54 resume output checksum mismatch: {path}")
            return existing
        raise RuntimeError("SLR54 resume fingerprint mismatch; use a fresh output directory.")

    rows = _slr54_rows(tsv_path)
    audio_files = [
        path for suffix in ("*.wav", "*.flac") for path in source_root.rglob(suffix)
    ]
    by_stem: dict[str, Path] = {}
    duplicate_stems: set[str] = set()
    for path in audio_files:
        if path.stem in by_stem:
            duplicate_stems.add(path.stem)
        else:
            by_stem[path.stem] = path.resolve()
    if duplicate_stems:
        raise RuntimeError(f"Ambiguous SLR54 audio stems: {sorted(duplicate_stems)[:5]}")

    valid: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in rows:
        path = by_stem.get(Path(row["file_id"]).stem)
        transcript = _normalize_slr54_text(row["transcript"])
        if path is None:
            rejected.append({**row, "reason": "missing_audio"})
        elif not transcript:
            rejected.append({**row, "reason": "empty_transcript"})
        else:
            valid.append({**row, "audio_path": str(path), "normalized_text": transcript})
    if not valid:
        raise RuntimeError("No usable SLR54 records remain after alignment checks.")

    speaker_split = _assign_speaker_splits(valid, seed)
    max_workers = max(1, workers or os.cpu_count() or 1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        infos = list(executor.map(lambda row: _audio_info(Path(row["audio_path"])), valid))
    for row, (frames, rate, duration) in zip(valid, infos):
        row.update({"frames": frames, "sample_rate": rate, "duration_seconds": duration})
        row["split"] = speaker_split[row["speaker_id"]]
        row["text_sha256"] = _text_sha256(row["normalized_text"])

    target_root.mkdir(parents=True, exist_ok=True)
    split_summaries: dict[str, Any] = {}
    for split in ("train", "dev", "test"):
        split_rows = sorted((row for row in valid if row["split"] == split), key=lambda row: row["file_id"])
        split_dir = target_root / split
        _write_idx(split_dir / "idx2wav", ((row["file_id"], row["audio_path"]) for row in split_rows))
        _write_idx(split_dir / "idx2wav_len", ((row["file_id"], row["frames"]) for row in split_rows))
        _write_idx(split_dir / "idx2no-punc_text", ((row["file_id"], row["normalized_text"]) for row in split_rows))
        _write_idx(split_dir / "idx2spk", ((row["file_id"], row["speaker_id"]) for row in split_rows))
        hash_path = split_dir / "text_hashes.txt"
        _write_idx(hash_path, ((row["text_sha256"], "") for row in split_rows))
        split_summaries[split] = {
            "utterances": len(split_rows),
            "speakers": len({row["speaker_id"] for row in split_rows}),
            "hours": sum(row["duration_seconds"] for row in split_rows) / 3600,
            "paths": {
                name: str(split_dir / name)
                for name in ("idx2wav", "idx2wav_len", "idx2no-punc_text", "idx2spk")
            },
            "text_hashes": str(hash_path),
            "checksums": {
                name: _sha256(split_dir / name)
                for name in ("idx2wav", "idx2wav_len", "idx2no-punc_text", "idx2spk", "text_hashes.txt")
            },
        }

    speakers = {split: {speaker for speaker, assigned in speaker_split.items() if assigned == split} for split in ("train", "dev", "test")}
    overlap = {
        "train_dev": sorted(speakers["train"] & speakers["dev"]),
        "train_test": sorted(speakers["train"] & speakers["test"]),
        "dev_test": sorted(speakers["dev"] & speakers["test"]),
    }
    if any(overlap.values()):
        raise RuntimeError(f"Speaker-disjoint split invariant failed: {overlap}")
    rejected_path = target_root / "rejected.jsonl"
    _write_jsonl(rejected_path, rejected)
    archives = {str(path.relative_to(source_root)): _verify_archive(path) for path in archive_paths}
    manifest = {
        "schema_version": 1,
        "created_at": _utc_now(),
        "input_root": str(source_root),
        "input_fingerprint": fingerprint,
        "archives": archives,
        "input_rows": len(rows),
        "usable_rows": len(valid),
        "rejected_rows": len(rejected),
        "rejected_sha256": _sha256(rejected_path),
        "splits": split_summaries,
        "speaker_overlap": overlap,
        "speaker_disjoint": True,
        "license": "CC BY-SA 4.0 (verify the downloaded SLR54 release metadata)",
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def _load_hashes(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.split(maxsplit=1)[0] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _record_syllables(record: Mapping[str, Any], tokenizer: Any, lookup: Sequence[str], text: str) -> list[str]:
    supplied = record.get("syllables")
    if supplied and isinstance(supplied, list):
        return [str(item) for item in supplied if str(item).strip()]
    return [token for token in tokenizer.tokenize(text, lookup) if token.strip()]


def _balanced_sample(records: Sequence[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    if count >= len(records):
        return list(records)
    ordered = sorted(records, key=lambda row: _text_sha256(f"{seed}:{row['utterance_id']}"))
    strata: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    for row in ordered:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        source = row.get("source_corpora")
        source_key = "+".join(sorted(map(str, source))) if isinstance(source, list) else str(source or "unknown")
        length_bin = min(4, len(row.get("syllables", [])) // 20)
        key = f"{source_key}|{metadata.get('sector', 'unknown')}|rare={bool(row.get('rare_syllables'))}|len={length_bin}"
        strata[key].append(row)
    output: list[dict[str, Any]] = []
    keys = sorted(strata)
    while len(output) < count and keys:
        for key in list(keys):
            if strata[key]:
                output.append(strata[key].popleft())
                if len(output) == count:
                    break
            if not strata[key]:
                keys.remove(key)
    return output


def _select_rare_support(records: Sequence[dict[str, Any]], rare: set[str], extra_needed: int, seed: int) -> list[dict[str, Any]]:
    """Greedy set cover repeated to provide extra distinct voice realizations."""
    if not rare or extra_needed <= 0:
        return []
    candidates = [row for row in records if rare.intersection(row["syllables"])]
    selected: list[dict[str, Any]] = []
    remaining = {token: extra_needed for token in rare}
    while any(value > 0 for value in remaining.values()):
        best: tuple[int, str, dict[str, Any]] | None = None
        for row in candidates:
            gain = sum(1 for token in set(row["syllables"]) if remaining.get(token, 0) > 0)
            if not gain:
                continue
            tie = _text_sha256(f"{seed}:{len(selected)}:{row['utterance_id']}")
            candidate = (gain, tie, row)
            if best is None or gain > best[0] or (gain == best[0] and tie < best[1]):
                best = candidate
        if best is None:
            break
        row = best[2]
        copy = dict(row)
        copy["required_rare_syllables"] = sorted(
            token for token in set(row["syllables"]) if remaining.get(token, 0) > 0
        )
        selected.append(copy)
        for token in copy["required_rare_syllables"]:
            remaining[token] -= 1
        if len(selected) > len(candidates) * extra_needed:
            break
    missing = {token: count for token, count in remaining.items() if count > 0}
    if missing:
        raise RuntimeError(f"Insufficient support for rare-tail voice realizations: {list(missing.items())[:10]}")
    return selected


def prepare_synthetic_asr(
    input_path: str | Path,
    slr54_manifest: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    workers: int | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Normalize selected text, mark evaluation overlap, and construct phase queues."""
    del workers  # deterministic preprocessing is streaming and CPU-light
    source_path = Path(input_path).expanduser().resolve()
    slr_path = Path(slr54_manifest).expanduser().resolve()
    run_dir = Path(output_dir).expanduser().resolve()
    config, resolved_config = _load_config(config_path)
    fingerprints = _fingerprint(config, resolved_config)
    fingerprints.update({"input_sha256": _sha256(source_path), "slr54_manifest_sha256": _sha256(slr_path)})
    completion = run_dir / "prepare.complete.json"
    if resume and completion.exists():
        existing = json.loads(completion.read_text(encoding="utf-8"))
        if existing.get("fingerprints") == fingerprints:
            for relative, expected in existing.get("output_checksums", {}).items():
                path = run_dir / relative
                if not path.is_file() or _sha256(path) != expected:
                    raise RuntimeError(f"Synthetic resume output checksum mismatch: {path}")
            return existing
        raise RuntimeError("Synthetic preparation resume fingerprint mismatch; use a fresh run directory.")

    tokenizer, lookup, vocab_source, tokenizer_source = _load_tokenizer(config, resolved_config)
    max_input_bytes = int(config["google_tts"].get("max_input_bytes", 4000))
    oversized_prompts = {
        name: len(str(prompt).encode("utf-8"))
        for name, prompt in config["google_tts"].get("styles", {}).items()
        if len(str(prompt).encode("utf-8")) > max_input_bytes
    }
    if oversized_prompts:
        raise RuntimeError(f"TTS style prompts exceed the provider byte limit: {oversized_prompts}")
    slr_manifest_data = json.loads(slr_path.read_text(encoding="utf-8"))
    evaluation_hashes: set[str] = set()
    for split in ("dev", "test"):
        evaluation_hashes.update(_load_hashes(Path(slr_manifest_data["splits"][split]["text_hashes"])))

    normalized: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    seen_source_hashes: set[str] = set()
    for index, raw in enumerate(_read_jsonl(source_path)):
        source_text = str(raw.get("text") or raw.get("normalized_text") or raw.get("source_text") or "")
        spoken = normalize_spoken_text(source_text)
        if spoken["status"] != "ok":
            quarantine.append({"input_index": index, **spoken})
            continue
        source_hash = str(raw.get("normalized_sha256") or _text_sha256(_normalize_spaces(source_text)))
        if source_hash in seen_source_hashes:
            quarantine.append({"input_index": index, "source_text": source_text, "reason": "duplicate_source_hash"})
            continue
        seen_source_hashes.add(source_hash)
        original_syllables = _record_syllables(raw, tokenizer, lookup, _normalize_spaces(source_text))
        asr_syllables = [token for token in tokenizer.tokenize(spoken["asr_text"], lookup) if token.strip()]
        required_rare = set(raw.get("rare_syllables") or [])
        disappeared = required_rare.difference(asr_syllables)
        if disappeared:
            quarantine.append({
                "input_index": index, "source_text": source_text,
                "reason": "required_rare_syllable_disappeared", "syllables": sorted(disappeared),
            })
            continue
        if len(spoken["tts_text"].encode("utf-8")) > max_input_bytes:
            quarantine.append({"input_index": index, "source_text": source_text, "reason": "tts_input_byte_limit"})
            continue
        utterance_id = f"syn_{index:08d}_{source_hash[:10]}"
        row = {
            "utterance_id": utterance_id,
            "source_text": spoken["source_text"],
            "tts_text": spoken["tts_text"],
            "asr_text": spoken["asr_text"],
            "normalized_sha256": source_hash,
            "spoken_sha256": _text_sha256(spoken["asr_text"]),
            "syllables": asr_syllables,
            "unique_syllables": sorted(set(asr_syllables)),
            "rare_syllables": sorted(required_rare),
            "normalization_actions": spoken["actions"],
            "evaluation_overlap": _text_sha256(_normalize_slr54_text(source_text)) in evaluation_hashes,
            "source_corpora": raw.get("source_corpora", raw.get("source")),
            "metadata": raw.get("metadata", {key: raw.get(key) for key in ("tense", "polarity", "gender", "sector") if raw.get(key) is not None}),
            "selection_reason": raw.get("selection_reason"),
            "similarity_exception": raw.get("similarity_exception", False),
            "original_syllables": original_syllables,
        }
        normalized.append(row)

    target = int(config.get("run", {}).get("target_canonical", 20000))
    if len(normalized) != target:
        raise RuntimeError(
            f"Expected exactly {target:,} usable canonical texts, got {len(normalized):,}; "
            f"{len(quarantine):,} were quarantined. Correct the input before synthesis."
        )
    frequency = Counter(token for row in normalized for token in row["syllables"])
    rare_threshold = int(config.get("rare_tail", {}).get("occurrence_threshold", 20))
    rare = {token for token, count in frequency.items() if count < rare_threshold}
    for row in normalized:
        row["rare_syllables"] = sorted(set(row["syllables"]).intersection(rare))
    required_realizations = int(config.get("rare_tail", {}).get("required_voice_realizations", 3))
    rare_support = _select_rare_support(normalized, rare, required_realizations - 1, int(config["run"].get("seed", 42)))

    run_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(run_dir / "manifests" / "prepared_texts.jsonl", normalized)
    _write_jsonl(run_dir / "manifests" / "quarantined_texts.jsonl", quarantine)
    _write_jsonl(run_dir / "manifests" / "rare_extra_support.jsonl", rare_support)
    (run_dir / "manifests" / "attainable_syllables.txt").write_text(
        "\n".join(sorted(frequency)) + "\n", encoding="utf-8"
    )
    pins_dir = run_dir / "pins"
    pins_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(vocab_source, pins_dir / "nepali_syllables_lookup.vocab")
    shutil.copy2(tokenizer_source, pins_dir / "syllabic_tokenizer.py")
    run_config = json.loads(json.dumps(config))
    run_config.setdefault("tokenizer", {})["vocabulary"] = "pins/nepali_syllables_lookup.vocab"
    run_config["tokenizer"]["source"] = "pins/syllabic_tokenizer.py"
    (run_dir / "synthetic_asr_config.yaml").write_text(
        yaml.safe_dump(run_config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    run_fingerprints = _fingerprint(run_config, run_dir / "synthetic_asr_config.yaml")
    _atomic_json(run_dir / "manifests" / "slr54_reference.json", slr_manifest_data)
    summary = {
        "schema_version": 1,
        "created_at": _utc_now(),
        "fingerprints": fingerprints,
        "run_fingerprints": run_fingerprints,
        "input_records": len(normalized) + len(quarantine),
        "prepared_canonical": len(normalized),
        "quarantined": len(quarantine),
        "evaluation_overlap": sum(bool(row["evaluation_overlap"]) for row in normalized),
        "observed_syllables": len(frequency),
        "expected_attainable_syllables": int(config.get("tokenizer", {}).get("attainable_inventory", 1358)),
        "rare_syllables": len(rare),
        "rare_extra_sentences": len(rare_support),
        "rare_frequency": dict(sorted((token, frequency[token]) for token in rare)),
        "prepared_manifest": str(run_dir / "manifests" / "prepared_texts.jsonl"),
        "slr54_manifest": str(slr_path),
        "output_checksums": {
            str(path.relative_to(run_dir)): _sha256(path)
            for path in (
                run_dir / "manifests" / "prepared_texts.jsonl",
                run_dir / "manifests" / "quarantined_texts.jsonl",
                run_dir / "manifests" / "rare_extra_support.jsonl",
                run_dir / "manifests" / "attainable_syllables.txt",
                run_dir / "synthetic_asr_config.yaml",
                run_dir / "pins" / "nepali_syllables_lookup.vocab",
                run_dir / "pins" / "syllabic_tokenizer.py",
            )
        },
    }
    _atomic_json(completion, summary)
    return summary


def _load_run_config(run_dir: Path) -> tuple[dict[str, Any], Path]:
    path = run_dir / "synthetic_asr_config.yaml"
    if not path.exists():
        raise RuntimeError(f"Prepared run configuration is missing: {path}")
    config, resolved = _load_config(path)
    completion = run_dir / "prepare.complete.json"
    if completion.exists():
        expected = json.loads(completion.read_text(encoding="utf-8")).get("run_fingerprints")
        actual = _fingerprint(config, resolved)
        if expected and actual != expected:
            raise RuntimeError(
                "Prepared run configuration, vocabulary, or tokenizer source changed; "
                "use a fresh run directory instead of mixing artifacts."
            )
    return config, resolved


def _require_legal_review(run_dir: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    legal_path = run_dir / str(config.get("run", {}).get("legal_review_file", "legal_review.json"))
    if not legal_path.exists():
        raise RuntimeError(
            f"Synthesis is blocked until the project owner reviews provider and dataset terms. "
            f"Copy configs/legal_review.example.json to {legal_path} and complete it."
        )
    review = json.loads(legal_path.read_text(encoding="utf-8"))
    required = {
        "approved": True,
        "usage": "academic_private",
        "google_terms_reviewed": True,
        "kaggle_private_transfer_approved": True,
    }
    mismatches = {key: {"required": expected, "actual": review.get(key)} for key, expected in required.items() if review.get(key) != expected}
    if mismatches:
        raise RuntimeError(f"Legal-review gate is incomplete: {mismatches}")
    return review


def _configured_voices(config: Mapping[str, Any]) -> list[dict[str, str]]:
    voices = [dict(voice) for voice in config.get("google_tts", {}).get("voices", [])]
    names = [voice.get("name") for voice in voices]
    if not voices or len(names) != len(set(names)) or any(not name for name in names):
        raise RuntimeError("google_tts.voices must contain unique non-empty voice names")
    return voices


def _qualified_voices(run_dir: Path, config: Mapping[str, Any], phase: str) -> list[dict[str, str]]:
    voices = _configured_voices(config)
    if phase == "audition":
        return voices
    if phase == "full":
        pilot_gate_path = run_dir / "qc" / "pilot_qc_gate.json"
        if not pilot_gate_path.exists() or not json.loads(pilot_gate_path.read_text(encoding="utf-8")).get("passed"):
            raise RuntimeError("Full synthesis is blocked until the imported 2,000-text pilot passes its QC gate.")
    path = run_dir / "qc" / "voice_qualification.json"
    if not path.exists():
        raise RuntimeError(
            "Pilot/full synthesis requires qc/voice_qualification.json generated by importing "
            "audition Kaggle QC. Export, run, and import the audition results first."
        )
    qualification = json.loads(path.read_text(encoding="utf-8"))
    selected = qualification.get("selected_voices", [])
    by_name = {voice["name"]: voice for voice in voices}
    result = [by_name[name] for name in selected if name in by_name]
    female = sum(voice.get("gender") == "female" for voice in result)
    male = sum(voice.get("gender") == "male" for voice in result)
    expected_female = int(config["google_tts"].get("selected_female", 4))
    expected_male = int(config["google_tts"].get("selected_male", 4))
    if female != expected_female or male != expected_male:
        raise RuntimeError(
            f"Voice qualification must contain {expected_female} female and {expected_male} male voices; "
            f"got {female} and {male}."
        )
    return result


def _style_for(row: Mapping[str, Any], rare_tail: bool) -> str:
    if rare_tail:
        return "neutral"
    metadata = row.get("metadata") or {}
    sector = metadata.get("sector") if isinstance(metadata, dict) else None
    if sector in {"formal", "news", "education", "business", "health"}:
        return "formal"
    if sector in {"conversational", "entertainment", "sports"}:
        return "conversational"
    return "neutral"


def _phase_jobs(run_dir: Path, config: Mapping[str, Any], phase: str, voices: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    records = list(_read_jsonl(run_dir / "manifests" / "prepared_texts.jsonl"))
    seed = int(config.get("run", {}).get("seed", 42))
    if phase == "audition":
        texts = _balanced_sample(records, int(config["phases"].get("audition_texts", 50)), seed)
        return [
            {
                **row,
                "job_id": f"aud_{row['utterance_id']}_{voice['name'].lower()}",
                "phase": phase,
                "realization": "audition",
                "voice": voice["name"],
                "voice_gender": voice.get("gender"),
                "style": _style_for(row, False),
                "required_rare_syllables": row.get("rare_syllables", []),
            }
            for row in texts for voice in voices
        ]
    limit = int(config["phases"].get("pilot_texts" if phase == "pilot" else "full_texts", len(records)))
    texts = _balanced_sample(records, limit, seed) if phase == "pilot" else records[:limit]
    canonical: list[dict[str, Any]] = []
    for index, row in enumerate(texts):
        voice = voices[index % len(voices)]
        canonical.append({
            **row,
            "job_id": f"{phase[:3]}_{row['utterance_id']}",
            "phase": phase,
            "realization": "canonical",
            "voice": voice["name"],
            "voice_gender": voice.get("gender"),
            "style": _style_for(row, False),
            "required_rare_syllables": row.get("rare_syllables", []),
        })
    if phase == "full":
        supports = list(_read_jsonl(run_dir / "manifests" / "rare_extra_support.jsonl"))
        canonical_voice = {job["utterance_id"]: job["voice"] for job in canonical}
        used_by_utterance: dict[str, set[str]] = defaultdict(set)
        used_by_token: dict[str, set[str]] = defaultdict(set)
        for job in canonical:
            for token in set(job.get("rare_syllables", [])):
                used_by_token[token].add(job["voice"])
        extra: list[dict[str, Any]] = []
        for ordinal, row in enumerate(supports):
            used = used_by_utterance[row["utterance_id"]] | {canonical_voice[row["utterance_id"]]}
            required_tokens = set(row.get("required_rare_syllables", []))
            token_used = set().union(*(used_by_token[token] for token in required_tokens)) if required_tokens else set()
            voice = next((candidate for candidate in voices if candidate["name"] not in used | token_used), None)
            if voice is None:
                voice = min(
                    (candidate for candidate in voices if candidate["name"] not in used),
                    key=lambda candidate: (
                        sum(candidate["name"] in used_by_token[token] for token in required_tokens),
                        candidate["name"],
                    ),
                    default=None,
                )
            if voice is None:
                raise RuntimeError(f"Not enough distinct qualified voices for {row['utterance_id']}")
            used_by_utterance[row["utterance_id"]].add(voice["name"])
            for token in required_tokens:
                used_by_token[token].add(voice["name"])
            extra.append({
                **row,
                "job_id": f"rare_{ordinal:06d}_{row['utterance_id']}",
                "phase": phase,
                "realization": "rare_extra",
                "voice": voice["name"],
                "voice_gender": voice.get("gender"),
                "style": "neutral",
            })
        canonical.extend(extra)
        regeneration_path = run_dir / "manifests" / "regeneration_requests.jsonl"
        if regeneration_path.exists():
            by_utterance = {row["utterance_id"]: row for row in records}
            canonical_index = {job["utterance_id"]: index for index, job in enumerate(canonical) if job["realization"] == "canonical"}
            for request in _read_jsonl(regeneration_path):
                row = by_utterance[request["utterance_id"]]
                attempt = int(request["attempt"])
                base_index = canonical_index[row["utterance_id"]]
                voice = voices[(base_index + attempt) % len(voices)]
                canonical.append({
                    **row,
                    "job_id": f"regen{attempt}_{row['utterance_id']}",
                    "phase": "full",
                    "realization": f"regeneration_{attempt}",
                    "voice": voice["name"],
                    "voice_gender": voice.get("gender"),
                    "style": _style_for(row, bool(row.get("rare_syllables"))),
                    "required_rare_syllables": row.get("rare_syllables", []),
                    "regenerates_job_id": request.get("failed_job_id"),
                })
    return canonical


def _init_job_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30, check_same_thread=False)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS synthesis_jobs (
            job_id TEXT PRIMARY KEY,
            phase TEXT NOT NULL,
            status TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            voice TEXT NOT NULL,
            master_path TEXT,
            training_path TEXT,
            audio_sha256 TEXT,
            duration_seconds REAL,
            input_tokens_estimate INTEGER,
            audio_tokens INTEGER,
            estimated_cost_usd REAL,
            error TEXT,
            updated_at TEXT NOT NULL
        )
        """
    )
    connection.commit()
    return connection


def _legal_job_manifest(run_dir: Path, phase: str, jobs: Sequence[Mapping[str, Any]]) -> Path:
    path = run_dir / "manifests" / f"synthesis_{phase}_jobs.jsonl"
    if path.exists():
        previous = list(_read_jsonl(path))
        previous_ids = [row["job_id"] for row in previous]
        current_ids = [row["job_id"] for row in jobs]
        if previous_ids == current_ids:
            return path
        if previous_ids != current_ids[: len(previous_ids)]:
            raise RuntimeError(f"Existing {phase} job manifest does not match deterministic reconstruction")
    _write_jsonl(path, jobs)
    return path


def _forecast_cost(jobs: Sequence[Mapping[str, Any]], config: Mapping[str, Any], seconds_per_syllable: float) -> dict[str, float]:
    budget = config["budget"]
    input_bytes = sum(len(str(job["tts_text"]).encode("utf-8")) for job in jobs)
    syllables = sum(len(job.get("syllables", [])) for job in jobs)
    input_tokens = math.ceil(input_bytes / 4)
    seconds = syllables * seconds_per_syllable
    audio_tokens = math.ceil(seconds * float(budget["audio_tokens_per_second"]))
    input_cost = input_tokens / 1_000_000 * float(budget["input_usd_per_million_tokens"])
    audio_cost = audio_tokens / 1_000_000 * float(budget["audio_usd_per_million_tokens"])
    return {
        "input_tokens": input_tokens,
        "audio_tokens": audio_tokens,
        "seconds": seconds,
        "input_usd": input_cost,
        "audio_usd": audio_cost,
        "total_usd": input_cost + audio_cost,
    }


def _existing_spend(connection: sqlite3.Connection) -> float:
    row = connection.execute("SELECT COALESCE(SUM(estimated_cost_usd), 0) FROM synthesis_jobs WHERE status='succeeded'").fetchone()
    return float(row[0])


class _RateLimiter:
    def __init__(self, requests_per_minute: int):
        self.limit = requests_per_minute
        self.events: deque[float] = deque()
        self.lock = threading.Lock()

    def wait(self) -> None:
        while True:
            with self.lock:
                now = time.monotonic()
                while self.events and now - self.events[0] >= 60:
                    self.events.popleft()
                if len(self.events) < self.limit:
                    self.events.append(now)
                    return
                delay = max(0.05, 60 - (now - self.events[0]))
            time.sleep(min(delay, 1.0))


def _convert_audio(master: Path, training: Path, rate: int) -> None:
    training.parent.mkdir(parents=True, exist_ok=True)
    temporary = training.with_suffix(".tmp.wav")
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(master), "-ac", "1", "-ar", str(rate), "-c:a", "pcm_s16le", str(temporary)],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg conversion failed: {result.stderr[-500:]}")
    os.replace(temporary, training)


def _google_synthesize(job: Mapping[str, Any], config: Mapping[str, Any], master: Path, limiter: _RateLimiter) -> None:
    from google.api_core import exceptions as google_exceptions  # type: ignore
    from google.cloud import texttospeech  # type: ignore

    tts = config["google_tts"]
    styles = tts.get("styles", {})
    retry_attempts = int(tts.get("retry_attempts", 5))
    retry_base = float(tts.get("retry_base_seconds", 2.0))
    last_error: Exception | None = None
    for attempt in range(retry_attempts):
        try:
            limiter.wait()
            client = texttospeech.TextToSpeechClient()
            synthesis_input = texttospeech.SynthesisInput(
                text=job["tts_text"], prompt=styles.get(job["style"], styles.get("neutral", ""))
            )
            voice = texttospeech.VoiceSelectionParams(
                language_code=tts["language_code"], name=job["voice"], model_name=tts["model"]
            )
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.LINEAR16,
                sample_rate_hertz=int(tts.get("master_sample_rate", 24000)),
            )
            response = client.synthesize_speech(
                request={"input": synthesis_input, "voice": voice, "audio_config": audio_config}, timeout=120
            )
            master.parent.mkdir(parents=True, exist_ok=True)
            temporary = master.with_suffix(".tmp.wav")
            temporary.write_bytes(response.audio_content)
            os.replace(temporary, master)
            return
        except (google_exceptions.TooManyRequests, google_exceptions.ServiceUnavailable, google_exceptions.InternalServerError) as error:
            last_error = error
            time.sleep(retry_base * (2**attempt) + random.Random(f"{job['job_id']}:{attempt}").random())
    raise RuntimeError(f"TTS failed after {retry_attempts} attempts: {last_error}")


def synthesize_synthetic_asr(
    run_dir: str | Path,
    phase: str,
    max_usd: float,
    workers: int | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Synthesize one resumable phase using only an already configured TTS API."""
    root = Path(run_dir).expanduser().resolve()
    if phase not in {"audition", "pilot", "full"}:
        raise ValueError("phase must be audition, pilot, or full")
    config, config_path = _load_run_config(root)
    _load_tokenizer(config, config_path)
    _require_legal_review(root, config)
    if not (root / "prepare.complete.json").exists():
        raise RuntimeError("Run prepare-synthetic-asr before synthesis")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("FFmpeg is required on the existing VM")
    voices = _qualified_voices(root, config, phase)
    jobs = _phase_jobs(root, config, phase, voices)
    manifest_path = _legal_job_manifest(root, phase, jobs)
    connection = _init_job_db(root / "state" / "synthesis.sqlite3")
    for job in jobs:
        connection.execute(
            "INSERT OR IGNORE INTO synthesis_jobs(job_id,phase,status,voice,updated_at) VALUES(?,?,?,?,?)",
            (job["job_id"], phase, "pending", job["voice"], _utc_now()),
        )
    connection.commit()
    if not resume:
        succeeded = connection.execute("SELECT COUNT(*) FROM synthesis_jobs WHERE phase=? AND status='succeeded'", (phase,)).fetchone()[0]
        if succeeded:
            raise RuntimeError(f"{succeeded} completed {phase} jobs already exist; pass --resume")

    calibration_path = root / "reports" / "duration_calibration.json"
    if calibration_path.exists():
        calibration = json.loads(calibration_path.read_text())
        seconds_per_syllable = float(
            calibration.get("conservative_seconds_per_syllable", calibration["seconds_per_syllable"])
        )
    else:
        seconds_per_syllable = float(config["budget"].get("conservative_seconds_per_syllable", 0.35))
    pending_ids = {
        row[0] for row in connection.execute("SELECT job_id FROM synthesis_jobs WHERE phase=? AND status!='succeeded'", (phase,))
    }
    pending = [job for job in jobs if job["job_id"] in pending_ids]
    forecast = _forecast_cost(pending, config, seconds_per_syllable)
    spent = _existing_spend(connection)
    tts_cap = min(float(max_usd) - float(config["budget"].get("reserve_usd", 5)), float(config["budget"].get("tts_usd", 85)))
    if spent + forecast["total_usd"] > tts_cap:
        raise RuntimeError(
            f"Budget guard blocked {phase}: tracked spend ${spent:.2f} + forecast ${forecast['total_usd']:.2f} "
            f"exceeds TTS allowance ${tts_cap:.2f}."
        )
    _atomic_json(root / "reports" / f"{phase}_cost_forecast.json", {
        "created_at": _utc_now(), "phase": phase, "pending_jobs": len(pending),
        "tracked_spend_usd": spent, "allowance_usd": tts_cap, **forecast,
    })

    limiter = _RateLimiter(int(config["google_tts"].get("requests_per_minute", 120)))
    database_lock = threading.Lock()
    budget_lock = threading.Lock()
    stop_for_budget = threading.Event()
    actual_phase_spend = 0.0
    errors: list[dict[str, str]] = []

    def run_job(job: Mapping[str, Any]) -> None:
        nonlocal actual_phase_spend
        master = root / "audio" / "master_24k" / phase / f"{job['job_id']}.wav"
        training = root / "audio" / "train_16k" / phase / f"{job['job_id']}.wav"
        try:
            if stop_for_budget.is_set():
                raise RuntimeError("Budget guard stopped new TTS requests after observed spend reached the allowance")
            if not (resume and master.exists() and training.exists()):
                _google_synthesize(job, config, master, limiter)
                _convert_audio(master, training, int(config["google_tts"].get("training_sample_rate", 16000)))
            frames, rate, duration = _audio_info(training)
            del frames, rate
            budget = config["budget"]
            input_tokens = math.ceil(len(str(job["tts_text"]).encode("utf-8")) / 4)
            audio_tokens = math.ceil(duration * float(budget["audio_tokens_per_second"]))
            cost = input_tokens / 1_000_000 * float(budget["input_usd_per_million_tokens"]) + audio_tokens / 1_000_000 * float(budget["audio_usd_per_million_tokens"])
            with budget_lock:
                actual_phase_spend += cost
                if spent + actual_phase_spend >= tts_cap:
                    stop_for_budget.set()
            with database_lock:
                connection.execute(
                    """UPDATE synthesis_jobs SET status='succeeded',attempts=attempts+1,master_path=?,training_path=?,
                    audio_sha256=?,duration_seconds=?,input_tokens_estimate=?,audio_tokens=?,estimated_cost_usd=?,error=NULL,updated_at=? WHERE job_id=?""",
                    (str(master), str(training), _sha256(training), duration, input_tokens, audio_tokens, cost, _utc_now(), job["job_id"]),
                )
                connection.commit()
        except Exception as error:  # noqa: BLE001 - one failed network/audio job must be checkpointed
            with database_lock:
                connection.execute(
                    "UPDATE synthesis_jobs SET status='failed',attempts=attempts+1,error=?,updated_at=? WHERE job_id=?",
                    (str(error)[:1000], _utc_now(), job["job_id"]),
                )
                connection.commit()
                errors.append({"job_id": str(job["job_id"]), "error": str(error)})

    concurrency = min(max(1, workers or os.cpu_count() or 1), 16)
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        list(executor.map(run_job, pending))
    succeeded_rows = connection.execute(
        "SELECT job_id,duration_seconds,estimated_cost_usd FROM synthesis_jobs WHERE phase=? AND status='succeeded'", (phase,)
    ).fetchall()
    connection.close()
    if succeeded_rows:
        succeeded_by_id = {row[0]: float(row[1]) for row in succeeded_rows}
        total_syllables = sum(
            len(job.get("syllables", [])) for job in jobs if job["job_id"] in succeeded_by_id
        )
        if total_syllables:
            ratios = [
                succeeded_by_id[job["job_id"]] / max(1, len(job.get("syllables", [])))
                for job in jobs if job["job_id"] in succeeded_by_id
            ]
            mean_ratio = sum(ratios) / len(ratios)
            p95_ratio = _quantile(ratios, 0.95) or mean_ratio
            _atomic_json(calibration_path, {
                "updated_at": _utc_now(), "source_phase": phase,
                "seconds_per_syllable": mean_ratio,
                "p95_seconds_per_syllable": p95_ratio,
                "conservative_seconds_per_syllable": max(
                    float(config["budget"].get("conservative_seconds_per_syllable", 0.35)),
                    p95_ratio * 1.20,
                ),
            })
    summary = {
        "phase": phase,
        "job_manifest": str(manifest_path),
        "jobs": len(jobs),
        "succeeded": len(succeeded_rows),
        "failed": len(errors),
        "duration_hours": sum(float(row[1]) for row in succeeded_rows) / 3600,
        "estimated_tts_cost_usd": sum(float(row[2]) for row in succeeded_rows),
        "errors": errors[:100],
    }
    _atomic_json(root / "reports" / f"synthesis_{phase}_summary.json", summary)
    if errors:
        raise RuntimeError(f"{len(errors)} {phase} synthesis jobs failed; rerun with --resume")
    return summary


def _wav_metrics(path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np

    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frames = handle.getnframes()
        raw = handle.readframes(frames)
    if sample_width != 2:
        return {"readable": True, "channels": channels, "sample_width": sample_width, "sample_rate": sample_rate, "frames": frames, "structural_pass": False, "failure": "not_pcm16"}
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    absolute = np.abs(samples)
    duration = len(samples) / sample_rate if sample_rate else 0.0
    peak = float(absolute.max()) if len(absolute) else 0.0
    rms = float(np.sqrt(np.mean(samples * samples))) if len(samples) else 0.0
    clip_level = float(config["qc"].get("clipping_amplitude", 0.999))
    clipping_ratio = float(np.mean(absolute >= clip_level)) if len(absolute) else 0.0
    threshold = max(rms * 0.1, 10 ** (-50 / 20))
    active = np.flatnonzero(absolute >= threshold)
    if len(active):
        leading = float(active[0] / sample_rate)
        trailing = float((len(samples) - 1 - active[-1]) / sample_rate)
        longest_silence = 0
        last_active = int(active[0])
        for index in active[1:]:
            longest_silence = max(longest_silence, int(index) - last_active - 1)
            last_active = int(index)
        internal_silence = longest_silence / sample_rate
    else:
        leading = trailing = internal_silence = duration
    structural_pass = (
        channels == 1 and sample_width == 2 and sample_rate == int(config["google_tts"].get("training_sample_rate", 16000))
        and float(config["qc"].get("min_duration_seconds", 0.2)) <= duration <= float(config["qc"].get("max_duration_seconds", 655.0))
        and rms > 0 and clipping_ratio <= float(config["qc"].get("max_clipping_ratio", 0.001))
    )
    return {
        "readable": True, "channels": channels, "sample_width": sample_width,
        "sample_rate": sample_rate, "frames": frames, "duration_seconds": duration,
        "peak": peak, "rms": rms, "clipping_ratio": clipping_ratio,
        "leading_silence_seconds": leading, "trailing_silence_seconds": trailing,
        "max_internal_silence_seconds": internal_silence, "structural_pass": structural_pass,
    }


def _quantile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return float(ordered[position])


def validate_synthetic_audio(
    run_dir: str | Path,
    stage: str = "cpu",
    workers: int | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    if stage != "cpu":
        raise ValueError("Only stage=cpu is supported on the existing VM")
    root = Path(run_dir).expanduser().resolve()
    config, _config_path = _load_run_config(root)
    database = root / "state" / "synthesis.sqlite3"
    if not database.exists():
        raise RuntimeError("No synthesized jobs found")
    connection = sqlite3.connect(database)
    jobs = connection.execute(
        "SELECT job_id,phase,training_path,master_path,audio_sha256,duration_seconds FROM synthesis_jobs WHERE status='succeeded' ORDER BY job_id"
    ).fetchall()
    connection.close()
    if not jobs:
        raise RuntimeError("No successful synthesis jobs are available for CPU QC")
    existing: dict[str, dict[str, Any]] = {}
    metrics_path = root / "qc" / "cpu_metrics.jsonl"
    if resume and metrics_path.exists():
        existing = {row["job_id"]: row for row in _read_jsonl(metrics_path)}

    def inspect(row: tuple[Any, ...]) -> dict[str, Any]:
        job_id, phase, training_path, master_path, stored_hash, stored_duration = row
        path = Path(training_path)
        if job_id in existing and path.exists() and existing[job_id].get("audio_sha256") == _sha256(path):
            return existing[job_id]
        base = {"job_id": job_id, "phase": phase, "training_path": training_path, "master_path": master_path}
        try:
            metrics = _wav_metrics(path, config)
            master_frames, master_rate, master_duration = _audio_info(Path(master_path))
            del master_frames, master_rate
            duration_delta = abs(float(metrics.get("duration_seconds", 0)) - master_duration)
            metrics["conversion_duration_delta_seconds"] = duration_delta
            metrics["conversion_integrity"] = duration_delta <= 0.05
            metrics["audio_sha256"] = _sha256(path)
            metrics["stored_hash_match"] = metrics["audio_sha256"] == stored_hash
            metrics["stored_duration_delta_seconds"] = abs(float(metrics.get("duration_seconds", 0)) - float(stored_duration))
            metrics["cpu_pass"] = bool(metrics["structural_pass"] and metrics["conversion_integrity"] and metrics["stored_hash_match"])
            return {**base, **metrics}
        except Exception as error:  # noqa: BLE001 - malformed audio is a QC result, not a pipeline crash
            return {**base, "readable": False, "structural_pass": False, "cpu_pass": False, "failure": str(error)[:500]}

    max_workers = max(1, workers or os.cpu_count() or 1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        metrics = list(executor.map(inspect, jobs))
    hashes = Counter(row.get("audio_sha256") for row in metrics if row.get("audio_sha256"))
    duplicate_hashes = {value for value, count in hashes.items() if count > 1}
    for row in metrics:
        row["duplicate_audio_hash"] = row.get("audio_sha256") in duplicate_hashes
        if row["duplicate_audio_hash"]:
            row["cpu_pass"] = False
    _write_jsonl(metrics_path, metrics)

    audition = [row for row in metrics if row["phase"] == "audition" and row.get("cpu_pass")]
    calibration = {
        "source": "audition",
        "records": len(audition),
        "leading_silence_p99": _quantile([row["leading_silence_seconds"] for row in audition], 0.99),
        "trailing_silence_p99": _quantile([row["trailing_silence_seconds"] for row in audition], 0.99),
        "internal_silence_p99": _quantile([row["max_internal_silence_seconds"] for row in audition], 0.99),
        "rms_p01": _quantile([row["rms"] for row in audition], 0.01),
        "rms_p99": _quantile([row["rms"] for row in audition], 0.99),
    }
    _atomic_json(root / "qc" / "cpu_threshold_calibration.json", calibration)
    summary = {
        "created_at": _utc_now(), "records": len(metrics),
        "passed": sum(bool(row.get("cpu_pass")) for row in metrics),
        "failed": sum(not bool(row.get("cpu_pass")) for row in metrics),
        "duplicate_audio_hashes": len(duplicate_hashes),
        "calibration": calibration,
        "metrics_path": str(metrics_path),
    }
    _atomic_json(root / "reports" / "cpu_qc_summary.json", summary)
    return summary


def _load_job_records(root: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in sorted((root / "manifests").glob("synthesis_*_jobs.jsonl")):
        for row in _read_jsonl(path):
            if row["job_id"] in records:
                raise RuntimeError(f"Duplicate synthesis job ID across manifests: {row['job_id']}")
            records[row["job_id"]] = row
    return records


def export_kaggle_qc(
    run_dir: str | Path,
    output_dir: str | Path,
    shard_size_gb: float = 4.0,
    workers: int | None = None,
) -> dict[str, Any]:
    """Package CPU-passed audio for a private Kaggle dataset; does not upload it."""
    root = Path(run_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    config, config_path = _load_run_config(root)
    _require_legal_review(root, config)
    metrics_path = root / "qc" / "cpu_metrics.jsonl"
    if not metrics_path.exists():
        raise RuntimeError("Run CPU validation before Kaggle export")
    records = _load_job_records(root)
    database = sqlite3.connect(root / "state" / "synthesis.sqlite3")
    synthesis = {
        row[0]: {"training_path": row[1], "audio_sha256": row[2]}
        for row in database.execute("SELECT job_id,training_path,audio_sha256 FROM synthesis_jobs WHERE status='succeeded'")
    }
    database.close()
    prior_whisper_path = root / "qc" / "whisper_results.jsonl"
    prior_mms_path = root / "qc" / "mms_results.jsonl"
    prior_whisper = {
        row["job_id"]: row for row in _read_jsonl(prior_whisper_path)
    } if prior_whisper_path.exists() else {}
    prior_mms = {
        row["job_id"]: row for row in _read_jsonl(prior_mms_path)
    } if prior_mms_path.exists() else {}
    max_cer = float(config["qc"].get("max_cer", 0.35))

    def qc_complete(job_id: str) -> bool:
        whisper_row = prior_whisper.get(job_id)
        if not whisper_row:
            return False
        requires_mms = (
            bool(records[job_id].get("required_rare_syllables"))
            or float(whisper_row.get("cer", 1.0)) > max_cer
            or abs(float(whisper_row.get("cer", 1.0)) - max_cer) <= 0.05
            or float(whisper_row.get("rare_syllable_recall", 1.0)) < 1.0
        )
        return not requires_mms or job_id in prior_mms

    synthesized_ready = [
        job_id for job_id in sorted(records)
        if job_id in synthesis
        and Path(synthesis[job_id]["training_path"]).is_file()
    ]
    already_qc_complete = [job_id for job_id in synthesized_ready if qc_complete(job_id)]
    eligible = [job_id for job_id in synthesized_ready if not qc_complete(job_id)]
    if not eligible:
        raise RuntimeError("No synthesized recordings are pending Kaggle QC")
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Kaggle output directory is not empty; use a fresh directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    max_bytes = int(shard_size_gb * 1024**3)
    shards: list[list[str]] = []
    current: list[str] = []
    current_size = 0
    for job_id in eligible:
        size = Path(synthesis[job_id]["training_path"]).stat().st_size
        if current and current_size + size > max_bytes:
            shards.append(current)
            current, current_size = [], 0
        current.append(job_id)
        current_size += size
    if current:
        shards.append(current)
    # Reserve ten top-level entries for manifests, code, config, tokenizer, and notebook.
    if len(shards) > 40:
        raise RuntimeError(
            f"Kaggle export needs {len(shards)} archives; increase --shard-size-gb "
            "to keep the complete private dataset at or below 50 top-level files"
        )

    def write_shard(spec: tuple[int, list[str]]) -> tuple[str, str, list[dict[str, Any]]]:
        shard_index, job_ids = spec
        shard_name = f"audio_{shard_index:03d}.tar.gz"
        shard_path = output / shard_name
        rows_for_shard: list[dict[str, Any]] = []
        with tarfile.open(shard_path, "w:gz") as archive:
            for job_id in job_ids:
                audio_path = Path(synthesis[job_id]["training_path"])
                arcname = f"audio/{job_id}.wav"
                archive.add(audio_path, arcname=arcname, recursive=False)
                row = records[job_id]
                rows_for_shard.append({
                    "job_id": job_id, "utterance_id": row["utterance_id"], "phase": row["phase"],
                    "realization": row["realization"], "voice": row["voice"],
                    "reference_text": row["asr_text"], "syllables": row["syllables"],
                    "rare_syllables": row.get("required_rare_syllables", row.get("rare_syllables", [])),
                    "audio_sha256": synthesis[job_id]["audio_sha256"], "archive": shard_name,
                    "archive_member": arcname,
                })
        return shard_name, _sha256(shard_path), rows_for_shard

    max_workers = min(max(1, workers or os.cpu_count() or 1), max(1, len(shards)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        shard_results = list(executor.map(write_shard, enumerate(shards)))
    manifest_rows = [row for _name, _checksum, rows in shard_results for row in rows]
    checksums = {name: checksum for name, checksum, _rows in shard_results}
    manifest_path = output / "qc_input_manifest.jsonl"
    _write_jsonl(manifest_path, manifest_rows)
    checksums[manifest_path.name] = _sha256(manifest_path)
    shutil.copy2(_PROJECT_ROOT / "notebooks" / "kaggle_synthetic_asr_qc.ipynb", output / "kaggle_synthetic_asr_qc.ipynb")
    shutil.copy2(_SCRIPT_ROOT / "kaggle_qc_runner.py", output / "kaggle_qc_runner.py")
    shutil.copy2(config_path, output / "synthetic_asr_config.yaml")
    _tokenizer, _lookup, vocab_path, tokenizer_path = _load_tokenizer(config, config_path)
    shutil.copy2(vocab_path, output / "nepali_syllables_lookup.vocab")
    shutil.copy2(tokenizer_path, output / "syllabic_tokenizer.py")
    dataset_metadata = {
        "title": f"Private Nepali synthetic ASR QC {root.name}",
        "id": "REPLACE_WITH_KAGGLE_USERNAME/nepali-synthetic-asr-qc",
        "licenses": [{"name": "other"}],
        "isPrivate": True,
    }
    _atomic_json(output / "dataset-metadata.json", dataset_metadata)
    export_manifest = {
        "created_at": _utc_now(), "private_dataset_required": True,
        "records": len(manifest_rows), "archives": len(shards), "checksums": checksums,
        "already_qc_complete_skipped": len(already_qc_complete),
        "models": {key: config["qc"][key] for key in (
            "whisper_model", "whisper_revision", "mms_model", "mms_revision", "mms_language_adapter",
            "speaker_embedding_model", "speaker_embedding_revision",
        )},
        "no_google_credentials_in_package": True,
    }
    package_bytes = sum(path.stat().st_size for path in output.iterdir() if path.is_file())
    export_manifest["package_bytes"] = package_bytes
    export_manifest["top_level_files"] = sum(path.is_file() for path in output.iterdir())
    if package_bytes > 200 * 1024**3:
        raise RuntimeError(f"Kaggle package is {package_bytes / 1024**3:.1f} GiB, above the 200 GiB limit")
    _atomic_json(output / "kaggle_export_manifest.json", export_manifest)
    _atomic_json(root / "reports" / "kaggle_export_manifest.json", export_manifest)
    return export_manifest


def _read_result_table(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".parquet":
        try:
            from pyarrow import parquet  # type: ignore
        except ImportError as error:
            raise RuntimeError("pyarrow is required to import Kaggle Parquet results") from error
        return parquet.read_table(path).to_pylist()
    return list(_read_jsonl(path))


def _discover_result_rows(results_dir: Path, stem: str) -> list[dict[str, Any]]:
    paths = sorted(results_dir.rglob(f"{stem}.parquet"))
    if not paths:
        paths = sorted(results_dir.rglob(f"{stem}.jsonl"))
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(_read_result_table(path))
    return rows


def _unique_results(rows: Sequence[Mapping[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for raw in rows:
        row = dict(raw)
        job_id = str(row.get("job_id", ""))
        if not job_id:
            raise RuntimeError(f"{label} result is missing job_id")
        if job_id in output:
            duplicates.append(job_id)
        output[job_id] = row
    if duplicates:
        raise RuntimeError(f"Duplicated {label} results: {sorted(set(duplicates))[:10]}")
    return output


def _voice_qualification(
    run_dir: Path,
    config: Mapping[str, Any],
    records: Mapping[str, Mapping[str, Any]],
    whisper: Mapping[str, Mapping[str, Any]],
    mms: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any] | None:
    voice_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for job_id, result in whisper.items():
        record = records.get(job_id)
        if record and record.get("phase") == "audition":
            voice_rows[str(record["voice"])].append(dict(result))
    configured = _configured_voices(config)
    if any(voice["name"] not in voice_rows for voice in configured):
        return None
    if any(not all(row.get("speaker_embedding") for row in voice_rows[voice["name"]]) for voice in configured):
        return None
    centroids: dict[str, list[float]] = {}
    for voice in configured:
        embeddings = [row["speaker_embedding"] for row in voice_rows[voice["name"]]]
        centroid = [sum(values) / len(values) for values in zip(*embeddings)]
        norm = math.sqrt(sum(value * value for value in centroid)) or 1.0
        centroids[voice["name"]] = [value / norm for value in centroid]

    def separation(name: str) -> float:
        similarities = []
        for other, centroid in centroids.items():
            if other != name:
                similarities.append(sum(left * right for left, right in zip(centroids[name], centroid)))
        return 1.0 - max(similarities) if similarities else 0.0

    separations = {voice["name"]: separation(voice["name"]) for voice in configured}
    sep_values = list(separations.values())
    sep_low, sep_high = min(sep_values), max(sep_values)
    scored: list[dict[str, Any]] = []
    for voice in configured:
        rows = voice_rows[voice["name"]]
        cers = sorted(float(row.get("cer", 1.0)) for row in rows)
        rare_recalls = [
            float(mms[job_id].get("rare_syllable_recall", whisper[job_id].get("rare_syllable_recall", 0.0)))
            for job_id in (row["job_id"] for row in rows) if job_id in whisper
        ]
        cpu = [row for row in _read_jsonl(run_dir / "qc" / "cpu_metrics.jsonl") if row["job_id"] in {item["job_id"] for item in rows}]
        normalized_separation = (separations[voice["name"]] - sep_low) / max(1e-12, sep_high - sep_low)
        score = (
            (1 - (cers[len(cers) // 2] if cers else 1.0)) * 0.55
            + (sum(rare_recalls) / len(rare_recalls) if rare_recalls else 0.0) * 0.25
            + (sum(bool(row.get("cpu_pass")) for row in cpu) / len(cpu) if cpu else 0.0) * 0.10
            + normalized_separation * 0.10
        )
        scored.append({
            "voice": voice["name"], "gender": voice.get("gender"), "score": score,
            "median_cer": cers[len(cers) // 2] if cers else None,
            "mean_rare_syllable_recall": sum(rare_recalls) / len(rare_recalls) if rare_recalls else None,
            "speaker_embedding_separation": separations[voice["name"]],
            "records": len(rows),
        })
    selected: list[str] = []
    for gender, count in (("female", int(config["google_tts"].get("selected_female", 4))), ("male", int(config["google_tts"].get("selected_male", 4)))):
        ranked = sorted((row for row in scored if row["gender"] == gender), key=lambda row: (-row["score"], row["voice"]))
        if len(ranked) < count:
            raise RuntimeError(f"Not enough qualified {gender} audition voices")
        selected.extend(row["voice"] for row in ranked[:count])
    return {
        "created_at": _utc_now(),
        "method": "audition_whisper_cer_rare_recall_cpu_integrity_wavlm_separation",
        "speaker_embedding_model": config["qc"]["speaker_embedding_model"],
        "selected_voices": selected,
        "scores": scored,
    }


def import_kaggle_qc(
    run_dir: str | Path,
    results_dir: str | Path,
    resume: bool = False,
) -> dict[str, Any]:
    """Validate and idempotently import compact Kaggle QC outputs."""
    root = Path(run_dir).expanduser().resolve()
    source = Path(results_dir).expanduser().resolve()
    config, _config_path = _load_run_config(root)
    records = _load_job_records(root)
    cpu = {row["job_id"]: row for row in _read_jsonl(root / "qc" / "cpu_metrics.jsonl")}
    export_manifest_path = root / "reports" / "kaggle_export_manifest.json"
    if not export_manifest_path.exists():
        raise RuntimeError("The run has no recorded Kaggle export manifest")
    export_manifest = json.loads(export_manifest_path.read_text(encoding="utf-8"))
    expected_input_manifest = export_manifest["checksums"]["qc_input_manifest.jsonl"]
    result_manifests = sorted(source.rglob("kaggle_manifest.json"))
    if not result_manifests:
        raise RuntimeError(f"No kaggle_manifest.json found under {source}")
    for manifest_path in result_manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("input_manifest_sha256") != expected_input_manifest:
            raise RuntimeError(f"Kaggle input-manifest checksum mismatch: {manifest_path}")
        for filename, expected_hash in manifest.get("outputs", {}).items():
            output_path = manifest_path.parent / filename
            if not output_path.is_file() or _sha256(output_path) != expected_hash:
                raise RuntimeError(f"Kaggle result checksum mismatch: {output_path}")
    whisper_new = _unique_results(_discover_result_rows(source, "whisper_results"), "Whisper")
    mms_new = _unique_results(_discover_result_rows(source, "mms_results"), "MMS")
    if not whisper_new:
        raise RuntimeError(f"No whisper_results.parquet or whisper_results.jsonl found under {source}")
    whisper_path = root / "qc" / "whisper_results.jsonl"
    mms_path = root / "qc" / "mms_results.jsonl"
    whisper = {row["job_id"]: row for row in _read_jsonl(whisper_path)} if resume and whisper_path.exists() else {}
    mms = {row["job_id"]: row for row in _read_jsonl(mms_path)} if resume and mms_path.exists() else {}
    for label, incoming, accumulated in (("Whisper", whisper_new, whisper), ("MMS", mms_new, mms)):
        for job_id, row in incoming.items():
            if job_id not in records or job_id not in cpu:
                raise RuntimeError(f"Unknown {label} job_id: {job_id}")
            expected_hash = cpu[job_id].get("audio_sha256")
            if row.get("audio_sha256") != expected_hash:
                raise RuntimeError(f"{label} audio checksum mismatch for {job_id}")
            existing = accumulated.get(job_id)
            if existing and existing != row:
                raise RuntimeError(f"Conflicting resumed {label} result for {job_id}")
            accumulated[job_id] = row
    expected_whisper_model = config["qc"]["whisper_model"]
    expected_whisper_revision = config["qc"]["whisper_revision"]
    if any(
        row.get("model") != expected_whisper_model or row.get("revision") != expected_whisper_revision
        for row in whisper_new.values()
    ):
        raise RuntimeError(f"Whisper result model/revision must be {expected_whisper_model}/{expected_whisper_revision}")
    for job_id, row in whisper_new.items():
        if records[job_id].get("phase") == "audition" and (
            row.get("speaker_embedding_model") != config["qc"]["speaker_embedding_model"]
            or row.get("speaker_embedding_revision") != config["qc"]["speaker_embedding_revision"]
        ):
            raise RuntimeError(f"Audition speaker-embedding model/revision mismatch for {job_id}")
    expected_mms_model = config["qc"]["mms_model"]
    if any(
        row.get("model") != expected_mms_model
        or row.get("revision") != config["qc"]["mms_revision"]
        or row.get("language_adapter") != config["qc"]["mms_language_adapter"]
        for row in mms_new.values()
    ):
        raise RuntimeError(f"MMS result model/adapter must be {expected_mms_model}/{config['qc']['mms_language_adapter']}")
    _write_jsonl(whisper_path, (whisper[key] for key in sorted(whisper)))
    _write_jsonl(mms_path, (mms[key] for key in sorted(mms)))

    max_cer = float(config["qc"].get("max_cer", 0.35))
    required_mms = {
        job_id for job_id, row in whisper.items()
        if records[job_id].get("required_rare_syllables")
        or float(row.get("cer", 1.0)) > max_cer
        or abs(float(row.get("cer", 1.0)) - max_cer) <= 0.05
        or float(row.get("rare_syllable_recall", 1.0)) < 1.0
    }
    missing_mms = sorted(required_mms.difference(mms))
    qualification = _voice_qualification(root, config, records, whisper, mms)
    if qualification:
        _atomic_json(root / "qc" / "voice_qualification.json", qualification)
    phase_gates: dict[str, Any] = {}
    for phase in ("audition", "pilot", "full"):
        phase_ids = {job_id for job_id, record in records.items() if record.get("phase") == phase}
        if not phase_ids:
            continue
        complete = phase_ids.issubset(whisper) and phase_ids.issubset(cpu)
        passed = 0
        for job_id in phase_ids:
            whisper_row, cpu_row = whisper.get(job_id), cpu.get(job_id)
            if not whisper_row or not cpu_row or not cpu_row.get("cpu_pass"):
                continue
            required_rare = records[job_id].get("required_rare_syllables", [])
            accurate = float(whisper_row.get("cer", 1.0)) <= max_cer and float(whisper_row.get("wer", 1.0)) <= float(config["qc"].get("max_wer", 0.60))
            rare_ok = not required_rare or (
                job_id in mms and float(mms[job_id].get("rare_syllable_recall", 0.0)) >= 1.0
                and float(whisper_row.get("rare_syllable_recall", 0.0)) >= 1.0
            )
            passed += bool(accurate and rare_ok)
        rate = passed / len(phase_ids)
        minimum = float(config["qc"].get("min_pilot_acceptance_rate", 0.95)) if phase == "pilot" else 1.0
        gate = {
            "created_at": _utc_now(), "phase": phase, "records": len(phase_ids),
            "complete": complete, "passed_records": passed, "acceptance_rate": rate,
            "minimum_acceptance_rate": minimum, "passed": bool(complete and rate >= minimum and not missing_mms),
        }
        phase_gates[phase] = gate
        _atomic_json(root / "qc" / f"{phase}_qc_gate.json", gate)
    summary = {
        "imported_at": _utc_now(), "whisper_records": len(whisper), "mms_records": len(mms),
        "known_cpu_passed_records": sum(bool(row.get("cpu_pass")) for row in cpu.values()),
        "required_mms_records": len(required_mms), "missing_mms_records": missing_mms,
        "complete_whisper_for_current_cpu_set": set(cpu).issubset(whisper),
        "complete_required_mms": not missing_mms,
        "voice_qualification_created": qualification is not None,
        "phase_gates": phase_gates,
    }
    _atomic_json(root / "reports" / "kaggle_import_summary.json", summary)
    if missing_mms:
        raise RuntimeError(f"MMS results are missing for {len(missing_mms)} rare/disputed recordings")
    return summary


def _write_speechain_vocab(root: Path, config: Mapping[str, Any], config_path: Path) -> Path:
    module, lookup, vocab_source, tokenizer_source = _load_tokenizer(config, config_path)
    token_root = root / "speechain" / "token" / "syllable"
    token_root.mkdir(parents=True, exist_ok=True)
    analytical = [
        token for token in vocab_source.read_text(encoding="utf-8").splitlines()
        if token.strip()
    ]
    selectable = [token for token in analytical if token not in {"।", "॥", "?", "!"}]
    unemittable = [token for token in selectable if module.tokenize(token, lookup) != [token]]
    if unemittable:
        raise RuntimeError(f"SpeeChain syllable round-trip preflight failed for {unemittable[:10]}")
    # SpeeChain's GPU CTC path requires the blank index to be zero.
    tokens = ["<blank>", "<sos/eos>", "<unk>", "<space>"] + selectable
    vocab_path = token_root / "vocab"
    vocab_path.write_text("\n".join(tokens) + "\n", encoding="utf-8")
    shutil.copy2(vocab_source, token_root / "lookup.vocab")
    shutil.copy2(tokenizer_source, token_root / "syllabic_tokenizer.py")
    _atomic_json(token_root / "tokenizer_manifest.json", {
        "created_at": _utc_now(), "vocabulary_sha256": _sha256(vocab_source),
        "tokenizer_sha256": _sha256(tokenizer_source), "speechain_vocab_sha256": _sha256(vocab_path),
        "tokens": len(tokens), "blank_index": 0, "sos_eos_index": 1,
        "space_token": "<space>", "round_trip_verified": len(selectable),
    })
    return token_root


def _speechain_iterator(data: Sequence[Mapping[str, str]], *, training: bool) -> dict[str, Any]:
    feats = [item["idx2wav"] for item in data]
    texts = [item["idx2text"] for item in data]
    lengths = [item["idx2len"] for item in data]
    return {
        "type": "block.BlockIterator" if training else "abs.Iterator",
        "conf": {
            "dataset_type": "speech_text.SpeechTextDataset",
            "dataset_conf": {
                "main_data": {
                    "feat": feats if len(feats) > 1 else feats[0],
                    "text": texts if len(texts) > 1 else texts[0],
                },
                "use_speed_perturb": False,
            },
            "data_len": lengths if len(lengths) > 1 else lengths[0],
            "shuffle": training,
            "is_descending": True,
            **({"batch_len": 240000} if training else {}),
        },
    }


def _render_recipe(
    recipe_dir: Path,
    name: str,
    train_sets: Sequence[Mapping[str, str]],
    dev: Mapping[str, str],
    test: Mapping[str, str],
    token_root: Path,
    seed: int,
    init_model: str | None = None,
) -> tuple[Path, Path]:
    """Write data/train YAMLs accepted by SpeeChain's runner interface."""
    data_cfg = {
        "train": _speechain_iterator(train_sets, training=True),
        "valid": _speechain_iterator([dev], training=False),
        "test": {"slr54_test": _speechain_iterator([test], training=False)},
    }
    model_conf: dict[str, Any] = {
        "customize_conf": {
            "ctc_weight": 0.5,
            "token_type": "syllable",
            "token_path": str(token_root),
        }
    }
    if init_model:
        model_conf["pretrained_model"] = {"path": init_model}
    train_cfg = {
        "model": {
            "model_type": "ar_asr.ARASR",
            "model_conf": model_conf,
            "module_conf": {
                "frontend": {
                    "type": "frontend.speech2mel.Speech2MelSpec",
                    "conf": {
                        "sr": 16000, "preemphasis": 0.97, "hop_length": 0.010,
                        "win_length": 0.025, "n_mels": 80,
                    },
                },
                "normalize": {"norm_type": "utterance", "mean_norm": True, "std_norm": True},
                "specaug": True,
                "enc_prenet": {
                    "type": "prenet.conv2d.Conv2dPrenet",
                    "conf": {
                        "conv_dims": [256, 256], "conv_kernel": 3, "conv_stride": 2,
                        "conv_batchnorm": True, "conv_activation": "LeakyReLU", "lnr_dims": 256,
                    },
                },
                "encoder": {
                    "type": "conformer.encoder.ConformerEncoder",
                    "conf": {
                        "posenc_dropout": 0.2, "emb_scale": False, "d_model": 256,
                        "num_heads": 4, "num_layers": 8, "att_dropout": 0.2,
                        "fdfwd_dim": 1024, "fdfwd_activation": "GELU", "fdfwd_dropout": 0.2,
                        "res_dropout": 0.2, "layernorm_first": True,
                    },
                },
                "dec_emb": {"type": "prenet.embed.EmbedPrenet", "conf": {"embedding_dim": 256}},
                "decoder": {
                    "type": "transformer.decoder.TransformerDecoder",
                    "conf": {
                        "posenc_dropout": 0.2, "posenc_scale": False, "emb_layernorm": True,
                        "emb_scale": False, "d_model": 256, "num_heads": 4, "num_layers": 4,
                        "att_dropout": 0.2, "fdfwd_dim": 1024, "fdfwd_activation": "GELU",
                        "fdfwd_dropout": 0.2, "res_dropout": 0.2, "layernorm_first": True,
                    },
                },
            },
            "criterion_conf": {"ce_loss": {"label_smoothing": 0.05}, "ctc_loss": True},
        },
        "optim_sches": {
            "type": "noam.Noamlr",
            "conf": {
                "optim_type": "Adam",
                "optim_conf": {"lr": 0.002, "betas": [0.9, 0.98], "eps": 1.0e-9},
                "warmup_steps": 4000,
            },
        },
        "experiment_metadata": {
            "seed": seed,
            "training_execution": False,
            "required_metrics": ["cer", "wer", "syllable_error_rate", "rare_syllable_recall"],
            "shared_architecture": "conformer8_transformer4_d256_ctc05",
        },
    }
    recipe_dir.mkdir(parents=True, exist_ok=True)
    data_path = recipe_dir / f"{name}_data.yaml"
    train_path = recipe_dir / f"{name}_train.yaml"
    data_path.write_text(yaml.safe_dump(data_cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    train_path.write_text(yaml.safe_dump(train_cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return data_path, train_path


def _export_speechain(
    root: Path,
    accepted: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    config_path: Path,
) -> dict[str, Any]:
    synthetic = [row for row in accepted if not row.get("evaluation_overlap")]
    synthetic_dir = root / "speechain" / "synthetic" / "train"
    _write_idx(synthetic_dir / "idx2wav", ((row["job_id"], row["training_path"]) for row in synthetic))
    _write_idx(synthetic_dir / "idx2wav_len", ((row["job_id"], row["frames"]) for row in synthetic))
    _write_idx(synthetic_dir / "idx2no-punc_text", ((row["job_id"], row["asr_text"]) for row in synthetic))
    _write_idx(synthetic_dir / "idx2spk", ((row["job_id"], f"gemini_{row['voice']}") for row in synthetic))
    _write_idx(synthetic_dir / "idx2source", ((row["job_id"], json.dumps(row.get("source_corpora"), ensure_ascii=False)) for row in synthetic))
    _write_idx(synthetic_dir / "idx2voice", ((row["job_id"], row["voice"]) for row in synthetic))
    _write_idx(synthetic_dir / "idx2qc_status", ((row["job_id"], "accepted_automated_two_model") for row in synthetic))
    token_root = _write_speechain_vocab(root, config, config_path)
    integration_source = _PROJECT_ROOT / "speechain_integration"
    integration_target = root / "speechain" / "integration"
    shutil.copytree(integration_source, integration_target, dirs_exist_ok=True)
    evaluation_dir = root / "speechain" / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    prepare_summary = json.loads((root / "prepare.complete.json").read_text(encoding="utf-8"))
    (evaluation_dir / "rare_syllables.txt").write_text(
        "\n".join(sorted(prepare_summary.get("rare_frequency", {}))) + "\n", encoding="utf-8"
    )
    slr = json.loads((root / "manifests" / "slr54_reference.json").read_text(encoding="utf-8"))
    real = {split: {
        "idx2wav": slr["splits"][split]["paths"]["idx2wav"],
        "idx2len": slr["splits"][split]["paths"]["idx2wav_len"],
        "idx2text": slr["splits"][split]["paths"]["idx2no-punc_text"],
        "idx2spk": slr["splits"][split]["paths"]["idx2spk"],
    } for split in ("train", "dev", "test")}
    syn = {
        "idx2wav": str(synthetic_dir / "idx2wav"),
        "idx2len": str(synthetic_dir / "idx2wav_len"),
        "idx2text": str(synthetic_dir / "idx2no-punc_text"),
        "idx2spk": str(synthetic_dir / "idx2spk"),
    }
    recipe_dir = root / "speechain" / "recipes"
    recipe_pairs: list[tuple[str, Path, Path, int]] = []
    for seed in (42, 43, 44):
        for name, train_sets, checkpoint in (
            ("real_only", [real["train"]], None),
            ("real_plus_synthetic", [real["train"], syn], None),
            ("synthetic_pretrain", [syn], None),
            (
                "real_finetune",
                [real["train"]],
                f"/REPLACE_WITH_SYNTHETIC_PRETRAIN_CHECKPOINT/seed{seed}.pth",
            ),
        ):
            data_path, train_path = _render_recipe(
                recipe_dir, f"{name}_seed{seed}", train_sets, real["dev"], real["test"],
                token_root, seed, checkpoint,
            )
            recipe_pairs.append((name, data_path, train_path, seed))
    command_lines = [
        "# SpeeChain training commands (documentation only; not executed by this pipeline)",
        "",
        "Install the exported adapter first. Replace the fine-tuning checkpoint placeholder after synthetic pretraining.",
        "",
    ]
    for name, data_path, train_path, seed in recipe_pairs:
        command_lines.extend([
            f"## {name}, seed {seed}",
            "```bash",
            "python speechain/runner.py \\",
            f"  --train True --seed {seed} \\",
            f"  --data_cfg {data_path} \\",
            f"  --train_cfg {train_path} \\",
            f"  --train_result_path /REPLACE_WITH_RESULT_ROOT/{name}_seed{seed}",
            "```",
            "",
        ])
    (recipe_dir / "TRAINING_COMMANDS.md").write_text("\n".join(command_lines), encoding="utf-8")
    return {
        "synthetic_training_records": len(synthetic),
        "excluded_evaluation_overlap": len(accepted) - len(synthetic),
        "token_path": str(token_root), "recipe_yaml_files": len(recipe_pairs) * 2,
        "training_commands_documented": str(recipe_dir / "TRAINING_COMMANDS.md"),
        "adapter_installer": str(integration_target / "install_adapter.py"),
        "evaluation_script": str(integration_target / "evaluate_results.py"),
    }


def _final_distribution(records: Sequence[Mapping[str, Any]], inventory: set[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from syllable_metrics import distribution_statistics, rarity_counts

    frequency = Counter(token for row in records for token in row["syllables"])
    total = sum(frequency.values())
    rows = [
        {
            "syllable": token,
            "frequency": int(frequency.get(token, 0)),
            "relative_frequency": frequency.get(token, 0) / total if total else 0.0,
            "below_20": frequency.get(token, 0) < 20,
        }
        for token in sorted(inventory)
    ]
    voice = Counter(str(row["voice"]) for row in records)
    style = Counter(str(row["style"]) for row in records)
    sources = Counter()
    metadata: dict[str, Counter[str]] = defaultdict(Counter)
    for row in records:
        value = row.get("source_corpora")
        for source in value if isinstance(value, list) else [value or "unknown"]:
            sources[str(source)] += 1
        if isinstance(row.get("metadata"), dict):
            for axis, label in row["metadata"].items():
                metadata[str(axis)][str(label)] += 1
    return {
        "recognized_syllable_tokens": total,
        "unique_syllables": sum(count > 0 for count in frequency.values()),
        "inventory_size": len(inventory),
        "frequency_total_reconciled": total == sum(len(row["syllables"]) for row in records),
        "balance": distribution_statistics(frequency, inventory=inventory),
        "rarity": rarity_counts(frequency),
        "voice_distribution": dict(sorted(voice.items())),
        "style_distribution": dict(sorted(style.items())),
        "source_contribution": dict(sorted(sources.items())),
        "metadata_distribution": {
            axis: dict(sorted(counts.items())) for axis, counts in sorted(metadata.items())
        },
        "duration_hours": sum(float(row.get("duration_seconds", 0)) for row in records) / 3600,
        "similarity_exception_records": sum(bool(row.get("similarity_exception")) for row in records),
    }, rows


def _write_distribution_files(root: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    csv_path = reports / "canonical_syllable_distribution.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["syllable", "frequency", "relative_frequency", "below_20"])
        writer.writeheader()
        writer.writerows(rows)
    parquet_path = reports / "canonical_syllable_distribution.parquet"
    try:
        import pyarrow as pa  # type: ignore
        from pyarrow import parquet  # type: ignore

        parquet.write_table(pa.Table.from_pylist(list(rows)), parquet_path, compression="zstd")
    except ImportError as error:
        raise RuntimeError("pyarrow is required for the final syllable-distribution Parquet") from error
    return {"csv": _sha256(csv_path), "parquet": _sha256(parquet_path)}


def finalize_synthetic_asr(run_dir: str | Path, workers: int | None = None) -> dict[str, Any]:
    """Apply final invariants and export accepted manifests and SpeeChain metadata."""
    del workers
    root = Path(run_dir).expanduser().resolve()
    config, config_path = _load_run_config(root)
    prepared = list(_read_jsonl(root / "manifests" / "prepared_texts.jsonl"))
    prepared_by_id = {row["utterance_id"]: row for row in prepared}
    records = _load_job_records(root)
    cpu = {row["job_id"]: row for row in _read_jsonl(root / "qc" / "cpu_metrics.jsonl")}
    whisper = {row["job_id"]: row for row in _read_jsonl(root / "qc" / "whisper_results.jsonl")}
    mms = {row["job_id"]: row for row in _read_jsonl(root / "qc" / "mms_results.jsonl")} if (root / "qc" / "mms_results.jsonl").exists() else {}
    max_cer, max_wer = float(config["qc"].get("max_cer", 0.35)), float(config["qc"].get("max_wer", 0.60))
    accepted_jobs: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    for job_id, record in records.items():
        if record.get("phase") != "full":
            continue
        cpu_row, whisper_row, mms_row = cpu.get(job_id), whisper.get(job_id), mms.get(job_id)
        reasons = []
        if not cpu_row or not cpu_row.get("cpu_pass"):
            reasons.append("cpu_qc")
        if not whisper_row:
            reasons.append("missing_whisper")
        elif float(whisper_row.get("cer", 1.0)) > max_cer or float(whisper_row.get("wer", 1.0)) > max_wer:
            reasons.append("whisper_accuracy")
        if whisper_row and (
            float(whisper_row.get("cer", 1.0)) > max_cer
            or abs(float(whisper_row.get("cer", 1.0)) - max_cer) <= 0.05
        ) and not mms_row:
            reasons.append("missing_mms_disputed")
        required_rare = set(record.get("required_rare_syllables", []))
        if required_rare:
            if not mms_row:
                reasons.append("missing_mms")
            elif float(mms_row.get("rare_syllable_recall", 0.0)) < 1.0:
                reasons.append("mms_rare_recall")
            if whisper_row and float(whisper_row.get("rare_syllable_recall", 0.0)) < 1.0:
                reasons.append("whisper_rare_recall")
        combined = {
            **record,
            "training_path": cpu_row.get("training_path") if cpu_row else None,
            "audio_sha256": cpu_row.get("audio_sha256") if cpu_row else None,
            "frames": cpu_row.get("frames") if cpu_row else None,
            "duration_seconds": cpu_row.get("duration_seconds") if cpu_row else None,
            "whisper_cer": whisper_row.get("cer") if whisper_row else None,
            "whisper_wer": whisper_row.get("wer") if whisper_row else None,
            "mms_cer": mms_row.get("cer") if mms_row else None,
            "qc_status": "accepted" if not reasons else "quarantined",
            "failure_reasons": reasons,
        }
        (accepted_jobs if not reasons else quarantine).append(combined)

    canonical_candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rare_extras: list[dict[str, Any]] = []
    for row in accepted_jobs:
        if row["realization"] == "canonical" or row["realization"].startswith("regeneration"):
            canonical_candidates[row["utterance_id"]].append(row)
        elif row["realization"] == "rare_extra":
            rare_extras.append(row)
    canonical: list[dict[str, Any]] = []
    for utterance_id in sorted(prepared_by_id):
        candidates = canonical_candidates.get(utterance_id, [])
        if candidates:
            canonical.append(min(candidates, key=lambda row: (float(row.get("whisper_cer", 1.0)), row["job_id"])))
    target = int(config["run"].get("target_canonical", 20000))
    coverage = {token for row in canonical for token in row["syllables"]}
    inventory_path = root / "manifests" / "attainable_syllables.txt"
    attainable = {line for line in inventory_path.read_text(encoding="utf-8").splitlines() if line}
    expected_minimum = int(config.get("tokenizer", {}).get("attainable_inventory", 1358))
    missing_syllables = sorted(attainable.difference(coverage))
    rare_frequency = json.loads((root / "prepare.complete.json").read_text(encoding="utf-8")).get("rare_frequency", {})
    realizations: dict[str, set[str]] = defaultdict(set)
    for row in canonical + rare_extras:
        for token in set(row["syllables"]):
            if token in rare_frequency:
                realizations[token].add(row["voice"])
    rare_shortfall = {
        token: len(realizations[token]) for token in rare_frequency
        if len(realizations[token]) < int(config["rare_tail"].get("required_voice_realizations", 3))
    }
    issues: list[str] = []
    if len(canonical) != target:
        issues.append(f"canonical recordings {len(canonical)}/{target}")
    if len({row["job_id"] for row in canonical + rare_extras}) != len(canonical) + len(rare_extras):
        issues.append("duplicate accepted utterance IDs")
    if len({row["normalized_sha256"] for row in canonical}) != len(canonical):
        issues.append("duplicate canonical normalized-text hashes")
    if len(attainable) < expected_minimum:
        issues.append(f"prepared attainable inventory {len(attainable)}/{expected_minimum}")
    if missing_syllables:
        issues.append(f"syllable coverage missing {len(missing_syllables)} prepared types")
    if rare_shortfall:
        issues.append(f"rare voice-realization shortfall for {len(rare_shortfall)} syllables")
    if len({row["audio_sha256"] for row in canonical + rare_extras}) != len(canonical) + len(rare_extras):
        issues.append("duplicate accepted audio hashes")
    total_cost = 0.0
    connection = sqlite3.connect(root / "state" / "synthesis.sqlite3")
    total_cost = float(connection.execute("SELECT COALESCE(SUM(estimated_cost_usd),0) FROM synthesis_jobs WHERE status='succeeded'").fetchone()[0])
    connection.close()
    estimated_google_total = total_cost + float(config["budget"].get("vm_runtime_storage_usd", 10))
    if total_cost > float(config["budget"].get("tts_usd", 85)):
        issues.append(f"TTS tracked spend ${total_cost:.2f} exceeds allowance")
    if estimated_google_total > float(config["budget"].get("total_usd", 100)):
        issues.append(f"Estimated Google Cloud total ${estimated_google_total:.2f} exceeds budget")
    distribution, distribution_rows = _final_distribution(canonical, attainable)
    if not distribution["frequency_total_reconciled"]:
        issues.append("syllable frequency totals do not reconcile")

    regeneration_requests: list[dict[str, Any]] = []
    missing_utterances = sorted(set(prepared_by_id).difference(row["utterance_id"] for row in canonical))
    for utterance_id in missing_utterances:
        related = [
            row for row in quarantine
            if row["utterance_id"] == utterance_id
            and (row["realization"] == "canonical" or str(row["realization"]).startswith("regeneration"))
        ]
        attempts = [
            int(match.group(1)) for row in related
            if (match := re.match(r"regeneration_(\d+)", str(row["realization"])))
        ]
        next_attempt = max(attempts, default=0) + 1
        if next_attempt <= 2:
            failed_job = max(related, key=lambda row: row["job_id"])["job_id"] if related else f"ful_{utterance_id}"
            regeneration_requests.append({
                "utterance_id": utterance_id, "attempt": next_attempt,
                "failed_job_id": failed_job, "requested_at": _utc_now(),
            })
    if regeneration_requests:
        regeneration_path = root / "manifests" / "regeneration_requests.jsonl"
        previous_requests = list(_read_jsonl(regeneration_path)) if regeneration_path.exists() else []
        known = {(row["utterance_id"], int(row["attempt"])) for row in previous_requests}
        combined_requests = previous_requests + [
            row for row in regeneration_requests if (row["utterance_id"], int(row["attempt"])) not in known
        ]
        _write_jsonl(regeneration_path, combined_requests)

    report = {
        "run_id": _text_sha256(str(root) + _fingerprint(config, config_path)["configuration_sha256"])[:16],
        "finalized_at": _utc_now(), "status": "accepted" if not issues else "incomplete",
        "issues": issues, "canonical_accepted": len(canonical), "rare_extras_accepted": len(rare_extras),
        "syllable_coverage": len(coverage), "expected_syllable_coverage": len(attainable),
        "configured_minimum_syllable_coverage": expected_minimum,
        "missing_syllables": missing_syllables,
        "rare_voice_shortfall": rare_shortfall, "tracked_tts_cost_usd": total_cost,
        "estimated_vm_runtime_storage_usd": float(config["budget"].get("vm_runtime_storage_usd", 10)),
        "estimated_google_cloud_total_usd": estimated_google_total,
        "regeneration_requests": len(regeneration_requests),
        "maximum_alternate_voice_attempts": 2,
        "pronunciation_validation": "Automated Whisper and MMS verification; no native-speaker certification.",
        "resource_policy": {"new_cloud_resources": False, "asr_training_executed": False},
        "canonical_distribution": distribution,
    }
    _write_jsonl(root / "qc" / "quarantine.jsonl", quarantine)
    _atomic_json(root / "reports" / "final_report.json", report)
    if issues:
        regeneration_note = (
            f" {len(regeneration_requests)} alternate-voice jobs were queued; rerun full synthesis, CPU QC, "
            "Kaggle export/import, and finalization."
            if regeneration_requests else " No further automatic regeneration is available."
        )
        raise RuntimeError(
            "Final synthetic corpus is incomplete; no accepted manifest or SpeeChain export written: "
            + "; ".join(issues) + "." + regeneration_note
        )
    _write_jsonl(root / "final" / "canonical_20k.jsonl", canonical)
    _write_jsonl(root / "final" / "rare_voice_extras.jsonl", rare_extras)
    _write_jsonl(root / "final" / "accepted_all.jsonl", canonical + rare_extras)
    report["distribution_checksums"] = _write_distribution_files(root, distribution_rows)
    report["speechain"] = _export_speechain(root, canonical + rare_extras, config, config_path)
    report["output_checksums"] = {
        name: _sha256(root / "final" / name)
        for name in ("canonical_20k.jsonl", "rare_voice_extras.jsonl", "accepted_all.jsonl")
    }
    _atomic_json(root / "reports" / "final_report.json", report)
    final_manifest = {
        "schema_version": 1,
        "run_id": report["run_id"],
        "accepted_at": report["finalized_at"],
        "status": "accepted",
        "fingerprints": _fingerprint(config, config_path),
        "legal_review_sha256": _sha256(root / str(config["run"].get("legal_review_file", "legal_review.json"))),
        "model_revisions": {
            key: config["qc"][key]
            for key in (
                "whisper_model", "whisper_revision", "mms_model", "mms_revision",
                "mms_language_adapter", "speaker_embedding_model", "speaker_embedding_revision",
            )
        },
        "software_versions": _software_versions((
            "google-cloud-texttospeech", "numpy", "pyarrow", "soundfile", "PyYAML",
        )),
        "platform": {"python": platform.python_version(), "operating_system": platform.platform()},
        "outputs": report["output_checksums"],
        "distribution_outputs": report["distribution_checksums"],
        "pronunciation_certification": "automated_whisper_and_mms_only_no_native_speaker",
        "new_cloud_resources_created": False,
        "asr_training_executed": False,
    }
    _atomic_json(root / "final" / "manifest.json", final_manifest)
    return report
