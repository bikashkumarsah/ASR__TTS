"""Report CER, WER, syllable error rate, and rare-syllable recall."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path


def load_idx2text(path: Path) -> dict[str, str]:
    rows = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            key, text = line.split(maxsplit=1)
        except ValueError as error:
            raise ValueError(f"Expected ID and text at {path}:{line_number}") from error
        if key in rows:
            raise ValueError(f"Duplicate ID in {path}: {key}")
        rows[key] = text
    return rows


def normalize(text: str) -> str:
    value = unicodedata.normalize("NFC", text)
    value = re.sub(r"[^\u0900-\u097F\s]", " ", value).replace("।", " ").replace("॥", " ")
    return re.sub(r"\s+", " ", value).strip()


def distance(left: list[str], right: list[str]) -> int:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--token-path", required=True)
    parser.add_argument("--rare-syllables", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    token_root = Path(args.token_path)
    spec = importlib.util.spec_from_file_location("evaluation_syllabic_tokenizer", token_root / "syllabic_tokenizer.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the pinned syllable tokenizer")
    tokenizer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tokenizer)
    lookup = tokenizer.get_lookup_tokens(str(token_root / "lookup.vocab"))
    reference = load_idx2text(Path(args.reference))
    hypothesis = load_idx2text(Path(args.hypothesis))
    if set(reference) != set(hypothesis):
        raise RuntimeError(
            f"Reference/hypothesis ID mismatch: missing={len(set(reference) - set(hypothesis))}, "
            f"unknown={len(set(hypothesis) - set(reference))}"
        )
    rare = {line for line in Path(args.rare_syllables).read_text(encoding="utf-8").splitlines() if line}
    char_edits = char_total = word_edits = word_total = syllable_edits = syllable_total = 0
    rare_reference = rare_matched = 0
    for key in sorted(reference):
        ref, hyp = normalize(reference[key]), normalize(hypothesis[key])
        ref_chars, hyp_chars = list(ref.replace(" ", "")), list(hyp.replace(" ", ""))
        ref_words, hyp_words = ref.split(), hyp.split()
        ref_syllables = [token for token in tokenizer.tokenize(ref, lookup) if token.strip()]
        hyp_syllables = [token for token in tokenizer.tokenize(hyp, lookup) if token.strip()]
        char_edits += distance(ref_chars, hyp_chars)
        char_total += len(ref_chars)
        word_edits += distance(ref_words, hyp_words)
        word_total += len(ref_words)
        syllable_edits += distance(ref_syllables, hyp_syllables)
        syllable_total += len(ref_syllables)
        ref_counts, hyp_counts = Counter(ref_syllables), Counter(hyp_syllables)
        for token in rare:
            rare_reference += ref_counts[token]
            rare_matched += min(ref_counts[token], hyp_counts[token])
    metrics = {
        "utterances": len(reference),
        "cer": char_edits / max(1, char_total),
        "wer": word_edits / max(1, word_total),
        "syllable_error_rate": syllable_edits / max(1, syllable_total),
        "rare_syllable_recall": rare_matched / max(1, rare_reference),
        "rare_reference_occurrences": rare_reference,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

