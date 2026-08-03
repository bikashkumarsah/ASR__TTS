#!/usr/bin/env python3
"""Generate a syllabic vocab from transcript text.

Reads one transcript per line, tokenizes each line with the syllabic tokenizer,
deduplicates tokens while preserving first-seen order, and writes the result to
the requested vocab file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from syllabic_tokenizer import get_lookup_tokens, tokenize


def read_transcripts(input_file):
    with open(input_file, "r", encoding="utf-8") as fh:
        for line in fh:
            text = line.strip()
            if text:
                yield text


def build_vocab(input_file, lookup_file):
    lookup_vocab = get_lookup_tokens(lookup_file)
    for transcript in read_transcripts(input_file):
        for token in tokenize(transcript, lookup_vocab, debug=False):
            if token:
                yield token


def unique_tokens(tokens):
    seen = set()
    for token in tokens:
        if token not in seen:
            seen.add(token)
            yield token


def write_vocab(tokens, output_file):
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as fh:
        for token in tokens:
            fh.write(token + "\n")
            count += 1
    return count


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate a unique syllabic vocab from transcripts.")
    parser.add_argument("--input_file", required=True, help="Transcript file with one line of text per record")
    parser.add_argument("--output_file", required=True, help="Path to write the vocab file")
    parser.add_argument(
        "--lookup_file",
        default="dataset/nepali_syllables_lookup.vocab",
        help="Lookup vocab used by the syllabic tokenizer",
    )
    args = parser.parse_args(argv)

    count = write_vocab(unique_tokens(build_vocab(args.input_file, args.lookup_file)), args.output_file)
    print(f"Wrote {count} unique tokens to {args.output_file}")


if __name__ == "__main__":
    main()
