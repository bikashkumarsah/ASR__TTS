# extract the unique transcripts from the tsv files of oslr splits
# https://github.com/rupakraj/slr54nepali-curated
#
#    get the files and dump to the dataset/oslr_transcripts folder
#       `wget https://github.com/rupakraj/slr54nepali-curated/blob/main/utt_{split}.tsv -O dataset/oslr_transcripts/utt_{split}.tsv`
#    valid splits are: `train`, `test`, `valid`

from pathlib import Path


def extract_unique(input_path):
    tokens = []
    seen = set()
    with input_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")

            if len(parts) >= 3:
                text = parts[2].strip()
            elif len(parts) >= 2:
                text = parts[-1].strip()
            else:
                continue

            if text and text not in seen:
                seen.add(text)
                tokens.append(text)
    return tokens


def main():
    root = Path("dataset/oslr_transcripts")
    if not root.exists():
        raise SystemExit(f"Path not found: {root}")

    lines_all = { }
    for tsv in sorted(root.glob("*.tsv")):
        lines_all[tsv.stem] = extract_unique(tsv)

    merged_lines = set()
    for split, lines in lines_all.items():
        merged_lines.update(lines)

    print(f"{len(merged_lines)} unique transcripts")

    # save the unique transcripts to a file
    output_path = Path("dataset/oslr_transcripts/unique_transcripts.txt")
    with output_path.open("w", encoding="utf-8") as fh:
        for line in sorted(merged_lines):
            fh.write(line + "\n")

if __name__ == "__main__":
    main()
