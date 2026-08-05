# Nepali ASR/TTS — Syllable Tokenizer and Balanced Corpus Builder

This repository contains a pronunciation-aware Nepali syllable tokenizer and a
repeatable pipeline for building a balanced ASR corpus. The builder extracts
sentences, annotates metadata, selects balanced batches, writes reports, and
merges the batches into a final JSONL corpus.

## Requirements

Use Python 3.10+ and install the project dependencies from the repository root:

```bash
python -m pip install -r tokenizer/syllable-tokenizer/requirements.txt
```

### Optional: download the primary text corpus

The full build can use
[IRIIS-RESEARCH/Nepali-Text-Corpus](https://huggingface.co/datasets/IRIIS-RESEARCH/Nepali-Text-Corpus)
(6.4M articles; approximately 27.5 GB). Place it beside this repository or
pass its location with `--nepali-corpus`.

```bash
python -m pip install -U "huggingface_hub[cli]"
huggingface-cli download --repo-type dataset \
  IRIIS-RESEARCH/Nepali-Text-Corpus --local-dir ./Nepali-Text-Corpus
```

## Build a 50k corpus

Run the commands from `tokenizer/syllable-tokenizer/scripts`.

```bash
cd tokenizer/syllable-tokenizer/scripts

# Cloud Linux: use every CPU core allocated to the instance.
CLOUD_WORKERS="$(nproc)"

# Destructive: clears the existing pool, batches, reports, state, and merged corpus.
python -m dataset_builder.pipeline reset

# Build the candidate pool, annotate every chunk, and select batch 1.
# --max-corpus 0 means no sentence limit for the parquet source.
python -m dataset_builder.pipeline run-batch \
  --batch-id 1 --target-size 5000 --max-corpus 0 --max-chunks 0 \
  --workers "$CLOUD_WORKERS"

# Reuse the annotated pool for the remaining batches.
for b in {2..10}; do
  python -m dataset_builder.pipeline run-batch \
    --batch-id "$b" --target-size 5000 --workers "$CLOUD_WORKERS"
done

# Merge and inspect the result.
python -m dataset_builder.pipeline merge --output ../dataset/asr_corpus/corpus_50k.jsonl
python -m dataset_builder.pipeline status
```

If the corpus is not in an auto-detected sibling directory, provide it
explicitly on batch 1:

```bash
python -m dataset_builder.pipeline run-batch \
  --batch-id 1 --target-size 5000 --max-corpus 0 \
  --nepali-corpus /absolute/path/to/Nepali-Text-Corpus
```

## Memory-safe operation

The pipeline never loads all pool chunks into one Python list during batch
selection. It streams each `pool_chunk_*.jsonl` file and keeps only a bounded,
reservoir-sampled set of candidates per metadata cell. This remains bounded
even when the pool has thousands of chunks.

The default is 2,000 candidates per cell. Lower it when memory is tight;
increase it only when you want a wider candidate choice for syllable balancing.
By default, `--workers` uses every CPU core visible to Python. On Cloud Linux,
the `CLOUD_WORKERS="$(nproc)"` setting above makes that allocation explicit.

```bash
python -m dataset_builder.pipeline run-batch \
  --batch-id 2 --target-size 5000 --max-candidates-per-cell 500 --workers 1
```

`--workers 1` uses the smallest RAM footprint. The recommended cloud setting
is `--workers "$(nproc)"`, which uses all allocated cores for extraction,
annotation, streaming, selection, and hash-partitioned deduplication. Each
worker keeps only a bounded candidate reservoir rather than the complete pool.
`--max-chunks 0` is the safe default: all chunks must be annotated before
selection, and the pipeline stops with a clear error if a partial annotation is
requested.

For a quick smoke test before a full build, limit the extraction input rather
than limiting annotation:

```bash
python -m dataset_builder.pipeline reset
python -m dataset_builder.pipeline run-batch \
  --batch-id 1 --target-size 100 --max-shards 1 --max-corpus 2000 --workers 1
```

## Outputs and resume behavior

All generated files are under `tokenizer/syllable-tokenizer/dataset/asr_corpus/`:

- `pool/` — extracted and annotated candidate chunks
- `batches/batch_###.jsonl` — selected batches
- `reports/batch_###_report.json` — distribution reports
- `corpus_state.json` — selected IDs and cumulative balancing state
- `corpus_50k.jsonl` — merged, de-duplicated output

To resume, rerun the next `run-batch` command. Existing selected IDs in
`corpus_state.json` are excluded automatically. Use `reset` only when you
intend to discard that generated state.

## Tokenizer usage

```bash
cd tokenizer/syllable-tokenizer
python scripts/generate_vocab_lookup.py
python scripts/syllabic_tokenizer.py
```

See [the tokenizer README](tokenizer/syllable-tokenizer/README.md) for library
usage, vocabulary generation, and the academic citation.
