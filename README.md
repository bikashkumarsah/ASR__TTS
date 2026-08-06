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

The default is 2,000 candidates per cell. This is a global per-cell cap, not a
cap per worker: when several workers are used, the pipeline divides the cap
between them and interleaves pool chunks across workers. This prevents the RAM
limit from multiplying with the number of cloud CPU cores. Lower the cap when
memory is tight; increase it only when you want a wider candidate choice for
syllable balancing.
By default, `--workers` uses every CPU core visible to Python. On Cloud Linux,
the `CLOUD_WORKERS="$(nproc)"` setting above makes that allocation explicit.

```bash
python -m dataset_builder.pipeline run-batch \
  --batch-id 2 --target-size 5000 --max-candidates-per-cell 500 --workers 1
```

`--workers 1` uses the smallest RAM footprint. The recommended cloud setting
is `--workers "$(nproc)"`, which uses all allocated cores for extraction,
annotation, streaming, selection, and hash-partitioned deduplication. Each
worker keeps only its share of a bounded global candidate reservoir rather than
the complete pool.
`--max-chunks 0` is the safe default: all chunks must be annotated before
selection, and the pipeline stops with a clear error if a partial annotation is
requested.

### Use additional cloud RAM for coverage expansion

More RAM does not make the selector read the entire pool. Instead, it permits a
larger *bounded* reservoir, giving coverage-priority selection a broader choice
of sentences in every metadata cell. The completed 100k run used about 10 GB
with the default 2,000 candidates per cell. For a 120 GB instance, start at
8,000 candidates per cell (four times the default) rather than allocating all
available RAM: Python worker and merge peaks need substantial headroom. Keep
all CPU cores enabled, monitor the first batch, and lower the cap if the
instance approaches its memory limit.

```bash
# Run from tokenizer/syllable-tokenizer/scripts after the 100k run.
CLOUD_WORKERS="$(nproc)"

# Preserve the completed 100k corpus before extending the active state.
mkdir -p ../dataset/asr_corpus_baseline_100k
cp ../dataset/asr_corpus/corpus_100k_coverage.jsonl ../dataset/asr_corpus_baseline_100k/
cp ../dataset/asr_corpus/corpus_state.json ../dataset/asr_corpus_baseline_100k/
cp -R ../dataset/asr_corpus/reports ../dataset/asr_corpus_baseline_100k/

# Reuse the existing extracted and annotated pool. Do not reset or force-extract.
for b in {21..30}; do
  python -m dataset_builder.pipeline run-batch \
    --batch-id "$b" --target-size 5000 \
    --coverage-priority 16 --max-candidates-per-cell 8000 \
    --workers "$CLOUD_WORKERS"
done

python -m dataset_builder.pipeline merge \
  --output ../dataset/asr_corpus/corpus_150k_coverage.jsonl
python -m dataset_builder.pipeline status
```

If the first batch stays below roughly 60 GB of RAM, `--max-candidates-per-cell
12000` is a reasonable next step. Do not change the cap part way through the
same 5k batch; changing it between completed batches is safe because only the
selected IDs and cumulative frequencies are persisted.

### Resume after a streaming-worker failure

Do not reset or rerun a completed batch. Pull the latest code, then rerun only
the failed batch; `corpus_state.json` excludes the 5,000 records already
selected in batch 1.

```bash
git pull origin main
cd tokenizer/syllable-tokenizer/scripts
CLOUD_WORKERS="$(nproc)"
python -m dataset_builder.pipeline run-batch \
  --batch-id 2 --target-size 5000 --workers "$CLOUD_WORKERS"
```

For a quick smoke test before a full build, limit the extraction input rather
than limiting annotation:

```bash
python -m dataset_builder.pipeline reset
python -m dataset_builder.pipeline run-batch \
  --batch-id 1 --target-size 100 --max-shards 1 --max-corpus 2000 --workers 1
```

## Expand coverage from the existing pool

After a completed 50k build, reuse the annotated pool and state to add
coverage-focused batches. Do not run `reset` or `--force-extract`: existing
selected IDs are excluded automatically, and `--coverage-priority` adds a
strong bonus for syllables not yet present in `corpus_state.json`.

```bash
cd tokenizer/syllable-tokenizer/scripts
CLOUD_WORKERS="$(nproc)"

# Preserve the completed 50k corpus as a comparison baseline.
mkdir -p ../dataset/asr_corpus_baseline_50k
cp ../dataset/asr_corpus/corpus_50k.jsonl ../dataset/asr_corpus_baseline_50k/
cp ../dataset/asr_corpus/corpus_state.json ../dataset/asr_corpus_baseline_50k/
cp -R ../dataset/asr_corpus/reports ../dataset/asr_corpus_baseline_50k/

for b in {11..20}; do
  python -m dataset_builder.pipeline run-batch \
    --batch-id "$b" --target-size 5000 \
    --coverage-priority 8 --workers "$CLOUD_WORKERS"
done

python -m dataset_builder.pipeline merge \
  --output ../dataset/asr_corpus/corpus_100k_coverage.jsonl
python -m dataset_builder.pipeline status
```

This creates a 100k expanded corpus while preserving the first 50k records as
the baseline. Compare the resulting `corpus_state.json` coverage and CV with
the 50k baseline after each batch.

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
