# Nepali ASR/TTS — Syllable Tokenizer and Balanced Corpus Builder

This repository contains a pronunciation-aware Nepali syllable tokenizer and a
repeatable pipeline for building a balanced ASR corpus. The builder extracts
sentences, annotates metadata, selects balanced batches, writes reports, and
merges the batches into a final JSONL corpus.

The existing-VM Gemini-TTS and private-Kaggle QC workflow is documented in
[`docs/synthetic_asr_existing_vm_kaggle.md`](docs/synthetic_asr_existing_vm_kaggle.md).

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
larger *bounded* per-cell reservoir, giving the balancer a broader choice of
sentences. For a 120 GB instance, `--max-candidates-per-cell 8000` is a
headroom-safe starting point. Keep all cores enabled and do not try to allocate
all RAM: Python worker and merge peaks vary with the record size.

For the final coverage stage, the pipeline uses a second, much smaller bounded
index: up to 256 examples for each requested source syllable. This is more
effective than random reservoir expansion for rare syllables, because it
guarantees that each source-supported target has candidates available to the
coverage seed selector.

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

## Source-aware coverage recovery and final 50k curation

The lookup vocabulary contains 1,782 tokens, but the supplied source scan
contains 1,182 distinct syllables. The final target is therefore the
source-supported inventory, not all lookup entries: a syllable absent from the
candidate pool cannot be recovered by selection.

After the completed 150k run, the supplied state has 1,072 of the 1,182
source-supported syllables (90.69%). The commands below recover the remaining
source targets in a strict Batch 31, then create a **fresh**, standalone,
balanced 50k corpus with all source-supported syllables. Neither command
resets or re-extracts the pool. `build-final` does not change the incremental
state or the existing batch files.

```bash
# Run from tokenizer/syllable-tokenizer/scripts in the cloud repository.
cd tokenizer/syllable-tokenizer/scripts
CLOUD_WORKERS="$(nproc)"

# Establish the exact set that exists in the annotated source pool.
python -m dataset_builder.pipeline coverage-inventory \
  --output ../dataset/asr_corpus/source_syllable_inventory.json \
  --workers "$CLOUD_WORKERS"

# Recover every currently absent source syllable, or fail without writing a
# partial batch. The remaining records are selected with metadata balancing.
python -m dataset_builder.pipeline run-batch \
  --batch-id 31 --target-size 5000 \
  --coverage-targets ../dataset/asr_corpus/source_syllable_inventory.json \
  --coverage-candidates-per-syllable 256 --require-full-coverage \
  --max-candidates-per-cell 8000 --workers "$CLOUD_WORKERS"
python -m dataset_builder.pipeline status

# Curate a new 50k delivery dataset. This is non-destructive: it reads the
# pool only, verifies 100% source-target coverage, and writes no batch/state.
python -m dataset_builder.pipeline build-final \
  --coverage-targets ../dataset/asr_corpus/source_syllable_inventory.json \
  --target-size 50000 --max-candidates-per-cell 8000 \
  --coverage-candidates-per-syllable 256 --workers "$CLOUD_WORKERS" \
  --output ../dataset/asr_corpus/final_50k_all_syllables.jsonl \
  --report-output ../dataset/asr_corpus/reports/final_50k_all_syllables_report.json
```

Download `final_50k_all_syllables.jsonl` as the final dataset, with
`final_50k_all_syllables_report.json` as its coverage and distribution proof.
The final command fails instead of emitting a corpus if even one inventory
syllable is missing or if an active metadata label deviates by more than two
percentage points from uniform distribution.

## Outputs and resume behavior

All generated files are under `tokenizer/syllable-tokenizer/dataset/asr_corpus/`:

- `pool/` — extracted and annotated candidate chunks
- `batches/batch_###.jsonl` — selected batches
- `reports/batch_###_report.json` — distribution reports
- `corpus_state.json` — selected IDs and cumulative balancing state
- `source_syllable_inventory.json` — source-supported coverage target set
- `final_50k_all_syllables.jsonl` — standalone final delivery corpus

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

## Analyze syllables across five public Nepali corpora

The repository also includes a separate, non-destructive analysis workflow for
IRIIS, Sakonii, cleaned CC100 Nepali, the IEEE DataPort `compiled.txt`, and
Boredoom17. It produces per-corpus reports, a source-native combined view, and
an authoritative exact-deduplicated union without changing the ASR corpus pool,
batches, state, or final 50k dataset.

Use the fixed lookup tokenizer with every cloud CPU core:

```bash
cd tokenizer/syllable-tokenizer/scripts
CLOUD_WORKERS="$(nproc)"

python -m corpus_analysis.pipeline validate-inputs \
  --config ../configs/nepali_corpus_analysis.yaml \
  --input-root /data/nepali-corpora

python -m corpus_analysis.pipeline analyze \
  --config ../configs/nepali_corpus_analysis.yaml \
  --input-root /data/nepali-corpora \
  --output-root ../dataset_analysis \
  --workers "$CLOUD_WORKERS"
```

See [the complete Google Cloud and reporting guide](docs/nepali_corpus_syllable_analysis.md)
for pinned downloads, the 100k-record smoke test, result interpretation,
validation, GCS upload, and the minimum files to retrieve from the cloud.
