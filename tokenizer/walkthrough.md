# Walkthrough — Balanced ASR Dataset Builder

## Summary

Implemented the complete 5-stage incremental pipeline for building balanced Nepali ASR recording datasets under the syllable-tokenizer project. The pipeline extracts sentences from two source files, annotates them with rule-based metadata, and selects balanced 5k-sentence batches targeting uniform syllable-type and metadata distributions.

## Files Created

### New Module: `scripts/dataset_builder/`

| File | Purpose |
|------|---------|
| [\_\_init\_\_.py](file:///Users/bikashkumarsah/Downloads/research_intern/tokenizer/syllable-tokenizer/scripts/dataset_builder/__init__.py) | Package init |
| [extract.py](file:///Users/bikashkumarsah/Downloads/research_intern/tokenizer/syllable-tokenizer/scripts/dataset_builder/extract.py) | Stage 1: Streaming HTML extraction from `compiled.txt` (1.8GB, chunked reads), sentence parsing from `Source_book.txt`, quality filters (5–80 syllables), SHA-256 dedup |
| [rules.yaml](file:///Users/bikashkumarsah/Downloads/research_intern/tokenizer/syllable-tokenizer/scripts/dataset_builder/rules.yaml) | Configurable regex patterns for tense/polarity/gender/sector annotation — edit rules without code changes |
| [annotate.py](file:///Users/bikashkumarsah/Downloads/research_intern/tokenizer/syllable-tokenizer/scripts/dataset_builder/annotate.py) | Stage 2: Rule-based tagger using patterns from `rules.yaml` with count-based disambiguation |
| [syllable_stats.py](file:///Users/bikashkumarsah/Downloads/research_intern/tokenizer/syllable-tokenizer/scripts/dataset_builder/syllable_stats.py) | Stage 3a: Corpus-level syllable frequency, CV computation, vocab coverage metrics |
| [analyze.py](file:///Users/bikashkumarsah/Downloads/research_intern/tokenizer/syllable-tokenizer/scripts/dataset_builder/analyze.py) | Stage 3b: Per-batch JSON reports with metadata crosstabs, deviation-from-uniform metrics, top/rare syllable lists |
| [balance.py](file:///Users/bikashkumarsah/Downloads/research_intern/tokenizer/syllable-tokenizer/scripts/dataset_builder/balance.py) | Stage 4: Two-objective greedy selection — metadata stratification (hard) + syllable deficit scoring (soft), with incremental rebalancing |
| [pipeline.py](file:///Users/bikashkumarsah/Downloads/research_intern/tokenizer/syllable-tokenizer/scripts/dataset_builder/pipeline.py) | Stage 5: CLI orchestrator with `run-batch`, `analyze`, `merge`, `status` subcommands |

### Modified

| File | Change |
|------|--------|
| [requirements.txt](file:///Users/bikashkumarsah/Downloads/research_intern/tokenizer/syllable-tokenizer/requirements.txt) | Added `tqdm`, `pyyaml` |

### Generated Outputs (from test runs)

```
dataset/asr_corpus/
├── pool/                          # Annotated candidate pool
│   ├── pool_chunk_0000.jsonl      # ~50k records
│   ├── pool_chunk_0001.jsonl      # ~32k records
│   ├── extraction_stats.json
│   └── annotation_stats.json
├── batches/
│   ├── batch_001.jsonl            # 5,000 selected sentences
│   └── batch_002.jsonl            # 5,000 selected sentences
├── reports/
│   ├── batch_001_report.json
│   └── batch_002_report.json
├── corpus_state.json              # Cumulative state for incremental batches
└── corpus_50k.jsonl               # Merged output (currently 10k from 2 batches)
```

## Validation Results

### Batch 1 (5,000 sentences)

| Metric | Result |
|--------|--------|
| Sentences selected | 5,000 |
| Unique syllables | 657 |
| Syllable CV | 2.6279 |
| Coverage | 36.9% of 1,781 vocab tokens |
| Median sentence length | 52 syllables |
| Dedup | Zero duplicates (SHA-256 hash check) |

### Batch 2 (5,000 sentences, incremental)

| Metric | Result |
|--------|--------|
| Sentences selected | 5,000 (no overlap with batch 1) |
| Cumulative total | 10,000 |
| Unique syllables (cumulative) | 690 |
| Cumulative CV | 2.7022 |
| Pool reuse | ✓ Skipped extraction + annotation |
| State persistence | ✓ `corpus_state.json` correctly loaded/updated |

### All CLI Commands Verified

| Command | Status |
|---------|--------|
| `run-batch --batch-id N` | ✓ Works for batch 1 (full pipeline) and batch 2 (incremental) |
| `analyze --batch-id N` | ✓ Regenerates report from existing batch file |
| `merge --output PATH` | ✓ Deduplicates and merges all batches |
| `status` | ✓ Shows cumulative corpus metrics |

## Usage

```bash
# From tokenizer/syllable-tokenizer/scripts:

# Run batch 1 (extracts pool + selects 5k)
python -m dataset_builder.pipeline run-batch --batch-id 1 --target-size 5000

# Run batch 2–10 (reuses pool, excludes prior selections)
python -m dataset_builder.pipeline run-batch --batch-id 2 --target-size 5000

# Analyze an existing batch
python -m dataset_builder.pipeline analyze --batch-id 1

# Check corpus status
python -m dataset_builder.pipeline status

# Merge all batches into final corpus
python -m dataset_builder.pipeline merge --output ../dataset/asr_corpus/corpus_50k.jsonl
```

## Observations & Tuning Notes

> [!NOTE]
> The pool extracted 82,174 candidate sentences (12,277 duplicates removed) from ~100k raw compiled.txt sentences + 2,306 book sentences. For the full 50k corpus, increase `--max-compiled` (the 1.8GB file contains millions of sentences).

> [!IMPORTANT]
> **Metadata imbalances to address before remaining batches:**
> - **Future tense** is underrepresented (~5%) — the pool naturally has few future-tense sentences. Consider adding more future-tense patterns to `rules.yaml`.
> - **Masculine gender** is sparse (~12%) — Nepali text defaults to neutral; this is expected for rule-based annotation.
> - **Literature sector** caps at ~2k sentences from `Source_book.txt` — the plan anticipated this limitation.
> - Tune patterns in [rules.yaml](file:///Users/bikashkumarsah/Downloads/research_intern/tokenizer/syllable-tokenizer/scripts/dataset_builder/rules.yaml) between batches to improve annotation accuracy.
