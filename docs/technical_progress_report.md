# Technical Progress Report: Nepali ASR/TTS Corpus

This Markdown file is the authoritative end-of-day progress record. The earlier
LaTeX report is preserved as historical material and is not compiled by this
workflow.

## Project objective

Create a verified 50,000-sentence Nepali corpus with complete attainable
lookup-syllable coverage, strengthened rare-syllable representation, an
ASR-realistic square-root-tempered distribution, and reduced lexical and
semantic repetition.

## Historical baseline

The previous final corpus contains 50,000 records, 3,154,230 recognized
syllable tokens under the earlier tokenizer and reported 1,168 source-pool
syllables. The later full-vocabulary-window experiment remains a separate,
checksum-pinned result. Comparisons with the pronunciation-aware tokenizer
paper now use the fixed four-code-point window and must not combine statistics
from the two tokenizer definitions.

## Current workflow

- The authoritative vocabulary is `dataset/nepali_syllables_lookup.vocab`.
- The vocabulary path, SHA-256 checksum, and 1,781 non-whitespace-entry
  integrity count are pinned in `configs/final_50k_diverse.yaml`. Coverage uses
  the 1,778 selectable entries after excluding `।`, `?`, and `!`.
- Every resumable stage also records the SHA-256 of
  `scripts/syllabic_tokenizer.py`; the combined configuration fingerprint
  changes whenever that tokenizer source changes.
- The default tokenizer uses the paper's fixed four-code-point lookup window.
  It emits 1,244 of the 1,778 selectable lookup entries; the 534 entries of
  length five to seven are explicitly reported as structurally unemittable.
- The former full-vocabulary-window behavior is available only through the
  explicit `max_token_length=None` compatibility option. Existing prepared
  runs remain reproducible through their pinned tokenizer copies.
- Coverage is reported separately for tokenizer structure, five-corpus source
  support, and the final selected corpus.
- The five public corpora are streamed and exactly deduplicated before
  selection; three bounded passes retain overlapping source provenance without
  caching records that are later evicted from the shortlist.
- Character 5-gram MinHash-LSH identifies lexical near-duplicates.
- Quantized multilingual E5 embeddings, empirical cosine calibration, and
  FAISS clustering control semantic repetition. Exact `IndexFlatIP` search
  enforces hard cutoffs and audits final neighbours.
- LaBSE is held out from selection and is used only for comparative semantic
  acceptance against the retokenized historical 50k corpus.
- Attainable coverage and a five-supporting-sentence rare floor take priority
  over similarity; all resulting exceptions are recorded.
- Metadata balance is a soft objective and source composition is unconstrained.

## Resource envelope

The workflow targets a 64 GB / 12 vCPU cloud runner.

- Preparation submits a bounded window of tokenization batches to its worker
  pool, so parent memory during the three streaming passes is capped by the
  window rather than by the total number of source sentences.
- The embedding topologies in `configs/final_50k_diverse.yaml` are sized so
  every candidate layout fits 12 vCPU; a layout set that cannot fit the
  available workers now reports the fallback instead of silently degrading to a
  single process.
- The embedding worker fleet is capped at 40 GB of summed concurrent resident
  memory.
- The candidate matrix for a 1.2M pool is loaded into a preallocated array and
  the raw copy is released after calibration, so only one candidates x
  dimensions matrix is resident during selection.
- Every stage prints its peak resident set size, because an out-of-memory kill
  leaves no traceback.

## Daily verified runs

### 2026-08-30: four-window paper-comparison mode

- Restored the fixed four-code-point greedy lookup window used by the reference
  pronunciation-aware tokenizer paper.
- Retokenizing the verified 20,000 source texts produces 1,183 observed source
  types; after spoken-form digit expansion, the TTS/ASR transcript inventory is
  1,173 types. Whitespace is excluded from both analytical counts.
- The existing full-vocabulary-window result (1,358 source and 1,348 spoken
  types) remains historical and reproducible, but it is not used as the direct
  denominator for the paper comparison.
- The synthetic-ASR configurations pin `lookup_window_size: 4`, and fresh run
  directories are required after this tokenizer-source change.
- This is an algorithm-controlled comparison rather than an exact replication:
  the reference paper reports 650 pronunciation-aware tokens, while the project
  continues to use its separately constructed, checksum-pinned lookup file.

Run sections below are generated idempotently from the machine-readable final
report:

```bash
python -m dataset_builder.pipeline update-progress \
  --run-report ../dataset/asr_corpus/reports/final_50k_diverse_rare/report.json \
  --progress-file ../../../docs/technical_progress_report.md
```

Do not manually claim final metrics before the complete cloud report has passed
its acceptance checks.
