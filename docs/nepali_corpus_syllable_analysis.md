# Nepali Corpus Syllable Distribution Analysis

This workflow analyzes the five published corpora independently and in two
combined views. It does not read from or modify the derived ASR 50k corpus.

- **Per-corpus:** preserves each publisher's rows, including repetitions.
- **Source-native combined:** sums all five published datasets.
- **Exact-deduplicated combined:** hashes normalized text and counts every
  distinct normalized text once. This is the authoritative overall result.

The distinction matters because Sakonii contains material derived from CC100
and Boredoom17 contains material derived from IRIIS. Repetition cannot introduce
a new observed lookup syllable, but it can bias every frequency statistic.

## Google Cloud resources

Use one Compute Engine `n2-standard-32` VM with 32 vCPUs, 128 GB RAM, a 500 GB
balanced persistent boot disk, Ubuntu 24.04 LTS, and no GPU. The analyzer keeps
RAM bounded: only a small number of record batches are in flight, while exact
text hashes are stored in SQLite on disk. Unused RAM is intentional safety
headroom for Arrow decompression, worker processes, the OS cache, and varying
record sizes.

Create the VM from Cloud Shell (adjust the project and zone if necessary):

```bash
gcloud compute instances create nepali-syllable-analysis \
  --zone=asia-south1-a \
  --machine-type=n2-standard-32 \
  --boot-disk-type=pd-balanced \
  --boot-disk-size=500GB \
  --image-family=ubuntu-2404-lts-amd64 \
  --image-project=ubuntu-os-cloud
```

SSH to the VM, then prepare the repository and dependencies:

```bash
sudo apt-get update
sudo apt-get install -y git python3-venv awscli tmux
git clone https://github.com/bikashkumarsah/ASR__TTS.git
cd ASR__TTS
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r tokenizer/syllable-tokenizer/requirements.txt
sudo mkdir -p /data/nepali-corpora
sudo chown -R "$USER":"$USER" /data/nepali-corpora
```

## Download exact source revisions

Run the provided downloader rather than using the derived 50k corpus. It pins
the Hugging Face `main` revisions to their current commit hashes and records
those hashes in `/data/nepali-corpora/source_manifest.json`. The public IEEE
DataPort file is downloaded from S3 and recorded with its file SHA-256.

```bash
cd tokenizer/syllable-tokenizer/scripts
CLOUD_WORKERS="$(nproc)"

python -m corpus_analysis.download \
  --config ../configs/nepali_corpus_analysis.yaml \
  --input-root /data/nepali-corpora \
  --workers "$CLOUD_WORKERS"
```

The expected local layout is:

```text
/data/nepali-corpora/
├── IRIIS-RESEARCH__Nepali-Text-Corpus/
├── Sakonii__nepalitext-language-model-dataset/
├── himalaya-ai__cc100-nepali/
├── ieee_dataport/compiled.txt
├── Boredoom17__Nepali-Corpus/
└── source_manifest.json
```

Validate all paths and Parquet schemas before starting CPU-intensive work:

```bash
python -m corpus_analysis.pipeline validate-inputs \
  --config ../configs/nepali_corpus_analysis.yaml \
  --input-root /data/nepali-corpora
```

After the pinned download, preserve the reusable raw source snapshot in GCS:

```bash
RAW_BUCKET="gs://<your-bucket>/nepali-syllable-analysis/raw"
gcloud storage rsync --recursive /data/nepali-corpora "$RAW_BUCKET"
```

On a replacement VM, reverse the two paths in that command to restore the
same inputs and `source_manifest.json` without downloading the public sources
again.

## Required smoke test and deterministic comparison

Analyze the first 100,000 records from every corpus once with one worker and
once with all cores. The comparison command fails if any summary, quality
metric, or syllable frequency differs.

```bash
python -m corpus_analysis.pipeline analyze \
  --config ../configs/nepali_corpus_analysis.yaml \
  --input-root /data/nepali-corpora \
  --output-root ../dataset_analysis \
  --max-records-per-corpus 100000 --workers 1 --run-id smoke_workers_1

python -m corpus_analysis.pipeline analyze \
  --config ../configs/nepali_corpus_analysis.yaml \
  --input-root /data/nepali-corpora \
  --output-root ../dataset_analysis \
  --max-records-per-corpus 100000 \
  --workers "$CLOUD_WORKERS" --run-id smoke_all_workers

python -m corpus_analysis.pipeline compare-runs \
  --left ../dataset_analysis/smoke_workers_1 \
  --right ../dataset_analysis/smoke_all_workers
```

## Full analysis

Run the complete scan inside `tmux` so it survives an SSH disconnection:

```bash
tmux new -s nepali-analysis
CLOUD_WORKERS="$(nproc)"
python -m corpus_analysis.pipeline analyze \
  --config ../configs/nepali_corpus_analysis.yaml \
  --input-root /data/nepali-corpora \
  --output-root ../dataset_analysis \
  --workers "$CLOUD_WORKERS"
```

The command prints the timestamped output directory. It writes:

```text
dataset_analysis/<run-id>/
├── manifest.json
├── validation.json
├── per_corpus/<corpus>/
│   ├── summary.json
│   ├── syllable_frequency.csv
│   ├── syllable_frequency.parquet
│   ├── unobserved_syllables.txt
│   ├── quality_metrics.json
│   ├── metadata_breakdown.json
│   └── metadata_breakdown.csv
├── combined/source_native/
├── combined/deduplicated/
├── overlap/
│   ├── corpus_overlap.csv
│   ├── duplicate_contribution.csv
│   ├── unique_contribution.csv
│   └── native_vs_deduplicated.csv
├── figures/
├── logs/
│   ├── analysis.log
│   └── exact_dedup.sqlite3
└── derived_50k_comparison/
```

The fixed lookup tokenizer defines the vocabulary boundary. Consequently,
`unique_observed_syllables` is the number of Devanagari lookup entries actually
emitted, not an open-ended linguistic syllable discovery count. The lookup
file's whitespace token is retained in the manifest's raw token count but is
excluded from the analyzed syllable denominator. Review
`unobserved_syllables.txt` together with `unmatched_devanagari_ratio` to
distinguish unseen lookup entries from Devanagari content the tokenizer could
not recognize.

Exact deduplication uses SHA-256 of NFC-normalized, HTML-cleaned,
Devanagari-and-whitespace-only text. It removes exact normalized duplicates;
semantic and near-duplicate detection is deliberately outside the primary
analysis.

## Upload and retrieve the results

Upload the complete run to a GCS bucket before deleting the VM:

```bash
RUN_ID="<printed-run-id>"
BUCKET="gs://<your-bucket>/nepali-syllable-analysis/$RUN_ID"
gcloud storage rsync --recursive ../dataset_analysis/"$RUN_ID" "$BUCKET"
```

For a complete reproducible record, download the entire run directory. If only
the authoritative final findings are needed, retrieve these files:

```text
manifest.json
validation.json
combined/deduplicated/summary.json
combined/deduplicated/quality_metrics.json
combined/deduplicated/syllable_frequency.csv
combined/deduplicated/unobserved_syllables.txt
overlap/corpus_overlap.csv
overlap/duplicate_contribution.csv
overlap/unique_contribution.csv
overlap/native_vs_deduplicated.csv
```

The Parquet frequency table is equivalent to the CSV and is only needed for
columnar downstream analysis. The SQLite hash index and logs are useful for an
audit but are not required to present the final statistics.

## Repository validation

Run the bounded fixture before publishing code changes:

```bash
cd tokenizer/syllable-tokenizer/scripts
python -m unittest discover -s tests -v
python -m compileall -q corpus_analysis tests
```

The fixture exercises JSONL, Parquet, and streaming HTML readers; normalization;
single- and multi-worker equality; exact within- and cross-corpus deduplication;
report generation; validation invariants; and output checksums.
