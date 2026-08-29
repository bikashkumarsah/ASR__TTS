# Nepali Synthetic ASR: Existing Google Cloud VM + Kaggle GPU

This workflow turns the verified 20,000-text corpus into a quality-controlled
synthetic ASR package. It never creates or resizes a VM, GPU, bucket, or other
Google Cloud compute resource. The existing VM performs preparation, Gemini-TTS
synthesis, CPU audio checks, packaging, result import, acceptance, and SpeeChain
export. Kaggle runs transcription QC only. No ASR training command is executed by
the pipeline.

## Safety and acceptance boundaries

- Complete `legal_review.json` before the first paid TTS call. The gate records
  review of Google service terms, private Kaggle transfer, and academic/private use.
- Billing alerts are advisory. The pipeline additionally forecasts each phase,
  maintains a per-request cost ledger, reserves $5, and stops before its configured
  TTS allowance is exceeded.
- Gemini-TTS Nepali is a Preview service. Pronunciation is accepted only after CPU
  integrity checks, Whisper verification for every clip, and MMS-Nepali verification
  for every rare or disputed clip.
- The final report explicitly states that the audio has automated two-model
  validation but no native-speaker certification.
- SLR54 dev/test remain human-speech-only evaluation sets. Exact transcript overlap
  is excluded from the synthetic training manifests.

Provider references: [Gemini-TTS](https://docs.cloud.google.com/text-to-speech/docs/gemini-tts),
[Text-to-Speech quotas](https://docs.cloud.google.com/text-to-speech/quotas),
[pricing](https://cloud.google.com/text-to-speech/pricing),
[Google Cloud service terms](https://cloud.google.com/terms/service-terms),
[OpenSLR SLR54](https://www.openslr.org/54/),
[Kaggle dataset limits](https://www.kaggle.com/docs/datasets), and
[Kaggle notebook limits](https://www.kaggle.com/docs/notebooks).

## 1. Install on the existing VM

Run from the existing repository checkout. Do not clone another checkout if this
one already contains the prepared 20k input.

```bash
cd ~/ASR__TTS
git pull origin main
source .venv/bin/activate
python -m pip install -r tokenizer/syllable-tokenizer/requirements.txt
python -m pip install -r tokenizer/syllable-tokenizer/requirements-synthetic-asr.txt

sudo apt-get update
sudo apt-get install -y ffmpeg

cd tokenizer/syllable-tokenizer/scripts
CLOUD_WORKERS="$(nproc)"
WORK_ROOT="/data/nepali-synthetic-20k"
INPUT_20K="/data/input/final_20k_diverse_rare.jsonl"
```

Application Default Credentials and a Google project must already be authorized
for Cloud Text-to-Speech. The preflight checks access but never enables an API or
creates a resource. Configure Kaggle CLI credentials only on this VM; do not copy
Google credentials into the Kaggle dataset.

```bash
python -m dataset_builder.pipeline synthetic-asr-preflight \
  --work-dir "$WORK_ROOT" \
  --config ../configs/synthetic_asr_gemini.yaml

python -m json.tool "$WORK_ROOT/preflight.json" | less
```

If `ready` is false, fix only the reported permission, package, disk, or credential
issue on this existing VM and rerun preflight.

## 2. Prepare SLR54 and the 20k text

Download/extract SLR54 under `/data/slr54` so that one `utt_spk_text.tsv` and its
aligned WAV/FLAC files are present. The command checks archive hashes, alignment,
audio metadata, and speaker-disjointness.

```bash
python -m dataset_builder.pipeline prepare-slr54-speechain \
  --input-root /data/slr54 \
  --output-dir "$WORK_ROOT/slr54" \
  --seed 42 --workers "$CLOUD_WORKERS" --resume

python -m dataset_builder.pipeline prepare-synthetic-asr \
  --input "$INPUT_20K" \
  --slr54-manifest "$WORK_ROOT/slr54/manifest.json" \
  --config ../configs/synthetic_asr_gemini.yaml \
  --output-dir "$WORK_ROOT" \
  --workers "$CLOUD_WORKERS" --resume
```

Review the preparation quarantine. The exact-20k gate intentionally stops if any
input is ambiguous, duplicated, over the provider byte limit, or loses a required
rare syllable. Correct the source record or its approved spoken form; do not silently
drop it.

Copy and complete the legal gate only after reviewing the referenced terms:

```bash
cp ../configs/legal_review.example.json "$WORK_ROOT/legal_review.json"
${EDITOR:-nano} "$WORK_ROOT/legal_review.json"
```

## 3. Audition, qualify, pilot, and synthesize

TTS calls are quota-bound. `--workers` provides CPU capacity for conversion, while
the shared limiter defaults to the configured ceiling. When a project has a lower
quota, pass a lower runtime value such as `--requests-per-minute 5`; an override may
lower but never raise the checked-in safety ceiling and does not invalidate prepared
checkpoints.

```bash
python -m dataset_builder.pipeline synthesize-synthetic-asr \
  --run-dir "$WORK_ROOT" --phase audition --max-usd 100 \
  --workers "$CLOUD_WORKERS" --resume

python -m dataset_builder.pipeline validate-synthetic-audio \
  --run-dir "$WORK_ROOT" --stage cpu --workers "$CLOUD_WORKERS" --resume
```

Export and run Kaggle QC as described below. Import the audition results; this ranks
voices by Whisper CER, rare-syllable recall, CPU integrity, and WavLM speaker-vector
separation, then selects four female and four male voices. Pilot/full synthesis will
not proceed without this qualification file.

```bash
python -m dataset_builder.pipeline import-kaggle-qc \
  --run-dir "$WORK_ROOT" --results-dir /data/kaggle_results/audition --resume

python -m dataset_builder.pipeline synthesize-synthetic-asr \
  --run-dir "$WORK_ROOT" --phase pilot --max-usd 100 \
  --workers "$CLOUD_WORKERS" --resume

python -m dataset_builder.pipeline validate-synthetic-audio \
  --run-dir "$WORK_ROOT" --stage cpu --workers "$CLOUD_WORKERS" --resume
```

Run/export/import the pilot QC and inspect `reports/` before authorizing the full
phase. Then run:

```bash
python -m dataset_builder.pipeline synthesize-synthetic-asr \
  --run-dir "$WORK_ROOT" --phase full --max-usd 100 \
  --workers "$CLOUD_WORKERS" --resume

python -m dataset_builder.pipeline validate-synthetic-audio \
  --run-dir "$WORK_ROOT" --stage cpu --workers "$CLOUD_WORKERS" --resume
```

## 4. Private Kaggle QC exchange

Use a fresh export directory for each stage. The exporter includes only synthesized
recordings that have not already completed their required Kaggle checks, writes fewer
than 50 top-level audio archives, checksums every shard, and includes the notebook and
pinned tokenizer. It does not upload anything itself.

```bash
KAGGLE_INPUT="$WORK_ROOT/kaggle_input_full"
python -m dataset_builder.pipeline export-kaggle-qc \
  --run-dir "$WORK_ROOT" --output-dir "$KAGGLE_INPUT" \
  --shard-size-gb 4 --workers "$CLOUD_WORKERS"
```

Edit only the `id` in `dataset-metadata.json`, keep `isPrivate: true`, then upload
with the authenticated Kaggle CLI:

```bash
kaggle datasets create -p "$KAGGLE_INPUT" --private
```

Create a Kaggle GPU notebook, attach that private dataset, and open
`kaggle_synthetic_asr_qc.ipynb`. Whisper runs on every exported clip. MMS with the
`npi` adapter runs on rare-tail clips, Whisper outliers, missing-rare cases, and the
threshold-dispute band. If one 12-hour session is insufficient, set non-overlapping
`START_SHARD` and `END_SHARD` values and save each compact output separately.

Download only these four outputs from each session:

- `whisper_results.parquet`
- `mms_results.parquet`
- `kaggle_qc_summary.json`
- `kaggle_manifest.json`

Place all session output directories below `/data/kaggle_results/full`, then import:

```bash
python -m dataset_builder.pipeline import-kaggle-qc \
  --run-dir "$WORK_ROOT" --results-dir /data/kaggle_results/full --resume
```

The importer rejects duplicate IDs, unknown IDs, checksum mismatches, wrong model
identities, and missing required MMS coverage.

## 5. Final acceptance and delivery

```bash
python -m dataset_builder.pipeline finalize-synthetic-asr \
  --run-dir "$WORK_ROOT" --workers "$CLOUD_WORKERS"
```

Finalization writes no accepted manifest unless all invariants pass: exactly 20,000
canonical recordings, all 1,348 spoken syllables, three distinct accepted voices for
every below-20 syllable, unique audio hashes, complete model coverage, no evaluation
overlap in training metadata, and tracked TTS spend within budget. The source-text
inventory remains 1,358 lookup types; the other ten are Devanagari digit glyphs
(`०`–`९`) that spoken-form normalization expands into words and therefore excludes
from the phonetic coverage denominator.

The files to retain or download are:

```text
final/canonical_20k.jsonl
final/rare_voice_extras.jsonl
final/accepted_all.jsonl
final/manifest.json
audio/train_16k/full/
reports/final_report.json
reports/canonical_syllable_distribution.csv
reports/canonical_syllable_distribution.parquet
qc/cpu_metrics.jsonl
qc/whisper_results.jsonl
qc/mms_results.jsonl
speechain/
synthetic_asr_config.yaml
legal_review.json
prepare.complete.json
```

SpeeChain recipe YAMLs are generated for seeds 42, 43, and 44 for real-only,
real-plus-synthetic, synthetic pretraining, and real fine-tuning. They are preparation
artifacts only; the pipeline contains no training invocation.

Before a later training experiment, install the exported adapter into the existing
SpeeChain checkout (first with `--check`, then without it):

```bash
python "$WORK_ROOT/speechain/integration/install_adapter.py" \
  --speechain-root ~/ASR__TTS/SpeeChain --check
python "$WORK_ROOT/speechain/integration/install_adapter.py" \
  --speechain-root ~/ASR__TTS/SpeeChain
```
