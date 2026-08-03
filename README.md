# ASR__TTS

Nepali Syllable Tokenizer & Balanced ASR Dataset Builder Pipeline.

## 📥 Dataset Download Instructions

To reproduce the candidate extraction from the **Nepali-Text-Corpus** (6.4M articles, 27.5 GB), run the following command on your machine/cloud instance:

```bash
# Install HuggingFace CLI if needed
pip install -U "huggingface_hub[cli]"

# Download the Nepali-Text-Corpus dataset
huggingface-cli download --repo-type dataset IRIIS-RESEARCH/Nepali-Text-Corpus --local-dir ./Nepali-Text-Corpus
```

Alternatively, using `hf`:
```bash
hf download --repo-type dataset IRIIS-RESEARCH/Nepali-Text-Corpus --local-dir ./Nepali-Text-Corpus
```

---

## ⚙️ Requirements & Installation

```bash
pip install -r tokenizer/syllable-tokenizer/requirements.txt
```

---

## 🚀 Running the Pipeline

### Quick Start (Full 10-Batch Build)

```bash
cd tokenizer/syllable-tokenizer/scripts

# Reset any previous state
python -m dataset_builder.pipeline reset

# Extract and run Batch 1 across all parquet shards
python -m dataset_builder.pipeline run-batch --batch-id 1 --target-size 5000 --max-corpus 0

# Run Batches 2 through 10
for b in {2..10}; do
  python -m dataset_builder.pipeline run-batch --batch-id $b --target-size 5000
done

# Merge into final 50k corpus
python -m dataset_builder.pipeline merge --output ../dataset/asr_corpus/corpus_50k.jsonl

# Check status
python -m dataset_builder.pipeline status
```
