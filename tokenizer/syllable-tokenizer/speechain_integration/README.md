# SpeeChain syllable integration

This bundle is tracked in the ASR/TTS repository because a local `SpeeChain/`
checkout is intentionally ignored. It installs the checksum-pinned
`SyllableTokenizer` into an existing SpeeChain checkout and enables
`token_type: syllable` in ASR and LM models.

Check compatibility, then install:

```bash
python tokenizer/syllable-tokenizer/speechain_integration/install_adapter.py \
  --speechain-root /path/to/SpeeChain --check

python tokenizer/syllable-tokenizer/speechain_integration/install_adapter.py \
  --speechain-root /path/to/SpeeChain
```

The installer is idempotent and saves the original model files beside them with
the suffix `.synthetic-asr.bak`. It does not run training. Concrete experiment
recipes are generated only after final dataset acceptance.

After later inference, use `evaluate_results.py` with the exported token directory,
reference/hypothesis `idx2text` files, and `speechain/evaluation/rare_syllables.txt`
to calculate CER, WER, syllable error rate, and rare-syllable recall consistently.
