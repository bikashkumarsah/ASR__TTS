# Pronunciation-Aware Syllable Tokenizer for Nepali

## Abstract
The Automatic Speech Recognition (ASR) has come up with significant advancements over the course of several decades, transitioning from a rule-based method to a statistical approach, and ultimately to the use of end-to-end (E2E) frameworks. This phenomenon continues with the progression of machine learning and deep learning methodologies. The E2E approach for ASR has demonstrated predominant success in the case of resourceful languages with larger annotated corpus. However, the accuracy is quite low for low-resourced languages such as Nepali. In this regard, language-specific tools such as tokenizers seem to play a vital role in improving the performance of the E2E model for low-resourced languages like Nepali. In this paper, we propose a pronunciation-aware syllable tokenizer for the Nepali language which improves the results of the E2E model. Our experiment confirm that the introduction of the proposed tokenizer yields better performance with the Character Error Rate (CER) 8.09% compared to other language-independent tokenizers.


## How it works?
The image below shows the working of the proposed tokenizer. The deails are explained in the [paper](https://aclanthology.org/2023.icon-1.4/) (Algorithm 1: Syllabic Tokenizer for
Nepali).
<p align="left">
  <img src="media/working.png" alt="Working" width="400"/>
</p>


### Generating the vocabulary lookup
The script `scripts/generate_vocab_lookup.py` generates the vocabulary lookup save to `dataset/nepali_syllable_vocab_lookup.vocab` file.

```bash
python scripts/generate_vocab_lookup.py
```

### Tokenier
The tokenizer is simple function that takes a sentence as input and returns the tokenized output. The script `scripts/syllabic_tokenizer.py` demonstrates the usage of the tokenizer.

```bash
python scripts/syllabic_tokenizer.py
```

Use tokenizer as library
```python
>>> import scripts.syllabic_tokenizer as tokenizer
>>> lookup = tokenizer.get_lookup_tokens()
>>> len(lookup)
1782
>>> tokens = tokenizer.tokenize('के छ तिम्रो हालचाल?', lookup)
>>> len(tokens)
12
>>> tokens
['के', ' ', 'छ', ' ', 'ति', 'म्रो', ' ', 'हा', 'ल', 'चा', 'ल', '?']
>>>
```


### Generating the token vocabulary for your dataset
Foe example the text dataset for the SLR54 can be generated using the script `scripts/extract_slr54_transcripts.py` as follows:

```bash
python scripts/extract_slr54_transcripts.py
```

To generate the vocabulary we can use the script `scripts/generate_vocab.py` as follows:

syntax:

```bash
python scripts/generate_vocab.py --input_file <input-file> --output_file <out-file> --lookup_file dataset/nepali_syllables_lookup.vocab
```

```bash
python scripts/generate_vocab.py --input_file dataset/oslr_transcripts/unique_transcripts.txt --output_file dataset/slr54.vocab --lookup_file dataset/nepali_syllables_lookup.vocab
```




## Citation
If you use this tokenizer, cite the [following paper](https://aclanthology.org/2023.icon-1.4/):
```
@inproceedings{rupak-raj-etal-2023-pronunciation,
    title = "Pronunciation-Aware Syllable Tokenizer for {N}epali Automatic Speech Recognition System",
    author = "Ghimire, Rupak Raj  and
      Bal, Bal Krishna  and
      Prasain, Balaram  and
      Poudyal, Prakash",
    editor = "D. Pawar, Jyoti  and
      Lalitha Devi, Sobha",
    booktitle = "Proceedings of the 20th International Conference on Natural Language Processing (ICON)",
    month = dec,
    year = "2023",
    address = "Goa University, Goa, India",
    publisher = "NLP Association of India (NLPAI)",
    url = "https://aclanthology.org/2023.icon-1.4/",
    pages = "36--43"
}
```

__Note__: The date of publication of the literature is 2023, but code is reproduced and published in 2026 May. Please use the date of publication of the literature for citation.