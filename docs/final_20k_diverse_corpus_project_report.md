# Technical Project Report: Diverse and Rare-Aware Nepali Syllable Corpus

**Report date:** 28 August 2026  
**Verified run ID:** `20260824T102044Z`  
**Current deliverable:** 20,000-sentence pilot corpus  
**Long-term target:** 50,000-sentence final corpus  
**Project area:** Nepali ASR/TTS corpus design and syllable coverage

## Executive summary

This project developed and validated a cloud-CPU pipeline for constructing a
Nepali text corpus that combines complete source-supported syllable coverage,
stronger representation of rare syllables, and reduced lexical and semantic
redundancy. The pipeline processed five public Nepali text sources, performed
global exact deduplication, created a bounded 1.2-million-sentence candidate
shortlist, identified lexical near-duplicates with MinHash-LSH, and used
quantized multilingual E5 embeddings with FAISS-based selection.

The verified pilot contains exactly 20,000 unique normalized sentences and
991,876 recognized syllable-token occurrences. It covers all 1,358 syllables
that the corrected tokenizer found in eligible source sentences. Every
construction invariant passed: the output size is exact, normalized hashes are
unique, attainable coverage is complete, the rare-support floor is satisfied,
all emitted tokens belong to the pinned vocabulary, and recomputed frequency
totals agree with the published tables.

The central result is that coverage increased from the earlier reported 1,182
types to 1,358 primarily because a tokenizer defect was corrected. The earlier
implementation searched no more than four Unicode code points at each
position. The corrected implementation derives its search window from the
longest pinned vocabulary entry. Of the current 1,358 observed types, 175 are
five- or six-code-point entries that the old tokenizer could not emit as whole
tokens. The increase is therefore mainly a correction in measurement and
segmentation rather than the discovery of 176 previously unknown linguistic
syllables.

The run is a successful technical pilot, not the final project endpoint. Its
successful report did not include a size-matched historical baseline, and the
final 50,000-sentence construction and comparison remain to be completed.

## 1. Project objective

The project aims to construct an ASR-oriented Nepali text corpus with four
properties:

1. Complete coverage of every lookup-vocabulary syllable supported by eligible
   source sentences.
2. A minimum representation floor for rare syllables whenever sufficient
   distinct supporting sentences exist.
3. Lower lexical and semantic repetition than frequency-only selection.
4. An ASR-realistic frequency structure based on a square-root-tempered source
   distribution rather than an artificial uniform distribution.

Normalized Shannon entropy, the Gini coefficient, and Jensen-Shannon
divergence from the tempered target are the principal distribution metrics.
The coefficient of variation is retained only as a diagnostic because it is
strongly affected by the deliberately preserved rare tail.

## 2. Data sources and analytical scope

The candidate inventory incorporates provenance from five public corpora:

| Source | Recorded revision or provenance |
|---|---|
| IRIIS-RESEARCH/Nepali-Text-Corpus | `167b56b327996ca2c5fe7f9a6b638aecbb2ae13b` |
| Sakonii/nepalitext-language-model-dataset | `9cdc23864fa3f8c5c96059524bb46fcdb40d3a53` |
| himalaya-ai/cc100-nepali | `478da53d02ab0bfb9dde90ec38e7cc514426fbd1` |
| IEEE DataPort `compiled.txt` | Existing local input; download was skipped in the recorded run |
| Boredoom17/Nepali-Corpus | `be2b638c2802f4587614c34bd4f7cff430276412` |

The source-native stream contained 233,478,947 eligible record
contributions. Exact normalized-text hashing reduced this to 63,863,537 unique
eligible sentences, removing 169,615,410 duplicate contributions. The exact
duplicate rate was therefore 72.65%. This high rate is consistent with known
source relationships: Boredoom17 contains IRIIS-derived material, while
Sakonii contains CC100-derived material.

The authoritative combined inventory is the exact-deduplicated union. Source
memberships are still retained on each selected record so that cross-source
overlap remains auditable.

## 3. Tokenizer and vocabulary control

The only authoritative lookup vocabulary is
`dataset/nepali_syllables_lookup.vocab`. The verified run recorded:

| Vocabulary layer | Entries |
|---|---:|
| Raw file entries | 1,782 |
| Non-whitespace analytical entries | 1,781 |
| Selectable entries after excluding `।`, `?`, and `!` | 1,778 |
| Structurally emittable entries | 1,778 |
| Source-supported attainable entries | 1,358 |

The vocabulary SHA-256 is
`2f5c5ef869018f9af151a853a0db42fadfd38f0ee1f06de8e6a9a72b39b417c5`.
The tokenizer source SHA-256 is
`c4ca12d037db6c6af43523d4f453b119eaadbbc516d07aaf8f6a3fb4f2afce30`.
Both hashes are included in preparation, embedding, selection, reporting, and
resume-state validation.

### 3.1 Explanation of the coverage increase

The old tokenizer used a fixed four-code-point longest-match window. The new
tokenizer calculates the window from the longest vocabulary entry, which is
seven code points. The current observed inventory has the following length
distribution:

| Unicode code-point length | Observed syllable types |
|---|---:|
| 1–4 | 1,183 |
| 5 | 132 |
| 6 | 43 |
| **Total** | **1,358** |

Thus, 175 currently observed syllables are longer than the old tokenizer's
maximum search window. Removing these longer entries leaves 1,183, essentially
the earlier 1,182 raw-scan count.

The archived full-corpus analysis dated `20260817T161231Z` reports 1,193 rather
than 1,182 in its exact-deduplicated combined summary. Direct set comparison
between that archived result and the corrected inventory shows 175 newly
emittable long entries and 10 entries no longer emitted or selected, giving a
net increase of 165 to 1,358. All 175 additions are five or six code points
long. This confirms that tokenizer correction, not corpus selection alone, is
the principal cause of the increase.

Examples of newly emittable entries include `क्राँ`, `क्रिँ`, `क्ष्रि`,
`त्त्र`, `द्ध्र`, `द्यौँ`, `श्र्रि`, and `ह्रौँ`.

## 4. Corpus-construction methodology

### 4.1 Normalization and eligibility

Each text was HTML-unescaped, cleaned to Devanagari and whitespace, and
whitespace-normalized. Eligible sentences contain between 5 and 80 recognized
syllable tokens. SHA-256 of normalized text provides exact global
deduplication while retaining every contributing source name.

### 4.2 Candidate preparation

The source data were processed in three deterministic streaming passes:

1. Count sentence and syllable occurrence/document frequencies.
2. Establish the source-supported attainable syllable inventory.
3. Construct a bounded 1.2-million-record shortlist with rare-syllable
   support.

The shortlist represents 1.88% of the 63.86-million-sentence exact-deduplicated
eligible pool. Bounded streaming and disk-backed state prevented the full
source collection from being loaded into memory.

### 4.3 Lexical redundancy control

Character five-gram MinHash signatures used 128 permutations. MinHash-LSH
generated candidate pairs, followed by exact Jaccard verification at a
threshold of 0.85. This stage identified 61,610 lexical near-duplicate
candidates, equivalent to 5.13% of the shortlist.

### 4.4 Semantic representation and calibration

Candidate sentences were encoded as `query: <normalized_text>` using
`intfloat/multilingual-e5-small`, revision
`614241f622f53c4eeff9890bdc4f31cfecc418b3`. The model was exported to ONNX
and dynamically quantized to INT8. The quantized model checksum was
`f2a98a9b22860770fec52cb192a0754d691130ec385b62a1dc420f4bd008e999`.

A 100,000-sentence calibration sample was used to characterize the empirical
cosine distribution. Raw normalized embeddings were retained as the calibrated
representation. The derived cosine thresholds were:

| Background upper-tail allowance | Cosine threshold |
|---:|---:|
| 2.0% | 0.88273853 |
| 1.0% | 0.88957322 |
| 0.5% | 0.89568830 |

The thresholds are empirical for this model and Nepali candidate pool; they
are not interpreted as universal semantic-similarity constants.

### 4.5 Rare-aware diverse selection

The selector used exact FAISS inner-product search and clustering over the
calibrated embeddings. Selection scores combined:

| Objective | Weight |
|---|---:|
| Tempered syllable deficit and rarity gain | 0.60 |
| Semantic novelty | 0.35 |
| Metadata underrepresentation | 0.05 |

The target syllable distribution was proportional to the square root of the
source occurrence frequency. Every attainable syllable was required to appear,
with a minimum target of `min(5, distinct supporting sentences)`. Coverage and
rare-floor records could override lexical or semantic thresholds, but every
override was recorded.

## 5. Verified final results

### 5.1 Construction and integrity

| Validation item | Result |
|---|---:|
| Output records | 20,000 |
| Unique record IDs | 20,000 |
| Unique normalized hashes | 20,000 |
| Unique normalized texts | 20,000 |
| JSON parsing failures | 0 |
| Text-hash mismatches | 0 |
| Syllable-count mismatches | 0 |
| Report artifact checksum failures | 0 of 19 |
| CSV/Parquet frequency agreement | Exact |

The output JSONL SHA-256 is
`89d1877e56c2337713b31f5c0049625ac43be2d9195fc0f3b5bdbb32c7084bc0`.

All six required construction invariants passed:

- exact output size;
- unique normalized hashes;
- complete attainable coverage;
- rare-frequency floor satisfied;
- every token belongs to the pinned selectable vocabulary; and
- frequency totals equal the recognized-token total.

### 5.2 Coverage

| Coverage layer | Result |
|---|---:|
| Tokenizer structural coverage | 1,778 / 1,778 (100%) |
| Five-corpus source support | 1,358 / 1,778 (76.38%) |
| Final coverage of attainable inventory | 1,358 / 1,358 (100%) |
| Vocabulary entries without source support | 420 |

The final corpus therefore contains every selectable lookup syllable observed
in eligible source text. The 420 uncovered entries are absent from the source
inventory and cannot be recovered through selection alone.

Source evidence is extremely limited for a portion of the inventory:

| Distinct supporting source sentences | Syllable types |
|---:|---:|
| 1 | 65 |
| 2--4 | 88 |
| 5--9 | 62 |
| 10--99 | 204 |
| At least 100 | 939 |

No rare-floor violation was found.

### 5.3 Syllable-frequency distribution

| Metric | Result |
|---|---:|
| Recognized syllable tokens | 991,876 |
| Observed syllable types | 1,358 |
| Mean occurrences per type | 730.39 |
| Median occurrences per type | 36 |
| Normalized Shannon entropy | 0.71370910 |
| Gini coefficient | 0.89416726 |
| Jensen-Shannon divergence from tempered target | 0.08131945 |
| Coefficient of variation, diagnostic | 3.7296 |
| Hapax types | 62 |
| Types with fewer than 5 occurrences | 152 |
| Types with fewer than 10 occurrences | 455 |
| Types with fewer than 20 occurrences | 561 |

The five most frequent syllables are `र` (31,852), `न` (27,877), `को`
(27,614), `स` (25,957), and `मा` (24,865).

The distribution remains strongly long-tailed:

| Cumulative token share | Required syllable types |
|---:|---:|
| 50% | 32 |
| 80% | 106 |
| 90% | 180 |
| 95% | 270 |
| 99% | 604 |

The ten most frequent types account for 24.11% of all tokens, and the top 50
account for 62.12%. The remaining 754 observed types beyond the first 604
collectively contribute only 1% of token occurrences. Complete coverage has
therefore been achieved without eliminating the natural common-syllable core.

### 5.4 Sentence-length characteristics

| Statistic | Syllables | Whitespace-delimited words | Characters |
|---|---:|---:|---:|
| Minimum | 5 | 1 | 7 |
| Median | 51 | 15 | 100 |
| Mean | 49.59 | 14.94 | 97.76 |
| 95th percentile | 78 | 24 | 154 |
| Maximum | 80 | 63 | 181 |

All 20,000 texts are NFC-normalized and contain only Devanagari-block
non-whitespace characters. No Latin text, URL, email address, Unicode
replacement character, or control character was detected.

### 5.5 Lexical and semantic diversity

The final corpus contains no exact text duplicates. It uses 6,659 semantic
clusters, and no cluster contributes more than nine selected sentences.

| Diversity metric | Result |
|---|---:|
| Exact duplicate records | 0 |
| Records carrying a pool lexical-near-duplicate flag | 664 (3.32%) |
| Rare-coverage selections | 2,404 (12.02%) |
| Recorded similarity exceptions | 2,263 (11.32%) |
| Exception rows with a direct lexical conflict | 24 |
| Selector-model nearest-neighbour median | 0.89290559 |
| Selector-model nearest-neighbour 95th percentile | 0.91687852 |
| Selector-model nearest-neighbour 99th percentile | 0.94477093 |

The 2,263 exceptions span 601 required rare syllables. Most exceptions are
semantic-threshold overrides rather than direct lexical conflicts. Inspection
of the highest-similarity cases shows that rare coverage sometimes preserves
near-identical sentences that differ only in spacing, inflection, or Unicode
nasalization. This is a deliberate coverage trade-off, but it also demonstrates
the need for linguistic review of rare types.

### 5.6 Source provenance

Source membership counts overlap because an exactly deduplicated sentence can
occur in several published corpora:

| Source membership | Records | Share of final records |
|---|---:|---:|
| Boredoom17 Nepali Corpus | 15,652 | 78.26% |
| IRIIS Nepali Text Corpus | 14,763 | 73.82% |
| Sakonii Nepali LM dataset | 4,774 | 23.87% |
| CC100 Nepali | 3,771 | 18.86% |
| IEEE compiled corpus | 615 | 3.08% |

Because memberships overlap, these percentages intentionally sum to more than
100%. In total, 86.17% of selected records have exactly two source memberships.
The dominant combinations are IRIIS plus Boredoom17 (13,944 records, 69.72%)
and CC100 plus Sakonii (3,186 records, 15.93%).

The primary stored source assignment is more concentrated: IRIIS supplies
14,763 records (73.82%), Sakonii 4,188 (20.94%), Boredoom17 768 (3.84%), IEEE
250 (1.25%), and CC100 31 (0.16%). This concentration is expected because the
design retains source realism and imposes no hard source quota.

### 5.7 Metadata composition

Metadata balance was a soft objective. The resulting distribution is:

| Axis | Largest category | Share | Smallest category | Share |
|---|---|---:|---|---:|
| Tense | Mixed | 43.15% | Future | 4.14% |
| Polarity | Neutral | 63.09% | Positive | 16.70% |
| Gender | Neutral | 68.04% | Feminine | 11.47% |
| Sector | News | 53.88% | Education | 1.68% |

News and formal text together account for 76.89% of the corpus. This is
consistent with the source mixture and current no-quota design. It is not a
balanced-domain corpus and should not be described as such.

## 6. Cloud execution and resource use

The verified run executed on Google Cloud Linux with Python 3.12.3. It
requested 12 workers and used a fixed embedding topology of six processes with
two CPU threads per process, thereby using all 12 available CPU cores.

| Operational metric | Result |
|---|---:|
| Candidate shortlist | 1,200,000 sentences |
| Chosen embedding topology | 6 processes x 2 threads |
| Benchmark throughput | 280.259 sentences/second |
| Full embedding elapsed time | 3,530.685 seconds (58.84 minutes) |
| Estimated embedding-worker RSS | 11.99 GiB |
| Successful finalize peak RSS | 14.48 GiB |
| Highest peak across delivered run attempts | 15.44 GiB |
| FAISS threads | 12 |

The result confirms that the workflow is CPU-scalable and memory-safe on the
12-vCPU/64-GB resource envelope. Additional RAM alone would not increase
syllable coverage; coverage is controlled by tokenizer correctness, source
support, and candidate selection.

## 7. Interpretation of the supplied figures

The syllable frequency-rank figure shows a smooth decline on logarithmic axes,
from frequencies above 30,000 for the most common syllables to a forced
count-one tail. The absence of isolated spikes indicates a coherent
long-tailed distribution, while the steep slope confirms that uniform balance
was neither achieved nor intended.

The cumulative token-coverage figure rises sharply: 32 types explain half the
tokens and 106 types explain 80%. It then approaches one slowly as hundreds of
rare types are added. This figure is the clearest visual justification for the
rare-aware selection stage: ordinary frequency-weighted sampling could preserve
most token mass while still omitting a large fraction of the syllable
inventory.

## 8. Reproducibility and validation status

The run records the vocabulary checksum, tokenizer checksum, configuration
fingerprint, source revisions, model revision, quantized-model checksum, random
seed, worker topology, Python version, output checksum, report checksum, and
checksums for all 19 report artifacts.

Independent post-run validation confirmed:

- all 20,000 JSONL rows parse successfully;
- stored text hashes equal freshly computed hashes;
- all record IDs, normalized hashes, texts, and pool IDs are unique;
- the JSONL and report hashes match the manifest;
- all 19 artifact checksums match;
- recomputed token and type frequencies equal both CSV and Parquet tables;
- independently recomputed entropy, normalized entropy, Gini, and
  Jensen-Shannon divergence match the report; and
- the CSV and Parquet syllable tables contain the same 1,781 analytical rows.

## 9. Limitations and risks

### 9.1 Pilot size and missing accepted baseline

The accepted run contains 20,000 sentences. Its successful report has no
baseline comparison. A preserved failed report compares this 20k candidate
against the previous 50k corpus and fails the target-JSD criterion, but that
comparison is size-mismatched and is not a valid final acceptance test.

The project must not claim that the pilot improves all balance and diversity
metrics over the historical 50k corpus until both datasets are retokenized and
compared at the same 50,000-record size.

### 9.2 Linguistic validity of the rare tail

Source-observed does not necessarily mean linguistically valid. Some rare
entries contain unusual conjunct, anusvara, chandrabindu, or spelling variants.
The 62 hapax types and the high-similarity rare-coverage exceptions should be
manually classified as valid form, named entity, foreign transliteration,
orthographic variant, or source noise before the final 50k build.

### 9.3 Provenance completeness

The run manifest says the IEEE download was skipped, while the inventory and
final records contain IEEE provenance. This is compatible with reusing an
existing local IEEE file, but the delivery does not include that file's SHA-256
or the complete `source_manifest.json`. Those artifacts should be included in
the final reproducibility package.

### 9.4 Delivery-package size

The compressed delivery is approximately 809 MB because the transformed
embedding matrix alone is approximately 881 MB uncompressed. A result-only
archive for supervision and publication should exclude embeddings and
selection checkpoints while retaining the final JSONL, report, manifest,
frequency tables, figures, exceptions, source distributions, and logs.

### 9.5 Metadata imbalance

The final pilot is dominated by news/formal text and neutral metadata labels.
This is acceptable for the present ASR-realistic objective, but it is unsuitable
if balanced domains, gender labels, sentiment, or tense become hard downstream
requirements.

## 10. Recommendations for the final 50k run

1. Conduct a linguistic audit of the 62 hapax types and all suspicious
   five- and six-code-point additions before freezing the attainable inventory.
2. Build exactly 50,000 unique records using the verified pool, model,
   vocabulary, tokenizer, and fixed CPU topology.
3. Retokenize the historical 50k baseline with the same pinned tokenizer and
   compare both datasets over identical shared support.
4. Require the new 50k to retain complete validated attainable coverage and the
   rare-support floor.
5. Apply the predefined weight profiles only if the default `60/35/5` profile
   fails the size-matched acceptance criteria.
6. Record the Git commit SHA and include the actual configuration, vocabulary,
   tokenizer source, source manifest, and IEEE input checksum in the final
   delivery.
7. Store failed attempts under a separate `failed_attempts/` directory so they
   cannot be confused with the accepted report.
8. Produce two archives: a compact result package for review and a complete
   checkpoint package for reproducibility and resume.

## 11. Conclusion

The project has successfully demonstrated a reproducible, CPU-based method for
constructing a diverse and rare-aware Nepali corpus from a very large,
overlapping multi-source text collection. The verified 20k pilot is
exact-deduplicated, internally consistent, and complete over all 1,358 corrected
source-supported syllables. It preserves the realistic high-frequency core
while explicitly protecting the rare tail and controlling redundancy with
lexical and semantic methods.

The result also establishes an important methodological correction: the
earlier 1,182-syllable estimate was constrained by a four-code-point tokenizer
window. The corrected tokenizer exposes longer vocabulary entries and provides
a more defensible attainable inventory. The remaining work is to validate the
linguistic quality of the rarest types, close the IEEE provenance gap, and run
the final size-matched 50k construction and comparison.

## 12. Key deliverables

- `final_20k_diverse_rare.jsonl`: verified pilot corpus.
- `report.json`: machine-readable metrics and construction acceptance.
- `manifest.json`: checksums, model revision, tokenizer fingerprint, and
  execution environment.
- `syllable_frequency.csv` and `syllable_frequency.parquet`: complete
  analytical vocabulary distribution.
- `coverage_exceptions.csv`: auditable coverage-driven similarity overrides.
- `similarity_metrics.json`: empirical calibration and final nearest-neighbour
  statistics.
- `source_distribution.*` and `metadata_distribution.*`: provenance and
  assigned metadata composition.
- `technical_progress_report.md`: chronological project record.

## References

1. Ghimire, R. R., Bal, B. K., Prasain, B., and Poudyal, P. (2023).
   [Pronunciation-Aware Syllable Tokenizer for Nepali Automatic Speech
   Recognition System](https://aclanthology.org/2023.icon-1.4/). Proceedings of
   ICON 2023, pp. 36–43.
2. Wang, L. et al. (2024). [Text Embeddings by Weakly-Supervised Contrastive
   Pre-training](https://arxiv.org/abs/2212.03533), arXiv:2212.03533.
3. Johnson, J., Douze, M., and Jegou, H. (2017). [Billion-scale similarity
   search with GPUs](https://arxiv.org/abs/1702.08734), arXiv:1702.08734.
