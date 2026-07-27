# Thesis-ready dataset and output length statistics

## Suggested revision to Section 3.6 (Dataset)

The experiments use the BioLaySumm 2025 dataset, which contains biomedical articles paired with human-written lay summaries from PLOS and eLife. PLOS summaries are written by article authors, whereas eLife summaries are prepared by editors in consultation with the authors. The complete PLOS dataset contains 24,773 training, 1,376 validation, and 142 test articles; the corresponding eLife splits contain 4,346, 241, and 142 articles, respectively. The main experiment uses 284 articles sampled from the validation splits (142 per source), stratified by reference-summary word count. Table 3.3 reports overall length statistics for the experimental sample, the direct-generation baseline, and the final ATLAS summaries. All lengths are whitespace-delimited word counts, and SD denotes the sample standard deviation (n - 1).

Across all 284 articles, the source texts contain 8,351.44 words on average (SD = 3,351.21; range = 1,990--23,048), while the reference summaries contain 290.57 words on average (SD = 112.63; range = 77--672).

Direct generation produces substantially longer and more variable summaries than ATLAS. The direct-generation means are 322.24 words for GPT-4.1 Mini, 429.45 for Gemini 3 Flash Preview, and 503.39 for Gemini 2.5 Flash. The corresponding ATLAS means are 188.81, 203.48, and 190.06 words. ATLAS also has markedly smaller standard deviations (10.08--16.10 words, compared with 46.23--97.22 words for direct generation), reflecting the framework's dataset-aware length constraints.

## Table 3.3. Overall word-count statistics for the experimental sample and system outputs

| Text | System | N | Mean (SD) | Min--Max |
|---|---|---:|---:|---:|
| Source article | Human-authored source | 284 | 8,351.44 (3,351.21) | 1,990--23,048 |
| Reference lay summary | Human-written reference | 284 | 290.57 (112.63) | 77--672 |
| Direct-generation summary | GPT-4.1 Mini | 284 | 322.24 (65.96) | 187--785 |
| Direct-generation summary | Gemini 2.5 Flash | 284 | 503.39 (97.22) | 240--782 |
| Direct-generation summary | Gemini 3 Flash Preview | 284 | 429.45 (46.23) | 319--567 |
| ATLAS final summary | GPT-4.1 Mini | 284 | 188.81 (10.08) | 175--245 |
| ATLAS final summary | Gemini 2.5 Flash | 284 | 190.06 (10.84) | 175--242 |
| ATLAS final summary | Gemini 3 Flash Preview | 284 | 203.48 (16.10) | 175--244 |

*Note.* Statistics are calculated on the 284 validation articles used in the main experiment (142 PLOS and 142 eLife). Word counts use the same whitespace-delimited definition as the ATLAS pipeline. SD is the sample standard deviation (n - 1). Direct-generation and ATLAS rows use the same 284 article IDs. ATLAS values refer to the final rewritten summary.

## Scope note

Keep the existing Table 3.2 for full train/validation/test split sizes, and insert this as Table 3.3. Only overall statistics are shown; they describe the actual 284-article experimental sample, not every article in the full training and validation pools.
