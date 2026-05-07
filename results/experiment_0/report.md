# Experiment 0: Test Bank Quality Control — Final Report

**Completion**: 2026-05-07T19:43:25  
**Final test bank size**: 325 questions (out of 509 generated)

## Executive Summary

This experiment builds and validates a benchmark of 1T3F multiple-choice questions
derived from BioLaySumm 2025 lay summaries (PLOS + eLife). The pipeline applies
three independent quality-control tests — Setup Gold (logical uniqueness),
Setup Blind (distractor strength), and Setup Counterfactual (context-faithfulness)
— and filters the question bank to retain only items that pass strict criteria.

Starting from 40 articles -> 509 atomic facts -> 509 multiple-choice questions ->
final bank of 325 questions after QC filtering.

Headline metrics:
- ACC_Gold = 97.05% (logical uniqueness)
- ACC_CF_Faithful (gpt-4.1) = 87.50% raw / 100.0% in final bank
- ACC_CF_Faithful (gpt-4o-mini) = 88.78% raw / 100.0% in final bank
- Total cost: $6.547997

## 1. Methodology

Stages: AF extraction -> 1T3F generation -> Setup Gold -> Setup Blind -> Setup Counterfactual -> strict filtering.

## 2. Data

BioLaySumm 2025 PLOS + eLife, each 20 validation articles, seed=42.

## 3. Results

### 3.1 Atomic Fact Extraction (Stage 1)
- Total AFs: 509
- Cost: $0.026402

### 3.2 Question Generation (Stage 2)
- Final generated questions (v2): 509
- Strategy distribution: {'direction_reversal': 479, 'entity_swap': 506, 'detail_fabrication': 430, 'numerical_perturbation': 112}
- Cost: $2.18096

### 3.3 Setup Gold (Stage 3)
| Source | n | Correct | ACC | 95% CI |
|---|---:|---:|---:|---:|
| PLOS | 509 | 494 | 0.9831 | 0.9520-0.9820 |
| eLife | 509 | 494 | 0.9639 | 0.9520-0.9820 |
| Overall | 509 | 494 | 0.9705 | 0.9520-0.9820 |

### 3.4 Setup Blind (Stage 4)
- gpt-4.1 ACC: 0.8802
- gpt-4o-mini ACC: 0.8271
- Random baseline: 0.25

### 3.5 Setup Counterfactual (Stage 5)
- v1 faithful: gpt-4.1=0.6870, gpt-4o-mini=0.6896
- v2 faithful: gpt-4.1=0.8750, gpt-4o-mini=0.8878

### 3.6 Filtering & Final Test Bank (Stage 6)
```
509 questions
  |
  |- x Failed Setup Gold:       15
  v
494
  |
  |- x No counterfactual built:  115
  |    (no_entity_swap: 9, entity_not_found: 96, no_clean_swap: 2, low_confidence: 7, no_variant_mapping: 1, no_replacement: 0)
  v
379
  |
  |- x CF parse error:           0
  |- x Prior on gpt-4.1 only:    12
  |- x Prior on gpt-4o-mini only:9
  |- x Prior on both:            32
  |- x Other (any model):        1
  v
Final test bank: 325
```

### 3.7 Per-Source Distribution in Final Bank
| Source | Final count | % of bank |
|---|---:|---:|
| PLOS | 105 | 32.31% |
| eLife | 220 | 67.69% |

### 3.8 Per-Strategy Distribution in Final Bank
| Strategy | # distractors in bank |
|---|---:|
| numerical_perturbation | 75 |
| entity_swap | 329 |
| direction_reversal | 303 |
| detail_fabrication | 268 |

## 4. Limitations

1. Counterfactual modification is restricted to entity_swap distractors; other
   strategies (numerical, direction_reversal, detail_fabrication) are not
   counterfactually validated.
2. Variant detection may still miss obscure abbreviations.
3. Setup Blind in lay-summary domain can remain higher than chance due to overlap
   between source content and pretraining knowledge.
4. AF extraction remains dependent on LLM decontextualization quality.

## 5. Final Test Bank Schema

See `results/experiment_0/final_test_bank.jsonl`.

## 6. Use in Downstream Experiments

For Exp 1 / Exp 3:
- Use `lay_summary` as source-of-truth context
- Use `correct_letter` as gold answer
- `counterfactual_lay_summary` and `promoted_distractor_letter` are available for robustness checks

## 7. Reproducibility

- Random seed: 42
- Sampling indices: see `data/raw/sampling_metadata.json`
- All prompts: see `prompts/`
- Filter trace: see `results/experiment_0/filter_trace.csv`

## Total Cost

$6.547997 USD
