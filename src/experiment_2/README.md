# Experiment 2: Module 2 Filtering Necessity and Reliability

This pipeline evaluates:

1. Part A: whether Module 2 filtering is necessary.
2. Part B: whether automatic Module 1 keep/skip labels are reliable enough for light human review.

The official run uses 50 BioLaySumm validation articles:

- PLOS: 25
- eLife: 25

Outputs are written under:

- `data/experiment_2/`
- `results/experiment_2/`

## Stages

```bash
python3 src/experiment_2/run_exp2.py stage00_prepare_inputs --n-per-source 25
python3 src/experiment_2/run_exp2.py stage01_extract_article_af --model gpt-4.1
python3 src/experiment_2/run_exp2.py stage02_coverage_setup_c --model gpt-4.1
python3 src/experiment_2/run_exp2.py stage02b_direct_coverage --model gpt-4.1
python3 src/experiment_2/run_exp2.py stage03_keep_skip --model gpt-4.1
python3 src/experiment_2/run_exp2.py stage04_eval
python3 src/experiment_2/run_exp2.py stage05_sample_human_check --n 80
```

## Method Summary

Part A extracts article-level atomic facts (`AF_article`) and uses a Setup C-style 1T3F + None-of-the-above judgement to determine whether each fact is covered by the expert lay summary.

Coverage recall is:

```text
covered AF_article count / total AF_article count
```

Low recall means many article facts are safe simplifications, so treating every article fact as required would create many false alarms.

Part B uses a direct LLM coverage judgement with a supporting span as the keep/skip reference standard, then compares automatic article-only keep/skip labels against that direct coverage-derived ground truth:

```text
covered by expert summary     -> ground truth keep
not covered by expert summary -> ground truth skip
```

The most important error is dangerous skip:

```text
ground truth keep, but module predicts skip
```
