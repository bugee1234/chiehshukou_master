# Experiment 3: AF-Guided Iterative Lay Summarization

This pipeline implements the Experiment 3 flow:

1. Load BioLaySumm 2025 Task 1.1 PLOS/eLife test articles.
2. Extract candidate selected keep AFs using Experiment 2 Part C.
3. Generate 1T3F questions for every keep AF.
4. Generate an initial lay summary from the article.
5. Check whether the summary covers each keep AF using Experiment 1 Setup C: 1T3F + E(None) with retrieved summary context.
6. Rewrite the summary using the error/missing list.
7. Repeat until no errors or the rewrite limit is reached.
8. Export initial and rewritten summaries for BioLaySumm official evaluation.

## Recommended First Pilot

```bash
python src/experiment_3/run_exp3.py run_all --run-name pilot_val_n20_gpt41_mini --split validation --model gpt-4.1-mini --n-per-source 10 --max-rewrites 2
```

This runs 20 validation articles total: 10 PLOS and 10 eLife. Validation is used because the public HuggingFace test split has blank references.

## Full Test Run

```bash
python src/experiment_3/run_exp3.py stage00_prepare_inputs --run-name full_test_gpt41_mini --full-test
python src/experiment_3/run_exp3.py stage01_extract_keep_af --run-name full_test_gpt41_mini --model gpt-4.1-mini
python src/experiment_3/run_exp3.py stage02_generate_questions --run-name full_test_gpt41_mini --model gpt-4.1-mini
python src/experiment_3/run_exp3.py stage03_generate_initial_summaries --run-name full_test_gpt41_mini --model gpt-4.1-mini
python src/experiment_3/run_exp3.py stage04_check_iteration --run-name full_test_gpt41_mini --model gpt-4.1-mini --iteration 0
python src/experiment_3/run_exp3.py stage05_rewrite_iteration --run-name full_test_gpt41_mini --model gpt-4.1-mini --from-iteration 0
```

Continue checking/rewrite stages until the configured limit.

## Evaluation Inputs

After `run_all`, files are written under:

```text
data/experiment_3/runs/<run_name>/06_eval_inputs/
  plos_initial.txt
  elife_initial.txt
  plos_rewritten.txt
  elife_rewritten.txt
  paired_summaries.jsonl
```

Ground truth JSONL files are written under:

```text
data/experiment_3/runs/<run_name>/00_inputs/
  PLOS_test.jsonl
  eLife_test.jsonl
```

In the evaluation environment, run:

```bash
python src/experiment_3/evaluate_exp3_outputs.py --run-name pilot_val_n20_gpt41_mini --variant both
```

This calls `evaluation/evaluation_final.py::evaluate_all` with loaded prediction/reference lists and writes:

```text
results/experiment_3/<run_name>/official_metrics/
  initial_scores.txt
  rewritten_scores.txt
  rewritten_minus_initial.txt
```

Important: the HuggingFace `test` split is correct for generation/submission alignment, but its `summary` fields are blank. Use `validation` for local metric evaluation unless official hidden test references are available.
