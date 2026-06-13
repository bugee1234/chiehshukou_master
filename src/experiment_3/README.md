# Experiment 3: AF-Guided Iterative Lay Summarization

This pipeline implements the Experiment 3 flow:

1. Load BioLaySumm 2025 Task 1.1 PLOS/eLife test articles.
2. Extract candidate selected keep AFs using Experiment 2 Part C.
3. Generate 1T3F questions for every keep AF.
4. Merge overlapping article source spans into evidence packets, order them by article position, and generate an evidence-first lay summary whose sentences each cite one evidence packet.
5. Run sentence-level factuality checking and minimal repair on the initial summary while preserving the pre-repair initial version for evaluation.
6. Check whether the summary covers every keep AF using Experiment 1 Setup C: 1T3F + E(None) with the full summary as context by default.
7. Rewrite the summary using the error/missing list and each AF's article source span.
8. Check every rewritten-summary sentence against the article as supported, partial, unsupported, or contradicted.
9. Apply code-enforced minimal edits only to problematic sentences.
10. Recheck all keep AF questions, not only the previous error subset, and repeat until no errors or the rewrite limit is reached.
11. Export the pre-repair evidence-first initial summary and final repaired summary for BioLaySumm official evaluation.

## Recommended First Pilot

```bash
python src/experiment_3/run_exp3.py run_all --run-name pilot_val_n20_gpt41_mini --split validation --model gpt-4.1-mini --n-per-source 10 --max-rewrites 2
```

## AF-Guided Full-Article Context Pilot

This variant keeps the imported final keep AFs as the content plan, gives the
generator the full article to recover qualifiers and scope, and stops after
sentence-level minimal repair without running the coverage rewrite loop:

```bash
python src/experiment_3/run_exp3.py run_all \
  --run-name exp3_module2_val_n10_gemini3_flash_preview_fullarticle_af \
  --model-key gemini3_flash_preview_minimal \
  --module2-run-name full_val_n284_judge_gemini31_flash_lite \
  --module2-model-key gemini3_flash_preview_minimal \
  --n-per-source 5 \
  --max-workers 8 \
  --initial-only
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
