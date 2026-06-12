# Experiment 3 Module 2 Reproducible Artifacts

This folder builds a reusable artifact bank for Experiment 3 without running summary generation or rewrite.

The goal is:

1. Select BioLaySumm validation articles with non-empty expert summaries.
2. For each pipeline model, extract candidate keep atomic facts from articles.
3. Use one fixed judge model to simulate Module 2 human review against the expert summary.
4. Keep only candidate facts judged to match the expert summary.
5. For each pipeline model, generate its own 1T3F + NOTA question bank for its final keep AFs.

## Default Model Roles

Pipeline models:

- `gpt41_mini`
- `gemini3_flash_preview_minimal`
- `gemini25_flash_non_thinking`

Fixed Module 2 judge:

- `gemini31_flash_lite`

The question generator is intentionally the same as the pipeline model for each model's artifact branch. This makes the output an end-to-end pipeline artifact, not a shared-question benchmark.

## Output Layout

```text
data/experiment_3_module2_repro/runs/<run_name>/
  00_inputs/
    articles.jsonl
    input_metadata.json
  01_candidate_keep_af/<model_key>/
    candidate_keep_af.jsonl
    candidate_keep_af_metadata.json
  02_module2_judgements/<model_key>/
    module2_judgements.jsonl
    module2_judgements_metadata.json
  03_final_keep_af/<model_key>/
    final_keep_af.jsonl
    final_keep_af_metadata.json
  04_questions/<model_key>/
    questions_1t3f_nota.jsonl
    question_metadata.json
  05_reports/
    model_artifact_summary.json
    run_summary.json
  api_usage_calls.csv
  api_usage_summary.json
```

## Final Keep Rule

A candidate keep AF is retained only when all conditions hold:

```text
covered_by_expert_summary == true
coverage_type in {exact, paraphrase, lay_generalization}
confidence in {high, medium}
supporting_span is non-empty
parse_error == false
```

## Pilot Command: 10 Articles

This selects 5 PLOS validation articles and 5 eLife validation articles.
By default, `--selection-mode first` is used, so these 10 articles are a subset of the later 284-article run when the same `--run-name` is reused.

```powershell
python src\experiment_3_module2_repro\run_module2_repro.py run_all `
  --run-name full_val_n284_judge_gemini31_flash_lite `
  --split validation `
  --n-per-source 5 `
  --selection-mode first `
  --models gpt41_mini gemini3_flash_preview_minimal gemini25_flash_non_thinking `
  --judge-model gemini31_flash_lite `
  --max-workers 8
```

## Full Validation Command: 284 Articles

This selects 142 PLOS validation articles and 142 eLife validation articles.

```powershell
python src\experiment_3_module2_repro\run_module2_repro.py run_all `
  --run-name full_val_n284_judge_gemini31_flash_lite `
  --split validation `
  --n-per-source 142 `
  --selection-mode first `
  --models gpt41_mini gemini3_flash_preview_minimal gemini25_flash_non_thinking `
  --judge-model gemini31_flash_lite `
  --max-workers 8
```

## Notes

- The validation split is required unless you provide another split/source with non-empty expert summaries.
- The public test split may have empty references; the script fails fast if any selected article lacks an expert summary.
- The run is resumable by default. Use `--no-resume` only when you intentionally want to rebuild outputs.
- Use the same `--run-name` and `--selection-mode first` for pilot and full runs if you want pilot outputs to be reused in the full 284-article run.
- API usage and estimated cost are written to `api_usage_calls.csv` and `api_usage_summary.json`.
