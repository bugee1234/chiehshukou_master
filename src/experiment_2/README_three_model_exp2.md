# Experiment 2 Three-Model Part B/C Comparison

This runner repeats Experiment 2 Part B and Part C with three language models while keeping the current prompts and core logic unchanged.

It does not write to the existing `data/experiment_2/` or `results/experiment_2/` directories. Outputs are written under:

```text
data/experiment_2_three_model_comparison/runs/
```

## Models

```text
gpt41_mini              -> OpenAI gpt-4.1-mini
gemini31_flash_lite     -> Google Gemini gemini-3.1-flash-lite
deepseek_v4_flash       -> DeepSeek deepseek-v4-flash
```

## Environment

Add these keys to `.env`:

```bash
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...
DEEPSEEK_API_KEY=...
```

## Smoke Test: 5 Articles

Run all three models:

```bash
python3 src/experiment_2/run_three_model_exp2.py --n-articles 5
```

Run one model only:

```bash
python3 src/experiment_2/run_three_model_exp2.py --n-articles 5 --models gpt41_mini
python3 src/experiment_2/run_three_model_exp2.py --n-articles 5 --models gemini31_flash_lite
python3 src/experiment_2/run_three_model_exp2.py --n-articles 5 --models deepseek_v4_flash
```

## Formal Run: 20 Articles

```bash
python3 src/experiment_2/run_three_model_exp2.py --n-articles 20
```

If you want a fully fresh rerun that ignores existing partial outputs:

```bash
python3 src/experiment_2/run_three_model_exp2.py --n-articles 20 --no-resume
```

## Comparison Table

After runs finish, the summary table is:

```text
data/experiment_2_three_model_comparison/runs/n5/comparison_summary.csv
data/experiment_2_three_model_comparison/runs/n20/comparison_summary.csv
```

Regenerate the table without making API calls:

```bash
python3 src/experiment_2/run_three_model_exp2.py --n-articles 5 --comparison-only
python3 src/experiment_2/run_three_model_exp2.py --n-articles 20 --comparison-only
```

## Output Layout

Each model writes:

```text
runs/n<N>/<model_key>/
  selected_articles.jsonl
  selected_articles_metadata.json
  run_metadata.json
  api_usage_calls.csv
  api_usage_summary.json
  model_summary.json
  part_b/
    01_article_af/
    02_direct_coverage_gt/
    03_keep_skip/
    04_eval/
      confusion_matrix.csv
  part_c/
    01_summary_af/
    02_selected_keep_af/
    03_eval/
      partc_summary.csv
```

The runner records wall-clock time, API call time, token usage, estimated cost, and cost currency.
