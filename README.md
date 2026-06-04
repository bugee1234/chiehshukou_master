# Detecting Information Errors and Omissions in Medical Lay Summaries

This repository contains experiment code for automated factual consistency evaluation of biomedical lay summaries.

## Setup

1. Clone this repository.
2. Create a virtual environment:
   - macOS/Linux:
     ```bash
     python -m venv .venv
     source .venv/bin/activate
     ```
   - Windows (PowerShell):
     ```powershell
     python -m venv .venv
     .venv\Scripts\Activate.ps1
     ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create environment file:
   - macOS/Linux:
     ```bash
     cp .env.example .env
     ```
   - Windows (PowerShell):
     ```powershell
     Copy-Item .env.example .env
     ```
5. Fill your API key in `.env`.

## Folder Structure

- `src/`: active Python code, shared utilities, and maintained experiment runners.
- `src/experiment_1/`: maintained Experiment 1 runners and Experiment 1-specific prompts.
- `src/experiment_2/`: maintained Experiment 2 runner and Experiment 2-specific prompts.
- `scripts/`: legacy and one-off scripts, mainly for the original/Experiment 0 pipeline and older Experiment 1 runs.
- `prompts/`: root-level prompt templates used by the legacy scripts.
- `data/`: complete reproducible run artifacts, including intermediate files, model outputs, per-stage predictions, and many per-stage metrics.
- `results/`: selected final/reportable summaries copied out of some pipelines. This directory is not a complete mirror of `data/`.
- `backups/`: manual snapshots made before reruns or prompt changes.
- `logs/`: runtime logs.

## Current Output Conventions

The project currently uses a pragmatic, historically grown layout. In practice, `data/` is the complete run ledger, while `results/` contains selected summaries for reporting. Some metrics therefore appear under `data/`, and only some are copied into `results/`.

### Legacy / Experiment 0 Pipeline

The root `scripts/` pipeline writes intermediate artifacts to:

- `data/raw/`
- `data/atomic_facts/`
- `data/questions/`
- `data/solver_results/`

The final Experiment 0 report files are written to:

- `results/experiment_0/final_test_bank.jsonl`
- `results/experiment_0/final_metrics.json`
- `results/experiment_0/filter_trace.csv`
- `results/experiment_0/statistics.json`
- `results/experiment_0/report.md`

### Experiment 1

The maintained Experiment 1 v2 runner is:

```bash
python src/experiment_1/run_1a_v2.py ...
```

It writes the main run artifacts to:

- `data/experiment_1/1a_v2/00_registry/`
- `data/experiment_1/1a_v2/01_inputs/`
- `data/experiment_1/1a_v2/02_perturbation/`
- `data/experiment_1/1a_v2/03_rag/`
- `data/experiment_1/1a_v2/04_judgement/`
- `data/experiment_1/1a_v2/05_metrics/`

The fair A/C comparison runner writes isolated outputs to:

- `data/experiment_1/1a_v2_fair_ac/04_judgement/`
- `data/experiment_1/1a_v2_fair_ac/05_metrics/`

Some `results/experiment_1/` folders exist, but the main reproducible stage outputs and metrics are under `data/experiment_1/`.

### Experiment 2

The maintained Experiment 2 runner is:

```bash
python src/experiment_2/run_exp2.py ...
```

Its full stage artifacts are written to:

- `data/experiment_2/00_inputs/`
- `data/experiment_2/01_article_af/`
- `data/experiment_2/01b_keep_af_ultra_recall/`
- `data/experiment_2/02_coverage_setup_c/`
- `data/experiment_2/02b_direct_coverage/`
- `data/experiment_2/03_keep_skip/`
- `data/experiment_2/04_eval/`
- `data/experiment_2/05_human_check/`
- `data/experiment_2/part_c_selective_keep_af/`

For Experiment 2 variants such as `ultra_recall`, `stage04_eval` writes evaluation files to both:

- `data/experiment_2/04_eval/<variant>/`
- `results/experiment_2/<variant>/`

For example, both locations may contain:

- `metrics_summary.json`
- `confusion_matrix.csv`

The `data/` copy should be treated as the canonical full run artifact. The `results/` copy is a convenience/reporting export.

### Experiment 2 Part C

Part C currently writes only under:

- `data/experiment_2/part_c_selective_keep_af/01_summary_af/`
- `data/experiment_2/part_c_selective_keep_af/02_selected_keep_af/`
- `data/experiment_2/part_c_selective_keep_af/03_eval/`
- `data/experiment_2/part_c_selective_keep_af/04_fair_eval/`
- `data/experiment_2/part_c_selective_keep_af/05_precision_eval/`

Part C does not currently export a duplicate summary to `results/experiment_2/`. If a Part C result is needed for reporting, check `data/experiment_2/part_c_selective_keep_af/03_eval/` or the relevant backup under `backups/`.

### Practical Rule

When looking for the most complete and reproducible artifact, start in `data/experiment_*`. When looking for a compact reporting summary, check `results/experiment_*`, but verify whether that experiment actually exports there.

## Progress

- [x] Step 1: Project setup
- [x] Step 2
- [x] Step 3
- [x] Step 4
- [x] Step 5
- [x] Step 6
- [x] Step 7

Final test bank: see `results/experiment_0/final_test_bank.jsonl` (325 questions)  
Full report: `results/experiment_0/report.md`
