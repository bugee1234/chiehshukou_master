# Experiment 3 Two-Computer Collaboration

This document defines the ownership and Git workflow for running Experiment 3 generation on one computer and official evaluation on another. Codex sessions on both computers should read this file before changing Experiment 3 artifacts.

## Branch Roles

- `main` is the integration branch and the source of truth for completed generation inputs, summaries, evaluation code, and accepted evaluation results.
- `exp3-pilot-evaluation` is the long-lived evaluation branch. It receives new runs from `main`, computes official metrics, and sends evaluation-only changes back to `main` through a pull request.

## File Ownership

The generation computer owns:

- `src/experiment_3/run_exp3.py`
- `src/experiment_3/prompts/`
- `data/experiment_3/runs/<run-name>/00_inputs/`
- `data/experiment_3/runs/<run-name>/03_summaries/`
- `data/experiment_3/runs/<run-name>/06_eval_inputs/`
- Other Experiment 3 pipeline artifacts for the run

The evaluation computer owns:

- `evaluation/`
- `src/experiment_3/evaluate_exp3_outputs.py`
- `results/experiment_3/<run-name>/official_metrics/`

The evaluation computer must never regenerate, edit, resolve conflicts by overwriting, or reformat generation-owned input and summary files.

## Run Naming

Every evaluation must use a unique, immutable run name. Do not reuse a run name after its summaries have changed. Create a new run name instead, for example:

```text
pilot_val_n20_gpt41_mini_v2
full_val_gpt41_mini_20260615
```

This prevents a result directory from silently referring to different summaries over time.

## Generation Computer Workflow

Work on `main`, commit the complete run artifacts, and push them before asking the evaluation computer to start:

```bash
git switch main
git pull --rebase origin main
git add src/experiment_3 data/experiment_3/runs/<run-name>
git commit -m "Prepare experiment 3 run <run-name>"
git push origin main
```

Before pushing, verify that `06_eval_inputs` contains initial and rewritten PLOS/eLife files and that references are non-empty.

## Evaluation Computer Workflow

The evaluation worktree must be clean before synchronization. Bring the latest generation work into the evaluation branch with:

```bash
git switch exp3-pilot-evaluation
git fetch origin
git rebase origin/main
```

If rebase reports a conflict under `data/experiment_3/runs/`, preserve the generation artifacts from `origin/main`. Do not replace them with stale evaluation-branch copies.

Run official metrics:

```bash
python src/experiment_3/evaluate_exp3_outputs.py --run-name <run-name> --variant both
```

Generate the reusable leaderboard comparison:

```bash
python evaluation/compare_task1_1_leaderboard.py --metrics-dir results/experiment_3/<run-name>/official_metrics
```

Commit only evaluation-owned files:

```bash
git add evaluation src/experiment_3/evaluate_exp3_outputs.py results/experiment_3/<run-name>/official_metrics
git diff --cached --name-only
git commit -m "Evaluate experiment 3 <run-name>"
git push -u origin exp3-pilot-evaluation
```

Open a pull request from `exp3-pilot-evaluation` to `main`. The pull request must not contain generation input, summary, or prompt changes.

## Returning Results To The Generation Computer

After the evaluation pull request is merged, the generation computer runs:

```bash
git switch main
git pull --rebase origin main
```

It will then receive the official metric files and leaderboard comparison without switching to the evaluation branch.

## Local-Only Environment Files

Never commit evaluation environments, downloaded repositories, model caches, or manual NLTK downloads:

```text
.venv/
.cache/
.nltk_data/
AlignScore/
summac/
*.zip
```

## Evaluation Portability Notes

`evaluation/evaluation_final.py` intentionally uses lazy imports for radiology-only metrics so BioLaySumm evaluation does not require `f1chexbert`, `radgraph`, or `sentence-transformers`. The SummaC checkpoint path is resolved from the repository root rather than a Linux-only absolute path. These are evaluation environment fixes and do not change Experiment 3 summaries.

The published Task 1.1 table is stored in `evaluation/task1_1_leaderboard.csv`. Comparisons against it are descriptive because local validation runs and the official hidden test leaderboard use different article sets.
