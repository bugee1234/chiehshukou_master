# Local Test Machine Setup

This note is for the local test machine at:

```text
C:\Users\bugee\OneDrive\桌面\chiehshukou_master
```

It is not the lab evaluation computer. Use it for small local checks such as:

- 10 to 20 article evaluation tests
- metric smoke tests
- expert-summary self-check diagnostics

Do not treat this machine as the source of official final metrics. Final runs should
still be computed on the lab evaluation computer.

Codex sessions on this machine should read this file before running local
evaluation commands.

## Purpose

This machine has two separate Python environments:

- `.venv`
  General project environment used for the repo's ordinary workflows.
- `.venv-eval`
  Local evaluation-only environment for BioLaySumm metrics, AlignScore, and SummaC.

Keep them separate. Do not install evaluation dependencies into `.venv` unless
there is a specific reason.

## Environment Switching

If PowerShell currently shows `(.venv)` and you want local evaluation:

```powershell
deactivate
.\.venv-eval\Scripts\Activate.ps1
```

If PowerShell currently shows `(.venv-eval)` and you want the ordinary project
environment again:

```powershell
deactivate
.\.venv\Scripts\Activate.ps1
```

Confirm the active evaluation interpreter with:

```powershell
python --version
```

Expected local evaluation interpreter:

```text
Python 3.9.13
```

## Local Evaluation Environment

The local evaluation stack on this machine is:

- Python `3.9.13`
- venv `.venv-eval`
- PyTorch `2.4.1+cu121`
- GPU `NVIDIA GeForce RTX 3050 Laptop GPU`
- local NLTK data under `.nltk_data`
- local Hugging Face cache under `C:\hf_cache`
- local cloned repos `AlignScore/` and `summac/`

This machine only has 4 GB VRAM, and about 1 GB may already be occupied by
Windows, the browser, the IDE, or other desktop GPU processes. Small runs are
fine, but do not assume the lab machine's headroom.

## Required Environment Variables

Always set these before local evaluation:

```powershell
$env:NLTK_DATA="$PWD\.nltk_data"
$env:HF_HOME="C:\hf_cache"
```

Use `C:\hf_cache`, not the repo-local `.cache\huggingface`, for local testing on
this machine. The repo path contains non-ASCII characters, and SummaC tokenizer
loading was more reliable with the ASCII-only cache path.

## Quick Startup Check

From the repo root, inside `(.venv-eval)`:

```powershell
$env:NLTK_DATA="$PWD\.nltk_data"
$env:HF_HOME="C:\hf_cache"
python -c "import torch, nltk; print(torch.__version__); print(torch.cuda.is_available()); print(nltk.data.find('tokenizers/punkt_tab'))"
```

Expected behavior:

- PyTorch imports successfully
- CUDA reports `True`
- `punkt_tab` resolves under `.nltk_data`

## Local Test Commands

### Experiment 3 Official-Style Evaluation

Use this only for local testing, not as the final authoritative evaluation:

```powershell
$env:NLTK_DATA="$PWD\.nltk_data"
$env:HF_HOME="C:\hf_cache"
python src/experiment_3/evaluate_exp3_outputs.py --run-name <run-name> --variant both
```

### Experiment 3 Reference Self-Check

This uses the expert reference summary as the prediction while keeping the
original article as the document:

```powershell
$env:NLTK_DATA="$PWD\.nltk_data"
$env:HF_HOME="C:\hf_cache"
python src/experiment_3/evaluate_exp3_outputs.py --run-name <run-name> --variant reference-self-check
```

### Module 2 Repro Expert-Summary Diagnostics

This machine also has a local helper script for the `experiment_3_module2_repro`
pilot runs:

```powershell
$env:NLTK_DATA="$PWD\.nltk_data"
$env:HF_HOME="C:\hf_cache"
python src\experiment_3_module2_repro\evaluate_expert_summary_diagnostics.py --run-name pilot_val_n10_judge_gemini31_flash_lite --variant both
```

That command writes two diagnostics:

- `reference_self_check`
  `generated = expert summary`, `reference = expert summary`, `document = article`
- `reference_as_document_self_check`
  `generated = expert summary`, `reference = expert summary`, `document = expert summary`

Results are written under:

```text
results/experiment_3_module2_repro/<run-name>/diagnostics/
```

## Interpreting Local Self-Checks

For self-check style diagnostics on this machine:

- `ROUGE`, `BLEU`, `METEOR`, and `BERTScore` should reach or nearly reach their
  maxima when prediction equals reference.
- `FKGL`, `DCRS`, and `CLI` simply describe the summary text.
- `AlignScore` and `SummaC` may remain well below `1.0` when the document is the
  original article, because they are document-grounded factuality metrics rather
  than identity checks.
- `AlignScore` and `SummaC` should become much higher when the document itself is
  replaced with the expert summary.

This pattern is expected and does not by itself indicate a bug.

## Local Compatibility Fixes

This machine needed two local source compatibility fixes inside the ignored
evaluation repos:

- `AlignScore/src/alignscore/model.py`
  uses `torch.optim.AdamW` so it works with the installed `transformers`
  version.
- `summac/summac/model_summac.py`
  uses `truncation="only_first"` instead of the old
  `truncation_strategy="only_first"` argument.

These live under ignored local-only directories:

```text
AlignScore/
summac/
```

They are not part of the main tracked repo code.

## Git and Local-Only Files

The following are local-only and should stay uncommitted:

```text
.venv-eval/
.cache/
.nltk_data/
AlignScore/
summac/
```

`C:\hf_cache` is outside the repo and is also local-only.

The helper script below is normal tracked project code and should not be added
to `.gitignore`:

```text
src/experiment_3_module2_repro/evaluate_expert_summary_diagnostics.py
```

## What Not To Do

- Do not overwrite or regenerate generation-owned files under
  `data/experiment_3/runs/` or `data/experiment_3_module2_repro/runs/` unless
  that is the explicit task.
- Do not assume this machine matches the lab evaluation machine.
- Do not report this machine's small-run results as the final official numbers.
- Do not switch `HF_HOME` back to the repo-local path for local SummaC tests on
  this machine unless you are intentionally debugging cache/path behavior.

## Related Files

- `evaluation/EXPERIMENT_3_COLLABORATION.md`
- `evaluation/EXPERIMENT_3_EVALUATION_SETUP.md`
- `src/experiment_3_module2_repro/evaluate_expert_summary_diagnostics.py`
