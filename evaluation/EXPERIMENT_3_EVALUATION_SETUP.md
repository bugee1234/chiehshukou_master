# Experiment 3 Evaluation Computer Setup

Read this file together with `EXPERIMENT_3_COLLABORATION.md` before running
Experiment 3 evaluation on this computer.

If you are on the local test machine at
`C:\Users\bugee\OneDrive\桌面\chiehshukou_master` rather than the lab evaluation
computer, read `evaluation/LOCAL_TEST_MACHINE_SETUP.md` first. That file records
the local `.venv-eval` workflow, ASCII-only Hugging Face cache path, and
small-run testing commands used on this machine.

## Local Environment

- Repository: `C:\Users\user\Desktop\chiehshukou_master`
- Python: repo-local `.venv` (Python 3.9.23)
- PyTorch: 2.4.1+cu121
- GPU: NVIDIA RTX A4000 16 GB
- Local model cache: `.cache/huggingface/`
- Local NLTK data: `.nltk_data/`
- Local evaluation repositories: `AlignScore/` and `summac/`

These directories are local-only and must not be committed:

```text
.venv/
.cache/
.nltk_data/
AlignScore/
summac/
```

Do not recreate the environment or arbitrarily upgrade or downgrade PyTorch,
Transformers, PyTorch Lightning, LENS, AlignScore, or SummaC. The installed
versions have been tested together.

## PowerShell Startup

Run commands from the repository root. Use the venv Python path explicitly so
PowerShell cannot accidentally select a Conda interpreter:

```powershell
$env:NLTK_DATA="$PWD\.nltk_data"
$env:HF_HOME="$PWD\.cache\huggingface"
.\.venv\Scripts\python.exe -c "import sys, torch, nltk; print(sys.executable); print(torch.cuda.is_available()); print(nltk.data.find('tokenizers/punkt_tab'))"
```

Expected results include the repo-local `.venv` interpreter, `True` for CUDA,
and a `punkt_tab` path under `.nltk_data`.

The local `.venv/Lib/site-packages/sitecustomize.py` configures `certifi` for
SSL. Do not remove it. This computer's Windows certificate handling may
otherwise fail with `[ASN1: NOT_ENOUGH_DATA]`.

## Run Evaluation

After completing the read-only checks in `EXPERIMENT_3_COLLABORATION.md`, run:

```powershell
$env:NLTK_DATA="$PWD\.nltk_data"; $env:HF_HOME="$PWD\.cache\huggingface"; .\.venv\Scripts\python.exe src/experiment_3/evaluate_exp3_outputs.py --run-name <run-name> --variant both
```

After the command reports that it wrote official metrics, run:

```powershell
.\.venv\Scripts\python.exe evaluation/compare_task1_1_leaderboard.py --metrics-dir results/experiment_3/<run-name>/official_metrics
```

Warnings about deprecated `pkg_resources` or LENS not using an available GPU
do not by themselves indicate evaluation failure.

## Reference Self-Check

To validate the metric pipeline on a run, use each expert reference summary as
the prediction for the same article:

```powershell
$env:NLTK_DATA="$PWD\.nltk_data"; $env:HF_HOME="$PWD\.cache\huggingface"; .\.venv\Scripts\python.exe src/experiment_3/evaluate_exp3_outputs.py --run-name <run-name> --variant reference-self-check
```

Results are written separately from official metrics under:

```text
results/experiment_3/<run-name>/diagnostics/reference_self_check/
```

ROUGE, BLEU, METEOR, and BERTScore should reach or approach their self-match
maximums. Readability metrics still describe the expert summaries themselves.
LENS, AlignScore, and SummaC also use the source article, so they are not
expected to equal one.

## Input Format

`evaluate_exp3_outputs.py` reads predictions from
`06_eval_inputs/paired_summaries.jsonl`. This preserves article boundaries for
summaries containing paragraph newlines. References come from
`00_inputs/PLOS_test.jsonl` and `00_inputs/eLife_test.jsonl`.

Do not edit or regenerate files under `data/experiment_3/runs/` on the
evaluation computer.
