# BioLaySumm 2025 off-the-shelf backbone pilot20

## What this experiment is

The BioLaySumm 2025 overview paper reports two Task 1 baseline backbones:

- `Qwen/Qwen2.5-7B-Instruct`
- `meta-llama/Meta-Llama-3-8B-Instruct`

The organizers trained both backbones on the BioLaySumm 2025 training data,
using the whole article as input and the lay summary as output. The published
leaderboard therefore labels them `Baseline-qwen2.5-7B-sft` and
`Baseline-llama3-8B-sft`.

The BioLaySumm Hugging Face organization currently publishes the datasets but
no model checkpoints. This experiment consequently runs the original Instruct
checkpoints without BioLaySumm task-specific SFT. It is an off-the-shelf,
zero-shot backbone comparison, not a reproduction of either official SFT
baseline. "Before fine-tuning" here means before *BioLaySumm task-specific*
fine-tuning; both Instruct checkpoints already received their own general
instruction post-training.

## Data and comparability

The run name is `official_backbones_zero_shot_pilot20_v1`. It uses the exact
20-article pilot already forced into
`v11_constellation_validation284_fixed_m12_pilot20`:

- PLOS validation: 10 articles
- eLife validation: 10 articles

The same IDs are used by the existing ATLAS pilot run
`pilot_n20_v11_fixed_m12_gemini25_same_model_phaseb_factuality_chase`. Do not
describe these as official hidden-test articles. The final 284-article thesis
run also uses 142 sampled validation articles from each source, despite the
current thesis draft calling them "test articles."

Only the raw article is visible during generation. Expert summaries and metric
inputs remain isolated until evaluation. Generation uses deterministic greedy
decoding, 8-bit weights, at most 7,680 input tokens, and at most 512 generated
tokens. The common 7,680-token input limit is required by the original Llama 3
8B context window. If an article is longer, only its tail is truncated and the
per-article token counts are recorded in `generation_metadata.json`.

## One-computer environment setup

Run all commands from the repository root in PowerShell. Keep local generation
separate from the protected official-metric environment:

```powershell
C:\Python312\python.exe -m venv .venv-local-llm
.\.venv-local-llm\Scripts\python.exe -m pip install --upgrade pip
.\.venv-local-llm\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu126
.\.venv-local-llm\Scripts\python.exe -m pip install -r requirements-local-llm.txt
```

Llama 3 is gated. Accept Meta's model terms on its Hugging Face model page,
then authenticate once:

```powershell
.\.venv-local-llm\Scripts\hf.exe auth login
```

Confirm CUDA and quantization support without changing `.venv`:

```powershell
.\.venv-local-llm\Scripts\python.exe -c "import torch, transformers, bitsandbytes; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0)); print(transformers.__version__, bitsandbytes.__version__)"
```

## Generation through evaluation

The resumable one-command workflow is:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_official_backbones_pilot20.ps1
```

It uses `.venv-local-llm` for the two local checkpoints and the existing
`.venv` for all ten BioLaySumm metrics. It does not compare the 20 validation
articles to the official hidden-test leaderboard.

Generation artifacts are written under:

```text
data/laysumm_direct_baseline/runs/official_backbones_zero_shot_pilot20_v1/
```

Evaluation artifacts and the final three-row comparison table are written
separately under:

```text
results/laysumm_direct_baseline/runs/official_backbones_zero_shot_pilot20_v1/
```

The final files are `official_backbones_vs_atlas.csv`, `.json`, and `.md`.

## Suggested thesis wording

> BioLaySumm 2025 採用 Qwen2.5-7B-Instruct 與 LLaMA3-8B-Instruct 作為 Task 1
> 的基礎模型，並以完整文章為輸入、人工通俗摘要為輸出，在官方訓練集上進行監督式微調。
> 然而，主辦單位未公開該兩個任務微調後的模型權重，因此本研究無法直接重現官方
> `-sft` baseline。作為補充比較，本研究改以相同的原始 Instruct checkpoints，在未經
> BioLaySumm 任務微調的情況下進行 zero-shot 推論。此結果應視為 off-the-shelf
> backbone comparison，而非官方 baseline 分數的重現。

> 此補充實驗使用主實驗 284 篇 validation sample 中既有的 20 篇固定子集（PLOS 與
> eLife 各 10 篇），並與相同 20 篇上的 ATLAS 結果比較。兩個本機模型皆採 8-bit
> 量化與 greedy decoding。為符合原始 Llama 3 8B 的 8,192-token context window，輸入
> 上限設為 7,680 tokens，輸出上限為 512 tokens；超過上限的文章由尾端截斷。因此，
> 此小樣本結果僅用來補充觀察未經任務微調之 backbone 與 ATLAS 的差異，不應與官方
> hidden-test leaderboard 分數作直接等價推論。

