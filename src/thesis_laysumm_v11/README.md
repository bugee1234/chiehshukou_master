# thesis_laysumm_v11

V11 is the cross-model-stability candidate for the thesis lay-summary pipeline.
It intentionally does not add sentence-level metric gating.

Main changes from V10:

- Module 2 remains fixed to `gemini25_flash_non_thinking` by the pilot runner,
  even when the generation model is GPT-4.1 mini or Gemini 3 Flash preview.
- Generation is tuned for conservative, evidence-local summaries so the three
  generation models choose more similar claims.
- Rewrite is no longer mandatory for every article. V11 skips LLM rewrite when
  the generated or expanded candidate is structurally valid and there is no
  severe 1T3F feedback.
- The selector prefers `generated`, `expanded_generated`, or
  `readability_trim` unless a factual repair has severe feedback and passes a
  stricter conservative gate.
- Length repair only adds the smallest row-local content needed to pass the
  minimum; it does not chase the target length.
- The validator and sanity checker reject incomplete sentences, duplicated
  sentences, mojibake-like text, missing mandatory candidates, unexpected
  Module 2 judges, and skipped rewrites without recorded skip reasons.

Expected run name:

- `pilot_n20_v11_factuality_chase`

Default models:

- `gpt41_mini`
- `gemini3_flash_preview_minimal`
- `gemini25_flash_non_thinking`

Default Module 2 judge:

- `gemini25_flash_non_thinking`

Full three-model n=20 command:

```powershell
.\src\thesis_laysumm_v11\run_pilot_n20_v11.ps1 -Mode factuality_chase -RunNamePrefix pilot_n20_v11 -ArticlesSourceRun pilot_n20_v2 -ModelKeys @("gpt41_mini", "gemini3_flash_preview_minimal", "gemini25_flash_non_thinking") -JudgeModelKey gemini25_flash_non_thinking -MaxWorkers 8
```

Post-run sanity check:

```powershell
foreach ($m in @("gpt41_mini", "gemini3_flash_preview_minimal", "gemini25_flash_non_thinking")) {
  .\.venv\Scripts\python.exe -m src.thesis_laysumm_v11.sanity_check --run-name pilot_n20_v11_factuality_chase --model-key $m --require-eval
}
```
