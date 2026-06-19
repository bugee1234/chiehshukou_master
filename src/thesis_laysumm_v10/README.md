# thesis_laysumm_v10

V10 is the final 1T3F-guided rewrite candidate for the thesis lay-summary
pipeline.

Main changes from V9:

- Module 2 is fixed to `gemini25_flash_non_thinking` by the pilot runner, even
  when the generation model is GPT-4.1 mini or Gemini 3 Flash preview.
- Every article invokes rewrite. `rewritten_summary` is no longer a no-op for
  no-error cases.
- The rewrite candidate is selected when it passes the V10 gain gate: valid
  length, no mojibake, no bad sentence fragments, no worse wrong-count proxy,
  and no major readability regression.
- eLife factuality-chase length is tightened to 185-255 words with a 205-word
  target so the first generated summary should reach the minimum without
  damaging expansion.
- The validator and sanity checker reject incomplete sentences, duplicated
  sentences, mojibake-like text, missing mandatory rewrite candidates, and
  unexpected Module 2 judges.

Expected run name:

- `pilot_n20_v10_factuality_chase`

Default models:

- `gpt41_mini`
- `gemini3_flash_preview_minimal`
- `gemini25_flash_non_thinking`

Default Module 2 judge:

- `gemini25_flash_non_thinking`

Typical command:

```powershell
.\src\thesis_laysumm_v10\run_pilot_n20_v10.ps1 -Mode factuality_chase
```

Post-run sanity check:

```powershell
foreach ($m in @("gpt41_mini", "gemini3_flash_preview_minimal", "gemini25_flash_non_thinking")) {
  .\.venv\Scripts\python.exe -m src.thesis_laysumm_v10.sanity_check --run-name pilot_n20_v10_factuality_chase --model-key $m --require-eval
}
```
