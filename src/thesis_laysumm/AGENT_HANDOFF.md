# Thesis LaySumm Pipeline — Agent Handoff (2026-06-15)

Paste this file (or the summary below) when starting a new Cursor agent session.

---

## Mission

Redo **论文 Experiment 3** as a clean pipeline under `src/thesis_laysumm/`.  
**Do NOT modify** `src/experiment_3*` (reference only).

**Final scale (not yet run):** validation random 142+142 (PLOS+eLife), 3 models.  
**Current pilot:** `pilot_n20_v1` — 20 articles (10 PLOS + 10 eLife, seed=42), model `gemini3_flash_preview_minimal` only.

---

## User workflow rules (MANDATORY)

1. **Step by step** — propose design/prompt/schema **before** implementing; wait for user OK.
2. **Do not auto-run LLM API or long GPU jobs** unless user explicitly asks. Give PowerShell commands; user runs and pastes output.
3. Verify **UTF-8 / jsonl** readability after each stage.
4. **Git commit only when user explicitly requests.**

---

## Core design (why redo)

- Factuality anchor = **article abstract**, not expert summary mimicry.
- Expert summary = few-shot style reference only (step 4).
- Module 2 judge = abstract sentences, not expert summary.
- Module 3: true MC option = AF verbatim; checker uses full generated summary.
- One rewrite round from wrong quiz answers + correct AF.

---

## Path layout

| Purpose | Path |
|---------|------|
| Code | `src/thesis_laysumm/` |
| Data | `data/laysumm_pipeline/runs/<run_name>/` |
| Results | `results/laysumm_pipeline/runs/<run_name>/<model_key>/` |

### Data stages (`pilot_n20_v1`)

```
00_articles/          articles.jsonl, PLOS/eLife_eval_references.jsonl
01_module1/<model>/   candidate_keep_af.jsonl
02_module2/<model>/   module2_judgements.jsonl, final_keep_af.jsonl
03_questions/<model>/ questions_1t3f_nota.jsonl
04_summaries/<model>/ generated_summaries.jsonl
05_module3_answers/   module3_answers.jsonl, wrong_answers.jsonl
06_rewritten/<model>/ rewritten_summaries.jsonl
```

---

## Scripts

| Script | Steps |
|--------|-------|
| `prepare_articles.py` | Step 0 |
| `run_module1.py` | Step 1 |
| `run_module2.py` | Step 2 |
| `run_module3_questions.py` | Step 3 |
| `run_phase_b.py` | Steps 4–6 (`--step` flag) |
| `run_evaluate.py` | Phase C |
| `prepare_few_shot_summary_examples.py` | Held-out few-shot for step 4 |

### `run_phase_b.py` steps

- `generate-summary` — step 4
- `repair-summaries` — local repair for step 4 parse errors (skip sentences with empty `af_ids`)
- `answer-questions` — step 5 (lenient JSON parse + `repair-answers`)
- `rewrite-summaries` — step 6

---

## Pilot `pilot_n20_v1` completion status

### Phase A — DONE

- 20 articles, 800 final keep AF, 800 questions, parse_errors=0, true_mismatch=0
- PLOS indices: [51, 178, 209, 228, 285, 457, 501, 563, 1116, 1309]
- eLife indices: [6, 26, 28, 35, 57, 62, 70, 163, 188, 189]

### Phase B — DONE

**Step 4 (generate):** 20/20 summaries. 5 initial parse errors fixed via relaxed validation + `repair-summaries`. 8 sentences skipped (empty `af_ids`). Mean ~410 words (expert ~283).

**Step 5 (answer):** 800/800 answered. Accuracy **76.7%** (613 correct). 98% of wrong = predicted **E** (summary insufficient for AF). 1 parse error fixed via re-run + lenient JSON. `wrong_answers.jsonl` = 187 rows.

**Step 6 (rewrite):** 20/20 rewritten. parse_errors=0. Mean ~492 words (+82 vs generated). All ≥ expert length.

### Phase C — IN PROGRESS

- `src/thesis_laysumm/run_evaluate.py` implemented.
- User must run in **`.venv-eval`**, NOT `.venv` (no torch in `.venv`).
- Partial results: `generated_scores.json` exists; `rewritten_scores.json` may still be pending.
- Current `--compare-leaderboard` only does **per-metric** rank among 17 teams (via `compare_task1_1_leaderboard.py`). It does **NOT** compute official **Final** composite rank.
- **TODO (user requested):** Add official-style Final ranking to `run_evaluate.py` (logic from `evaluation/rank.py`):
  - Min-max normalize 10 metrics across published leaderboard + our system row(s)
  - Invert FKGL, DCRS, CLI (lower is better)
  - Relevance = mean(ROUGE, BLEU, METEOR, BERTScore)
  - Readability = mean(FKGL, DCRS, CLI, LENS)
  - Factuality = mean(AlignScore, SummaC)
  - Final = mean(Relevance, Readability, Factuality)
  - Output `official_style_rank.json` / csv for generated and rewritten vs leaderboard teams
  - Propose design first, then implement after user OK (unless user already approved in handoff prompt)
- **TODO:** Finish full eval (`--variant both --compare-leaderboard`), analyze results.

---

## Two Python environments (this machine)

| Env | Use |
|-----|-----|
| `.venv` | LLM pipeline (Phase A/B) |
| `.venv-eval` | Official metrics (Phase C) |

```powershell
# Phase C only:
deactivate
.\.venv-eval\Scripts\Activate.ps1
$env:NLTK_DATA="$PWD\.nltk_data"
$env:HF_HOME="C:\hf_cache"
.\.venv-eval\Scripts\python.exe -m src.thesis_laysumm.run_evaluate `
  --run-name pilot_n20_v1 `
  --model-key gemini3_flash_preview_minimal `
  --variant both `
  --compare-leaderboard
```

See `evaluation/LOCAL_TEST_MACHINE_SETUP.md`.

---

## Known issues / quirks

- `data/.../pilot_n20_v1/api_usage_calls.csv/` may be a **directory** (legacy). `llm_utils.save_usage` handles nested path; user can clean up manually.
- `UsageTracker.save()` takes directory not file path.
- Leaderboard comparison is **descriptive only** (20-article validation pilot vs official test 142+142).
- Do not use deleted run `pilot_n20_seed42/`.

---

## Reference code (read only)

- `src/experiment_2/` — AF definitions
- `src/experiment_3/` — old flow (Module2 used expert summary — deprecated)
- `src/experiment_1/prompts_1a_v2_fair_ac/setup_c_system.txt` — MC checker policy
- `evaluation/evaluation_final.py`, `evaluation/compare_task1_1_leaderboard.py`

---

## Suggested next tasks for new agent

1. **Add official Final rank** to `run_evaluate.py` (see Phase C TODO above; reference `evaluation/rank.py`).
2. Help user **finish Phase C eval** in `.venv-eval`; analyze generated vs rewritten metrics.
3. Optional: re-run Module 3 on **rewritten** summaries to measure quiz accuracy improvement (not implemented yet).
4. Scale to full 284 articles / 3 models when user ready.
5. Do not touch `src/experiment_3*` unless user asks.

## Metric direction cheat sheet

| Higher is better | Lower is better |
|------------------|-----------------|
| ROUGE, BLEU, METEOR, BERTScore, LENS, AlignScore, SummaC | FKGL, DCRS, CLI |

Official rank uses all 10 via normalized category averages (not single-metric rank).

---

## Quick commands (Phase B, `.venv`)

```powershell
python -m src.thesis_laysumm.run_phase_b --run-name pilot_n20_v1 --model-key gemini3_flash_preview_minimal --step generate-summary
python -m src.thesis_laysumm.run_phase_b --run-name pilot_n20_v1 --model-key gemini3_flash_preview_minimal --step answer-questions --max-workers 8
python -m src.thesis_laysumm.run_phase_b --run-name pilot_n20_v1 --model-key gemini3_flash_preview_minimal --step rewrite-summaries --max-workers 4
```
