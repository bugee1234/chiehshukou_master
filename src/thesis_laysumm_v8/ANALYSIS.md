# V8 Data-Driven Tuning Notes

This version was tuned from the observed v6/v7 pilot results, not from a pure
prompt hunch.

## Observed Scores

Best n=20 factuality anchor:

- `pilot_n20_v6_factuality_chase/gpt41_mini/rewritten`
- final `0.5975`
- relevance `0.6009`
- readability `0.4797`
- factuality `0.7117`
- AlignScore `0.8128`
- SummaC `0.7195`

Best n=20 overall candidate among recent runs:

- `pilot_n20_v7_factuality_chase/gemini3_flash_preview_minimal/generated`
- final `0.6556`
- relevance `0.7356`
- readability `0.5995`
- factuality `0.6316`
- AlignScore `0.7550`
- SummaC `0.6981`

V7 GPT tradeoff:

- generated: final `0.6086`, relevance `0.6449`, factuality `0.6675`,
  AlignScore `0.7947`, SummaC `0.6955`
- rewritten: final `0.6133`, relevance `0.6841`, factuality `0.6667`,
  AlignScore `0.7884`, SummaC `0.7003`

So v7 improved relevance and fixed below-min length, but its rewrite/expansion
path lowered GPT AlignScore versus v6.

## Structural Findings

- v6 GPT generated/rewritten had 8/20 below-min summaries but the strongest
  AlignScore.
- v7 GPT rewritten had 0/20 below-min summaries and better relevance, but lower
  AlignScore.
- v7 Gemini generated was better than rewritten, so factual repair should not be
  automatically selected.
- v6 rewrite was always invoked. This helped GPT AlignScore slightly, but hurt
  Gemini AlignScore, so v8 keeps repair as a candidate instead of always using it.
- Across n=20 v5-v7, generated is the safer AlignScore choice for Gemini in
  every available run. For GPT, rewrite only clearly helps in v6, and the gain is
  small compared with the cross-model risk.

## V8 Changes

- Generation prompt now restores the v6-style priority: fewer fully supported
  claims are better than broad risky coverage.
- Sentence-level AlignScore diagnostics were added with
  `src.thesis_laysumm.diagnose_alignscore_sentences`.
- V7 generated sentence diagnostics showed that low AlignScore sentences are
  often not long or garbled; many are short inference sentences that are not
  self-contained when scored alone.
- The prompt now requires every sentence to name the relevant entity, method,
  condition, or result instead of using deictic subjects such as "This",
  "These", "It", or "This means".
- The prompt now bans unsupported textbook/background definitions, analogies,
  and causal verbs such as "causes", "ensures", "confirms", "allows", "makes",
  "drives", or "explains" unless the evidence row states the relation.
- Relevance is still targeted, but only through central disease/problem,
  intervention/method, result, number/comparison, or mechanism-chain facts that
  are directly evidence-local.
- Length repair is minimal. It now prefers `abstract_claim` and `core_keep_af`
  over freer `lay_context` or `expert_summary_hint`.
- Deterministic expansion no longer uses the extra fallback that added arbitrary
  lay-context or expert-hint sentences from any row.
- Selector keeps `generated`, `expanded_generated`, and `factual_repair` as
  internal candidates, and also adds a deterministic `readability_trim`
  candidate. Final output is one `rewritten_summary`.
- `readability_trim` is the safe improvement path: it only removes optional
  clarification sentences when the summary stays within length and readability
  proxy improves. It does not add or rewrite factual claims.
- `readability_trim` now prioritizes optional clarifications with high
  sentence-level AlignScore risk: deictic subjects, inference verbs, analogies,
  broad definitions, and overlong sentences.
- Factual repair must now pass a high-severity gate: at least four wrong
  feedback items, at least two wrong-count proxy reductions, and no readability
  or hard-word regression. This treats 1T3F as a candidate generator, not a final
  selector, because v6/v7 show that repair often lowers Gemini AlignScore.

## Expected Behavior

- For Gemini, v8 should usually preserve generated summaries unless repair is
  clearly necessary.
- For GPT, v8 should recover more of the v6 entailment posture while preserving
  the v7 fix for below-min length. It intentionally gives up the v6 "always
  rewrite" behavior because that is not cross-model robust.
- Rewritten summaries should still be able to improve over generated summaries:
  length failures can select `expanded_generated`, no-error summaries can select
  `readability_trim`, and severe factual failures can select `factual_repair`.
- The remaining weakness is DCRS/CLI. V8 does not chase these aggressively
  because the strongest factuality runs are already hurt by readability metrics,
  and over-simplifying can damage exact anchors.

## Sentence-Level AlignScore Findings

Commands run on v7 generated summaries:

```powershell
$env:NLTK_DATA="$PWD\.nltk_data"
$env:HF_HOME="C:\hf_cache"
$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"
.\.venv-eval\Scripts\python.exe -m src.thesis_laysumm.diagnose_alignscore_sentences --run-name pilot_n20_v7_factuality_chase --model-key gemini3_flash_preview_minimal --variant generated --threshold 0.65 --quiet
.\.venv-eval\Scripts\python.exe -m src.thesis_laysumm.diagnose_alignscore_sentences --run-name pilot_n20_v7_factuality_chase --model-key gpt41_mini --variant generated --threshold 0.65 --quiet
```

Observed results:

- Gemini v7 generated: 382 sentences, 104 below 0.65, mean sentence AlignScore
  `0.754478`.
- GPT v7 generated: 288 sentences, 66 below 0.65, mean sentence AlignScore
  `0.799427`.

Common low-scoring sentence patterns:

- deictic standalone claims: "This means...", "This shows...", "This change...",
  "These proteins...";
- unsupported causal/inferential verbs: "causes", "ensures", "confirms",
  "allows", "makes", "explains";
- broad definitions and background facts not anchored in the article sentence;
- analogies/metaphors;
- very dense molecular sentences combining multiple entities and relations.
