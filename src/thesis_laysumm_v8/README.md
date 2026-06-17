# thesis_laysumm_v8

V8 is a conservative candidate-reranking version of the evidence-slot
BioLaySumm pipeline. It keeps factual repair as a candidate, but no longer lets
repair automatically replace a strong generated summary.

Design goals:

- Keep v5's evidence-table structure and model-agnostic runs.
- Keep `rewritten_summary` as the evaluation-compatible final output field, but
  make it selector-gated rather than always LLM-rewritten.
- Use 1T3F failures to trigger factual-repair rewrite. If generated has no
  failed checks and passes length/encoding gates, keep generated.
- Keep three internal candidates per article whenever possible:
  `generated`, `expanded_generated`, `readability_trim`, and
  `factual_repair`.
- Select final `rewritten_summary` with validity, readability, and
  wrong-count proxy gates instead of the old "wrong feedback means use repair"
  rule.
- Repair below-minimum summaries with deterministic evidence-local expansion
  before evaluation.
- Improve readability with formula-aware constraints: shorter sentences, fewer
  hard words, and lower technical-term density.
- Preserve factual anchors: entities, direction, negation, comparison, number,
  organism, method, and condition when they control the claim.
- Prefer deleting or narrowing risky details over adding new technical content.
- Reduce DCRS/CLI pressure with first-mention exact terms plus short aliases.

Main stage paths:

- `02_5_evidence_table/<mode>/<model_key>`
- `03_questions_v8/<mode>/<model_key>`
- `04_summaries/<model_key>`
- `05_module3_answers_v8/<mode>/<model_key>`
- `06_rewritten/<model_key>`

Validation:

`validate_outputs.py` checks stage completeness, parse errors, slot/text
consistency, final length, selector metadata, mojibake-like output text, and
simple readability proxies.

Outputs remain compatible with `src.thesis_laysumm.run_evaluate` because final
rows still include `generated_summary` and `rewritten_summary`.

In V8, `rewritten_summary` means selected final summary. Each row also records
`selected_variant` and `candidate_summaries` so generated, expanded, and
factual-repair candidates can be audited.

The V8 selector is intentionally conservative:

- valid generated summaries with no wrong feedback are kept;
- no-error summaries may select `readability_trim` when optional clarification
  removal gives a clear readability gain without dropping below length;
- below-min generated summaries can fall back to `expanded_generated`;
- factual repair must reduce `wrong_count_proxy`;
- low-severity repairs must not worsen readability or hard-word density;
- severe repairs get limited slack, but still cannot add long-sentence or
  hard-word regressions.

