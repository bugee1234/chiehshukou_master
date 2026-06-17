# thesis_laysumm_v6

V6 is a factuality-first, readability-aware version of the evidence-slot
BioLaySumm pipeline.

Design goals:

- Keep v5's evidence-table structure and model-agnostic runs.
- Make `rewritten` the intended final output, not a fallback.
- Rewrite every article, even when generated has no failed 1T3F checks.
- Improve readability with formula-aware constraints: shorter sentences, fewer
  hard words, and lower technical-term density.
- Preserve factual anchors: entities, direction, negation, comparison, number,
  organism, method, and condition when they control the claim.
- Prefer deleting or narrowing risky details over adding new technical content.

Main stage paths:

- `02_5_evidence_table/<mode>/<model_key>`
- `03_questions_v6/<mode>/<model_key>`
- `04_summaries/<model_key>`
- `05_module3_answers_v6/<mode>/<model_key>`
- `06_rewritten/<model_key>`

Validation:

`validate_outputs.py` checks stage completeness, parse errors, slot/text
consistency, length, mojibake-like output text, and simple readability proxies.

Outputs remain compatible with `src.thesis_laysumm.run_evaluate` because final
rows still include `generated_summary` and `rewritten_summary`.
