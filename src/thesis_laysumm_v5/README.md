# thesis_laysumm_v5

V5 is a model-agnostic, evidence-slot controlled lay-summary pipeline.

The goal is not to route different datasets to different models. Instead, GPT and
Gemini run the same pipeline, with the same dataset profiles, so score gaps can be
interpreted as pipeline robustness rather than model cherry-picking.

Key changes from v4:

- Evidence rows are treated as final keep-AF groups and become summary slots.
- Generation returns controlled `slots`, each with a row-local `claim_sentence`
  and optional `clarification_sentence`.
- Dataset profiles are allowed:
  - `plos_concise`: concise, numeric, abstract-close summaries.
  - `elife_mechanism`: longer mechanism-aware summaries with row-local
    explanations.
- Length repair is explicit. If a generated or rewritten summary is below the
  minimum, v5 calls a separate expansion prompt that may only add supported,
  row-local clarification.
- Rewrite receives current slots and failed 1T3F feedback, so it can repair
  slot-level factual issues instead of freely rewriting the whole summary.
- `validate_outputs.py` supports staged validation:
  `evidence`, `questions`, `summaries`, `answers`, `rewritten`, `eval`, or `all`.

Outputs remain compatible with `src.thesis_laysumm.run_evaluate` because final
rows still include `generated_summary` and `rewritten_summary`.
