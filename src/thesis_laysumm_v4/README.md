# thesis_laysumm_v4

V4 is an adaptive factuality-chase pipeline. It keeps the strict evidence-table,
1T3F checking, and rewrite loop from v3, but tries to recover relevance and
readability by adding grounded lay explanations.

Key changes from v3:

- `factuality_chase` uses adaptive length targets based on expert summary length.
  Long expert-style summaries get a fuller target instead of being compressed to
  170-230 words.
- Evidence rows include `lay_context` and `story_role` in addition to the strict
  `core_keep_af`.
- Generation may add plain-language explanation only when it is supported by the
  same evidence row.
- Rewrite no longer skips below-minimum summaries just because 1T3F found no
  factual errors. It expands them using supported evidence.
- Rewrite should repair factual errors without shrinking the summary unless the
  unsupported wording cannot be repaired.
- V4 does not implement candidate generation or candidate selection.

The generated and rewritten summary paths remain compatible with
`src.thesis_laysumm.run_evaluate`.
