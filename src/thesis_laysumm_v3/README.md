# thesis_laysumm_v3

V3 keeps the v2 code intact and adds two modes:

- `balanced`: expert-guided, document-grounded summaries.
- `factuality_chase`: shorter, stricter, document-entailment-friendly summaries.

Both modes still use 1T3F:

1. v2 Module 1 extracts full-document candidate AFs.
2. v2 Module 2 judges candidate AFs against abstract sentences.
3. v3 builds an evidence table instead of canonicalizing facts back to abstract text.
4. v3 generates 1T3F questions for core claims and high-risk factual slots.
5. v3 generates summaries from evidence rows.
6. v3 answers 1T3F with the generated summary.
7. v3 rewrites with failed checks plus full-document evidence.

Outputs are compatible with `src.thesis_laysumm.run_evaluate`.

