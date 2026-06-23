# Direct zero-shot full-article baseline

This baseline is isolated from the V11 pipeline.

## Model-visible input

The generation model receives exactly one user message containing:

1. raw `article`
2. the instruction: `Write a lay summary ... for a general audience.`

It does not receive the expert summary, target length, abstract as a separately
identified field, evidence tables, questions, answers, module outputs, existing
pipeline summaries, metric names, or rewrite feedback. The model response is
stored without cleanup apart from stripping outer whitespace.

`00_inputs` contains only identifiers and raw `article` text.
Evaluation-only `document` and `expert_summary` fields live in `00_references`.
Generation reads only `00_inputs`; evaluation joins the references afterward.

## Experimental runs

- Pilot: the same 20 validation articles forced into the V11 final sample.
- Full: the same 284 validation articles used by the V11 final experiment.
- Pilot summaries can be seeded into the full run, so only the remaining 264
  articles are generated during the full experiment.
