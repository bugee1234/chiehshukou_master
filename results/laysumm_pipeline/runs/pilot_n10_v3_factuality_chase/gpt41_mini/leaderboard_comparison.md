# BioLaySumm Task 1.1 Leaderboard Comparison

> Caution: these experiment scores use a 20-article validation pilot (10 PLOS and 10 eLife), while the published leaderboard uses the official test set. Metric positions below are descriptive, not official ranks.

| Metric | Initial | Initial position | Rewritten | Rewritten position | Delta |
|---|---:|---:|---:|---:|---:|
| ROUGE | 0.287 | 12/17 | 0.283 | 13/17 | -0.004 |
| BLEU | 5.614 | 8/17 | 5.719 | 8/17 | +0.105 |
| METEOR | 0.241 | 11/17 | 0.241 | 11/17 | +0.000 |
| BERTScore | 0.837 | 14/17 | 0.837 | 14/17 | +0.000 |
| FKGL | 16.374 | 15/17 | 16.760 | 17/17 | +0.387 |
| DCRS | 14.081 | 17/17 | 14.192 | 17/17 | +0.111 |
| CLI | 18.127 | 17/17 | 18.531 | 17/17 | +0.404 |
| LENS | 60.589 | 11/17 | 58.431 | 12/17 | -2.158 |
| AlignScore | 0.823 | 4/17 | 0.823 | 4/17 | +0.000 |
| SummaC | 0.707 | 2/17 | 0.766 | 2/17 | +0.059 |

Positions are computed independently per metric against the 16 published systems plus the experiment variant. For FKGL, DCRS, and CLI, lower is treated as better.

## Official-style Final rank

> Pilot 20-article validation (10 PLOS + 10 eLife) versus official test 142+142. Final ranks are computed by min-max normalizing across the 16 published teams plus our row(s), following evaluation/rank.py; not the competition's published official_rank.

| Variant | Final rank | Final score | Relevance | Readability | Factuality |
|---|---:|---:|---:|---:|---:|
| generated | 14/18 | 0.4856 | 0.5400 | 0.2086 | 0.7082 |
| rewritten | 13/18 | 0.4932 | 0.5390 | 0.1672 | 0.7736 |

Full normalized table: `official_style_rank.csv` / `official_style_rank.json`.
