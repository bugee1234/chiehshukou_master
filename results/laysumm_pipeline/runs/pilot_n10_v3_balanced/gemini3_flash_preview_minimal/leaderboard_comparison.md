# BioLaySumm Task 1.1 Leaderboard Comparison

> Caution: these experiment scores use a 20-article validation pilot (10 PLOS and 10 eLife), while the published leaderboard uses the official test set. Metric positions below are descriptive, not official ranks.

| Metric | Initial | Initial position | Rewritten | Rewritten position | Delta |
|---|---:|---:|---:|---:|---:|
| ROUGE | 0.331 | 7/17 | 0.334 | 5/17 | +0.003 |
| BLEU | 6.306 | 7/17 | 6.802 | 7/17 | +0.496 |
| METEOR | 0.274 | 7/17 | 0.284 | 7/17 | +0.010 |
| BERTScore | 0.846 | 13/17 | 0.846 | 13/17 | -0.000 |
| FKGL | 15.061 | 14/17 | 15.024 | 14/17 | -0.037 |
| DCRS | 12.935 | 17/17 | 12.957 | 17/17 | +0.022 |
| CLI | 16.277 | 16/17 | 16.345 | 16/17 | +0.068 |
| LENS | 66.194 | 9/17 | 65.162 | 9/17 | -1.032 |
| AlignScore | 0.706 | 8/17 | 0.702 | 8/17 | -0.004 |
| SummaC | 0.582 | 9/17 | 0.583 | 9/17 | +0.001 |

Positions are computed independently per metric against the 16 published systems plus the experiment variant. For FKGL, DCRS, and CLI, lower is treated as better.

## Official-style Final rank

> Pilot 20-article validation (10 PLOS + 10 eLife) versus official test 142+142. Final ranks are computed by min-max normalizing across the 16 published teams plus our row(s), following evaluation/rank.py; not the competition's published official_rank.

| Variant | Final rank | Final score | Relevance | Readability | Factuality |
|---|---:|---:|---:|---:|---:|
| generated | 14/18 | 0.4821 | 0.6970 | 0.2937 | 0.4555 |
| rewritten | 13/18 | 0.4898 | 0.7294 | 0.2876 | 0.4525 |

Full normalized table: `official_style_rank.csv` / `official_style_rank.json`.
