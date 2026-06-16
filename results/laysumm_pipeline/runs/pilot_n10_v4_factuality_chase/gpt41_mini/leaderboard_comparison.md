# BioLaySumm Task 1.1 Leaderboard Comparison

> Caution: these experiment scores use a 20-article validation pilot (10 PLOS and 10 eLife), while the published leaderboard uses the official test set. Metric positions below are descriptive, not official ranks.

| Metric | Initial | Initial position | Rewritten | Rewritten position | Delta |
|---|---:|---:|---:|---:|---:|
| ROUGE | 0.321 | 9/17 | 0.324 | 8/17 | +0.004 |
| BLEU | 5.057 | 10/17 | 5.289 | 10/17 | +0.232 |
| METEOR | 0.271 | 7/17 | 0.276 | 7/17 | +0.004 |
| BERTScore | 0.844 | 13/17 | 0.843 | 13/17 | -0.000 |
| FKGL | 16.612 | 16/17 | 16.358 | 15/17 | -0.254 |
| DCRS | 13.505 | 17/17 | 13.593 | 17/17 | +0.088 |
| CLI | 17.140 | 17/17 | 17.263 | 17/17 | +0.123 |
| LENS | 68.119 | 9/17 | 67.811 | 9/17 | -0.308 |
| AlignScore | 0.712 | 8/17 | 0.708 | 8/17 | -0.005 |
| SummaC | 0.602 | 8/17 | 0.610 | 7/17 | +0.009 |

Positions are computed independently per metric against the 16 published systems plus the experiment variant. For FKGL, DCRS, and CLI, lower is treated as better.

## Official-style Final rank

> Pilot 20-article validation (10 PLOS + 10 eLife) versus official test 142+142. Final ranks are computed by min-max normalizing across the 16 published teams plus our row(s), following evaluation/rank.py; not the competition's published official_rank.

| Variant | Final rank | Final score | Relevance | Readability | Factuality |
|---|---:|---:|---:|---:|---:|
| generated | 14/18 | 0.4446 | 0.6375 | 0.2130 | 0.4832 |
| rewritten | 13/18 | 0.4517 | 0.6541 | 0.2127 | 0.4883 |

Full normalized table: `official_style_rank.csv` / `official_style_rank.json`.
