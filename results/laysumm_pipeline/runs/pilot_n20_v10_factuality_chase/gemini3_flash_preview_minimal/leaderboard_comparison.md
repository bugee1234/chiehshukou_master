# BioLaySumm Task 1.1 Leaderboard Comparison

> Caution: these experiment scores use a 20-article validation pilot (10 PLOS and 10 eLife), while the published leaderboard uses the official test set. Metric positions below are descriptive, not official ranks.

| Metric | Initial | Initial position | Rewritten | Rewritten position | Delta |
|---|---:|---:|---:|---:|---:|
| ROUGE | 0.320 | 9/17 | 0.320 | 9/17 | +0.000 |
| BLEU | 5.375 | 10/17 | 5.343 | 10/17 | -0.032 |
| METEOR | 0.241 | 11/17 | 0.241 | 10/17 | +0.001 |
| BERTScore | 0.851 | 10/17 | 0.851 | 10/17 | -0.000 |
| FKGL | 10.602 | 2/17 | 10.650 | 2/17 | +0.047 |
| DCRS | 12.474 | 17/17 | 12.501 | 17/17 | +0.027 |
| CLI | 14.798 | 13/17 | 14.760 | 13/17 | -0.038 |
| LENS | 75.624 | 5/17 | 75.285 | 5/17 | -0.339 |
| AlignScore | 0.771 | 4/17 | 0.773 | 4/17 | +0.002 |
| SummaC | 0.692 | 2/17 | 0.682 | 2/17 | -0.009 |

Positions are computed independently per metric against the 16 published systems plus the experiment variant. For FKGL, DCRS, and CLI, lower is treated as better.

## Official-style Final rank

> Pilot 20-article validation (10 PLOS + 10 eLife) versus official test 142+142. Final ranks are computed by min-max normalizing across the 16 published teams plus our row(s), following evaluation/rank.py; not the competition's published official_rank.

| Variant | Final rank | Final score | Relevance | Readability | Factuality |
|---|---:|---:|---:|---:|---:|
| generated | 8/18 | 0.6117 | 0.6248 | 0.5700 | 0.6402 |
| rewritten | 9/18 | 0.6075 | 0.6236 | 0.5670 | 0.6320 |

Full normalized table: `official_style_rank.csv` / `official_style_rank.json`.
