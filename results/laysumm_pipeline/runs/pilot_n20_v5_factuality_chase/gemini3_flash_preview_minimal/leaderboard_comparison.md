# BioLaySumm Task 1.1 Leaderboard Comparison

> Caution: these experiment scores use a 20-article validation pilot (10 PLOS and 10 eLife), while the published leaderboard uses the official test set. Metric positions below are descriptive, not official ranks.

| Metric | Initial | Initial position | Rewritten | Rewritten position | Delta |
|---|---:|---:|---:|---:|---:|
| ROUGE | 0.354 | 3/17 | 0.355 | 3/17 | +0.001 |
| BLEU | 8.287 | 4/17 | 8.598 | 4/17 | +0.311 |
| METEOR | 0.312 | 2/17 | 0.313 | 2/17 | +0.001 |
| BERTScore | 0.852 | 9/17 | 0.852 | 9/17 | -0.000 |
| FKGL | 13.837 | 13/17 | 13.903 | 13/17 | +0.066 |
| DCRS | 12.667 | 17/17 | 12.698 | 17/17 | +0.031 |
| CLI | 15.265 | 13/17 | 15.270 | 13/17 | +0.005 |
| LENS | 70.675 | 9/17 | 70.376 | 9/17 | -0.299 |
| AlignScore | 0.654 | 12/17 | 0.649 | 12/17 | -0.005 |
| SummaC | 0.593 | 8/17 | 0.584 | 9/17 | -0.009 |

Positions are computed independently per metric against the 16 published systems plus the experiment variant. For FKGL, DCRS, and CLI, lower is treated as better.

## Official-style Final rank

> Pilot 20-article validation (10 PLOS + 10 eLife) versus official test 142+142. Final ranks are computed by min-max normalizing across the 16 published teams plus our row(s), following evaluation/rank.py; not the competition's published official_rank.

| Variant | Final rank | Final score | Relevance | Readability | Factuality |
|---|---:|---:|---:|---:|---:|
| generated | 10/18 | 0.5608 | 0.8612 | 0.4033 | 0.4179 |
| rewritten | 11/18 | 0.5573 | 0.8715 | 0.3976 | 0.4028 |

Full normalized table: `official_style_rank.csv` / `official_style_rank.json`.
