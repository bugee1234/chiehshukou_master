# BioLaySumm Task 1.1 Leaderboard Comparison

> Caution: these experiment scores use a 20-article validation pilot (10 PLOS and 10 eLife), while the published leaderboard uses the official test set. Metric positions below are descriptive, not official ranks.

| Metric | Initial | Initial position | Rewritten | Rewritten position | Delta |
|---|---:|---:|---:|---:|---:|
| ROUGE | 0.336 | 4/17 | 0.339 | 4/17 | +0.003 |
| BLEU | 5.941 | 8/17 | 6.440 | 7/17 | +0.499 |
| METEOR | 0.276 | 7/17 | 0.285 | 7/17 | +0.009 |
| BERTScore | 0.849 | 11/17 | 0.847 | 12/17 | -0.001 |
| FKGL | 15.792 | 14/17 | 15.368 | 14/17 | -0.424 |
| DCRS | 12.922 | 17/17 | 12.775 | 17/17 | -0.146 |
| CLI | 16.684 | 16/17 | 16.425 | 16/17 | -0.259 |
| LENS | 68.095 | 9/17 | 68.198 | 9/17 | +0.102 |
| AlignScore | 0.619 | 13/17 | 0.604 | 14/17 | -0.015 |
| SummaC | 0.540 | 11/17 | 0.532 | 11/17 | -0.008 |

Positions are computed independently per metric against the 16 published systems plus the experiment variant. For FKGL, DCRS, and CLI, lower is treated as better.

## Official-style Final rank

> Pilot 20-article validation (10 PLOS + 10 eLife) versus official test 142+142. Final ranks are computed by min-max normalizing across the 16 published teams plus our row(s), following evaluation/rank.py; not the competition's published official_rank.

| Variant | Final rank | Final score | Relevance | Readability | Factuality |
|---|---:|---:|---:|---:|---:|
| generated | 14/18 | 0.4276 | 0.7074 | 0.2510 | 0.3245 |
| rewritten | 13/18 | 0.4414 | 0.7346 | 0.2882 | 0.3015 |

Full normalized table: `official_style_rank.csv` / `official_style_rank.json`.
