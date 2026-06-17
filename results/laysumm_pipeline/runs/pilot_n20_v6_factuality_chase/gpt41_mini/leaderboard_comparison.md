# BioLaySumm Task 1.1 Leaderboard Comparison

> Caution: these experiment scores use a 20-article validation pilot (10 PLOS and 10 eLife), while the published leaderboard uses the official test set. Metric positions below are descriptive, not official ranks.

| Metric | Initial | Initial position | Rewritten | Rewritten position | Delta |
|---|---:|---:|---:|---:|---:|
| ROUGE | 0.310 | 10/17 | 0.308 | 10/17 | -0.001 |
| BLEU | 4.766 | 11/17 | 4.781 | 11/17 | +0.015 |
| METEOR | 0.254 | 9/17 | 0.253 | 9/17 | -0.001 |
| BERTScore | 0.848 | 12/17 | 0.848 | 12/17 | -0.000 |
| FKGL | 11.694 | 2/17 | 11.703 | 2/17 | +0.009 |
| DCRS | 13.272 | 17/17 | 13.296 | 17/17 | +0.024 |
| CLI | 15.464 | 15/17 | 15.518 | 15/17 | +0.054 |
| LENS | 72.186 | 7/17 | 71.864 | 7/17 | -0.322 |
| AlignScore | 0.807 | 4/17 | 0.813 | 4/17 | +0.006 |
| SummaC | 0.714 | 2/17 | 0.720 | 2/17 | +0.005 |

Positions are computed independently per metric against the 16 published systems plus the experiment variant. For FKGL, DCRS, and CLI, lower is treated as better.

## Official-style Final rank

> Pilot 20-article validation (10 PLOS + 10 eLife) versus official test 142+142. Final ranks are computed by min-max normalizing across the 16 published teams plus our row(s), following evaluation/rank.py; not the competition's published official_rank.

| Variant | Final rank | Final score | Relevance | Readability | Factuality |
|---|---:|---:|---:|---:|---:|
| generated | 10/18 | 0.5965 | 0.6044 | 0.4848 | 0.7004 |
| rewritten | 9/18 | 0.5975 | 0.6009 | 0.4797 | 0.7117 |

Full normalized table: `official_style_rank.csv` / `official_style_rank.json`.
