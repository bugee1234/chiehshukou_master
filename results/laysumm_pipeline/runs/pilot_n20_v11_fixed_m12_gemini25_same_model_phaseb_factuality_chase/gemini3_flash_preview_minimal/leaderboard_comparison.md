# BioLaySumm Task 1.1 Leaderboard Comparison

> Caution: these experiment scores use a 20-article validation pilot (10 PLOS and 10 eLife), while the published leaderboard uses the official test set. Metric positions below are descriptive, not official ranks.

| Metric | Initial | Initial position | Rewritten | Rewritten position | Delta |
|---|---:|---:|---:|---:|---:|
| ROUGE | 0.328 | 8/17 | 0.328 | 8/17 | -0.000 |
| BLEU | 4.564 | 12/17 | 4.495 | 12/17 | -0.069 |
| METEOR | 0.253 | 10/17 | 0.255 | 9/17 | +0.002 |
| BERTScore | 0.850 | 10/17 | 0.850 | 10/17 | -0.000 |
| FKGL | 11.965 | 4/17 | 11.967 | 4/17 | +0.002 |
| DCRS | 12.990 | 17/17 | 12.993 | 17/17 | +0.002 |
| CLI | 16.223 | 15/17 | 16.216 | 15/17 | -0.007 |
| LENS | 68.745 | 9/17 | 68.562 | 9/17 | -0.183 |
| AlignScore | 0.867 | 2/17 | 0.865 | 2/17 | -0.002 |
| SummaC | 0.789 | 2/17 | 0.792 | 2/17 | +0.003 |

Positions are computed independently per metric against the 16 published systems plus the experiment variant. For FKGL, DCRS, and CLI, lower is treated as better.

## Official-style Final rank

> Pilot 20-article validation (10 PLOS + 10 eLife) versus official test 142+142. Final ranks are computed by min-max normalizing across the 16 published teams plus our row(s), following evaluation/rank.py; not the competition's published official_rank.

| Variant | Final rank | Final score | Relevance | Readability | Factuality |
|---|---:|---:|---:|---:|---:|
| generated | 7/18 | 0.6323 | 0.6278 | 0.4277 | 0.8414 |
| rewritten | 6/18 | 0.6325 | 0.6274 | 0.4272 | 0.8429 |

Full normalized table: `official_style_rank.csv` / `official_style_rank.json`.
