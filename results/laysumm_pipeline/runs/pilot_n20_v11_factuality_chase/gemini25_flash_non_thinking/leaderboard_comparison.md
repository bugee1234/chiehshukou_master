# BioLaySumm Task 1.1 Leaderboard Comparison

> Caution: these experiment scores use a 20-article validation pilot (10 PLOS and 10 eLife), while the published leaderboard uses the official test set. Metric positions below are descriptive, not official ranks.

| Metric | Initial | Initial position | Rewritten | Rewritten position | Delta |
|---|---:|---:|---:|---:|---:|
| ROUGE | 0.307 | 10/17 | 0.328 | 8/17 | +0.021 |
| BLEU | 6.411 | 7/17 | 8.751 | 3/17 | +2.340 |
| METEOR | 0.243 | 10/17 | 0.267 | 9/17 | +0.025 |
| BERTScore | 0.847 | 12/17 | 0.849 | 11/17 | +0.002 |
| FKGL | 12.035 | 4/17 | 11.919 | 4/17 | -0.115 |
| DCRS | 13.420 | 17/17 | 13.360 | 17/17 | -0.060 |
| CLI | 16.540 | 16/17 | 16.332 | 16/17 | -0.208 |
| LENS | 65.862 | 9/17 | 58.808 | 12/17 | -7.054 |
| AlignScore | 0.942 | 1/17 | 0.936 | 1/17 | -0.006 |
| SummaC | 0.915 | 2/17 | 0.922 | 1/17 | +0.007 |

Positions are computed independently per metric against the 16 published systems plus the experiment variant. For FKGL, DCRS, and CLI, lower is treated as better.

## Official-style Final rank

> Pilot 20-article validation (10 PLOS + 10 eLife) versus official test 142+142. Final ranks are computed by min-max normalizing across the 16 published teams plus our row(s), following evaluation/rank.py; not the competition's published official_rank.

| Variant | Final rank | Final score | Relevance | Readability | Factuality |
|---|---:|---:|---:|---:|---:|
| generated | 6/18 | 0.6735 | 0.6272 | 0.4013 | 0.9920 |
| rewritten | 3/18 | 0.7178 | 0.7633 | 0.3955 | 0.9946 |

Full normalized table: `official_style_rank.csv` / `official_style_rank.json`.
