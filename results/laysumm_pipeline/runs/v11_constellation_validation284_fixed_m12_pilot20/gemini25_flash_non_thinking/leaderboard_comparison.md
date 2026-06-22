# BioLaySumm Task 1.1 Leaderboard Comparison

> Caution: these experiment scores use a 20-article validation pilot (10 PLOS and 10 eLife), while the published leaderboard uses the official test set. Metric positions below are descriptive, not official ranks.

| Metric | Initial | Initial position | Rewritten | Rewritten position | Delta |
|---|---:|---:|---:|---:|---:|
| ROUGE | 0.307 | 10/17 | 0.329 | 8/17 | +0.022 |
| BLEU | 6.761 | 7/17 | 8.611 | 4/17 | +1.851 |
| METEOR | 0.240 | 11/17 | 0.264 | 9/17 | +0.024 |
| BERTScore | 0.848 | 12/17 | 0.849 | 11/17 | +0.001 |
| FKGL | 12.212 | 5/17 | 12.198 | 4/17 | -0.014 |
| DCRS | 13.488 | 17/17 | 13.454 | 17/17 | -0.034 |
| CLI | 16.385 | 16/17 | 16.268 | 16/17 | -0.117 |
| LENS | 66.642 | 9/17 | 59.110 | 12/17 | -7.532 |
| AlignScore | 0.929 | 1/17 | 0.924 | 1/17 | -0.005 |
| SummaC | 0.932 | 1/17 | 0.930 | 1/17 | -0.002 |

Positions are computed independently per metric against the 16 published systems plus the experiment variant. For FKGL, DCRS, and CLI, lower is treated as better.

## Official-style Final rank

> Pilot 20-article validation (10 PLOS + 10 eLife) versus official test 142+142. Final ranks are computed by min-max normalizing across the 16 published teams plus our row(s), following evaluation/rank.py; not the competition's published official_rank.

| Variant | Final rank | Final score | Relevance | Readability | Factuality |
|---|---:|---:|---:|---:|---:|
| generated | 5/18 | 0.6790 | 0.6333 | 0.4036 | 1.0000 |
| rewritten | 3/18 | 0.7114 | 0.7541 | 0.3868 | 0.9932 |

Full normalized table: `official_style_rank.csv` / `official_style_rank.json`.
