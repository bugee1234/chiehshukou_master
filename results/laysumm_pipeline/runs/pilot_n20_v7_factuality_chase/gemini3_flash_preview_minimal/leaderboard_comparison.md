# BioLaySumm Task 1.1 Leaderboard Comparison

> Caution: these experiment scores use a 20-article validation pilot (10 PLOS and 10 eLife), while the published leaderboard uses the official test set. Metric positions below are descriptive, not official ranks.

| Metric | Initial | Initial position | Rewritten | Rewritten position | Delta |
|---|---:|---:|---:|---:|---:|
| ROUGE | 0.338 | 4/17 | 0.337 | 4/17 | -0.001 |
| BLEU | 6.739 | 7/17 | 6.685 | 7/17 | -0.054 |
| METEOR | 0.274 | 7/17 | 0.274 | 7/17 | +0.000 |
| BERTScore | 0.851 | 10/17 | 0.850 | 10/17 | -0.001 |
| FKGL | 10.371 | 1/17 | 10.388 | 1/17 | +0.017 |
| DCRS | 12.190 | 17/17 | 12.214 | 17/17 | +0.024 |
| CLI | 14.151 | 13/17 | 14.188 | 13/17 | +0.037 |
| LENS | 74.769 | 5/17 | 73.908 | 5/17 | -0.861 |
| AlignScore | 0.755 | 5/17 | 0.754 | 5/17 | -0.001 |
| SummaC | 0.698 | 2/17 | 0.691 | 2/17 | -0.008 |

Positions are computed independently per metric against the 16 published systems plus the experiment variant. For FKGL, DCRS, and CLI, lower is treated as better.

## Official-style Final rank

> Pilot 20-article validation (10 PLOS + 10 eLife) versus official test 142+142. Final ranks are computed by min-max normalizing across the 16 published teams plus our row(s), following evaluation/rank.py; not the competition's published official_rank.

| Variant | Final rank | Final score | Relevance | Readability | Factuality |
|---|---:|---:|---:|---:|---:|
| generated | 5/18 | 0.6556 | 0.7356 | 0.5995 | 0.6316 |
| rewritten | 7/18 | 0.6488 | 0.7309 | 0.5928 | 0.6226 |

Full normalized table: `official_style_rank.csv` / `official_style_rank.json`.
