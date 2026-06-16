# BioLaySumm Task 1.1 Leaderboard Comparison

> Caution: these experiment scores use a 20-article validation pilot (10 PLOS and 10 eLife), while the published leaderboard uses the official test set. Metric positions below are descriptive, not official ranks.

| Metric | Initial | Initial position | Rewritten | Rewritten position | Delta |
|---|---:|---:|---:|---:|---:|
| ROUGE | 0.329 | 8/17 | 0.327 | 8/17 | -0.002 |
| BLEU | 4.965 | 10/17 | 4.878 | 10/17 | -0.087 |
| METEOR | 0.303 | 3/17 | 0.305 | 3/17 | +0.002 |
| BERTScore | 0.854 | 9/17 | 0.853 | 9/17 | -0.001 |
| FKGL | 14.643 | 14/17 | 14.749 | 14/17 | +0.106 |
| DCRS | 11.595 | 14/17 | 11.687 | 15/17 | +0.092 |
| CLI | 13.766 | 13/17 | 13.752 | 13/17 | -0.014 |
| LENS | 75.255 | 5/17 | 74.686 | 5/17 | -0.569 |
| AlignScore | 0.504 | 16/17 | 0.514 | 16/17 | +0.010 |
| SummaC | 0.474 | 16/17 | 0.474 | 16/17 | +0.000 |

Positions are computed independently per metric against the 16 published systems plus the experiment variant. For FKGL, DCRS, and CLI, lower is treated as better.

## Official-style Final rank

> Pilot 20-article validation (10 PLOS + 10 eLife) versus official test 142+142. Final ranks are computed by min-max normalizing across the 16 published teams plus our row(s), following evaluation/rank.py; not the competition's published official_rank.

| Variant | Final rank | Final score | Relevance | Readability | Factuality |
|---|---:|---:|---:|---:|---:|
| generated | 13/18 | 0.4468 | 0.7331 | 0.4684 | 0.1390 |
| rewritten | 14/18 | 0.4436 | 0.7263 | 0.4561 | 0.1485 |

Full normalized table: `official_style_rank.csv` / `official_style_rank.json`.
