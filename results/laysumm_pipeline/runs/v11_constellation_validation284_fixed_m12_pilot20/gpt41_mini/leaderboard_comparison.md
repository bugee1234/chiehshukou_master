# BioLaySumm Task 1.1 Leaderboard Comparison

> Caution: these experiment scores use a 20-article validation pilot (10 PLOS and 10 eLife), while the published leaderboard uses the official test set. Metric positions below are descriptive, not official ranks.

| Metric | Initial | Initial position | Rewritten | Rewritten position | Delta |
|---|---:|---:|---:|---:|---:|
| ROUGE | 0.286 | 12/17 | 0.324 | 8/17 | +0.038 |
| BLEU | 4.967 | 10/17 | 8.091 | 4/17 | +3.124 |
| METEOR | 0.218 | 14/17 | 0.258 | 9/17 | +0.040 |
| BERTScore | 0.847 | 12/17 | 0.849 | 11/17 | +0.003 |
| FKGL | 11.761 | 4/17 | 11.895 | 4/17 | +0.134 |
| DCRS | 13.541 | 17/17 | 13.490 | 17/17 | -0.051 |
| CLI | 15.944 | 15/17 | 15.924 | 15/17 | -0.020 |
| LENS | 68.626 | 9/17 | 58.982 | 12/17 | -9.645 |
| AlignScore | 0.851 | 4/17 | 0.858 | 4/17 | +0.007 |
| SummaC | 0.834 | 2/17 | 0.864 | 2/17 | +0.030 |

Positions are computed independently per metric against the 16 published systems plus the experiment variant. For FKGL, DCRS, and CLI, lower is treated as better.

## Official-style Final rank

> Pilot 20-article validation (10 PLOS + 10 eLife) versus official test 142+142. Final ranks are computed by min-max normalizing across the 16 published teams plus our row(s), following evaluation/rank.py; not the competition's published official_rank.

| Variant | Final rank | Final score | Relevance | Readability | Factuality |
|---|---:|---:|---:|---:|---:|
| generated | 9/18 | 0.6146 | 0.5207 | 0.4479 | 0.8754 |
| rewritten | 5/18 | 0.6855 | 0.7261 | 0.4148 | 0.9156 |

Full normalized table: `official_style_rank.csv` / `official_style_rank.json`.
