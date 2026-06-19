# BioLaySumm Task 1.1 Leaderboard Comparison

> Caution: these experiment scores use a 20-article validation pilot (10 PLOS and 10 eLife), while the published leaderboard uses the official test set. Metric positions below are descriptive, not official ranks.

| Metric | Initial | Initial position | Rewritten | Rewritten position | Delta |
|---|---:|---:|---:|---:|---:|
| ROUGE | 0.324 | 8/17 | 0.324 | 8/17 | -0.001 |
| BLEU | 5.193 | 10/17 | 5.188 | 10/17 | -0.005 |
| METEOR | 0.249 | 10/17 | 0.248 | 10/17 | -0.001 |
| BERTScore | 0.850 | 11/17 | 0.850 | 11/17 | -0.000 |
| FKGL | 11.629 | 2/17 | 11.588 | 2/17 | -0.042 |
| DCRS | 12.818 | 17/17 | 12.831 | 17/17 | +0.013 |
| CLI | 15.616 | 15/17 | 15.587 | 15/17 | -0.029 |
| LENS | 73.085 | 5/17 | 72.733 | 6/17 | -0.352 |
| AlignScore | 0.773 | 4/17 | 0.772 | 4/17 | -0.001 |
| SummaC | 0.674 | 3/17 | 0.681 | 3/17 | +0.006 |

Positions are computed independently per metric against the 16 published systems plus the experiment variant. For FKGL, DCRS, and CLI, lower is treated as better.

## Official-style Final rank

> Pilot 20-article validation (10 PLOS + 10 eLife) versus official test 142+142. Final ranks are computed by min-max normalizing across the 16 published teams plus our row(s), following evaluation/rank.py; not the competition's published official_rank.

| Variant | Final rank | Final score | Relevance | Readability | Factuality |
|---|---:|---:|---:|---:|---:|
| generated | 11/18 | 0.5800 | 0.6339 | 0.4830 | 0.6230 |
| rewritten | 10/18 | 0.5812 | 0.6302 | 0.4841 | 0.6294 |

Full normalized table: `official_style_rank.csv` / `official_style_rank.json`.
