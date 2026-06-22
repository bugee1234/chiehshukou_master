# BioLaySumm Task 1.1 Leaderboard Comparison

> Caution: these experiment scores use a 20-article validation pilot (10 PLOS and 10 eLife), while the published leaderboard uses the official test set. Metric positions below are descriptive, not official ranks.

| Metric | Initial | Initial position | Rewritten | Rewritten position | Delta |
|---|---:|---:|---:|---:|---:|
| ROUGE | 0.330 | 8/17 | 0.330 | 8/17 | -0.000 |
| BLEU | 5.926 | 8/17 | 5.939 | 8/17 | +0.013 |
| METEOR | 0.259 | 9/17 | 0.259 | 9/17 | +0.000 |
| BERTScore | 0.850 | 10/17 | 0.850 | 10/17 | -0.000 |
| FKGL | 11.943 | 4/17 | 11.911 | 4/17 | -0.032 |
| DCRS | 13.019 | 17/17 | 13.017 | 17/17 | -0.002 |
| CLI | 15.807 | 15/17 | 15.759 | 15/17 | -0.048 |
| LENS | 71.160 | 9/17 | 70.969 | 9/17 | -0.190 |
| AlignScore | 0.834 | 4/17 | 0.833 | 4/17 | -0.001 |
| SummaC | 0.808 | 2/17 | 0.807 | 2/17 | -0.001 |

Positions are computed independently per metric against the 16 published systems plus the experiment variant. For FKGL, DCRS, and CLI, lower is treated as better.

## Official-style Final rank

> Pilot 20-article validation (10 PLOS + 10 eLife) versus official test 142+142. Final ranks are computed by min-max normalizing across the 16 published teams plus our row(s), following evaluation/rank.py; not the competition's published official_rank.

| Variant | Final rank | Final score | Relevance | Readability | Factuality |
|---|---:|---:|---:|---:|---:|
| generated | 6/18 | 0.6544 | 0.6779 | 0.4549 | 0.8305 |
| rewritten | 7/18 | 0.6543 | 0.6776 | 0.4578 | 0.8276 |

Full normalized table: `official_style_rank.csv` / `official_style_rank.json`.
