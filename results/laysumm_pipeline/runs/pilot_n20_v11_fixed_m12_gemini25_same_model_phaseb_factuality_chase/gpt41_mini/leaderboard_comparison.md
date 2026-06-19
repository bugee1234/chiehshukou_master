# BioLaySumm Task 1.1 Leaderboard Comparison

> Caution: these experiment scores use a 20-article validation pilot (10 PLOS and 10 eLife), while the published leaderboard uses the official test set. Metric positions below are descriptive, not official ranks.

| Metric | Initial | Initial position | Rewritten | Rewritten position | Delta |
|---|---:|---:|---:|---:|---:|
| ROUGE | 0.297 | 10/17 | 0.320 | 9/17 | +0.023 |
| BLEU | 5.078 | 10/17 | 7.134 | 6/17 | +2.056 |
| METEOR | 0.230 | 12/17 | 0.252 | 10/17 | +0.022 |
| BERTScore | 0.847 | 12/17 | 0.850 | 11/17 | +0.003 |
| FKGL | 11.542 | 2/17 | 11.483 | 2/17 | -0.058 |
| DCRS | 13.543 | 17/17 | 13.520 | 17/17 | -0.024 |
| CLI | 16.117 | 15/17 | 15.946 | 15/17 | -0.170 |
| LENS | 68.143 | 9/17 | 59.959 | 12/17 | -8.184 |
| AlignScore | 0.862 | 2/17 | 0.866 | 2/17 | +0.003 |
| SummaC | 0.821 | 2/17 | 0.859 | 2/17 | +0.038 |

Positions are computed independently per metric against the 16 published systems plus the experiment variant. For FKGL, DCRS, and CLI, lower is treated as better.

## Official-style Final rank

> Pilot 20-article validation (10 PLOS + 10 eLife) versus official test 142+142. Final ranks are computed by min-max normalizing across the 16 published teams plus our row(s), following evaluation/rank.py; not the competition's published official_rank.

| Variant | Final rank | Final score | Relevance | Readability | Factuality |
|---|---:|---:|---:|---:|---:|
| generated | 8/18 | 0.6253 | 0.5561 | 0.4474 | 0.8725 |
| rewritten | 5/18 | 0.6783 | 0.6853 | 0.4322 | 0.9175 |

Full normalized table: `official_style_rank.csv` / `official_style_rank.json`.
