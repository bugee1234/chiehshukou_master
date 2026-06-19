# BioLaySumm Task 1.1 Leaderboard Comparison

> Caution: these experiment scores use a 20-article validation pilot (10 PLOS and 10 eLife), while the published leaderboard uses the official test set. Metric positions below are descriptive, not official ranks.

| Metric | Initial | Initial position | Rewritten | Rewritten position | Delta |
|---|---:|---:|---:|---:|---:|
| ROUGE | 0.293 | 10/17 | 0.305 | 10/17 | +0.012 |
| BLEU | 4.542 | 12/17 | 5.832 | 8/17 | +1.290 |
| METEOR | 0.236 | 11/17 | 0.251 | 10/17 | +0.015 |
| BERTScore | 0.848 | 12/17 | 0.848 | 11/17 | +0.001 |
| FKGL | 10.830 | 2/17 | 11.056 | 2/17 | +0.226 |
| DCRS | 13.346 | 17/17 | 13.381 | 17/17 | +0.035 |
| CLI | 15.380 | 15/17 | 15.586 | 15/17 | +0.206 |
| LENS | 70.686 | 9/17 | 66.902 | 9/17 | -3.784 |
| AlignScore | 0.837 | 4/17 | 0.840 | 4/17 | +0.003 |
| SummaC | 0.751 | 2/17 | 0.763 | 2/17 | +0.012 |

Positions are computed independently per metric against the 16 published systems plus the experiment variant. For FKGL, DCRS, and CLI, lower is treated as better.

## Official-style Final rank

> Pilot 20-article validation (10 PLOS + 10 eLife) versus official test 142+142. Final ranks are computed by min-max normalizing across the 16 published teams plus our row(s), following evaluation/rank.py; not the competition's published official_rank.

| Variant | Final rank | Final score | Relevance | Readability | Factuality |
|---|---:|---:|---:|---:|---:|
| generated | 9/18 | 0.6125 | 0.5482 | 0.5190 | 0.7702 |
| rewritten | 7/18 | 0.6333 | 0.6261 | 0.4867 | 0.7871 |

Full normalized table: `official_style_rank.csv` / `official_style_rank.json`.
