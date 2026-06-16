# BioLaySumm Task 1.1 Leaderboard Comparison

> Caution: these experiment scores use a 20-article validation pilot (10 PLOS and 10 eLife), while the published leaderboard uses the official test set. Metric positions below are descriptive, not official ranks.

| Metric | Initial | Initial position | Rewritten | Rewritten position | Delta |
|---|---:|---:|---:|---:|---:|
| ROUGE | 0.315 | 10/17 | 0.315 | 10/17 | -0.001 |
| BLEU | 4.484 | 12/17 | 4.384 | 12/17 | -0.099 |
| METEOR | 0.274 | 7/17 | 0.276 | 7/17 | +0.001 |
| BERTScore | 0.854 | 9/17 | 0.853 | 9/17 | -0.001 |
| FKGL | 13.824 | 13/17 | 13.892 | 13/17 | +0.068 |
| DCRS | 11.517 | 14/17 | 11.661 | 15/17 | +0.145 |
| CLI | 13.147 | 6/17 | 12.938 | 5/17 | -0.209 |
| LENS | 78.307 | 4/17 | 78.005 | 4/17 | -0.302 |
| AlignScore | 0.550 | 16/17 | 0.576 | 16/17 | +0.026 |
| SummaC | 0.500 | 15/17 | 0.504 | 15/17 | +0.004 |

Positions are computed independently per metric against the 16 published systems plus the experiment variant. For FKGL, DCRS, and CLI, lower is treated as better.

## Official-style Final rank

> Pilot 20-article validation (10 PLOS + 10 eLife) versus official test 142+142. Final ranks are computed by min-max normalizing across the 16 published teams plus our row(s), following evaluation/rank.py; not the competition's published official_rank.

| Variant | Final rank | Final score | Relevance | Readability | Factuality |
|---|---:|---:|---:|---:|---:|
| generated | 14/18 | 0.4716 | 0.6578 | 0.5447 | 0.2123 |
| rewritten | 13/18 | 0.4780 | 0.6520 | 0.5397 | 0.2423 |

Full normalized table: `official_style_rank.csv` / `official_style_rank.json`.
