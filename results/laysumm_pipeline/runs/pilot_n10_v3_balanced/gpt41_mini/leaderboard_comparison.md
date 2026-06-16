# BioLaySumm Task 1.1 Leaderboard Comparison

> Caution: these experiment scores use a 20-article validation pilot (10 PLOS and 10 eLife), while the published leaderboard uses the official test set. Metric positions below are descriptive, not official ranks.

| Metric | Initial | Initial position | Rewritten | Rewritten position | Delta |
|---|---:|---:|---:|---:|---:|
| ROUGE | 0.313 | 10/17 | 0.313 | 10/17 | -0.000 |
| BLEU | 5.326 | 10/17 | 5.612 | 8/17 | +0.286 |
| METEOR | 0.271 | 7/17 | 0.273 | 7/17 | +0.002 |
| BERTScore | 0.841 | 13/17 | 0.840 | 14/17 | -0.001 |
| FKGL | 16.161 | 15/17 | 16.356 | 15/17 | +0.195 |
| DCRS | 13.470 | 17/17 | 13.725 | 17/17 | +0.255 |
| CLI | 17.141 | 17/17 | 17.764 | 17/17 | +0.623 |
| LENS | 65.667 | 9/17 | 61.634 | 11/17 | -4.032 |
| AlignScore | 0.710 | 8/17 | 0.751 | 6/17 | +0.041 |
| SummaC | 0.630 | 7/17 | 0.703 | 2/17 | +0.073 |

Positions are computed independently per metric against the 16 published systems plus the experiment variant. For FKGL, DCRS, and CLI, lower is treated as better.

## Official-style Final rank

> Pilot 20-article validation (10 PLOS + 10 eLife) versus official test 142+142. Final ranks are computed by min-max normalizing across the 16 published teams plus our row(s), following evaluation/rank.py; not the competition's published official_rank.

| Variant | Final rank | Final score | Relevance | Readability | Factuality |
|---|---:|---:|---:|---:|---:|
| generated | 14/18 | 0.4624 | 0.6243 | 0.2504 | 0.5125 |
| rewritten | 13/18 | 0.4857 | 0.6309 | 0.1929 | 0.6333 |

Full normalized table: `official_style_rank.csv` / `official_style_rank.json`.
