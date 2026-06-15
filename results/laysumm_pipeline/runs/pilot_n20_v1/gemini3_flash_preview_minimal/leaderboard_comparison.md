# BioLaySumm Task 1.1 Leaderboard Comparison

> Caution: these experiment scores use a 20-article validation pilot (10 PLOS and 10 eLife), while the published leaderboard uses the official test set. Metric positions below are descriptive, not official ranks.

| Metric | Initial | Initial position | Rewritten | Rewritten position | Delta |
|---|---:|---:|---:|---:|---:|
| ROUGE | 0.318 | 10/17 | 0.301 | 10/17 | -0.016 |
| BLEU | 4.322 | 12/17 | 4.308 | 12/17 | -0.014 |
| METEOR | 0.321 | 1/17 | 0.332 | 1/17 | +0.011 |
| BERTScore | 0.849 | 11/17 | 0.843 | 13/17 | -0.007 |
| FKGL | 13.453 | 13/17 | 14.179 | 13/17 | +0.726 |
| DCRS | 11.612 | 14/17 | 12.268 | 17/17 | +0.656 |
| CLI | 13.641 | 12/17 | 14.973 | 13/17 | +1.331 |
| LENS | 75.284 | 5/17 | 67.217 | 9/17 | -8.068 |
| AlignScore | 0.536 | 16/17 | 0.606 | 14/17 | +0.069 |
| SummaC | 0.484 | 15/17 | 0.495 | 15/17 | +0.011 |

Positions are computed independently per metric against the 16 published systems plus the experiment variant. For FKGL, DCRS, and CLI, lower is treated as better.

## Official-style Final rank

> Pilot 20-article validation (10 PLOS + 10 eLife) versus official test 142+142. Final ranks are computed by min-max normalizing across the 16 published teams plus our row(s), following evaluation/rank.py; not the competition's published official_rank.

| Variant | Final rank | Final score | Relevance | Readability | Factuality |
|---|---:|---:|---:|---:|---:|
| generated | 13/18 | 0.4715 | 0.6847 | 0.5476 | 0.1822 |
| rewritten | 14/18 | 0.4360 | 0.6559 | 0.3897 | 0.2623 |

Full normalized table: `official_style_rank.csv` / `official_style_rank.json`.
