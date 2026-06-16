# BioLaySumm Task 1.1 Leaderboard Comparison

> Caution: these experiment scores use a 20-article validation pilot (10 PLOS and 10 eLife), while the published leaderboard uses the official test set. Metric positions below are descriptive, not official ranks.

| Metric | Initial | Initial position | Rewritten | Rewritten position | Delta |
|---|---:|---:|---:|---:|---:|
| ROUGE | 1.000 | 1/17 | 1.000 | 1/17 | +0.000 |
| BLEU | 100.000 | 1/17 | 100.000 | 1/17 | +0.000 |
| METEOR | 1.000 | 1/17 | 1.000 | 1/17 | +0.000 |
| BERTScore | 1.000 | 1/17 | 1.000 | 1/17 | +0.000 |
| FKGL | 13.886 | 13/17 | 13.886 | 13/17 | +0.000 |
| DCRS | 12.230 | 17/17 | 12.230 | 17/17 | +0.000 |
| CLI | 14.521 | 13/17 | 14.521 | 13/17 | +0.000 |
| LENS | 61.006 | 11/17 | 61.006 | 11/17 | +0.000 |
| AlignScore | 0.802 | 4/17 | 0.802 | 4/17 | +0.000 |
| SummaC | 0.560 | 9/17 | 0.560 | 9/17 | +0.000 |

Positions are computed independently per metric against the 16 published systems plus the experiment variant. For FKGL, DCRS, and CLI, lower is treated as better.

## Official-style Final rank

> Pilot 20-article validation (10 PLOS + 10 eLife) versus official test 142+142. Final ranks are computed by min-max normalizing across the 16 published teams plus our row(s), following evaluation/rank.py; not the competition's published official_rank.

| Variant | Final rank | Final score | Relevance | Readability | Factuality |
|---|---:|---:|---:|---:|---:|
| generated | 2/18 | 0.6422 | 1.0000 | 0.4017 | 0.5249 |
| rewritten | 1/18 | 0.6422 | 1.0000 | 0.4017 | 0.5249 |

Full normalized table: `official_style_rank.csv` / `official_style_rank.json`.
