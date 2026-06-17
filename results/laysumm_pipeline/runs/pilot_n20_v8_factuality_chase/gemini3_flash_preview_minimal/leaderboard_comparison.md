# BioLaySumm Task 1.1 Leaderboard Comparison

> Caution: these experiment scores use a 20-article validation pilot (10 PLOS and 10 eLife), while the published leaderboard uses the official test set. Metric positions below are descriptive, not official ranks.

| Metric | Initial | Initial position | Rewritten | Rewritten position | Delta |
|---|---:|---:|---:|---:|---:|
| ROUGE | 0.341 | 4/17 | 0.341 | 4/17 | +0.000 |
| BLEU | 5.976 | 8/17 | 5.976 | 8/17 | +0.000 |
| METEOR | 0.272 | 7/17 | 0.272 | 7/17 | +0.000 |
| BERTScore | 0.850 | 11/17 | 0.850 | 11/17 | +0.000 |
| FKGL | 12.138 | 4/17 | 12.138 | 4/17 | +0.000 |
| DCRS | 12.861 | 17/17 | 12.861 | 17/17 | +0.000 |
| CLI | 15.840 | 15/17 | 15.840 | 15/17 | +0.000 |
| LENS | 71.126 | 9/17 | 71.126 | 9/17 | +0.000 |
| AlignScore | 0.796 | 4/17 | 0.796 | 4/17 | +0.000 |
| SummaC | 0.747 | 2/17 | 0.747 | 2/17 | +0.000 |

Positions are computed independently per metric against the 16 published systems plus the experiment variant. For FKGL, DCRS, and CLI, lower is treated as better.

## Official-style Final rank

> Pilot 20-article validation (10 PLOS + 10 eLife) versus official test 142+142. Final ranks are computed by min-max normalizing across the 16 published teams plus our row(s), following evaluation/rank.py; not the competition's published official_rank.

| Variant | Final rank | Final score | Relevance | Readability | Factuality |
|---|---:|---:|---:|---:|---:|
| generated | 7/18 | 0.6276 | 0.7119 | 0.4454 | 0.7256 |
| rewritten | 8/18 | 0.6276 | 0.7119 | 0.4454 | 0.7256 |

Full normalized table: `official_style_rank.csv` / `official_style_rank.json`.
