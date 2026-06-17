# BioLaySumm Task 1.1 Leaderboard Comparison

> Caution: these experiment scores use a 20-article validation pilot (10 PLOS and 10 eLife), while the published leaderboard uses the official test set. Metric positions below are descriptive, not official ranks.

| Metric | Initial | Initial position | Rewritten | Rewritten position | Delta |
|---|---:|---:|---:|---:|---:|
| ROUGE | 0.316 | 10/17 | 0.324 | 8/17 | +0.008 |
| BLEU | 5.138 | 10/17 | 5.753 | 8/17 | +0.615 |
| METEOR | 0.266 | 9/17 | 0.275 | 7/17 | +0.009 |
| BERTScore | 0.849 | 11/17 | 0.848 | 11/17 | -0.000 |
| FKGL | 11.422 | 2/17 | 11.570 | 2/17 | +0.148 |
| DCRS | 12.985 | 17/17 | 12.966 | 17/17 | -0.020 |
| CLI | 15.021 | 13/17 | 15.123 | 13/17 | +0.101 |
| LENS | 71.913 | 7/17 | 67.271 | 9/17 | -4.642 |
| AlignScore | 0.795 | 4/17 | 0.788 | 4/17 | -0.006 |
| SummaC | 0.695 | 2/17 | 0.700 | 2/17 | +0.005 |

Positions are computed independently per metric against the 16 published systems plus the experiment variant. For FKGL, DCRS, and CLI, lower is treated as better.

## Official-style Final rank

> Pilot 20-article validation (10 PLOS + 10 eLife) versus official test 142+142. Final ranks are computed by min-max normalizing across the 16 published teams plus our row(s), following evaluation/rank.py; not the competition's published official_rank.

| Variant | Final rank | Final score | Relevance | Readability | Factuality |
|---|---:|---:|---:|---:|---:|
| generated | 9/18 | 0.6086 | 0.6449 | 0.5134 | 0.6675 |
| rewritten | 8/18 | 0.6133 | 0.6841 | 0.4891 | 0.6667 |

Full normalized table: `official_style_rank.csv` / `official_style_rank.json`.
