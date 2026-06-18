# BioLaySumm Task 1.1 Leaderboard Comparison

> Caution: these experiment scores use a 20-article validation pilot (10 PLOS and 10 eLife), while the published leaderboard uses the official test set. Metric positions below are descriptive, not official ranks.

| Metric | Initial | Initial position | Rewritten | Rewritten position | Delta |
|---|---:|---:|---:|---:|---:|
| ROUGE | 0.322 | 8/17 | 0.321 | 8/17 | -0.001 |
| BLEU | 5.229 | 10/17 | 5.224 | 10/17 | -0.004 |
| METEOR | 0.241 | 10/17 | 0.241 | 11/17 | -0.000 |
| BERTScore | 0.849 | 11/17 | 0.848 | 11/17 | -0.000 |
| FKGL | 10.787 | 2/17 | 10.742 | 2/17 | -0.045 |
| DCRS | 12.836 | 17/17 | 12.858 | 17/17 | +0.022 |
| CLI | 14.960 | 13/17 | 14.954 | 13/17 | -0.006 |
| LENS | 74.237 | 5/17 | 73.913 | 5/17 | -0.324 |
| AlignScore | 0.774 | 4/17 | 0.771 | 4/17 | -0.004 |
| SummaC | 0.679 | 3/17 | 0.666 | 3/17 | -0.014 |

Positions are computed independently per metric against the 16 published systems plus the experiment variant. For FKGL, DCRS, and CLI, lower is treated as better.

## Official-style Final rank

> Pilot 20-article validation (10 PLOS + 10 eLife) versus official test 142+142. Final ranks are computed by min-max normalizing across the 16 published teams plus our row(s), following evaluation/rank.py; not the competition's published official_rank.

| Variant | Final rank | Final score | Relevance | Readability | Factuality |
|---|---:|---:|---:|---:|---:|
| generated | 8/18 | 0.5987 | 0.6156 | 0.5504 | 0.6300 |
| rewritten | 10/18 | 0.5912 | 0.6122 | 0.5502 | 0.6111 |

Full normalized table: `official_style_rank.csv` / `official_style_rank.json`.
