# BioLaySumm Task 1.1 Leaderboard Comparison

> Caution: these experiment scores use a 20-article validation pilot (10 PLOS and 10 eLife), while the published leaderboard uses the official test set. Metric positions below are descriptive, not official ranks.

| Metric | Initial | Initial position | Rewritten | Rewritten position | Delta |
|---|---:|---:|---:|---:|---:|
| ROUGE | 0.336 | 4/17 | 0.335 | 4/17 | -0.000 |
| BLEU | 6.282 | 7/17 | 6.685 | 7/17 | +0.403 |
| METEOR | 0.270 | 7/17 | 0.271 | 7/17 | +0.001 |
| BERTScore | 0.851 | 9/17 | 0.850 | 10/17 | -0.001 |
| FKGL | 10.741 | 2/17 | 10.639 | 2/17 | -0.102 |
| DCRS | 12.284 | 17/17 | 12.310 | 17/17 | +0.026 |
| CLI | 14.418 | 13/17 | 14.348 | 13/17 | -0.070 |
| LENS | 75.137 | 5/17 | 74.538 | 5/17 | -0.599 |
| AlignScore | 0.753 | 6/17 | 0.744 | 7/17 | -0.009 |
| SummaC | 0.675 | 3/17 | 0.682 | 2/17 | +0.007 |

Positions are computed independently per metric against the 16 published systems plus the experiment variant. For FKGL, DCRS, and CLI, lower is treated as better.

## Official-style Final rank

> Pilot 20-article validation (10 PLOS + 10 eLife) versus official test 142+142. Final ranks are computed by min-max normalizing across the 16 published teams plus our row(s), following evaluation/rank.py; not the competition's published official_rank.

| Variant | Final rank | Final score | Relevance | Readability | Factuality |
|---|---:|---:|---:|---:|---:|
| generated | 7/18 | 0.6326 | 0.7143 | 0.5799 | 0.6037 |
| rewritten | 6/18 | 0.6368 | 0.7239 | 0.5835 | 0.6028 |

Full normalized table: `official_style_rank.csv` / `official_style_rank.json`.
