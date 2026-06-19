# BioLaySumm Task 1.1 Leaderboard Comparison

> Caution: these experiment scores use a 20-article validation pilot (10 PLOS and 10 eLife), while the published leaderboard uses the official test set. Metric positions below are descriptive, not official ranks.

| Metric | Initial | Initial position | Rewritten | Rewritten position | Delta |
|---|---:|---:|---:|---:|---:|
| ROUGE | 0.302 | 10/17 | 0.317 | 10/17 | +0.015 |
| BLEU | 5.744 | 8/17 | 7.410 | 5/17 | +1.666 |
| METEOR | 0.241 | 11/17 | 0.258 | 9/17 | +0.017 |
| BERTScore | 0.847 | 12/17 | 0.848 | 11/17 | +0.001 |
| FKGL | 11.746 | 4/17 | 11.764 | 4/17 | +0.018 |
| DCRS | 13.301 | 17/17 | 13.271 | 17/17 | -0.030 |
| CLI | 16.257 | 16/17 | 16.134 | 15/17 | -0.123 |
| LENS | 65.745 | 9/17 | 59.518 | 12/17 | -6.226 |
| AlignScore | 0.938 | 1/17 | 0.926 | 1/17 | -0.011 |
| SummaC | 0.901 | 2/17 | 0.902 | 2/17 | +0.000 |

Positions are computed independently per metric against the 16 published systems plus the experiment variant. For FKGL, DCRS, and CLI, lower is treated as better.

## Official-style Final rank

> Pilot 20-article validation (10 PLOS + 10 eLife) versus official test 142+142. Final ranks are computed by min-max normalizing across the 16 published teams plus our row(s), following evaluation/rank.py; not the competition's published official_rank.

| Variant | Final rank | Final score | Relevance | Readability | Factuality |
|---|---:|---:|---:|---:|---:|
| generated | 6/18 | 0.6679 | 0.5991 | 0.4252 | 0.9794 |
| rewritten | 4/18 | 0.6921 | 0.6948 | 0.4115 | 0.9699 |

Full normalized table: `official_style_rank.csv` / `official_style_rank.json`.
