# BioLaySumm Task 1.1 Leaderboard Comparison

> Caution: these experiment scores use a 20-article validation pilot (10 PLOS and 10 eLife), while the published leaderboard uses the official test set. Metric positions below are descriptive, not official ranks.

| Metric | Initial | Initial position | Rewritten | Rewritten position | Delta |
|---|---:|---:|---:|---:|---:|
| ROUGE | 0.296 | 10/17 | 0.307 | 10/17 | +0.011 |
| BLEU | 4.947 | 10/17 | 5.947 | 8/17 | +1.000 |
| METEOR | 0.242 | 10/17 | 0.252 | 10/17 | +0.010 |
| BERTScore | 0.846 | 13/17 | 0.847 | 12/17 | +0.001 |
| FKGL | 11.519 | 2/17 | 11.536 | 2/17 | +0.018 |
| DCRS | 13.512 | 17/17 | 13.434 | 17/17 | -0.078 |
| CLI | 15.954 | 15/17 | 15.882 | 15/17 | -0.072 |
| LENS | 68.570 | 9/17 | 64.920 | 9/17 | -3.650 |
| AlignScore | 0.862 | 2/17 | 0.863 | 2/17 | +0.001 |
| SummaC | 0.766 | 2/17 | 0.773 | 2/17 | +0.007 |

Positions are computed independently per metric against the 16 published systems plus the experiment variant. For FKGL, DCRS, and CLI, lower is treated as better.

## Official-style Final rank

> Pilot 20-article validation (10 PLOS + 10 eLife) versus official test 142+142. Final ranks are computed by min-max normalizing across the 16 published teams plus our row(s), following evaluation/rank.py; not the competition's published official_rank.

| Variant | Final rank | Final score | Relevance | Readability | Factuality |
|---|---:|---:|---:|---:|---:|
| generated | 9/18 | 0.6111 | 0.5655 | 0.4570 | 0.8109 |
| rewritten | 7/18 | 0.6329 | 0.6277 | 0.4516 | 0.8195 |

Full normalized table: `official_style_rank.csv` / `official_style_rank.json`.
