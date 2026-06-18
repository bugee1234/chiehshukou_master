# BioLaySumm Task 1.1 Leaderboard Comparison

> Caution: these experiment scores use a 20-article validation pilot (10 PLOS and 10 eLife), while the published leaderboard uses the official test set. Metric positions below are descriptive, not official ranks.

| Metric | Initial | Initial position | Rewritten | Rewritten position | Delta |
|---|---:|---:|---:|---:|---:|
| ROUGE | 0.291 | 10/17 | 0.302 | 10/17 | +0.012 |
| BLEU | 4.963 | 10/17 | 5.141 | 10/17 | +0.179 |
| METEOR | 0.233 | 11/17 | 0.245 | 10/17 | +0.013 |
| BERTScore | 0.846 | 12/17 | 0.846 | 12/17 | +0.000 |
| FKGL | 10.922 | 2/17 | 10.855 | 2/17 | -0.067 |
| DCRS | 13.456 | 17/17 | 13.256 | 17/17 | -0.200 |
| CLI | 15.595 | 15/17 | 15.240 | 13/17 | -0.355 |
| LENS | 69.613 | 9/17 | 53.507 | 12/17 | -16.106 |
| AlignScore | 0.822 | 4/17 | 0.821 | 4/17 | -0.001 |
| SummaC | 0.747 | 2/17 | 0.746 | 2/17 | -0.002 |

Positions are computed independently per metric against the 16 published systems plus the experiment variant. For FKGL, DCRS, and CLI, lower is treated as better.

## Official-style Final rank

> Pilot 20-article validation (10 PLOS + 10 eLife) versus official test 142+142. Final ranks are computed by min-max normalizing across the 16 published teams plus our row(s), following evaluation/rank.py; not the competition's published official_rank.

| Variant | Final rank | Final score | Relevance | Readability | Factuality |
|---|---:|---:|---:|---:|---:|
| generated | 9/18 | 0.5994 | 0.5467 | 0.5005 | 0.7512 |
| rewritten | 8/18 | 0.6037 | 0.5856 | 0.4771 | 0.7484 |

Full normalized table: `official_style_rank.csv` / `official_style_rank.json`.
