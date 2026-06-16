# BioLaySumm Task 1.1 Leaderboard Comparison

> Caution: these experiment scores use a 20-article validation pilot (10 PLOS and 10 eLife), while the published leaderboard uses the official test set. Metric positions below are descriptive, not official ranks.

| Metric | Initial | Initial position | Rewritten | Rewritten position | Delta |
|---|---:|---:|---:|---:|---:|
| ROUGE | 0.285 | 12/17 | 0.287 | 12/17 | +0.002 |
| BLEU | 4.962 | 10/17 | 5.428 | 10/17 | +0.466 |
| METEOR | 0.213 | 14/17 | 0.220 | 14/17 | +0.006 |
| BERTScore | 0.840 | 13/17 | 0.839 | 14/17 | -0.001 |
| FKGL | 13.494 | 13/17 | 14.027 | 13/17 | +0.533 |
| DCRS | 13.254 | 17/17 | 13.385 | 17/17 | +0.131 |
| CLI | 16.060 | 15/17 | 16.547 | 16/17 | +0.486 |
| LENS | 60.877 | 11/17 | 61.795 | 11/17 | +0.918 |
| AlignScore | 0.825 | 4/17 | 0.812 | 4/17 | -0.013 |
| SummaC | 0.759 | 2/17 | 0.735 | 2/17 | -0.024 |

Positions are computed independently per metric against the 16 published systems plus the experiment variant. For FKGL, DCRS, and CLI, lower is treated as better.

## Official-style Final rank

> Pilot 20-article validation (10 PLOS + 10 eLife) versus official test 142+142. Final ranks are computed by min-max normalizing across the 16 published teams plus our row(s), following evaluation/rank.py; not the competition's published official_rank.

| Variant | Final rank | Final score | Relevance | Readability | Factuality |
|---|---:|---:|---:|---:|---:|
| generated | 11/18 | 0.5366 | 0.4878 | 0.3547 | 0.7672 |
| rewritten | 13/18 | 0.5151 | 0.5098 | 0.3078 | 0.7277 |

Full normalized table: `official_style_rank.csv` / `official_style_rank.json`.
