# BioLaySumm Task 1.1 Leaderboard Comparison

> Caution: these experiment scores use a 20-article validation pilot (10 PLOS and 10 eLife), while the published leaderboard uses the official test set. Metric positions below are descriptive, not official ranks.

| Metric | Initial | Initial position | Rewritten | Rewritten position | Delta |
|---|---:|---:|---:|---:|---:|
| ROUGE | 0.334 | 5/17 | 0.333 | 7/17 | -0.001 |
| BLEU | 6.435 | 7/17 | 6.353 | 7/17 | -0.082 |
| METEOR | 0.296 | 4/17 | 0.293 | 5/17 | -0.002 |
| BERTScore | 0.846 | 12/17 | 0.846 | 13/17 | -0.001 |
| FKGL | 14.515 | 13/17 | 14.597 | 14/17 | +0.082 |
| DCRS | 13.578 | 17/17 | 13.642 | 17/17 | +0.064 |
| CLI | 16.682 | 16/17 | 16.749 | 16/17 | +0.067 |
| LENS | 66.732 | 9/17 | 65.942 | 9/17 | -0.790 |
| AlignScore | 0.758 | 5/17 | 0.757 | 5/17 | -0.001 |
| SummaC | 0.667 | 3/17 | 0.662 | 3/17 | -0.005 |

Positions are computed independently per metric against the 16 published systems plus the experiment variant. For FKGL, DCRS, and CLI, lower is treated as better.

## Official-style Final rank

> Pilot 20-article validation (10 PLOS + 10 eLife) versus official test 142+142. Final ranks are computed by min-max normalizing across the 16 published teams plus our row(s), following evaluation/rank.py; not the competition's published official_rank.

| Variant | Final rank | Final score | Relevance | Readability | Factuality |
|---|---:|---:|---:|---:|---:|
| generated | 10/18 | 0.5469 | 0.7397 | 0.3011 | 0.5999 |
| rewritten | 12/18 | 0.5373 | 0.7294 | 0.2892 | 0.5934 |

Full normalized table: `official_style_rank.csv` / `official_style_rank.json`.
