# BioLaySumm Task 1.1 Leaderboard Comparison

> Caution: these experiment scores use a 20-article validation pilot (10 PLOS and 10 eLife), while the published leaderboard uses the official test set. Metric positions below are descriptive, not official ranks.

| Metric | Initial | Initial position | Rewritten | Rewritten position | Delta |
|---|---:|---:|---:|---:|---:|
| ROUGE | 0.300 | 10/17 | 0.309 | 10/17 | +0.009 |
| BLEU | 4.910 | 10/17 | 5.237 | 10/17 | +0.327 |
| METEOR | 0.255 | 9/17 | 0.266 | 9/17 | +0.011 |
| BERTScore | 0.845 | 13/17 | 0.846 | 13/17 | +0.001 |
| FKGL | 12.740 | 11/17 | 12.680 | 10/17 | -0.060 |
| DCRS | 13.682 | 17/17 | 13.621 | 17/17 | -0.061 |
| CLI | 16.664 | 16/17 | 16.554 | 16/17 | -0.109 |
| LENS | 66.617 | 9/17 | 60.394 | 11/17 | -6.223 |
| AlignScore | 0.860 | 4/17 | 0.859 | 4/17 | -0.001 |
| SummaC | 0.761 | 2/17 | 0.748 | 2/17 | -0.013 |

Positions are computed independently per metric against the 16 published systems plus the experiment variant. For FKGL, DCRS, and CLI, lower is treated as better.

## Official-style Final rank

> Pilot 20-article validation (10 PLOS + 10 eLife) versus official test 142+142. Final ranks are computed by min-max normalizing across the 16 published teams plus our row(s), following evaluation/rank.py; not the competition's published official_rank.

| Variant | Final rank | Final score | Relevance | Readability | Factuality |
|---|---:|---:|---:|---:|---:|
| generated | 10/18 | 0.5874 | 0.5888 | 0.3698 | 0.8036 |
| rewritten | 9/18 | 0.5921 | 0.6284 | 0.3600 | 0.7880 |

Full normalized table: `official_style_rank.csv` / `official_style_rank.json`.
