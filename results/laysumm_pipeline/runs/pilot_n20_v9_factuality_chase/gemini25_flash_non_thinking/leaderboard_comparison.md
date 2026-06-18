# BioLaySumm Task 1.1 Leaderboard Comparison

> Caution: these experiment scores use a 20-article validation pilot (10 PLOS and 10 eLife), while the published leaderboard uses the official test set. Metric positions below are descriptive, not official ranks.

| Metric | Initial | Initial position | Rewritten | Rewritten position | Delta |
|---|---:|---:|---:|---:|---:|
| ROUGE | 0.295 | 10/17 | 0.314 | 10/17 | +0.019 |
| BLEU | 5.908 | 8/17 | 7.077 | 6/17 | +1.170 |
| METEOR | 0.236 | 11/17 | 0.256 | 9/17 | +0.020 |
| BERTScore | 0.847 | 12/17 | 0.849 | 11/17 | +0.001 |
| FKGL | 11.992 | 4/17 | 11.921 | 4/17 | -0.071 |
| DCRS | 13.464 | 17/17 | 13.399 | 17/17 | -0.064 |
| CLI | 16.519 | 16/17 | 16.417 | 16/17 | -0.102 |
| LENS | 66.486 | 9/17 | 56.440 | 12/17 | -10.046 |
| AlignScore | 0.947 | 1/17 | 0.935 | 1/17 | -0.013 |
| SummaC | 0.934 | 1/17 | 0.924 | 1/17 | -0.010 |

Positions are computed independently per metric against the 16 published systems plus the experiment variant. For FKGL, DCRS, and CLI, lower is treated as better.

## Official-style Final rank

> Pilot 20-article validation (10 PLOS + 10 eLife) versus official test 142+142. Final ranks are computed by min-max normalizing across the 16 published teams plus our row(s), following evaluation/rank.py; not the competition's published official_rank.

| Variant | Final rank | Final score | Relevance | Readability | Factuality |
|---|---:|---:|---:|---:|---:|
| generated | 6/18 | 0.6647 | 0.5881 | 0.4060 | 1.0000 |
| rewritten | 4/18 | 0.6809 | 0.6800 | 0.3842 | 0.9784 |

Full normalized table: `official_style_rank.csv` / `official_style_rank.json`.
