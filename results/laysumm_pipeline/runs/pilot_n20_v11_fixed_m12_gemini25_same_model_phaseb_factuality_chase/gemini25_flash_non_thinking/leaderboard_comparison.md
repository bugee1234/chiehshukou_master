# BioLaySumm Task 1.1 Leaderboard Comparison

> Caution: these experiment scores use a 20-article validation pilot (10 PLOS and 10 eLife), while the published leaderboard uses the official test set. Metric positions below are descriptive, not official ranks.

| Metric | Initial | Initial position | Rewritten | Rewritten position | Delta |
|---|---:|---:|---:|---:|---:|
| ROUGE | 0.305 | 10/17 | 0.324 | 8/17 | +0.019 |
| BLEU | 6.159 | 7/17 | 8.208 | 4/17 | +2.049 |
| METEOR | 0.239 | 11/17 | 0.264 | 9/17 | +0.025 |
| BERTScore | 0.847 | 12/17 | 0.848 | 11/17 | +0.001 |
| FKGL | 12.114 | 4/17 | 12.019 | 4/17 | -0.095 |
| DCRS | 13.457 | 17/17 | 13.382 | 17/17 | -0.074 |
| CLI | 16.588 | 16/17 | 16.434 | 16/17 | -0.154 |
| LENS | 66.331 | 9/17 | 58.473 | 12/17 | -7.858 |
| AlignScore | 0.941 | 1/17 | 0.933 | 1/17 | -0.009 |
| SummaC | 0.919 | 2/17 | 0.924 | 1/17 | +0.005 |

Positions are computed independently per metric against the 16 published systems plus the experiment variant. For FKGL, DCRS, and CLI, lower is treated as better.

## Official-style Final rank

> Pilot 20-article validation (10 PLOS + 10 eLife) versus official test 142+142. Final ranks are computed by min-max normalizing across the 16 published teams plus our row(s), following evaluation/rank.py; not the competition's published official_rank.

| Variant | Final rank | Final score | Relevance | Readability | Factuality |
|---|---:|---:|---:|---:|---:|
| generated | 6/18 | 0.6678 | 0.6116 | 0.3974 | 0.9943 |
| rewritten | 4/18 | 0.7046 | 0.7348 | 0.3866 | 0.9926 |

Full normalized table: `official_style_rank.csv` / `official_style_rank.json`.
