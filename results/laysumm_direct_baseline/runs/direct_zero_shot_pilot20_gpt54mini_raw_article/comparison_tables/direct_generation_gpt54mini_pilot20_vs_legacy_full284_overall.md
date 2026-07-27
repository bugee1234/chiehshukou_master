# Direct Generation Overall Metric Comparison

> GPT-5.4 Mini is pilot20 (PLOS10 + eLife10). The three legacy direct-generation rows are validation284 (PLOS142 + eLife142), so compare directionally rather than as a matched-sample test.

| System | Articles | ROUGE | BLEU | METEOR | BERTScore | FKGL | DCRS | CLI | LENS | AlignScore | SummaC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GPT-5.4 Mini direct | 20 | 0.3006 | 3.2573 | 0.2862 | 0.8383 | 12.0934 | 11.2572 | 12.9647 | 70.0438 | 0.4793 | 0.5310 |
| Gemini 2.5 Flash direct | 284 | 0.2889 | 3.2164 | 0.3105 | 0.8347 | 11.9512 | 11.3481 | 12.7981 | 64.5093 | 0.4583 | 0.5580 |
| Gemini 3 Flash Preview direct | 284 | 0.2902 | 2.8223 | 0.2971 | 0.8312 | 12.3264 | 11.4460 | 12.9154 | 58.6646 | 0.3754 | 0.5146 |
| GPT-4.1 Mini direct | 284 | 0.3132 | 4.7797 | 0.2938 | 0.8468 | 13.1719 | 11.7659 | 13.7379 | 75.6965 | 0.5027 | 0.4846 |
