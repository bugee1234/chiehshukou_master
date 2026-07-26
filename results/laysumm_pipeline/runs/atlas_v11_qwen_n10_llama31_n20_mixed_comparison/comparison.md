# ATLAS V11 mixed-sample backbone results

> This is a descriptive mixed-sample table, not a controlled head-to-head comparison: Qwen uses 5 PLOS + 5 eLife articles (n=10), whereas Llama 3.1 uses all 10 PLOS + 10 eLife pilot articles (n=20). Meta-Llama-3.1-8B-Instruct is an availability-driven proxy rather than the exact Meta-Llama-3-8B-Instruct official baseline backbone.

| System | n | PLOS | eLife | ROUGE | BLEU | METEOR | BERTScore | FKGL | DCRS | CLI | LENS | AlignScore | SummaC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ATLAS Phase B + Qwen2.5-7B-Instruct (OpenRouter, off-the-shelf) | 10 | 5 | 5 | 0.336 | 9.511 | 0.283 | 0.853 | 14.214 | 13.550 | 17.245 | 48.448 | 0.884 | 0.901 |
| ATLAS Phase B + Meta-Llama-3.1-8B-Instruct (OpenRouter proxy, off-the-shelf) | 20 | 10 | 10 | 0.323 | 7.830 | 0.263 | 0.849 | 13.934 | 13.352 | 16.840 | 54.762 | 0.903 | 0.863 |
