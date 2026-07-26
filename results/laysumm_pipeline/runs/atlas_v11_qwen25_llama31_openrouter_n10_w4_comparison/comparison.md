# ATLAS V11 OpenRouter Qwen 2.5 / Llama 3.1 proxy comparison (fixed upstream, n=10)

> Both systems use off-the-shelf Instruct checkpoints through OpenRouter without BioLaySumm-specific fine-tuning. Meta-Llama-3.1-8B-Instruct is an availability-driven proxy because the original Meta-Llama-3-8B-Instruct endpoint was unavailable at run time; it is not the exact official baseline backbone. The systems use the same 5 PLOS + 5 eLife articles and the same precomputed Gemini 2.5 evidence tables and verification questions. Only each backbone's Phase-B generation, question answering, and rewriting are model-specific. Gemini outputs are not evaluated in this run.

| System | ROUGE | BLEU | METEOR | BERTScore | FKGL | DCRS | CLI | LENS | AlignScore | SummaC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ATLAS Phase B + Qwen2.5-7B-Instruct (OpenRouter, off-the-shelf) | 0.336 | 9.511 | 0.283 | 0.853 | 14.214 | 13.550 | 17.245 | 48.448 | 0.884 | 0.901 |
| ATLAS Phase B + Meta-Llama-3.1-8B-Instruct (OpenRouter proxy, off-the-shelf) | 0.340 | 9.265 | 0.278 | 0.852 | 14.304 | 13.189 | 16.546 | 56.658 | 0.885 | 0.819 |
