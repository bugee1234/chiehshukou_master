# V10 three-model analysis

## Verdict
Among existing n=20 official-style pilot rows, v10 gemini25_flash_non_thinking rewritten is the highest final score: 0.6921. However, v10 is not uniformly best across all three models: GPT-4.1 mini rewritten is 0.6333 and Gemini 3 Flash preview rewritten is 0.6075.

## V10 rewritten summary
| model | final | relevance | readability | factuality | doc AlignScore | SummaC | sent mean AlignScore | low sent <0.65 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt41_mini | 0.6333 | 0.6261 | 0.4867 | 0.7871 | 0.8403 | 0.7634 | 0.8410 | 50/327 (15.3%) |
| gemini3_flash_preview_minimal | 0.6075 | 0.6236 | 0.5670 | 0.6320 | 0.7732 | 0.6823 | 0.7749 | 79/323 (24.5%) |
| gemini25_flash_non_thinking | 0.6921 | 0.6948 | 0.4115 | 0.9699 | 0.9262 | 0.9018 | 0.9260 | 18/325 (5.5%) |

## Biggest cross-model article spread
| article | dataset | mean AlignScore range | low-sentence range | means by model | low by model |
|---|---|---:|---:|---|---|
| laysumm_elife_0026 | eLife | 0.3937 | 9 | gemini25_flash_non_thinking=0.84078; gemini3_flash_preview_minimal=0.44705; gpt41_mini=0.656216 | gemini25_flash_non_thinking=3; gemini3_flash_preview_minimal=12; gpt41_mini=6 |
| laysumm_plos_0051 | PLOS | 0.2968 | 8 | gemini25_flash_non_thinking=0.92868; gemini3_flash_preview_minimal=0.631837; gpt41_mini=0.794177 | gemini25_flash_non_thinking=0; gemini3_flash_preview_minimal=8; gpt41_mini=5 |
| laysumm_elife_0070 | eLife | 0.2900 | 5 | gemini25_flash_non_thinking=0.947449; gemini3_flash_preview_minimal=0.65749; gpt41_mini=0.88308 | gemini25_flash_non_thinking=1; gemini3_flash_preview_minimal=6; gpt41_mini=1 |
| laysumm_plos_0563 | PLOS | 0.2808 | 5 | gemini25_flash_non_thinking=0.978607; gemini3_flash_preview_minimal=0.697811; gpt41_mini=0.766348 | gemini25_flash_non_thinking=0; gemini3_flash_preview_minimal=5; gpt41_mini=3 |
| laysumm_elife_0163 | eLife | 0.2802 | 5 | gemini25_flash_non_thinking=0.927928; gemini3_flash_preview_minimal=0.647773; gpt41_mini=0.765415 | gemini25_flash_non_thinking=1; gemini3_flash_preview_minimal=6; gpt41_mini=5 |
| laysumm_elife_0189 | eLife | 0.1393 | 5 | gemini25_flash_non_thinking=0.913399; gemini3_flash_preview_minimal=0.782279; gpt41_mini=0.921601 | gemini25_flash_non_thinking=1; gemini3_flash_preview_minimal=5; gpt41_mini=0 |
| laysumm_plos_1309 | PLOS | 0.2608 | 4 | gemini25_flash_non_thinking=0.889858; gemini3_flash_preview_minimal=0.629021; gpt41_mini=0.709948 | gemini25_flash_non_thinking=2; gemini3_flash_preview_minimal=6; gpt41_mini=4 |
| laysumm_elife_0188 | eLife | 0.1933 | 4 | gemini25_flash_non_thinking=0.991113; gemini3_flash_preview_minimal=0.797828; gpt41_mini=0.911549 | gemini25_flash_non_thinking=0; gemini3_flash_preview_minimal=4; gpt41_mini=2 |
| laysumm_plos_0457 | PLOS | 0.1607 | 4 | gemini25_flash_non_thinking=0.920496; gemini3_flash_preview_minimal=0.759762; gpt41_mini=0.830034 | gemini25_flash_non_thinking=1; gemini3_flash_preview_minimal=5; gpt41_mini=4 |
| laysumm_plos_0209 | PLOS | 0.1995 | 3 | gemini25_flash_non_thinking=0.89821; gemini3_flash_preview_minimal=0.698662; gpt41_mini=0.767211 | gemini25_flash_non_thinking=2; gemini3_flash_preview_minimal=5; gpt41_mini=5 |

## Files
- v10_three_model_summary.csv
- sentence_alignscore_all_models.csv
- article_alignscore_all_models.csv
- article_cross_model_spread.csv
- all_runs_official_style_top.csv
