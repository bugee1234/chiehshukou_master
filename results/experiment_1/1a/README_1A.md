# Experiment 1A (Hallucination Detection with Ground Truth)

## Binary Label Definition
- Positive (hallucination): F_del ∪ F_error
- Negative (non-hallucination): F_kept

## Fixed Settings
- Perturbation ratio: 30%
- Pilot: 10 articles (PLOS=5, eLife=5)
- Retrieval top-k: 3 (ablation later: k=1,3,5)
- Evaluation unit: AF-level

## Stage Output Contract
1) 00_registry
   - article_pool_excluding_exp0.jsonl
   - pilot_10_manifest.jsonl
2) 01_inputs
   - af_gold_for_1a.jsonl
3) 02_perturbation
   - perturbed_summary.jsonl
   - gt_labels_1a.jsonl
4) 03_rag
   - chunk_index_meta.json
   - retrieval_topk3.jsonl
5) 04_judgement
   - setup_a_predictions.jsonl
   - setup_b_predictions.jsonl
6) 05_metrics
   - metrics_setup_a.json
   - metrics_setup_b.json
   - confusion_matrix.csv
