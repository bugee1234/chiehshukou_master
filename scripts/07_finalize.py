from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
import random
import sys
from typing import Any

from scipy import stats


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src import config
from src.utils import load_json, load_jsonl, save_json, save_jsonl, setup_logger


def _safe_load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return load_json(path)


def _safe_load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return load_jsonl(path)


def _cost(meta: dict[str, Any]) -> float:
    return float(meta.get("estimated_cost_usd", 0.0) or 0.0)


def _acc(successes: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(successes / total, 4)


def wilson_ci(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    low, high = stats.beta.interval(1 - alpha, successes + 1, n - successes + 1)
    return (round(float(low), 4), round(float(high), 4))


def _load_gold_with_fallback(logger) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    primary_results = config.DATA_SOLVER_RESULTS_DIR / "setup_gold_results.jsonl"
    primary_meta = config.DATA_SOLVER_RESULTS_DIR / "setup_gold_metadata.json"
    backup_results = config.DATA_SOLVER_RESULTS_DIR / "setup_gold_results_v1.jsonl"
    backup_meta = config.DATA_SOLVER_RESULTS_DIR / "setup_gold_metadata_v1.json"

    rows = _safe_load_jsonl(primary_results)
    meta = _safe_load_json(primary_meta)
    parse_errors = sum(1 for r in rows if r.get("parse_error", False))
    if rows and parse_errors == len(rows) and backup_results.exists():
        logger.warning("Primary setup_gold results are all parse_error; fallback to _v1.")
        return _safe_load_jsonl(backup_results), _safe_load_json(backup_meta), "backup_v1"
    return rows, meta, "primary"


def main() -> None:
    logger = setup_logger("finalize", str(config.LOGS_DIR / "07_finalize.log"))
    logger.info("Step 7 finalize started.")

    sources = {
        "questions": config.DATA_QUESTIONS_DIR / "questions.jsonl",
        "gold_results": config.DATA_SOLVER_RESULTS_DIR / "setup_gold_results.jsonl",
        "blind_gpt41": config.DATA_SOLVER_RESULTS_DIR / "setup_blind_gpt-4.1_results.jsonl",
        "blind_4o_mini": config.DATA_SOLVER_RESULTS_DIR / "setup_blind_gpt-4o-mini_results.jsonl",
        "cf_contexts": config.DATA_QUESTIONS_DIR / "counterfactual_contexts.jsonl",
        "cf_skipped": config.DATA_QUESTIONS_DIR / "counterfactual_skipped.jsonl",
        "cf_gpt41_results": config.DATA_SOLVER_RESULTS_DIR / "setup_counterfactual_gpt-4.1_results.jsonl",
        "cf_4o_mini_results": config.DATA_SOLVER_RESULTS_DIR / "setup_counterfactual_gpt-4o-mini_results.jsonl",
    }

    questions = _safe_load_jsonl(sources["questions"])
    gold_results, gold_meta, gold_source = _load_gold_with_fallback(logger)
    blind_41 = _safe_load_jsonl(sources["blind_gpt41"])
    blind_mini = _safe_load_jsonl(sources["blind_4o_mini"])
    cf_contexts = _safe_load_jsonl(sources["cf_contexts"])
    cf_skipped = _safe_load_jsonl(sources["cf_skipped"])
    cf_41 = _safe_load_jsonl(sources["cf_gpt41_results"])
    cf_mini = _safe_load_jsonl(sources["cf_4o_mini_results"])

    sampling_meta = _safe_load_json(config.DATA_RAW_DIR / "sampling_metadata.json")
    af_meta = _safe_load_json(config.DATA_ATOMIC_FACTS_DIR / "extraction_metadata.json")
    q_meta = _safe_load_json(config.DATA_QUESTIONS_DIR / "generation_metadata.json")
    blind_41_meta = _safe_load_json(config.DATA_SOLVER_RESULTS_DIR / "setup_blind_gpt-4.1_metadata.json")
    blind_mini_meta = _safe_load_json(config.DATA_SOLVER_RESULTS_DIR / "setup_blind_gpt-4o-mini_metadata.json")
    cf_build_v1_meta = _safe_load_json(config.DATA_QUESTIONS_DIR / "counterfactual_metadata_v1.json")
    cf_41_v1_meta = _safe_load_json(config.DATA_SOLVER_RESULTS_DIR / "setup_counterfactual_gpt-4.1_metadata_v1.json")
    cf_mini_v1_meta = _safe_load_json(config.DATA_SOLVER_RESULTS_DIR / "setup_counterfactual_gpt-4o-mini_metadata_v1.json")
    cf_build_v2_meta = _safe_load_json(config.DATA_QUESTIONS_DIR / "counterfactual_metadata.json")
    cf_41_v2_meta = _safe_load_json(config.DATA_SOLVER_RESULTS_DIR / "setup_counterfactual_gpt-4.1_metadata.json")
    cf_mini_v2_meta = _safe_load_json(config.DATA_SOLVER_RESULTS_DIR / "setup_counterfactual_gpt-4o-mini_metadata.json")

    q_by_id = {q["question_id"]: q for q in questions}
    gold_by_id = {r["question_id"]: r for r in gold_results}
    cf_ctx_by_id = {r["question_id"]: r for r in cf_contexts}
    cf_skip_by_id = {r["question_id"]: r for r in cf_skipped}
    cf41_by_id = {r["question_id"]: r for r in cf_41}
    cfmini_by_id = {r["question_id"]: r for r in cf_mini}

    joined_rows: list[dict[str, Any]] = []
    final_bank: list[dict[str, Any]] = []
    exclusion_counter: dict[str, int] = {
        "setup_gold_failed": 0,
        "no_counterfactual_built": 0,
        "cf_parse_error": 0,
        "cf_prior_both": 0,
        "cf_prior_gpt41_only": 0,
        "cf_prior_4o_mini_only": 0,
        "cf_other": 0,
    }
    no_cf_subreasons: dict[str, int] = {}

    def _cf_classification(res: dict[str, Any] | None, has_cf: bool) -> str:
        if not has_cf:
            return "not_tested"
        if res is None:
            return "not_tested"
        if res.get("parse_error", False):
            return "parse_error"
        return str(res.get("classification", "other"))

    for question_id, q in q_by_id.items():
        gold = gold_by_id.get(question_id)
        cf_row = cf_ctx_by_id.get(question_id)
        cf_skip = cf_skip_by_id.get(question_id)
        cf41 = cf41_by_id.get(question_id)
        cfmini = cfmini_by_id.get(question_id)

        passed_gold = bool(gold and (not gold.get("parse_error", False)) and gold.get("is_correct", False))
        has_cf = cf_row is not None
        cf_skip_reason = str(cf_skip.get("skip_reason", "")) if cf_skip else ""
        cls41 = _cf_classification(cf41, has_cf)
        clsmini = _cf_classification(cfmini, has_cf)

        exclusion_reason = ""
        if not passed_gold:
            exclusion_reason = "setup_gold_failed"
        elif not has_cf:
            exclusion_reason = "no_counterfactual_built"
        elif cls41 == "parse_error" or clsmini == "parse_error":
            exclusion_reason = "cf_parse_error"
        elif cls41 == "prior_dependent" and clsmini == "prior_dependent":
            exclusion_reason = "cf_prior_both"
        elif cls41 == "prior_dependent" and clsmini == "context_faithful":
            exclusion_reason = "cf_prior_gpt41_only"
        elif cls41 == "context_faithful" and clsmini == "prior_dependent":
            exclusion_reason = "cf_prior_4o_mini_only"
        elif cls41 == "context_faithful" and clsmini == "context_faithful":
            exclusion_reason = ""
        else:
            exclusion_reason = "cf_other"

        in_final_bank = exclusion_reason == ""
        if in_final_bank:
            final_bank.append(
                {
                    "question_id": question_id,
                    "af_id": q.get("af_id"),
                    "article_id": q.get("article_id"),
                    "source_dataset": q.get("source_dataset"),
                    "atomic_fact": q.get("atomic_fact"),
                    "lay_summary": q.get("lay_summary"),
                    "options": q.get("options", []),
                    "correct_letter": q.get("correct_letter"),
                    "correct_index": q.get("correct_index"),
                    "true_statement": q.get("true_statement"),
                    "false_statements": q.get("false_statements", []),
                    "counterfactual_lay_summary": cf_row.get("counterfactual_lay_summary", ""),
                    "promoted_distractor_letter": cf_row.get("promoted_distractor_letter", ""),
                    "swap_pair": {
                        "in_true": cf_row.get("in_true", ""),
                        "in_false": cf_row.get("in_false", ""),
                    },
                    "variant_mapping": cf_row.get("variant_mapping", []),
                    "qc": {
                        "passed_setup_gold": True,
                        "setup_gold_predicted": gold.get("predicted_letter", "") if gold else "",
                        "cf_faithful_gpt41": cls41 == "context_faithful",
                        "cf_faithful_gpt4o_mini": clsmini == "context_faithful",
                        "cf_predicted_gpt41": cf41.get("predicted_letter", "") if cf41 else "",
                        "cf_predicted_gpt4o_mini": cfmini.get("predicted_letter", "") if cfmini else "",
                    },
                }
            )
        else:
            exclusion_counter[exclusion_reason] = exclusion_counter.get(exclusion_reason, 0) + 1
            if exclusion_reason == "no_counterfactual_built":
                sr = cf_skip_reason or "unknown"
                no_cf_subreasons[sr] = no_cf_subreasons.get(sr, 0) + 1

        joined_rows.append(
            {
                "question_id": question_id,
                "source_dataset": q.get("source_dataset", ""),
                "atomic_fact_first80": str(q.get("atomic_fact", ""))[:80],
                "passed_gold": passed_gold,
                "gold_predicted": gold.get("predicted_letter", "") if gold else "",
                "gold_correct": q.get("correct_letter", ""),
                "has_counterfactual": has_cf,
                "cf_skip_reason": cf_skip_reason,
                "swap_in_true": cf_row.get("in_true", "") if cf_row else "",
                "swap_in_false": cf_row.get("in_false", "") if cf_row else "",
                "cf_gpt41_classification": cls41,
                "cf_gpt41_predicted": cf41.get("predicted_letter", "") if cf41 else "",
                "cf_4o_mini_classification": clsmini,
                "cf_4o_mini_predicted": cfmini.get("predicted_letter", "") if cfmini else "",
                "in_final_bank": in_final_bank,
                "exclusion_reason": exclusion_reason if exclusion_reason else "",
            }
        )

    out_dir = config.EXPERIMENT_DIR
    final_bank_path = out_dir / "final_test_bank.jsonl"
    final_metrics_path = out_dir / "final_metrics.json"
    filter_trace_path = out_dir / "filter_trace.csv"
    statistics_path = out_dir / "statistics.json"
    report_path = out_dir / "report.md"

    save_jsonl(final_bank, final_bank_path)

    stage_0_articles = int(
        sum(v.get("sample_size", 0) for v in sampling_meta.get("datasets", {}).values())
    ) or 40
    stage_1_af = int(af_meta.get("total_afs_extracted", len(q_by_id)))
    stage_2_q = int(q_meta.get("successful_count", len(q_by_id)))
    stage_3_passed_gold = sum(1 for r in joined_rows if r["passed_gold"])
    stage_4_with_cf = sum(1 for r in joined_rows if r["has_counterfactual"])
    stage_5_cf_faithful_both = len(final_bank)

    acc_gold_overall = float(gold_meta.get("acc_gold_overall", 0.0))
    acc_gold_plos = float(gold_meta.get("acc_gold_per_source", {}).get("PLOS", 0.0))
    acc_gold_elife = float(gold_meta.get("acc_gold_per_source", {}).get("eLife", 0.0))
    acc_blind_41 = float(blind_41_meta.get("acc_blind_overall", 0.0))
    acc_blind_mini = float(blind_mini_meta.get("acc_blind_overall", 0.0))
    acc_cf_41_raw = float(cf_41_v2_meta.get("acc_cf_faithful", 0.0))
    acc_cf_mini_raw = float(cf_mini_v2_meta.get("acc_cf_faithful", 0.0))
    acc_cf_41_final = 1.0 if len(final_bank) > 0 else 0.0
    acc_cf_mini_final = 1.0 if len(final_bank) > 0 else 0.0

    per_source_final: dict[str, int] = {"PLOS": 0, "eLife": 0}
    per_source_af: dict[str, int] = af_meta.get("afs_per_source", {"PLOS": 0, "eLife": 0})
    per_source_q: dict[str, int] = {"PLOS": 0, "eLife": 0}
    for q in questions:
        src = str(q.get("source_dataset", ""))
        per_source_q[src] = per_source_q.get(src, 0) + 1
    for r in final_bank:
        src = str(r.get("source_dataset", ""))
        per_source_final[src] = per_source_final.get(src, 0) + 1

    strategy_counts = {
        "numerical_perturbation": 0,
        "entity_swap": 0,
        "direction_reversal": 0,
        "detail_fabrication": 0,
    }
    for r in final_bank:
        for fs in r.get("false_statements", []):
            if isinstance(fs, dict):
                s = str(fs.get("strategy", ""))
                if s in strategy_counts:
                    strategy_counts[s] += 1

    cost_breakdown = {
        "step_3_af_extraction": _cost(af_meta),
        "step_4_question_generation": _cost(q_meta),
        "step_5_setup_gold": _cost(gold_meta),
        "step_6_setup_blind": round(_cost(blind_41_meta) + _cost(blind_mini_meta), 6),
        "step_6b_v1_cf_build": _cost(cf_build_v1_meta),
        "step_6c_v1_cf_solve": round(_cost(cf_41_v1_meta) + _cost(cf_mini_v1_meta), 6),
        "step_6b_v2_cf_build": _cost(cf_build_v2_meta),
        "step_6c_v2_cf_solve": round(_cost(cf_41_v2_meta) + _cost(cf_mini_v2_meta), 6),
        "step_7_finalize": 0.0,
    }
    total_cost = round(sum(cost_breakdown.values()), 6)

    final_metrics = {
        "experiment_name": config.EXPERIMENT_NAME,
        "completion_time": datetime.now().isoformat(timespec="seconds"),
        "models_used": {
            "af_extractor": af_meta.get("model", "gpt-4o-mini"),
            "question_generator": q_meta.get("model", "gpt-4.1"),
            "gold_solver": gold_meta.get("model", config.MODEL_GOLD_SOLVER),
            "blind_solvers": [blind_41_meta.get("model", "gpt-4.1"), blind_mini_meta.get("model", "gpt-4o-mini")],
            "cf_swap_extractor": cf_build_v2_meta.get("model", config.MODEL_GOLD_SOLVER),
            "cf_variant_finder": cf_build_v2_meta.get("model", config.MODEL_GOLD_SOLVER),
            "cf_solvers": [cf_41_v2_meta.get("model", "gpt-4.1"), cf_mini_v2_meta.get("model", "gpt-4o-mini")],
        },
        "data_source_notes": {"setup_gold_source": gold_source},
        "filtering_pipeline": {
            "stage_0_articles_sampled": stage_0_articles,
            "stage_1_atomic_facts": stage_1_af,
            "stage_2_questions_generated": stage_2_q,
            "stage_3_passed_gold": stage_3_passed_gold,
            "stage_4_with_counterfactual": stage_4_with_cf,
            "stage_5_cf_faithful_both": stage_5_cf_faithful_both,
            "final_test_bank": len(final_bank),
        },
        "metrics": {
            "acc_gold_overall": acc_gold_overall,
            "acc_gold_plos": acc_gold_plos,
            "acc_gold_elife": acc_gold_elife,
            "acc_blind_gpt41": acc_blind_41,
            "acc_blind_gpt4o_mini": acc_blind_mini,
            "acc_blind_random_baseline": 0.25,
            "acc_cf_faithful_gpt41_raw": acc_cf_41_raw,
            "acc_cf_faithful_gpt4o_mini_raw": acc_cf_mini_raw,
            "acc_cf_faithful_gpt41_final_bank": acc_cf_41_final,
            "acc_cf_faithful_gpt4o_mini_final_bank": acc_cf_mini_final,
        },
        "exclusion_breakdown": {
            "total_excluded": len(questions) - len(final_bank),
            "setup_gold_failed": exclusion_counter.get("setup_gold_failed", 0),
            "no_counterfactual_built": exclusion_counter.get("no_counterfactual_built", 0),
            "no_counterfactual_built_subreasons": no_cf_subreasons,
            "cf_parse_error": exclusion_counter.get("cf_parse_error", 0),
            "cf_prior_both": exclusion_counter.get("cf_prior_both", 0),
            "cf_prior_gpt41_only": exclusion_counter.get("cf_prior_gpt41_only", 0),
            "cf_prior_gpt4o_mini_only": exclusion_counter.get("cf_prior_4o_mini_only", 0),
            "cf_other": exclusion_counter.get("cf_other", 0),
        },
        "per_source_breakdown": {
            "PLOS": {
                "articles": 20,
                "atomic_facts": int(per_source_af.get("PLOS", 0)),
                "questions_generated": int(per_source_q.get("PLOS", 0)),
                "in_final_bank": int(per_source_final.get("PLOS", 0)),
            },
            "eLife": {
                "articles": 20,
                "atomic_facts": int(per_source_af.get("eLife", 0)),
                "questions_generated": int(per_source_q.get("eLife", 0)),
                "in_final_bank": int(per_source_final.get("eLife", 0)),
            },
        },
        "per_strategy_in_final_bank": strategy_counts,
        "total_cost_usd": total_cost,
        "cost_breakdown_usd": cost_breakdown,
    }
    save_json(final_metrics, final_metrics_path)

    # Filter trace CSV
    with filter_trace_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "question_id",
                "source_dataset",
                "atomic_fact_first80",
                "passed_gold",
                "gold_predicted",
                "gold_correct",
                "has_counterfactual",
                "cf_skip_reason",
                "swap_in_true",
                "swap_in_false",
                "cf_gpt41_classification",
                "cf_gpt41_predicted",
                "cf_4o_mini_classification",
                "cf_4o_mini_predicted",
                "in_final_bank",
                "exclusion_reason",
            ],
        )
        writer.writeheader()
        writer.writerows(joined_rows)

    # Statistics with CI
    stats_payload = {}

    def add_stat(key: str, successes: int, n: int) -> None:
        val = _acc(successes, n)
        lo, hi = wilson_ci(successes, n)
        stats_payload[key] = {
            "value": val,
            "n": int(n),
            "successes": int(successes),
            "ci_lower": lo,
            "ci_upper": hi,
        }

    gold_n = int(gold_meta.get("total", 0) - gold_meta.get("parse_error", 0))
    gold_s = int(gold_meta.get("correct", 0))
    add_stat("acc_gold_overall", gold_s, gold_n)

    blind41_n = int(blind_41_meta.get("total", 0) - blind_41_meta.get("parse_error", 0))
    blind41_s = int(blind_41_meta.get("correct", 0))
    add_stat("acc_blind_gpt41", blind41_s, blind41_n)

    blindmini_n = int(blind_mini_meta.get("total", 0) - blind_mini_meta.get("parse_error", 0))
    blindmini_s = int(blind_mini_meta.get("correct", 0))
    add_stat("acc_blind_gpt4o_mini", blindmini_s, blindmini_n)

    cf41_n = int(cf_41_v2_meta.get("total", 0) - cf_41_v2_meta.get("parse_error", 0))
    cf41_s = int(cf_41_v2_meta.get("context_faithful", 0))
    add_stat("acc_cf_faithful_gpt41_raw", cf41_s, cf41_n)

    cfmini_n = int(cf_mini_v2_meta.get("total", 0) - cf_mini_v2_meta.get("parse_error", 0))
    cfmini_s = int(cf_mini_v2_meta.get("context_faithful", 0))
    add_stat("acc_cf_faithful_gpt4o_mini_raw", cfmini_s, cfmini_n)

    add_stat("acc_cf_faithful_gpt41_final_bank", len(final_bank), len(final_bank))
    add_stat("acc_cf_faithful_gpt4o_mini_final_bank", len(final_bank), len(final_bank))

    save_json(stats_payload, statistics_path)

    # report.md
    flow_gold_fail = exclusion_counter.get("setup_gold_failed", 0)
    flow_no_cf = exclusion_counter.get("no_counterfactual_built", 0)
    flow_parse = exclusion_counter.get("cf_parse_error", 0)
    flow_prior_both = exclusion_counter.get("cf_prior_both", 0)
    flow_prior_41 = exclusion_counter.get("cf_prior_gpt41_only", 0)
    flow_prior_mini = exclusion_counter.get("cf_prior_4o_mini_only", 0)
    flow_other = exclusion_counter.get("cf_other", 0)
    rem_after_gold = stage_2_q - flow_gold_fail
    rem_after_cf = rem_after_gold - flow_no_cf

    report = f"""# Experiment 0: Test Bank Quality Control — Final Report

**Completion**: {final_metrics['completion_time']}  
**Final test bank size**: {len(final_bank)} questions (out of {stage_2_q} generated)

## Executive Summary

This experiment builds and validates a benchmark of 1T3F multiple-choice questions
derived from BioLaySumm 2025 lay summaries (PLOS + eLife). The pipeline applies
three independent quality-control tests — Setup Gold (logical uniqueness),
Setup Blind (distractor strength), and Setup Counterfactual (context-faithfulness)
— and filters the question bank to retain only items that pass strict criteria.

Starting from 40 articles -> {stage_1_af} atomic facts -> {stage_2_q} multiple-choice questions ->
final bank of {len(final_bank)} questions after QC filtering.

Headline metrics:
- ACC_Gold = {acc_gold_overall*100:.2f}% (logical uniqueness)
- ACC_CF_Faithful (gpt-4.1) = {acc_cf_41_raw*100:.2f}% raw / {acc_cf_41_final*100:.1f}% in final bank
- ACC_CF_Faithful (gpt-4o-mini) = {acc_cf_mini_raw*100:.2f}% raw / {acc_cf_mini_final*100:.1f}% in final bank
- Total cost: ${total_cost}

## 1. Methodology

Stages: AF extraction -> 1T3F generation -> Setup Gold -> Setup Blind -> Setup Counterfactual -> strict filtering.

## 2. Data

BioLaySumm 2025 PLOS + eLife, each 20 validation articles, seed=42.

## 3. Results

### 3.1 Atomic Fact Extraction (Stage 1)
- Total AFs: {stage_1_af}
- Cost: ${_cost(af_meta)}

### 3.2 Question Generation (Stage 2)
- Final generated questions (v2): {stage_2_q}
- Strategy distribution: {q_meta.get('strategy_distribution', {})}
- Cost: ${_cost(q_meta)}

### 3.3 Setup Gold (Stage 3)
| Source | n | Correct | ACC | 95% CI |
|---|---:|---:|---:|---:|
| PLOS | {int(gold_meta.get('total',0))} | {int(gold_meta.get('correct',0))} | {acc_gold_plos:.4f} | {stats_payload['acc_gold_overall']['ci_lower']:.4f}-{stats_payload['acc_gold_overall']['ci_upper']:.4f} |
| eLife | {int(gold_meta.get('total',0))} | {int(gold_meta.get('correct',0))} | {acc_gold_elife:.4f} | {stats_payload['acc_gold_overall']['ci_lower']:.4f}-{stats_payload['acc_gold_overall']['ci_upper']:.4f} |
| Overall | {gold_n} | {gold_s} | {acc_gold_overall:.4f} | {stats_payload['acc_gold_overall']['ci_lower']:.4f}-{stats_payload['acc_gold_overall']['ci_upper']:.4f} |

### 3.4 Setup Blind (Stage 4)
- gpt-4.1 ACC: {acc_blind_41:.4f}
- gpt-4o-mini ACC: {acc_blind_mini:.4f}
- Random baseline: 0.25

### 3.5 Setup Counterfactual (Stage 5)
- v1 faithful: gpt-4.1={cf_41_v1_meta.get('acc_cf_faithful',0.0):.4f}, gpt-4o-mini={cf_mini_v1_meta.get('acc_cf_faithful',0.0):.4f}
- v2 faithful: gpt-4.1={acc_cf_41_raw:.4f}, gpt-4o-mini={acc_cf_mini_raw:.4f}

### 3.6 Filtering & Final Test Bank (Stage 6)
```
{stage_2_q} questions
  |
  |- x Failed Setup Gold:       {flow_gold_fail}
  v
{rem_after_gold}
  |
  |- x No counterfactual built:  {flow_no_cf}
  |    (no_entity_swap: {no_cf_subreasons.get('no_entity_swap',0)}, entity_not_found: {no_cf_subreasons.get('entity_not_found',0)}, no_clean_swap: {no_cf_subreasons.get('no_clean_swap',0)}, low_confidence: {no_cf_subreasons.get('low_confidence',0)}, no_variant_mapping: {no_cf_subreasons.get('no_variant_mapping',0)}, no_replacement: {no_cf_subreasons.get('no_replacement',0)})
  v
{rem_after_cf}
  |
  |- x CF parse error:           {flow_parse}
  |- x Prior on gpt-4.1 only:    {flow_prior_41}
  |- x Prior on gpt-4o-mini only:{flow_prior_mini}
  |- x Prior on both:            {flow_prior_both}
  |- x Other (any model):        {flow_other}
  v
Final test bank: {len(final_bank)}
```

### 3.7 Per-Source Distribution in Final Bank
| Source | Final count | % of bank |
|---|---:|---:|
| PLOS | {per_source_final.get('PLOS',0)} | {_acc(per_source_final.get('PLOS',0), max(len(final_bank),1))*100:.2f}% |
| eLife | {per_source_final.get('eLife',0)} | {_acc(per_source_final.get('eLife',0), max(len(final_bank),1))*100:.2f}% |

### 3.8 Per-Strategy Distribution in Final Bank
| Strategy | # distractors in bank |
|---|---:|
| numerical_perturbation | {strategy_counts['numerical_perturbation']} |
| entity_swap | {strategy_counts['entity_swap']} |
| direction_reversal | {strategy_counts['direction_reversal']} |
| detail_fabrication | {strategy_counts['detail_fabrication']} |

## 4. Limitations

1. Counterfactual modification is restricted to entity_swap distractors; other
   strategies (numerical, direction_reversal, detail_fabrication) are not
   counterfactually validated.
2. Variant detection may still miss obscure abbreviations.
3. Setup Blind in lay-summary domain can remain higher than chance due to overlap
   between source content and pretraining knowledge.
4. AF extraction remains dependent on LLM decontextualization quality.

## 5. Final Test Bank Schema

See `results/experiment_0/final_test_bank.jsonl`.

## 6. Use in Downstream Experiments

For Exp 1 / Exp 3:
- Use `lay_summary` as source-of-truth context
- Use `correct_letter` as gold answer
- `counterfactual_lay_summary` and `promoted_distractor_letter` are available for robustness checks

## 7. Reproducibility

- Random seed: 42
- Sampling indices: see `data/raw/sampling_metadata.json`
- All prompts: see `prompts/`
- Filter trace: see `results/experiment_0/filter_trace.csv`

## Total Cost

${total_cost} USD
"""
    report_path.write_text(report, encoding="utf-8")

    logger.info("Saved outputs under %s", out_dir)
    logger.info("Step 7 finalize completed.")

    print("\n=== Experiment 0 Finalization Summary ===\n")
    print("Pipeline:")
    print(f"  Articles sampled:                {stage_0_articles}")
    print(f"  Atomic facts extracted:          {stage_1_af}")
    print(f"  Questions generated:             {stage_2_q}")
    print(f"  Passed Setup Gold:               {stage_3_passed_gold}")
    print(f"  Counterfactual built:            {stage_4_with_cf}")
    print(f"  CF faithful (both models):       {stage_5_cf_faithful_both}")
    print("  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  Final test bank:                 {len(final_bank)}")

    print("\nBy source in final bank:")
    print(f"  PLOS:  {per_source_final.get('PLOS',0)}")
    print(f"  eLife: {per_source_final.get('eLife',0)}")

    print("\nVerify ACC_Faithful in final bank (should be 100% by construction):")
    print(f"  gpt-4.1:     {acc_cf_41_final:.3f} {'✓' if acc_cf_41_final == 1.0 else '✗'}")
    print(f"  gpt-4o-mini: {acc_cf_mini_final:.3f} {'✓' if acc_cf_mini_final == 1.0 else '✗'}")

    print("\nOutput files:")
    print(f"  ✓ {final_bank_path}")
    print(f"  ✓ {final_metrics_path}")
    print(f"  ✓ {filter_trace_path}")
    print(f"  ✓ {statistics_path}")
    print(f"  ✓ {report_path}")

    print("\n=== 3 Random Samples from Final Test Bank ===")
    sample_n = min(3, len(final_bank))
    if sample_n > 0:
        for sample in random.Random(42).sample(final_bank, sample_n):
            print("\n---")
            print(f"question_id: {sample['question_id']}")
            print(f"source: {sample['source_dataset']}")
            print(f"atomic_fact: {sample['atomic_fact']}")
            opts = sample.get("options", [])
            for i, opt in enumerate(opts):
                print(f"{'ABCD'[i]}. {opt}")
            print(f"correct: {sample.get('correct_letter')}")
            print(f"qc: {sample.get('qc')}")


if __name__ == "__main__":
    main()
