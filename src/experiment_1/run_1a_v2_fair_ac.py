from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src import config
from src.utils import OpenAIClient, load_jsonl, save_json, save_jsonl


# Read-only inputs from the existing 1a_v2 pipeline.
SRC_DATA_1A_V2_DIR = config.DATA_DIR / "experiment_1" / "1a_v2"
SRC_DIR_01 = SRC_DATA_1A_V2_DIR / "01_inputs"
SRC_DIR_02 = SRC_DATA_1A_V2_DIR / "02_perturbation"
SRC_DIR_03 = SRC_DATA_1A_V2_DIR / "03_rag"

# New isolated outputs for fair A/C comparison.
FAIR_DATA_DIR = config.DATA_DIR / "experiment_1" / "1a_v2_fair_ac"
FAIR_RESULTS_DIR = config.RESULTS_DIR / "experiment_1" / "1a_v2_fair_ac"
FAIR_DIR_04 = FAIR_DATA_DIR / "04_judgement"
FAIR_DIR_05 = FAIR_DATA_DIR / "05_metrics"

PROMPTS_DIR = ROOT_DIR / "src" / "experiment_1" / "prompts_1a_v2_fair_ac"


def _ensure_dirs() -> None:
    for p in [FAIR_DIR_04, FAIR_DIR_05, FAIR_RESULTS_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def _judgement_dir(top_k: int) -> Path:
    d = FAIR_DIR_04 / f"k{top_k}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _metrics_dir(top_k: int) -> Path:
    d = FAIR_DIR_05 / f"k{top_k}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _coerce_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v or "").strip().lower()
    return s in {"1", "true", "yes", "y"}


def _parse_setup_c_option_eval(obj: dict[str, Any]) -> dict[str, dict[str, Any]]:
    letters = ["A", "B", "C", "D"]
    out: dict[str, dict[str, Any]] = {
        k: {"status": "", "negation_flag": False, "evidence": "", "strictness_mismatch": False}
        for k in letters
    }
    raw_eval = obj.get("option_evaluation", {})
    if isinstance(raw_eval, dict):
        for k in letters:
            entry = raw_eval.get(k)
            if not isinstance(entry, dict):
                continue
            status = str(entry.get("status", "")).strip().lower()
            if status not in {"supported", "contradicted", "insufficient"}:
                status = ""
            out[k] = {
                "status": status,
                "negation_flag": _coerce_bool(entry.get("negation_flag", False)),
                "evidence": str(entry.get("evidence", "")).strip(),
                "strictness_mismatch": _coerce_bool(entry.get("strictness_mismatch", False)),
            }
    return out


def _decide_setup_c_answer(option_eval: dict[str, dict[str, Any]]) -> str:
    supported: list[str] = []
    for letter in ["A", "B", "C", "D"]:
        entry = option_eval.get(letter, {})
        status = str(entry.get("status", "")).strip().lower()
        has_negation = _coerce_bool(entry.get("negation_flag", False))
        strictness_mismatch = _coerce_bool(entry.get("strictness_mismatch", False))
        if status == "supported" and not has_negation and not strictness_mismatch:
            supported.append(letter)

    if len(supported) == 1:
        return supported[0]
    if len(supported) == 0:
        return "E"

    # If multiple options are marked supported, pick one with strongest explicit evidence.
    def evidence_score(letter: str) -> int:
        ev = str(option_eval.get(letter, {}).get("evidence", "")).strip()
        if not ev:
            return 0
        if ev.lower() in {"quote: none", "none"}:
            return 0
        return len(ev)

    return max(supported, key=evidence_score)


def _compute_binary_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    acc = (tp + tn) / max(len(y_true), 1)
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(acc, 4),
    }


def _load_prompt(path: Path) -> str:
    if not path.exists():
        raise RuntimeError(f"Prompt file missing: {path}")
    return path.read_text(encoding="utf-8")


def stage05_judgement(model: str, top_k: int, parallel_setups: bool = False, pilot_first_n: int | None = None) -> None:
    _ensure_dirs()
    retrieval = load_jsonl(SRC_DIR_03 / f"retrieval_topk{top_k}.jsonl")
    af_rows = load_jsonl(SRC_DIR_01 / "af_gold_for_1a_v2.jsonl")

    if pilot_first_n is not None:
        stats_path = SRC_DIR_01 / "pilot_articles_stats.json"
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        keep_ids = set(str(x) for x in stats.get("article_ids", [])[:pilot_first_n])
        retrieval = [r for r in retrieval if str(r.get("article_id", "")) in keep_ids]
        print(f"[stage05 fair] pilot_first_n={pilot_first_n} filtered_rows={len(retrieval)}")

    judgement_dir = _judgement_dir(top_k)
    out_a = judgement_dir / "setup_a_predictions.jsonl"
    out_c = judgement_dir / "setup_c_predictions.jsonl"
    out_meta = judgement_dir / "judgement_metadata.json"

    system_prompt_a = _load_prompt(PROMPTS_DIR / "setup_a_system.txt")
    system_prompt_c = _load_prompt(PROMPTS_DIR / "setup_c_system.txt")
    question_bank_path = SRC_DATA_1A_V2_DIR / "04_judgement" / "questions_1t3f.jsonl"
    if not question_bank_path.exists():
        raise RuntimeError(f"Missing existing 1T3F question bank: {question_bank_path}")
    questions_rows = load_jsonl(question_bank_path)

    questions = {str(r["af_id"]): r for r in questions_rows if isinstance(r.get("options"), list) and len(r["options"]) == 4}
    af_map = {str(r["af_id"]): r for r in af_rows}
    retrieval = [r for r in retrieval if str(r["af_id"]) in questions and str(r["af_id"]) in af_map]

    def run_setup_a() -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        local_client = OpenAIClient()
        for row in tqdm(retrieval, desc="stage05 fair | setup A judgement", total=len(retrieval)):
            chunks_txt = "\n\n".join([f"[{c['rank']}] {c['chunk_text']}" for c in row.get("retrieved_chunks", [])])
            user_msg = f"Claim:\n{row['fact']}\n\nContext Chunks:\n{chunks_txt}"
            hallucination = 1
            supporting_evidence = ""
            reason = ""
            parse_error = False
            try:
                resp = local_client.chat(
                    messages=[
                        {"role": "system", "content": system_prompt_a},
                        {"role": "user", "content": user_msg},
                    ],
                    model=model,
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                obj = json.loads(str(resp.get("content", "")))
                hallucination = int(obj.get("hallucination", 1))
                if hallucination not in (0, 1):
                    hallucination = 1
                supporting_evidence = str(obj.get("supporting_evidence", "")).strip()
                reason = str(obj.get("reason", "")).strip()
            except Exception:
                parse_error = True
                reason = "parse_or_runtime_error"
            rows.append(
                {
                    "af_id": row["af_id"],
                    "article_id": row["article_id"],
                    "source_dataset": row["source_dataset"],
                    "fact": row["fact"],
                    "top_k": row["top_k"],
                    "setup": "A_fair",
                    "pred_hallucination": hallucination,
                    "supporting_evidence": supporting_evidence,
                    "reason": reason,
                    "parse_error": parse_error,
                }
            )
        return rows

    def run_setup_c() -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        local_client = OpenAIClient()
        for row in tqdm(retrieval, desc="stage05 fair | setup C 1T3F+NOTA", total=len(retrieval)):
            af_id = str(row["af_id"])
            q = questions[af_id]
            opts = q["options"]
            chunks_txt = "\n\n".join([f"[{c['rank']}] {c['chunk_text']}" for c in row.get("retrieved_chunks", [])])
            user_msg = (
                "Context Chunks:\n"
                f"{chunks_txt}\n\n"
                "Question: Which option is supported by the context under the SAME strict policy as direct claim checking?\n"
                f"A. {opts[0]}\n"
                f"B. {opts[1]}\n"
                f"C. {opts[2]}\n"
                f"D. {opts[3]}\n"
                "E. None of the above\n"
            )
            pred_letter = ""
            reason = ""
            supporting_evidence = ""
            parse_error = False
            option_evaluation: dict[str, dict[str, Any]] = {}
            contrastive_variables: list[str] = []
            try:
                resp = local_client.chat(
                    messages=[
                        {"role": "system", "content": system_prompt_c},
                        {"role": "user", "content": user_msg},
                    ],
                    model=model,
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                obj = json.loads(str(resp.get("content", "")))
                option_evaluation = _parse_setup_c_option_eval(obj)
                pred_letter = _decide_setup_c_answer(option_evaluation)
                reason = str(obj.get("reasoning", "")).strip()
                contrastive_variables = [str(x) for x in obj.get("contrastive_variables", []) if str(x).strip()]
                if pred_letter in {"A", "B", "C", "D"}:
                    supporting_evidence = str(option_evaluation.get(pred_letter, {}).get("evidence", "")).strip() or "Quote: NONE"
                else:
                    supporting_evidence = "Quote: NONE"
            except Exception:
                parse_error = True
                reason = "parse_or_runtime_error"

            is_correct_true_pick = pred_letter == q["correct_letter"]
            pred_hallucination = 0 if is_correct_true_pick else 1
            rows.append(
                {
                    "af_id": row["af_id"],
                    "article_id": row["article_id"],
                    "source_dataset": row["source_dataset"],
                    "fact": row["fact"],
                    "top_k": row["top_k"],
                    "setup": "C_fair",
                    "question_id": q["question_id"],
                    "options": q["options"] + ["None of the above"],
                    "correct_letter": q["correct_letter"],
                    "predicted_letter": pred_letter,
                    "pred_hallucination": pred_hallucination,
                    "supporting_evidence": supporting_evidence,
                    "reason": reason,
                    "contrastive_variables": contrastive_variables,
                    "option_evaluation": option_evaluation,
                    "parse_error": parse_error,
                }
            )
        return rows

    if parallel_setups:
        with ThreadPoolExecutor(max_workers=2) as ex:
            f_a = ex.submit(run_setup_a)
            f_c = ex.submit(run_setup_c)
            rows_a = f_a.result()
            rows_c = f_c.result()
    else:
        rows_a = run_setup_a()
        rows_c = run_setup_c()

    save_jsonl(rows_a, out_a)
    save_jsonl(rows_c, out_c)
    save_json(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "model": model,
            "top_k": top_k,
            "total_rows": len(retrieval),
            "pilot_first_n": pilot_first_n,
            "question_bank_source": str(question_bank_path),
            "setup_a_parse_error": sum(1 for r in rows_a if r["parse_error"]),
            "setup_c_parse_error": sum(1 for r in rows_c if r["parse_error"]),
            "policy": "fair_ac_same_strictness_same_support_contradiction_definitions",
        },
        out_meta,
    )
    print(f"[stage05 fair] top_k={top_k} rows={len(retrieval)} -> {judgement_dir}")


def stage06_eval(top_k: int, pilot_first_n: int | None = None) -> None:
    _ensure_dirs()
    judgement_dir = _judgement_dir(top_k)
    metrics_dir = _metrics_dir(top_k)

    gt = load_jsonl(SRC_DIR_02 / "gt_labels_1a_v2.jsonl")
    a = load_jsonl(judgement_dir / "setup_a_predictions.jsonl")
    c = load_jsonl(judgement_dir / "setup_c_predictions.jsonl")

    if pilot_first_n is not None:
        stats_path = SRC_DIR_01 / "pilot_articles_stats.json"
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        keep_ids = set(str(x) for x in stats.get("article_ids", [])[:pilot_first_n])
        gt = [r for r in gt if str(r.get("article_id", "")) in keep_ids]
        a = [r for r in a if str(r.get("article_id", "")) in keep_ids]
        c = [r for r in c if str(r.get("article_id", "")) in keep_ids]

    out_a = metrics_dir / "metrics_setup_a_fair.json"
    out_c = metrics_dir / "metrics_setup_c_fair.json"
    out_cm = metrics_dir / "confusion_matrix_fair_ac.csv"
    out_task = metrics_dir / "metrics_taskwise_primary_fair_ac.json"

    a_valid = [r for r in a if not r.get("parse_error", False)]
    c_valid = [r for r in c if not r.get("parse_error", False)]

    gt_map = {str(r["af_id"]): int(r["hallucination_label"]) for r in gt}
    gt_type_map = {str(r["af_id"]): str(r["perturbation_type"]) for r in gt}
    a_map = {str(r["af_id"]): int(r["pred_hallucination"]) for r in a_valid}
    c_map = {str(r["af_id"]): int(r["pred_hallucination"]) for r in c_valid}

    def _metrics_for_setup(pred_map: dict[str, int]) -> dict[str, float]:
        af_ids_local = sorted(set(pred_map.keys()) & set(gt_map.keys()))
        y_true_local = [gt_map[k] for k in af_ids_local]
        y_pred_local = [pred_map[k] for k in af_ids_local]
        return _compute_binary_metrics(y_true_local, y_pred_local)

    ma = _metrics_for_setup(a_map)
    mc = _metrics_for_setup(c_map)
    save_json({"setup": "A_fair", **ma}, out_a)
    save_json({"setup": "C_fair", **mc}, out_c)

    lines = [
        "setup,tp,tn,fp,fn,precision,recall,f1,accuracy",
        f"A_fair,{ma['tp']},{ma['tn']},{ma['fp']},{ma['fn']},{ma['precision']},{ma['recall']},{ma['f1']},{ma['accuracy']}",
        f"C_fair,{mc['tp']},{mc['tn']},{mc['fp']},{mc['fn']},{mc['precision']},{mc['recall']},{mc['f1']},{mc['accuracy']}",
    ]
    out_cm.write_text("\n".join(lines) + "\n", encoding="utf-8")

    af_ids = sorted(gt_map.keys())

    def _task_slice(pred_map: dict[str, int], ptypes: set[str]) -> dict[str, Any]:
        af_ids_local = [af_id for af_id in af_ids if gt_type_map.get(af_id) in ptypes and af_id in pred_map]
        y_t = [gt_map[i] for i in af_ids_local]
        y_p = [pred_map[i] for i in af_ids_local]
        return {"n": len(af_ids_local), **_compute_binary_metrics(y_t, y_p)}

    taskwise = {
        "k": top_k,
        "pilot_first_n": pilot_first_n,
        "tasks": {
            "hard_omission_F_del": {
                "A_fair": _task_slice(a_map, {"F_del"}),
                "C_fair": _task_slice(c_map, {"F_del"}),
            },
            "error_detection_F_error": {
                "A_fair": _task_slice(a_map, {"F_error"}),
                "C_fair": _task_slice(c_map, {"F_error"}),
            },
            "retention_specificity_F_kept": {
                "A_fair": _task_slice(a_map, {"F_kept"}),
                "C_fair": _task_slice(c_map, {"F_kept"}),
            },
            "positive_combined_all": {
                "A_fair": _task_slice(a_map, {"F_del", "F_error"}),
                "C_fair": _task_slice(c_map, {"F_del", "F_error"}),
            },
        },
        "note": "Only framework differs (direct claim check vs 1T3F+NOTA) under shared strict policy.",
    }
    save_json(taskwise, out_task)
    print(f"[stage06 fair] top_k={top_k} A_fair_f1={ma['f1']} C_fair_f1={mc['f1']} -> {metrics_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 1A_v2 fair A/C comparison runner")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p5 = sub.add_parser("stage05_judgement")
    p5.add_argument("--model", type=str, default="gpt-4.1-mini")
    p5.add_argument("--top-k", type=int, default=5)
    p5.add_argument("--parallel-setups", action="store_true")
    p5.add_argument("--pilot-first-n", type=int, default=None)

    p6 = sub.add_parser("stage06_eval")
    p6.add_argument("--top-k", type=int, required=True)
    p6.add_argument("--pilot-first-n", type=int, default=None)

    args = parser.parse_args()
    if args.cmd == "stage05_judgement":
        stage05_judgement(
            model=args.model,
            top_k=args.top_k,
            parallel_setups=args.parallel_setups,
            pilot_first_n=args.pilot_first_n,
        )
    elif args.cmd == "stage06_eval":
        stage06_eval(top_k=args.top_k, pilot_first_n=args.pilot_first_n)


if __name__ == "__main__":
    main()
