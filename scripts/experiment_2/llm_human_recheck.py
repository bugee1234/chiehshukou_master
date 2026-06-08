from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src import config
from src.utils import OpenAIClient, load_jsonl, save_json, save_jsonl


DATA_EXP2_DIR = config.DATA_DIR / "experiment_2"
RESULTS_DIR = config.RESULTS_DIR / "experiment_2" / "llm_human_recheck"

PART_B_INPUT = DATA_EXP2_DIR / "05_human_check" / "ultra_recall" / "human_check_sample.csv"
PART_C_EVAL = DATA_EXP2_DIR / "part_c_selective_keep_af" / "03_eval" / "summary_af_coverage_by_selected_keep.jsonl"
PART_C_KEEP = DATA_EXP2_DIR / "part_c_selective_keep_af" / "02_selected_keep_af" / "selected_keep_af.jsonl"


SYSTEM_PROMPT = """You are acting as an independent human-audit style scientific fact coverage judge.
Use only the supplied texts. Do not use outside knowledge.
Be strict about factual coverage, but allow faithful paraphrase and useful lay simplification.

Return only JSON:
{
  "covered": true/false,
  "confidence": "high|medium|low",
  "supporting_evidence": "short quote or selected fact id(s), or empty string",
  "reasoning": "brief explanation"
}
"""


PART_B_USER_TEMPLATE = """Task: Decide whether the expert lay summary covers the article-level atomic fact.

Decision rule:
- covered=true if the core factual content of the atomic fact is stated, paraphrased, or clearly entailed by the lay summary.
- covered=false if the lay summary omits it, contradicts it, or only mentions a broad related topic without the specific claim.
- Technical details may be simplified in the lay summary; focus on whether the core fact survives.

Atomic fact:
{fact}

Expert lay summary:
{lay_summary}

Original pipeline judgement for reference only:
- llm_covered: {llm_covered}
- llm_supporting_span: {llm_supporting_span}
"""


PART_C_USER_TEMPLATE = """Task: Decide whether the selected keep facts cover the expert summary atomic fact.

Decision rule:
- covered=true if one or more selected keep facts state, paraphrase, or clearly entail the expert summary atomic fact.
- covered=false if the selected keep facts omit it, contradict it, or are merely topically related.
- A single selected keep fact can be enough if it covers the core claim; multiple partial facts can jointly cover it.

Expert summary atomic fact:
{summary_fact}

Selected keep facts available for this article:
{selected_keep_facts}

Original pipeline judgement for reference only:
- covered_by_selected_keep_af: {covered_by_selected_keep_af}
- original_supporting_keep_facts: {original_supporting_keep_facts}
"""


def _safe_name(text: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return safe.strip("_") or "run"


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y", "keep", "covered"}


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_json_response(content: str) -> dict[str, Any]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _judge(client: OpenAIClient, model: str, user_prompt: str) -> dict[str, Any]:
    resp = client.chat(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        model=model,
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    obj = _parse_json_response(str(resp.get("content", "")))
    return {
        "covered": _coerce_bool(obj.get("covered", False)),
        "confidence": str(obj.get("confidence", "")).strip(),
        "supporting_evidence": str(obj.get("supporting_evidence", "")).strip(),
        "reasoning": str(obj.get("reasoning", "")).strip(),
        "parse_error": False,
    }


def _balanced_sample(rows: list[dict[str, Any]], label_key: str, n: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    positives = [r for r in rows if _coerce_bool(r.get(label_key))]
    negatives = [r for r in rows if not _coerce_bool(r.get(label_key))]
    half = n // 2
    sample = rng.sample(positives, min(half, len(positives)))
    sample += rng.sample(negatives, min(n - len(sample), len(negatives)))
    if len(sample) < n:
        sampled_ids = {id(r) for r in sample}
        remaining = [r for r in rows if id(r) not in sampled_ids]
        sample += rng.sample(remaining, min(n - len(sample), len(remaining)))
    rng.shuffle(sample)
    return sample


def _agreement_metrics(rows: list[dict[str, Any]], reference_key: str, judge_key: str) -> dict[str, Any]:
    pairs = [
        (_coerce_bool(r.get(reference_key)), _coerce_bool(r.get(judge_key)))
        for r in rows
        if not _coerce_bool(r.get("llm_judge_parse_error", False))
    ]
    total = len(pairs)
    agree = sum(1 for ref, judge in pairs if ref == judge)
    ref_pos_judge_pos = sum(1 for ref, judge in pairs if ref and judge)
    ref_pos_judge_neg = sum(1 for ref, judge in pairs if ref and not judge)
    ref_neg_judge_pos = sum(1 for ref, judge in pairs if not ref and judge)
    ref_neg_judge_neg = sum(1 for ref, judge in pairs if not ref and not judge)
    expected = None
    kappa = None
    if total:
        ref_pos = ref_pos_judge_pos + ref_pos_judge_neg
        ref_neg = ref_neg_judge_pos + ref_neg_judge_neg
        judge_pos = ref_pos_judge_pos + ref_neg_judge_pos
        judge_neg = ref_pos_judge_neg + ref_neg_judge_neg
        observed = agree / total
        expected = ((ref_pos / total) * (judge_pos / total)) + ((ref_neg / total) * (judge_neg / total))
        kappa = None if expected == 1 else (observed - expected) / (1 - expected)
    return {
        "n_evaluated": total,
        "agreement_count": agree,
        "agreement_rate": round(agree / total, 4) if total else None,
        "cohen_kappa": round(kappa, 4) if kappa is not None else None,
        "confusion_matrix": {
            "reference_true__judge_true": ref_pos_judge_pos,
            "reference_true__judge_false": ref_pos_judge_neg,
            "reference_false__judge_true": ref_neg_judge_pos,
            "reference_false__judge_false": ref_neg_judge_neg,
        },
    }


def _write_confusion_csv(metrics: dict[str, Any], path: Path) -> None:
    cm = metrics["confusion_matrix"]
    lines = [
        "reference,judge_true,judge_false",
        f"true,{cm['reference_true__judge_true']},{cm['reference_true__judge_false']}",
        f"false,{cm['reference_false__judge_true']},{cm['reference_false__judge_false']}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_part_b(args: argparse.Namespace, client: OpenAIClient, out_root: Path) -> None:
    rows = _read_csv(Path(args.part_b_input))
    if args.n and len(rows) > args.n:
        rows = _balanced_sample(rows, "llm_covered", args.n, args.seed)

    out_dir = out_root / "part_b_ultra_recall"
    out_rows: list[dict[str, Any]] = []
    for row in tqdm(rows, desc="part_b llm human recheck"):
        parse_error = False
        try:
            judgement = _judge(
                client,
                args.model,
                PART_B_USER_TEMPLATE.format(
                    fact=row.get("fact", ""),
                    lay_summary=row.get("lay_summary", ""),
                    llm_covered=row.get("llm_covered", ""),
                    llm_supporting_span=row.get("llm_supporting_span", ""),
                ),
            )
        except Exception as exc:
            parse_error = True
            judgement = {
                "covered": False,
                "confidence": "",
                "supporting_evidence": "",
                "reasoning": f"parse_or_runtime_error: {exc}",
                "parse_error": True,
            }
        filled = dict(row)
        filled["human_covered"] = str(judgement["covered"])
        filled["human_notes"] = judgement["reasoning"]
        filled["llm_judge_covered"] = str(judgement["covered"])
        filled["llm_judge_confidence"] = judgement["confidence"]
        filled["llm_judge_supporting_evidence"] = judgement["supporting_evidence"]
        filled["llm_judge_reasoning"] = judgement["reasoning"]
        filled["llm_judge_parse_error"] = str(parse_error)
        filled["agreement_llm_covered_vs_judge"] = str(_coerce_bool(row.get("llm_covered")) == judgement["covered"])
        filled["agreement_module_keep_vs_judge"] = str(_coerce_bool(row.get("module_keep")) == judgement["covered"])
        filled["judge_error_type"] = (
            "dangerous_skip_by_judge"
            if judgement["covered"] and not _coerce_bool(row.get("module_keep"))
            else "not_dangerous_skip_by_judge"
        )
        out_rows.append(filled)

    metrics = {
        "part": "part_b_ultra_recall",
        "time": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "input_path": str(Path(args.part_b_input)),
        "n": len(out_rows),
        "label_distribution": dict(Counter(str(r["human_covered"]) for r in out_rows)),
        "original_llm_covered_vs_llm_judge": _agreement_metrics(out_rows, "llm_covered", "llm_judge_covered"),
        "module_keep_vs_llm_judge": _agreement_metrics(out_rows, "module_keep", "llm_judge_covered"),
        "parse_error_count": sum(1 for r in out_rows if _coerce_bool(r.get("llm_judge_parse_error"))),
        "usage": OpenAIClient.get_total_usage(),
    }
    _write_csv(out_rows, out_dir / "human_check_sample_filled_by_llm_judge.csv")
    save_jsonl(out_rows, out_dir / "human_check_sample_filled_by_llm_judge.jsonl")
    save_json(metrics, out_dir / "llm_judge_agreement_metrics.json")
    _write_confusion_csv(metrics["original_llm_covered_vs_llm_judge"], out_dir / "llm_covered_vs_llm_judge_confusion.csv")


def _selected_keep_by_article(path: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in load_jsonl(path):
        grouped[str(row["article_id"])].append(row)
    return grouped


def _format_selected_keep_facts(rows: list[dict[str, Any]], max_candidates: int | None) -> str:
    selected = rows[:max_candidates] if max_candidates and max_candidates > 0 else rows
    if not selected:
        return "NONE"
    return "\n".join(
        f"- {r.get('selected_keep_af_id', '')}: {r.get('fact', '')}"
        for r in selected
    )


def run_part_c(args: argparse.Namespace, client: OpenAIClient, out_root: Path) -> None:
    rows = load_jsonl(Path(args.part_c_eval))
    rows = _balanced_sample(rows, "covered_by_selected_keep_af", args.n, args.seed)
    by_article = _selected_keep_by_article(Path(args.part_c_keep))

    out_dir = out_root / "part_c_selective_keep_af"
    out_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(tqdm(rows, desc="part_c llm human recheck"), start=1):
        article_keep = by_article.get(str(row.get("article_id")), [])
        selected_keep_text = _format_selected_keep_facts(article_keep, args.part_c_max_candidates)
        original_support = row.get("supporting_keep_facts", row.get("supporting_keep_fact", ""))
        parse_error = False
        try:
            judgement = _judge(
                client,
                args.model,
                PART_C_USER_TEMPLATE.format(
                    summary_fact=row.get("summary_fact", ""),
                    selected_keep_facts=selected_keep_text,
                    covered_by_selected_keep_af=row.get("covered_by_selected_keep_af", ""),
                    original_supporting_keep_facts=original_support,
                ),
            )
        except Exception as exc:
            parse_error = True
            judgement = {
                "covered": False,
                "confidence": "",
                "supporting_evidence": "",
                "reasoning": f"parse_or_runtime_error: {exc}",
                "parse_error": True,
            }
        filled = dict(row)
        filled["sample_id"] = f"exp2_partc_human_{idx:03d}"
        filled["human_covered"] = judgement["covered"]
        filled["human_notes"] = judgement["reasoning"]
        filled["llm_judge_covered"] = judgement["covered"]
        filled["llm_judge_confidence"] = judgement["confidence"]
        filled["llm_judge_supporting_evidence"] = judgement["supporting_evidence"]
        filled["llm_judge_reasoning"] = judgement["reasoning"]
        filled["llm_judge_parse_error"] = parse_error
        filled["llm_judge_candidate_count"] = len(article_keep)
        filled["llm_judge_candidate_limit"] = args.part_c_max_candidates or "all"
        filled["agreement_original_coverage_vs_judge"] = (
            _coerce_bool(row.get("covered_by_selected_keep_af")) == judgement["covered"]
        )
        out_rows.append(filled)

    metrics = {
        "part": "part_c_selective_keep_af",
        "time": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "input_path": str(Path(args.part_c_eval)),
        "selected_keep_path": str(Path(args.part_c_keep)),
        "n": len(out_rows),
        "seed": args.seed,
        "balanced": True,
        "part_c_max_candidates": args.part_c_max_candidates or "all",
        "label_distribution": dict(Counter(str(r["human_covered"]) for r in out_rows)),
        "original_coverage_vs_llm_judge": _agreement_metrics(
            out_rows, "covered_by_selected_keep_af", "llm_judge_covered"
        ),
        "parse_error_count": sum(1 for r in out_rows if _coerce_bool(r.get("llm_judge_parse_error"))),
        "usage": OpenAIClient.get_total_usage(),
    }
    _write_csv(out_rows, out_dir / "partc_human_check_sample_filled_by_llm_judge.csv")
    save_jsonl(out_rows, out_dir / "partc_human_check_sample_filled_by_llm_judge.jsonl")
    save_json(metrics, out_dir / "llm_judge_agreement_metrics.json")
    _write_confusion_csv(metrics["original_coverage_vs_llm_judge"], out_dir / "original_coverage_vs_llm_judge_confusion.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-as-judge human recheck for Experiment 2 Part B and Part C.")
    parser.add_argument("--part", choices=["b", "c", "all"], default="all")
    parser.add_argument("--model", default="gpt-4.1")
    parser.add_argument("--n", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--part-b-input", type=Path, default=PART_B_INPUT)
    parser.add_argument("--part-c-eval", type=Path, default=PART_C_EVAL)
    parser.add_argument("--part-c-keep", type=Path, default=PART_C_KEEP)
    parser.add_argument(
        "--part-c-max-candidates",
        type=int,
        default=0,
        help="Max selected keep facts shown to the Part C judge per article. 0 means all.",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = _safe_name(args.run_name) if args.run_name else f"{_safe_name(args.model)}_{timestamp}"
    out_root = RESULTS_DIR / run_name
    out_root.mkdir(parents=True, exist_ok=True)

    client = OpenAIClient()
    if args.part in {"b", "all"}:
        run_part_b(args, client, out_root)
    if args.part in {"c", "all"}:
        run_part_c(args, client, out_root)

    save_json(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "part": args.part,
            "model": args.model,
            "n": args.n,
            "seed": args.seed,
            "output_root": str(out_root),
            "usage": OpenAIClient.get_total_usage(),
        },
        out_root / "run_metadata.json",
    )
    print(f"[done] outputs written to {out_root}")


if __name__ == "__main__":
    main()
