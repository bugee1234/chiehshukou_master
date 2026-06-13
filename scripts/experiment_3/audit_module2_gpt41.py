from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.experiment_2.run_three_model_exp2 import ProviderClient, UsageTracker
from src.utils import load_jsonl, save_json, save_jsonl


DEFAULT_MODULE2_RUN = "full_val_n284_judge_gemini31_flash_lite"
DEFAULT_MODEL_BRANCH = "gemini3_flash_preview_minimal"
DEFAULT_AUDIT_MODEL = "gpt41"


AUDIT_PROMPT = """You are independently auditing an automatically constructed biomedical lay-summary atomic fact.

Judge conservatively. Do not defer to the previous model's labels. Use the full article as the factual authority.

Evaluate these dimensions:
1. atomicity: whether the candidate expresses one independently verifiable core proposition.
2. article_support: whether the full article supports every material detail in the candidate.
3. source_span_support: whether the supplied source span alone supports every material detail in the candidate.
4. expert_coverage: whether the expert lay summary actually conveys the same core fact at comparable specificity.
5. redundancy: whether another listed final keep AF conveys substantially the same core fact or is a clear parent/child duplicate.
6. summary_priority: whether this fact should be required, optional, or excluded in a length-limited lay summary.
7. question_validity, when a multiple-choice question is supplied: whether the true option faithfully preserves the AF and exactly one A-D option is supported by the AF/article. E is None of the above.

Important distinctions:
- "partial" article or source-span support means at least one qualifier, entity, direction, comparison, causal relation, certainty level, population, or condition is unsupported.
- "broad_only" expert coverage means the expert summary mentions a related broader idea but does not convey this candidate's specific core claim.
- Do not mark a fact redundant merely because it shares a topic; the propositions must materially overlap.
- For rejected candidates, question_validity must be "not_applicable".

Return ONLY valid JSON:
{
  "atomicity": "atomic|compound|unclear",
  "article_support": "full|partial|none",
  "source_span_support": "full|partial|none",
  "expert_coverage": "full|broad_only|none|contradicted",
  "redundancy": "none|near_duplicate|parent_child_overlap",
  "redundant_af_ids": ["<af_id>"],
  "summary_priority": "required|optional|exclude",
  "question_validity": "valid|invalid_true_option|multiple_supported|invalid_distractor|not_applicable",
  "recommended_action": "keep_required|keep_optional|rewrite_af|reject",
  "confidence": "high|medium|low",
  "issues": ["<short issue>"],
  "reasoning": "<concise explanation grounded in the supplied texts>"
}

AUDIT STATUS:
{audit_status}

CANDIDATE AF:
ID: {af_id}
Fact: {fact}

SOURCE SPAN:
{source_span}

FULL ARTICLE:
{article}

EXPERT LAY SUMMARY:
{expert_summary}

OTHER FINAL KEEP AFS FOR THIS ARTICLE:
{other_afs}

MULTIPLE-CHOICE QUESTION:
{question}
"""


def _run_dir(run_name: str) -> Path:
    return ROOT_DIR / "data" / "experiment_3_module2_repro" / "runs" / run_name


def _format_other_afs(rows: list[dict[str, Any]], current_id: str) -> str:
    other = [r for r in rows if str(r["af_id"]) != current_id]
    if not other:
        return "NONE"
    return "\n".join(f"- {r['af_id']}: {r['fact']}" for r in other)


def _format_question(question: dict[str, Any] | None) -> str:
    if not question:
        return "NONE (rejected candidate)"
    options = question.get("options", {})
    return "\n".join(
        [f"Correct letter recorded by pipeline: {question.get('correct_letter', '')}"]
        + [f"{letter}. {options.get(letter, '')}" for letter in "ABCDE"]
    )


def _select_retained(
    final_rows: list[dict[str, Any]],
    per_article: int,
    total: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_article: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in final_rows:
        by_article[str(row["article_id"])].append(row)

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for article_id in sorted(by_article):
        rows = by_article[article_id]
        by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_type[str(row.get("module2_coverage_type", ""))].append(row)
        for group in by_type.values():
            rng.shuffle(group)

        picks: list[dict[str, Any]] = []
        for coverage_type in ["lay_generalization", "exact", "paraphrase"]:
            if by_type.get(coverage_type) and len(picks) < min(per_article, len(rows)):
                picks.append(by_type[coverage_type].pop())
        remaining = [r for group in by_type.values() for r in group]
        rng.shuffle(remaining)
        picks.extend(remaining[: max(0, min(per_article, len(rows)) - len(picks))])
        for row in picks:
            selected.append(row)
            selected_ids.add(str(row["af_id"]))

    if len(selected) < total:
        remaining = [r for r in final_rows if str(r["af_id"]) not in selected_ids]
        rng.shuffle(remaining)
        selected.extend(remaining[: total - len(selected)])
    return selected[:total]


def _select_rejected(
    judgements: list[dict[str, Any]],
    per_article: int,
    total: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed + 1)
    rejected = [r for r in judgements if not r.get("final_keep")]
    by_article: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rejected:
        by_article[str(row["article_id"])].append(row)

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for article_id in sorted(by_article):
        rows = by_article[article_id]
        by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_type[str(row.get("coverage_type", ""))].append(row)
        for group in by_type.values():
            rng.shuffle(group)
        picks: list[dict[str, Any]] = []
        for coverage_type in ["topic_only", "absent", "contradicted"]:
            if by_type.get(coverage_type) and len(picks) < min(per_article, len(rows)):
                picks.append(by_type[coverage_type].pop())
        remaining = [r for group in by_type.values() for r in group]
        rng.shuffle(remaining)
        picks.extend(remaining[: max(0, min(per_article, len(rows)) - len(picks))])
        for row in picks:
            selected.append(row)
            selected_ids.add(str(row["af_id"]))

    if len(selected) < total:
        remaining = [r for r in rejected if str(r["af_id"]) not in selected_ids]
        rng.shuffle(remaining)
        selected.extend(remaining[: total - len(selected)])
    return selected[:total]


def _validate_result(obj: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "atomicity": {"atomic", "compound", "unclear"},
        "article_support": {"full", "partial", "none"},
        "source_span_support": {"full", "partial", "none"},
        "expert_coverage": {"full", "broad_only", "none", "contradicted"},
        "redundancy": {"none", "near_duplicate", "parent_child_overlap"},
        "summary_priority": {"required", "optional", "exclude"},
        "question_validity": {
            "valid",
            "invalid_true_option",
            "multiple_supported",
            "invalid_distractor",
            "not_applicable",
        },
        "recommended_action": {"keep_required", "keep_optional", "rewrite_af", "reject"},
        "confidence": {"high", "medium", "low"},
    }
    out: dict[str, Any] = {}
    for field, values in allowed.items():
        value = str(obj.get(field, "")).strip()
        if value not in values:
            raise ValueError(f"invalid {field}: {value}")
        out[field] = value
    out["redundant_af_ids"] = [str(x) for x in obj.get("redundant_af_ids", []) if str(x).strip()]
    out["issues"] = [str(x) for x in obj.get("issues", []) if str(x).strip()]
    out["reasoning"] = str(obj.get("reasoning", "")).strip()
    return out


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = [
        "atomicity",
        "article_support",
        "source_span_support",
        "expert_coverage",
        "redundancy",
        "summary_priority",
        "question_validity",
        "recommended_action",
        "confidence",
    ]
    summary: dict[str, Any] = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "audited": len(rows),
        "retained": sum(1 for r in rows if r["audit_status"] == "retained"),
        "rejected": sum(1 for r in rows if r["audit_status"] == "rejected"),
        "parse_error_count": sum(1 for r in rows if r.get("parse_error")),
    }
    valid_rows = [r for r in rows if not r.get("parse_error")]
    summary["counts"] = {field: dict(Counter(str(r.get(field, "")) for r in valid_rows)) for field in fields}
    retained = [r for r in valid_rows if r["audit_status"] == "retained"]
    summary["retained_problem_rates"] = {
        "article_not_fully_supported": round(
            sum(r["article_support"] != "full" for r in retained) / len(retained), 4
        )
        if retained
        else 0.0,
        "source_span_not_fully_supported": round(
            sum(r["source_span_support"] != "full" for r in retained) / len(retained), 4
        )
        if retained
        else 0.0,
        "expert_not_full_coverage": round(
            sum(r["expert_coverage"] != "full" for r in retained) / len(retained), 4
        )
        if retained
        else 0.0,
        "compound_or_unclear": round(sum(r["atomicity"] != "atomic" for r in retained) / len(retained), 4)
        if retained
        else 0.0,
        "semantic_overlap": round(sum(r["redundancy"] != "none" for r in retained) / len(retained), 4)
        if retained
        else 0.0,
        "invalid_question": round(sum(r["question_validity"] != "valid" for r in retained) / len(retained), 4)
        if retained
        else 0.0,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit simulated Module 2 AFs with an independent GPT-4.1 judge.")
    parser.add_argument("--module2-run-name", default=DEFAULT_MODULE2_RUN)
    parser.add_argument("--model-branch", default=DEFAULT_MODEL_BRANCH)
    parser.add_argument("--audit-model-key", default=DEFAULT_AUDIT_MODEL)
    parser.add_argument("--retained", type=int, default=40)
    parser.add_argument("--rejected", type=int, default=20)
    parser.add_argument("--retained-per-article", type=int, default=4)
    parser.add_argument("--rejected-per-article", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    run_dir = _run_dir(args.module2_run_name)
    articles = {str(r["id"]): r for r in load_jsonl(run_dir / "00_inputs" / "articles.jsonl")}
    judgements = load_jsonl(run_dir / "02_module2_judgements" / args.model_branch / "module2_judgements.jsonl")
    final_rows = load_jsonl(run_dir / "03_final_keep_af" / args.model_branch / "final_keep_af.jsonl")
    questions = {
        str(r["af_id"]): r
        for r in load_jsonl(run_dir / "04_questions" / args.model_branch / "questions_1t3f_nota.jsonl")
    }
    final_by_article: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in final_rows:
        final_by_article[str(row["article_id"])].append(row)

    retained = _select_retained(final_rows, args.retained_per_article, args.retained, args.seed)
    rejected = _select_rejected(judgements, args.rejected_per_article, args.rejected, args.seed)
    sample = [dict(r, audit_status="retained") for r in retained] + [
        dict(r, audit_status="rejected") for r in rejected
    ]
    sample.sort(key=lambda r: (str(r["article_id"]), str(r["audit_status"]), str(r["af_id"])))

    out_dir = run_dir / "06_gpt41_audit" / args.model_branch
    out_path = out_dir / "audit_results.jsonl"
    existing = load_jsonl(out_path) if not args.no_resume and out_path.exists() else []
    done = {str(r["af_id"]) for r in existing if not r.get("parse_error")}
    usage = UsageTracker.from_csv(out_dir / "api_usage_calls.csv") if not args.no_resume else UsageTracker(rows=[])

    def worker(row: dict[str, Any]) -> dict[str, Any]:
        af_id = str(row["af_id"])
        article = articles[str(row["article_id"])]
        prompt = (
            AUDIT_PROMPT.replace("{audit_status}", str(row["audit_status"]))
            .replace("{af_id}", af_id)
            .replace("{fact}", str(row["fact"]))
            .replace("{source_span}", str(row.get("source_span", "")))
            .replace("{article}", str(article["document"]))
            .replace("{expert_summary}", str(article["expert_summary"]))
            .replace("{other_afs}", _format_other_afs(final_by_article[str(row["article_id"])], af_id))
            .replace("{question}", _format_question(questions.get(af_id)))
        )
        raw = ""
        try:
            client = ProviderClient(args.audit_model_key, usage)
            raw = client.chat(
                stage=f"module2_audit:{args.audit_model_key}",
                item_id=af_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            result = _validate_result(json.loads(raw))
            return {**row, **result, "audit_model_key": args.audit_model_key, "parse_error": False}
        except Exception as exc:
            return {
                **row,
                "audit_model_key": args.audit_model_key,
                "parse_error": True,
                "error": str(exc),
                "raw_preview": raw[:500],
            }

    todo = [r for r in sample if str(r["af_id"]) not in done]
    new_rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(worker, row) for row in todo]
        for future in tqdm(as_completed(futures), total=len(futures), desc="GPT-4.1 Module 2 audit"):
            new_rows.append(future.result())

    new_ids = {str(r["af_id"]) for r in new_rows}
    rows = [r for r in existing if str(r["af_id"]) not in new_ids] + new_rows
    rows.sort(key=lambda r: (str(r["article_id"]), str(r["audit_status"]), str(r["af_id"])))
    save_jsonl(rows, out_path)
    save_json(_summarize(rows), out_dir / "audit_summary.json")
    save_json(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "module2_run_name": args.module2_run_name,
            "model_branch": args.model_branch,
            "audit_model_key": args.audit_model_key,
            "seed": args.seed,
            "retained_requested": args.retained,
            "rejected_requested": args.rejected,
            "sample_size": len(sample),
            "sample_by_status": dict(Counter(str(r["audit_status"]) for r in sample)),
            "sample_by_article": dict(Counter(str(r["article_id"]) for r in sample)),
        },
        out_dir / "audit_metadata.json",
    )
    usage.save(out_dir)

    flat_fields = [
        "af_id",
        "article_id",
        "source_dataset",
        "audit_status",
        "coverage_type",
        "module2_coverage_type",
        "atomicity",
        "article_support",
        "source_span_support",
        "expert_coverage",
        "redundancy",
        "summary_priority",
        "question_validity",
        "recommended_action",
        "confidence",
        "parse_error",
    ]
    with (out_dir / "audit_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=flat_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[module2 audit] results -> {out_dir}")


if __name__ == "__main__":
    main()
