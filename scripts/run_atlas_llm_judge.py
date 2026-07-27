from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


RELEVANCE = ["ROUGE", "BLEU", "METEOR", "BERTScore"]
READABILITY = ["FKGL", "DCRS", "CLI", "LENS"]
FACTUALITY = ["AlignScore", "SummaC"]
LOWER_IS_BETTER = {"FKGL", "DCRS", "CLI"}


DEFAULT_PIPELINE_RUN = "v11_constellation_validation284_fixed_m12_pilot20"
DEFAULT_JUDGE_MODEL = "gpt-5.4-mini-2026-03-17"
DEFAULT_REASONING_EFFORT = "medium"
DEFAULT_OUTPUT_DIRNAME = "llm_judge_atlas_gpt54mini"
DEFAULT_ARTICLES_PER_DATASET = 10
MAX_ITEM_ATTEMPTS = 3

ATLAS_SYSTEMS: dict[str, dict[str, str]] = {
    "gemini25_flash_non_thinking": {
        "system_name": "Gemini 2.5 Flash + ATLAS",
        "generation_model": "gemini-2.5-flash",
    },
    "gemini3_flash_preview_minimal": {
        "system_name": "Gemini 3 Flash Preview + ATLAS",
        "generation_model": "gemini-3-flash-preview",
    },
    "gpt41_mini": {
        "system_name": "GPT-4.1 Mini + ATLAS",
        "generation_model": "gpt-4.1-mini",
    },
}

SYSTEM_PROMPT = """You are an independent evaluator of biomedical lay summaries.

Evaluate the candidate summary on three separate dimensions: Relevance, Readability, and Factuality. Use an integer score from 1 to 5 for each dimension and apply the supplied scoring rubric consistently.

For Relevance, determine whether the candidate summary covers the most important research purpose, methods, findings, and conclusions while remaining focused. The reference summary may help identify important information, but the candidate does not need to use the same wording or include exactly the same content.

For Readability, determine whether a reader without biomedical expertise can understand the candidate summary. Consider terminology, explanations, sentence structure, organization, and overall reading difficulty.

For Factuality, treat the source article as the factual authority. Check whether every substantive claim in the candidate summary is supported by the source article. Look for contradictions, exaggerations, incorrect numbers, unsupported causal relationships, and hallucinated information. Do not assume that a fluent or plausible statement is factually correct.

Do not use outside medical knowledge to correct, update, or supplement the source article. Do not infer the identity of the generation model or method. Evaluate each dimension independently.

Return only the structured result required by the response schema."""

USER_TEMPLATE = """Source biomedical article:
{source_article}

Reference lay summary:
{reference_summary}

Candidate summary:
{candidate_summary}"""

JUDGE_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "name": "biomedical_lay_summary_judge",
    "description": "Scores a candidate biomedical lay summary on relevance, readability, and factuality.",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "relevance": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
                    "reason": {"type": "string"},
                },
                "required": ["score", "reason"],
            },
            "readability": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
                    "reason": {"type": "string"},
                },
                "required": ["score", "reason"],
            },
            "factuality": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
                    "reason": {"type": "string"},
                },
                "required": ["score", "reason"],
            },
        },
        "required": ["relevance", "readability", "factuality"],
    },
}


@dataclass(frozen=True)
class JudgeItem:
    article_id: str
    dataset: str
    original_index: int | None
    model_key: str
    system_name: str
    generation_model: str
    source_article: str
    reference_summary: str
    candidate_summary: str

    @property
    def item_id(self) -> str:
        return f"{self.article_id}::{self.model_key}"


class ParallelExecutionError(RuntimeError):
    def __init__(self, message: str, partial_results: list[Any]):
        super().__init__(message)
        self.partial_results = partial_results


def load_jsonl(path: str | Path) -> list[Any]:
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_json(data: Any, path: str | Path) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_jsonl(records: Iterable[Any], path: str | Path) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def print_progress(desc: str, completed: int, total: int, row: dict[str, Any] | None = None) -> None:
    pct = (completed / total * 100.0) if total else 100.0
    detail = ""
    if row:
        detail = (
            f" dataset={row.get('dataset', '')}"
            f" item={row.get('judge_item_id', '')}"
            f" elapsed={float(row.get('elapsed_seconds') or 0.0):.1f}s"
        )
    print(f"[{desc}] {completed}/{total} ({pct:.1f}%){detail}", flush=True)


def run_parallel(
    items: Iterable[Any],
    worker: Callable[[Any], Any],
    *,
    max_workers: int,
    desc: str,
) -> list[Any]:
    item_list = list(items)
    if max_workers <= 1:
        out: list[Any] = []
        for index, item in enumerate(item_list, start=1):
            try:
                row = worker(item)
                out.append(row)
                print_progress(desc, index, len(item_list), row if isinstance(row, dict) else None)
            except Exception as exc:
                raise ParallelExecutionError(
                    f"{desc} failed after {len(out)} completed items: {exc}",
                    out,
                ) from exc
        return out
    out = []
    ex = ThreadPoolExecutor(max_workers=max_workers)
    futures = [ex.submit(worker, item) for item in item_list]
    failed = False
    try:
        for index, fut in enumerate(as_completed(futures), start=1):
            try:
                row = fut.result()
                out.append(row)
                print_progress(desc, index, len(futures), row if isinstance(row, dict) else None)
            except Exception as exc:
                failed = True
                for pending in futures:
                    pending.cancel()
                ex.shutdown(wait=False, cancel_futures=True)
                raise ParallelExecutionError(
                    f"{desc} failed after {len(out)} completed items: {exc}",
                    out,
                ) from exc
    finally:
        if not failed:
            ex.shutdown(wait=True, cancel_futures=False)
    return out


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def data_run_dir(pipeline_run: str) -> Path:
    return ROOT_DIR / "data" / "laysumm_pipeline" / "runs" / pipeline_run


def results_run_dir(pipeline_run: str) -> Path:
    return ROOT_DIR / "results" / "laysumm_pipeline" / "runs" / pipeline_run


def default_output_dir(pipeline_run: str, articles_per_dataset: int | None = None) -> Path:
    dirname = DEFAULT_OUTPUT_DIRNAME
    if articles_per_dataset is not None:
        dirname = f"{dirname}_per_dataset{articles_per_dataset}"
    return results_run_dir(pipeline_run) / dirname


def load_articles_by_id(pipeline_run: str) -> dict[str, dict[str, Any]]:
    path = data_run_dir(pipeline_run) / "00_articles" / "articles.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing articles file: {path}")
    rows = load_jsonl(path)
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        article_id = str(row.get("id") or row.get("article_id") or "")
        if not article_id:
            raise ValueError(f"Article row without id in {path}")
        if article_id in out:
            raise ValueError(f"Duplicate article id in {path}: {article_id}")
        out[article_id] = row
    return out


def load_judge_items(pipeline_run: str, system_keys: list[str]) -> list[JudgeItem]:
    articles = load_articles_by_id(pipeline_run)
    items: list[JudgeItem] = []
    for model_key in system_keys:
        if model_key not in ATLAS_SYSTEMS:
            raise ValueError(f"Unknown ATLAS system key: {model_key}. Valid: {sorted(ATLAS_SYSTEMS)}")
        summaries_path = (
            data_run_dir(pipeline_run) / "06_rewritten" / model_key / "rewritten_summaries.jsonl"
        )
        if not summaries_path.exists():
            raise FileNotFoundError(f"Missing ATLAS summaries file: {summaries_path}")
        seen: set[str] = set()
        for row in load_jsonl(summaries_path):
            article_id = str(row.get("article_id") or "")
            if not article_id:
                raise ValueError(f"Summary row without article_id in {summaries_path}")
            if article_id in seen:
                raise ValueError(f"Duplicate article_id in {summaries_path}: {article_id}")
            seen.add(article_id)
            article = articles.get(article_id)
            if article is None:
                raise ValueError(f"Summary article_id not found in articles.jsonl: {article_id}")
            candidate = str(row.get("rewritten_summary") or row.get("generated_summary") or "").strip()
            if not candidate:
                raise ValueError(f"Empty candidate summary for {article_id} in {summaries_path}")
            cfg = ATLAS_SYSTEMS[model_key]
            items.append(
                JudgeItem(
                    article_id=article_id,
                    dataset=str(article.get("source_dataset") or row.get("source_dataset") or ""),
                    original_index=_optional_int(article.get("original_index")),
                    model_key=model_key,
                    system_name=cfg["system_name"],
                    generation_model=cfg["generation_model"],
                    source_article=str(article.get("article") or "").strip(),
                    reference_summary=str(article.get("expert_summary") or "").strip(),
                    candidate_summary=candidate,
                )
            )
        missing = set(articles) - seen
        if missing:
            raise ValueError(f"{summaries_path} is missing {len(missing)} article summaries")
    return sorted(items, key=lambda x: (x.article_id, x.model_key))


def filter_items_by_articles_per_dataset(items: list[JudgeItem], articles_per_dataset: int | None) -> list[JudgeItem]:
    if articles_per_dataset is None:
        return items
    if articles_per_dataset <= 0:
        raise ValueError("--articles-per-dataset must be positive when provided")

    articles_by_dataset: dict[str, dict[str, tuple[int | None, str]]] = defaultdict(dict)
    for item in items:
        articles_by_dataset[item.dataset][item.article_id] = (item.original_index, item.article_id)

    selected_article_ids: set[str] = set()
    for dataset, article_map in sorted(articles_by_dataset.items()):
        ordered_articles = sorted(
            article_map.values(),
            key=lambda row: (row[0] is None, row[0] if row[0] is not None else 0, row[1]),
        )
        if len(ordered_articles) < articles_per_dataset:
            raise ValueError(
                f"Dataset {dataset} has only {len(ordered_articles)} articles; "
                f"cannot select {articles_per_dataset}"
            )
        selected_article_ids.update(article_id for _, article_id in ordered_articles[:articles_per_dataset])

    return [item for item in items if item.article_id in selected_article_ids]


def dataset_counts(items: list[JudgeItem]) -> dict[str, dict[str, int]]:
    article_ids: dict[str, set[str]] = defaultdict(set)
    item_counts: dict[str, int] = defaultdict(int)
    for item in items:
        article_ids[item.dataset].add(item.article_id)
        item_counts[item.dataset] += 1
    return {
        dataset: {
            "article_count": len(article_ids[dataset]),
            "judge_item_count": item_counts[dataset],
        }
        for dataset in sorted(article_ids)
    }


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def build_user_prompt(item: JudgeItem) -> str:
    return USER_TEMPLATE.format(
        source_article=item.source_article,
        reference_summary=item.reference_summary,
        candidate_summary=item.candidate_summary,
    )


def extract_response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text)
    chunks: list[str] = []
    for output in getattr(response, "output", []) or []:
        for content in getattr(output, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                chunks.append(str(text))
    return "".join(chunks)


def usage_dict(response: Any) -> dict[str, int | None]:
    usage = getattr(response, "usage", None)
    input_tokens = _usage_int(usage, "input_tokens")
    output_tokens = _usage_int(usage, "output_tokens")
    total_tokens = _usage_int(usage, "total_tokens")
    reasoning_tokens: int | None = None
    details = getattr(usage, "output_tokens_details", None)
    if details is not None:
        reasoning_tokens = _usage_int(details, "reasoning_tokens")
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": total_tokens,
    }


def _usage_int(obj: Any, field: str) -> int | None:
    if obj is None:
        return None
    value = getattr(obj, field, None)
    if value is None and isinstance(obj, dict):
        value = obj.get(field)
    return int(value) if value is not None else None


def validate_judgement(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Judge response is not a JSON object")
    out: dict[str, Any] = {}
    for dimension in ("relevance", "readability", "factuality"):
        block = payload.get(dimension)
        if not isinstance(block, dict):
            raise ValueError(f"Missing object for {dimension}")
        score = int(block.get("score"))
        if score < 1 or score > 5:
            raise ValueError(f"{dimension}.score outside 1..5: {score}")
        reason = str(block.get("reason") or "").strip()
        if not reason:
            raise ValueError(f"{dimension}.reason is empty")
        out[dimension] = {"score": score, "reason": reason}
    return out


class ResponsesJudgeClient:
    def __init__(self, *, model: str, reasoning_effort: str) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The OpenAI Python SDK is required for live judging. "
                "Install project requirements before running without --dry-run."
            ) from exc
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is missing. Set it before running the LLM judge.")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.reasoning_effort = reasoning_effort

    def judge_once(self, item: JudgeItem) -> tuple[dict[str, Any], dict[str, int | None], str]:
        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(item)},
            ],
            reasoning={"effort": self.reasoning_effort},
            text={"format": JUDGE_SCHEMA},
        )
        text = extract_response_text(response).strip()
        if not text:
            raise ValueError("Responses API returned no output_text")
        return validate_judgement(json.loads(text)), usage_dict(response), str(getattr(response, "id", "") or "")


def _done_ids(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    done: set[str] = set()
    for row in load_jsonl(out_path):
        item_id = str(row.get("judge_item_id") or "")
        if not item_id:
            raise ValueError(f"Existing row without judge_item_id in {out_path}")
        if item_id in done:
            raise ValueError(f"Duplicate judge_item_id in existing output: {item_id}")
        done.add(item_id)
    return done


def _result_row(
    *,
    item: JudgeItem,
    judge_model: str,
    reasoning_effort: str,
    judgement: dict[str, Any],
    usage: dict[str, int | None],
    retry_count: int,
    response_id: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "judge_item_id": item.item_id,
        "article_id": item.article_id,
        "dataset": item.dataset,
        "system_name": item.system_name,
        "generation_model": item.generation_model,
        "model_key": item.model_key,
        "judge_model": judge_model,
        "reasoning_effort": reasoning_effort,
        "relevance_score": judgement["relevance"]["score"],
        "relevance_reason": judgement["relevance"]["reason"],
        "readability_score": judgement["readability"]["score"],
        "readability_reason": judgement["readability"]["reason"],
        "factuality_score": judgement["factuality"]["score"],
        "factuality_reason": judgement["factuality"]["reason"],
        "normalized_relevance_score": normalize_judge_score(judgement["relevance"]["score"]),
        "normalized_readability_score": normalize_judge_score(judgement["readability"]["score"]),
        "normalized_factuality_score": normalize_judge_score(judgement["factuality"]["score"]),
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "reasoning_tokens": usage.get("reasoning_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "retry_count": retry_count,
        "response_id": response_id,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "evaluation_timestamp": utc_timestamp(),
        "candidate_sha256": sha256_text(item.candidate_summary),
    }


def normalize_judge_score(score: int | float) -> float:
    return (float(score) - 1.0) / 4.0


def run_judge(
    *,
    pipeline_run: str,
    output_dir: Path,
    system_keys: list[str],
    judge_model: str,
    reasoning_effort: str,
    max_workers: int,
    limit: int | None,
    articles_per_dataset: int | None,
    resume: bool,
    dry_run: bool,
    summarize_only: bool,
) -> Path:
    items = load_judge_items(pipeline_run, system_keys)
    items = filter_items_by_articles_per_dataset(items, articles_per_dataset)
    if limit is not None:
        items = items[:limit]
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "judge_evaluations.jsonl"
    if summarize_only:
        if not out_path.exists():
            raise FileNotFoundError(f"Cannot summarize before judge output exists: {out_path}")
        rows = load_jsonl(out_path)
        write_all_summaries(pipeline_run=pipeline_run, output_dir=output_dir, rows=rows, system_keys=system_keys)
        print(f"[llm judge] rebuilt summary tables from {len(rows)} existing rows -> {output_dir}")
        return out_path

    done = _done_ids(out_path) if resume else set()
    todo = [item for item in items if item.item_id not in done]
    save_json(
        {
            "created_or_updated_at": utc_timestamp(),
            "pipeline_run": pipeline_run,
            "judge_model": judge_model,
            "reasoning_effort": reasoning_effort,
            "structured_outputs": True,
            "responses_api": True,
            "system_keys": system_keys,
            "articles_per_dataset": articles_per_dataset,
            "total_requested_items": len(items),
            "dataset_counts": dataset_counts(items),
            "already_done_items": len(done),
            "todo_items": len(todo),
            "prompt_sha256": sha256_text(SYSTEM_PROMPT + "\n\n" + USER_TEMPLATE),
            "response_schema": JUDGE_SCHEMA,
            "source_articles_written_to_outputs": False,
        },
        output_dir / "run_metadata.json",
    )
    if dry_run:
        print(f"[llm judge] dry run OK: items={len(items)} todo={len(todo)} output_dir={output_dir}")
        return out_path

    client = ResponsesJudgeClient(model=judge_model, reasoning_effort=reasoning_effort)

    def worker(item: JudgeItem) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, MAX_ITEM_ATTEMPTS + 1):
            started = time.perf_counter()
            try:
                judgement, usage, response_id = client.judge_once(item)
                return _result_row(
                    item=item,
                    judge_model=judge_model,
                    reasoning_effort=reasoning_effort,
                    judgement=judgement,
                    usage=usage,
                    retry_count=attempt - 1,
                    response_id=response_id,
                    elapsed_seconds=time.perf_counter() - started,
                )
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"judge failed for {item.item_id} after {MAX_ITEM_ATTEMPTS} attempts: {last_error}")

    existing = load_jsonl(out_path) if resume and out_path.exists() else []
    try:
        new_rows = run_parallel(todo, worker, max_workers=max_workers, desc="LLM judge ATLAS")
    except ParallelExecutionError as exc:
        checkpoint = sorted(existing + exc.partial_results, key=lambda row: str(row["judge_item_id"]))
        save_jsonl(checkpoint, out_path)
        print(f"[llm judge] checkpointed {len(checkpoint)} rows before failure -> {out_path}")
        raise

    combined = sorted(existing + new_rows, key=lambda row: str(row["judge_item_id"]))
    save_jsonl(combined, out_path)
    print(f"[llm judge] wrote {len(combined)} rows -> {out_path}")
    write_all_summaries(pipeline_run=pipeline_run, output_dir=output_dir, rows=combined, system_keys=system_keys)
    return out_path


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def write_all_summaries(
    *,
    pipeline_run: str,
    output_dir: Path,
    rows: list[dict[str, Any]],
    system_keys: list[str],
) -> None:
    aggregate_rows = build_aggregate_rows(rows)
    write_csv(output_dir / "judge_aggregate.csv", aggregate_rows)
    save_json({"rows": aggregate_rows}, output_dir / "judge_aggregate.json")

    comparison_rows = build_comparison_rows(
        pipeline_run=pipeline_run,
        aggregate_rows=aggregate_rows,
        system_keys=system_keys,
    )
    write_csv(output_dir / "judge_vs_biolaysumm_comparison.csv", comparison_rows)
    save_json({"rows": comparison_rows}, output_dir / "judge_vs_biolaysumm_comparison.json")

    overall_rows = build_overall_284_comparison_rows(
        pipeline_run=pipeline_run,
        aggregate_rows=aggregate_rows,
        system_keys=system_keys,
    )
    write_csv(output_dir / "judge_vs_biolaysumm_284_overall.csv", overall_rows)
    save_json(
        {
            "comparison_basis": (
                "BioLaySumm columns use each ATLAS system's 284-article rewritten official-style "
                "normalized overall dimension scores from official_style_rank.json. "
                "LLM judge columns use this run's normalized score means, where "
                "normalized_judge_score = (score - 1) / 4."
            ),
            "rows": overall_rows,
        },
        output_dir / "judge_vs_biolaysumm_284_overall.json",
    )


def build_aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["system_name"]), str(row["dataset"]))].append(row)
        grouped[(str(row["system_name"]), "overall")].append(row)

    out: list[dict[str, Any]] = []
    for (system_name, dataset), group in sorted(grouped.items()):
        relevance = [float(r["normalized_relevance_score"]) for r in group]
        readability = [float(r["normalized_readability_score"]) for r in group]
        factuality = [float(r["normalized_factuality_score"]) for r in group]
        generation_model = str(group[0]["generation_model"])
        row = {
            "system_name": system_name,
            "generation_model": generation_model,
            "dataset": dataset,
            "article_count": len(group),
            "llm_judge_relevance": mean(relevance),
            "llm_judge_readability": mean(readability),
            "llm_judge_factuality": mean(factuality),
            "llm_judge_overall": mean([mean(relevance), mean(readability), mean(factuality)]),
            "raw_relevance_mean_1to5": mean([float(r["relevance_score"]) for r in group]),
            "raw_readability_mean_1to5": mean([float(r["readability_score"]) for r in group]),
            "raw_factuality_mean_1to5": mean([float(r["factuality_score"]) for r in group]),
            "input_tokens": sum(int(r.get("input_tokens") or 0) for r in group),
            "output_tokens": sum(int(r.get("output_tokens") or 0) for r in group),
            "reasoning_tokens": sum(int(r.get("reasoning_tokens") or 0) for r in group),
            "total_tokens": sum(int(r.get("total_tokens") or 0) for r in group),
        }
        out.append(row)
    return out


def build_comparison_rows(
    *,
    pipeline_run: str,
    aggregate_rows: list[dict[str, Any]],
    system_keys: list[str],
) -> list[dict[str, Any]]:
    system_by_name = {cfg["system_name"]: key for key, cfg in ATLAS_SYSTEMS.items()}
    out: list[dict[str, Any]] = []
    for agg in aggregate_rows:
        dataset = str(agg["dataset"])
        model_key = system_by_name[str(agg["system_name"])]
        if model_key not in system_keys:
            continue
        official = load_biolaysumm_dimension_scores(pipeline_run, model_key, dataset)
        out.append(
            {
                "system_name": agg["system_name"],
                "generation_model": agg["generation_model"],
                "dataset": dataset,
                "article_count": agg["article_count"],
                "biolaysumm_score_method": official.get("method"),
                "biolaysumm_relevance": official.get("relevance"),
                "llm_judge_relevance": agg["llm_judge_relevance"],
                "biolaysumm_readability": official.get("readability"),
                "llm_judge_readability": agg["llm_judge_readability"],
                "biolaysumm_factuality": official.get("factuality"),
                "llm_judge_factuality": agg["llm_judge_factuality"],
                "biolaysumm_overall": official.get("overall"),
                "llm_judge_overall": agg["llm_judge_overall"],
            }
        )
    return out


def build_overall_284_comparison_rows(
    *,
    pipeline_run: str,
    aggregate_rows: list[dict[str, Any]],
    system_keys: list[str],
) -> list[dict[str, Any]]:
    by_system_name = {
        str(row["system_name"]): row
        for row in aggregate_rows
        if str(row.get("dataset")) == "overall"
    }
    out: list[dict[str, Any]] = []
    for model_key in system_keys:
        cfg = ATLAS_SYSTEMS[model_key]
        system_name = cfg["system_name"]
        judge = by_system_name.get(system_name)
        official = load_biolaysumm_dimension_scores(pipeline_run, model_key, "overall")
        if judge is None:
            continue
        row = {
            "system_name": system_name,
            "generation_model": cfg["generation_model"],
            "biolaysumm_article_count": load_biolaysumm_article_count(pipeline_run, model_key),
            "llm_judge_article_count": judge["article_count"],
            "biolaysumm_score_method": official.get("method"),
            "llm_judge_score_method": "mean_normalized_1to5_scores_(score_minus_1_divided_by_4)",
            **load_biolaysumm_raw_overall_metrics(pipeline_run, model_key),
            "biolaysumm_284_relevance": official.get("relevance"),
            "llm_judge_relevance": judge["llm_judge_relevance"],
            "delta_llm_minus_biolaysumm_relevance": _delta(judge["llm_judge_relevance"], official.get("relevance")),
            "biolaysumm_284_readability": official.get("readability"),
            "llm_judge_readability": judge["llm_judge_readability"],
            "delta_llm_minus_biolaysumm_readability": _delta(judge["llm_judge_readability"], official.get("readability")),
            "biolaysumm_284_factuality": official.get("factuality"),
            "llm_judge_factuality": judge["llm_judge_factuality"],
            "delta_llm_minus_biolaysumm_factuality": _delta(judge["llm_judge_factuality"], official.get("factuality")),
            "biolaysumm_284_overall": official.get("overall"),
            "llm_judge_overall": judge["llm_judge_overall"],
            "delta_llm_minus_biolaysumm_overall": _delta(judge["llm_judge_overall"], official.get("overall")),
        }
        out.append(row)

    add_rank_columns(out, "biolaysumm_284_overall", "biolaysumm_284_rank")
    add_rank_columns(out, "llm_judge_overall", "llm_judge_rank")
    add_rank_columns(out, "biolaysumm_284_relevance", "biolaysumm_284_relevance_rank")
    add_rank_columns(out, "llm_judge_relevance", "llm_judge_relevance_rank")
    add_rank_columns(out, "biolaysumm_284_readability", "biolaysumm_284_readability_rank")
    add_rank_columns(out, "llm_judge_readability", "llm_judge_readability_rank")
    add_rank_columns(out, "biolaysumm_284_factuality", "biolaysumm_284_factuality_rank")
    add_rank_columns(out, "llm_judge_factuality", "llm_judge_factuality_rank")
    return out


def add_rank_columns(rows: list[dict[str, Any]], score_key: str, rank_key: str) -> None:
    ranked = sorted(
        [row for row in rows if row.get(score_key) is not None],
        key=lambda row: float(row[score_key]),
        reverse=True,
    )
    for rank, row in enumerate(ranked, start=1):
        row[rank_key] = rank
    for row in rows:
        row.setdefault(rank_key, None)


def _delta(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def load_biolaysumm_article_count(pipeline_run: str, model_key: str) -> int | None:
    path = results_run_dir(pipeline_run) / model_key / "rewritten_scores.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    count = payload.get("article_count")
    return int(count) if count is not None else None


def load_biolaysumm_raw_overall_metrics(pipeline_run: str, model_key: str) -> dict[str, float | None]:
    path = results_run_dir(pipeline_run) / model_key / "rewritten_scores.json"
    fields = {
        "biolaysumm_284_raw_ROUGE": None,
        "biolaysumm_284_raw_BLEU": None,
        "biolaysumm_284_raw_METEOR": None,
        "biolaysumm_284_raw_BERTScore": None,
        "biolaysumm_284_raw_FKGL": None,
        "biolaysumm_284_raw_DCRS": None,
        "biolaysumm_284_raw_CLI": None,
        "biolaysumm_284_raw_LENS": None,
        "biolaysumm_284_raw_AlignScore": None,
        "biolaysumm_284_raw_SummaC": None,
        "biolaysumm_284_raw_factuality_mean": None,
    }
    if not path.exists():
        return fields
    payload = json.loads(path.read_text(encoding="utf-8"))
    overall = payload.get("overall")
    if not isinstance(overall, dict):
        return fields
    for metric in ("ROUGE", "BLEU", "METEOR", "BERTScore", "FKGL", "DCRS", "CLI", "LENS", "AlignScore", "SummaC"):
        value = overall.get(metric)
        fields[f"biolaysumm_284_raw_{metric}"] = float(value) if value is not None else None
    align = fields["biolaysumm_284_raw_AlignScore"]
    summac = fields["biolaysumm_284_raw_SummaC"]
    if align is not None and summac is not None:
        fields["biolaysumm_284_raw_factuality_mean"] = (align + summac) / 2.0
    return fields


def load_biolaysumm_dimension_scores(
    pipeline_run: str,
    model_key: str,
    dataset: str,
) -> dict[str, float | str | None]:
    if dataset == "overall":
        official_rank = results_run_dir(pipeline_run) / model_key / "official_style_rank.json"
        if official_rank.exists():
            payload = json.loads(official_rank.read_text(encoding="utf-8"))
            rewritten = payload.get("our_systems", {}).get("rewritten")
            if isinstance(rewritten, dict):
                return {
                    "relevance": _optional_float(rewritten.get("relevance")),
                    "readability": _optional_float(rewritten.get("readability")),
                    "factuality": _optional_float(rewritten.get("factuality")),
                    "overall": _optional_float(rewritten.get("final_score")),
                    "method": "official_style_leaderboard_minmax_rewritten",
                }
    path = results_run_dir(pipeline_run) / model_key / "rewritten_scores.json"
    if not path.exists():
        return {
            "relevance": None,
            "readability": None,
            "factuality": None,
            "overall": None,
            "method": "missing_rewritten_scores_json",
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    block = payload.get(dataset)
    if not isinstance(block, dict):
        return {
            "relevance": None,
            "readability": None,
            "factuality": None,
            "overall": None,
            "method": "dataset_block_missing",
        }
    normalized = _normalize_metrics(block)
    relevance = mean([normalized[m] for m in RELEVANCE if m in normalized])
    readability = mean([normalized[m] for m in READABILITY if m in normalized])
    factuality = mean([normalized[m] for m in FACTUALITY if m in normalized])
    return {
        "relevance": relevance,
        "readability": readability,
        "factuality": factuality,
        "overall": mean([relevance, readability, factuality]),
        "method": (
            "native_metric_proxy_for_dataset: relevance averages ROUGE/BLEU/100/METEOR/BERTScore; "
            "readability keeps FKGL/DCRS/CLI/LENS native directions; factuality averages AlignScore/SummaC"
        ),
    }


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _normalize_metrics(metric_values: dict[str, Any]) -> dict[str, float]:
    # Comparison-only normalization within a single row, preserving the official
    # dimension groupings without recomputing leaderboard-pool ranks.
    out: dict[str, float] = {}
    for metric, value in metric_values.items():
        try:
            val = float(value)
        except (TypeError, ValueError):
            continue
        if metric in LOWER_IS_BETTER:
            # Readability error/count-style metrics are lower-is-better and do
            # not share a common native range. Leave them raw in comparison JSON
            # would be misleading, so map only impossible-empty cases elsewhere.
            out[metric] = val
        elif metric == "BLEU":
            out[metric] = val / 100.0
        else:
            out[metric] = val
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate ATLAS-generated BioLaySumm summaries with an OpenAI Responses API LLM judge."
    )
    parser.add_argument("--pipeline-run", default=DEFAULT_PIPELINE_RUN)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument("--system-key", action="append", choices=sorted(ATLAS_SYSTEMS), dest="system_keys")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--articles-per-dataset", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    return parser


def main() -> None:
    load_dotenv(ROOT_DIR / ".env")
    args = build_parser().parse_args()
    system_keys = args.system_keys or list(ATLAS_SYSTEMS)
    output_dir = args.output_dir or default_output_dir(args.pipeline_run, args.articles_per_dataset)
    run_judge(
        pipeline_run=args.pipeline_run,
        output_dir=output_dir,
        system_keys=system_keys,
        judge_model=args.judge_model,
        reasoning_effort=args.reasoning_effort,
        max_workers=args.max_workers,
        limit=args.limit,
        articles_per_dataset=args.articles_per_dataset,
        resume=not args.no_resume,
        dry_run=args.dry_run,
        summarize_only=args.summarize_only,
    )


if __name__ == "__main__":
    main()
