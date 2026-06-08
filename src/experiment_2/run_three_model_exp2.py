from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils import load_jsonl, save_json, save_jsonl


load_dotenv(ROOT_DIR / ".env")

DATA_THREE_MODEL_DIR = ROOT_DIR / "data" / "experiment_2_three_model_comparison"
RUNS_DIR = DATA_THREE_MODEL_DIR / "runs"
PROMPTS_EXP2_DIR = ROOT_DIR / "src" / "experiment_2" / "prompts"
INPUT_ARTICLES_PATH = ROOT_DIR / "data" / "experiment_2" / "00_inputs" / "articles_50.jsonl"

ARTICLE_AF_PROMPT = PROMPTS_EXP2_DIR / "article_af_extraction.txt"
DIRECT_COVERAGE_PROMPT = PROMPTS_EXP2_DIR / "direct_coverage.txt"
KEEP_SKIP_PROMPT = PROMPTS_EXP2_DIR / "keep_skip_ultra_recall.txt"
PARTC_SUMMARY_AF_PROMPT = PROMPTS_EXP2_DIR / "partc_summary_af_extraction.txt"
PARTC_SELECTED_KEEP_PROMPT = PROMPTS_EXP2_DIR / "selective_keep_af_ultra_recall.txt"
PARTC_EVAL_PROMPT = PROMPTS_EXP2_DIR / "partc_selected_keep_covers_summary_af.txt"


# Prices are per 1M tokens. Keep these fixed in the metadata so each run is
# reproducible even if provider pricing changes later.
MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "gpt41_mini": {
        "provider": "openai",
        "model": "gpt-4.1-mini",
        "input_price_per_1m": 0.40,
        "output_price_per_1m": 1.60,
        "currency": "USD",
    },
    "gemini31_flash_lite": {
        "provider": "gemini",
        "model": "gemini-3.1-flash-lite",
        "input_price_per_1m": 0.125,
        "output_price_per_1m": 0.75,
        "currency": "USD",
    },
    "deepseek_v4_flash": {
        "provider": "deepseek",
        "model": "deepseek-v4-flash",
        "input_price_per_1m": 0.14,
        "output_price_per_1m": 0.28,
        "currency": "USD",
        "pricing_note": "Uses DeepSeek cache-miss input price; cache-hit discounts are not estimated.",
    },
}


@dataclass
class UsageTracker:
    rows: list[dict[str, Any]]

    @classmethod
    def from_csv(cls, path: Path) -> "UsageTracker":
        if not path.exists():
            return cls(rows=[])
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        normalized: list[dict[str, Any]] = []
        for row in rows:
            if "estimated_cost" not in row and "estimated_cost_usd" in row:
                row["estimated_cost"] = row.get("estimated_cost_usd", "0")
                row["cost_currency"] = "USD"
            if "input_price_per_1m" not in row and "input_usd_per_1m" in row:
                row["input_price_per_1m"] = row.get("input_usd_per_1m", "0")
            if "output_price_per_1m" not in row and "output_usd_per_1m" in row:
                row["output_price_per_1m"] = row.get("output_usd_per_1m", "0")
            normalized.append(row)
        return cls(rows=normalized)

    def add(
        self,
        *,
        stage: str,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        elapsed_seconds: float,
        input_price_per_1m: float,
        output_price_per_1m: float,
        currency: str,
        item_id: str,
    ) -> None:
        cost = (prompt_tokens / 1_000_000 * input_price_per_1m) + (
            completion_tokens / 1_000_000 * output_price_per_1m
        )
        self.rows.append(
            {
                "time": datetime.now().isoformat(timespec="seconds"),
                "stage": stage,
                "provider": provider,
                "model": model,
                "item_id": item_id,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "elapsed_seconds": round(elapsed_seconds, 4),
                "input_price_per_1m": input_price_per_1m,
                "output_price_per_1m": output_price_per_1m,
                "cost_currency": currency,
                "estimated_cost": round(cost, 8),
            }
        )

    def summarize(self) -> dict[str, Any]:
        total_prompt = sum(int(r["prompt_tokens"]) for r in self.rows)
        total_completion = sum(int(r["completion_tokens"]) for r in self.rows)
        total_tokens = sum(int(r["total_tokens"]) for r in self.rows)
        total_elapsed = sum(float(r["elapsed_seconds"]) for r in self.rows)
        currencies = sorted({str(r.get("cost_currency", "")) for r in self.rows if r.get("cost_currency")})
        total_cost = sum(float(r["estimated_cost"]) for r in self.rows)
        by_stage: dict[str, dict[str, Any]] = {}
        for stage in sorted({str(r["stage"]) for r in self.rows}):
            sub = [r for r in self.rows if str(r["stage"]) == stage]
            by_stage[stage] = {
                "api_calls": len(sub),
                "prompt_tokens": sum(int(r["prompt_tokens"]) for r in sub),
                "completion_tokens": sum(int(r["completion_tokens"]) for r in sub),
                "total_tokens": sum(int(r["total_tokens"]) for r in sub),
                "api_elapsed_seconds": round(sum(float(r["elapsed_seconds"]) for r in sub), 4),
                "cost_currencies": sorted(
                    {str(r.get("cost_currency", "")) for r in sub if r.get("cost_currency")}
                ),
                "estimated_cost": round(sum(float(r["estimated_cost"]) for r in sub), 6),
            }
        return {
            "api_calls": len(self.rows),
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "total_tokens": total_tokens,
            "api_elapsed_seconds": round(total_elapsed, 4),
            "cost_currencies": currencies,
            "estimated_cost": round(total_cost, 6),
            "by_stage": by_stage,
        }

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        if self.rows:
            with (out_dir / "api_usage_calls.csv").open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(self.rows[0].keys()))
                writer.writeheader()
                writer.writerows(self.rows)
        save_json(self.summarize(), out_dir / "api_usage_summary.json")

    def subset(self, stage_prefix: str) -> "UsageTracker":
        return UsageTracker(rows=[r for r in self.rows if str(r["stage"]).startswith(stage_prefix)])


class ProviderClient:
    def __init__(self, model_key: str, usage: UsageTracker) -> None:
        cfg = MODEL_CONFIGS[model_key]
        self.model_key = model_key
        self.provider = str(cfg["provider"])
        self.model = str(cfg["model"])
        self.input_price_per_1m = float(cfg["input_price_per_1m"])
        self.output_price_per_1m = float(cfg["output_price_per_1m"])
        self.currency = str(cfg.get("currency", "USD"))
        self.usage = usage

        if self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY", "")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is missing. Please set it in .env.")
            self.client = OpenAI(api_key=api_key)
        elif self.provider == "deepseek":
            api_key = os.getenv("DEEPSEEK_API_KEY", "")
            if not api_key:
                raise ValueError("DEEPSEEK_API_KEY is missing. Please set it in .env.")
            self.client = OpenAI(base_url="https://api.deepseek.com", api_key=api_key)
        elif self.provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY", "")
            if not api_key:
                raise ValueError("GEMINI_API_KEY is missing. Please set it in .env.")
            try:
                from google import genai
                from google.genai import types as genai_types
            except ImportError as exc:
                raise ImportError("Please install google-genai: pip install google-genai") from exc
            self.genai_types = genai_types
            self.client = genai.Client(api_key=api_key)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=20), reraise=True)
    def chat(
        self,
        *,
        stage: str,
        item_id: str,
        messages: list[dict[str, str]],
        response_format: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> str:
        started = time.perf_counter()
        if self.provider == "gemini":
            content, prompt_tokens, completion_tokens, total_tokens = self._gemini_chat(
                messages=messages,
                response_format=response_format,
                temperature=temperature,
            )
        else:
            params: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
            }
            if response_format is not None:
                params["response_format"] = response_format
            response = self.client.chat.completions.create(**params)
            content = response.choices[0].message.content or ""
            usage_obj = response.usage
            prompt_tokens = int(getattr(usage_obj, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage_obj, "completion_tokens", 0) or 0)
            total_tokens = int(getattr(usage_obj, "total_tokens", 0) or 0)
        elapsed = time.perf_counter() - started
        self.usage.add(
            stage=stage,
            provider=self.provider,
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            elapsed_seconds=elapsed,
            input_price_per_1m=self.input_price_per_1m,
            output_price_per_1m=self.output_price_per_1m,
            currency=self.currency,
            item_id=item_id,
        )
        return content

    def _gemini_chat(
        self,
        *,
        messages: list[dict[str, str]],
        response_format: dict[str, Any] | None,
        temperature: float,
    ) -> tuple[str, int, int, int]:
        contents: list[Any] = []
        for msg in messages:
            role = str(msg.get("role", "user"))
            text = str(msg.get("content", ""))
            if role == "system":
                contents.append(text)
            elif role == "assistant":
                contents.append(self.genai_types.Content(role="model", parts=[self.genai_types.Part(text=text)]))
            else:
                contents.append(self.genai_types.Content(role="user", parts=[self.genai_types.Part(text=text)]))

        config_kwargs: dict[str, Any] = {"temperature": temperature}
        if response_format is not None:
            config_kwargs["response_mime_type"] = "application/json"
        config = self.genai_types.GenerateContentConfig(**config_kwargs)
        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )
        usage_obj = getattr(response, "usage_metadata", None)
        prompt_tokens = int(getattr(usage_obj, "prompt_token_count", 0) or 0)
        completion_tokens = int(getattr(usage_obj, "candidates_token_count", 0) or 0)
        total_tokens = int(getattr(usage_obj, "total_token_count", 0) or 0)
        return response.text or "", prompt_tokens, completion_tokens, total_tokens


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _coerce_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    return str(v or "").strip().lower() in {"1", "true", "yes", "y"}


def _chunk_words(text: str, chunk_words: int, overlap_words: int = 0) -> list[dict[str, Any]]:
    words = re.findall(r"\S+", str(text or ""))
    chunks: list[dict[str, Any]] = []
    step = max(1, chunk_words - max(0, overlap_words))
    for i in range(0, len(words), step):
        seg = words[i : i + chunk_words]
        if not seg:
            continue
        chunks.append(
            {
                "chunk_idx": len(chunks),
                "start_word": i,
                "end_word": i + len(seg),
                "text": " ".join(seg),
            }
        )
    return chunks


def _select_articles(n_articles: int) -> list[dict[str, Any]]:
    articles = load_jsonl(INPUT_ARTICLES_PATH)
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in articles:
        by_source[str(row["source_dataset"])].append(row)

    selected: list[dict[str, Any]] = []
    sources = ["PLOS", "eLife"]
    idx = 0
    while len(selected) < n_articles:
        progressed = False
        for source in sources:
            if len(selected) >= n_articles:
                break
            if idx < len(by_source[source]):
                selected.append(by_source[source][idx])
                progressed = True
        if not progressed:
            break
        idx += 1
    if len(selected) < n_articles:
        raise RuntimeError(f"Only found {len(selected)} articles, requested {n_articles}")
    return selected


def _binary_metrics(tp: int, tn: int, fp: int, fn: int) -> dict[str, Any]:
    total = tp + tn + fp + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": round((tp + tn) / total, 4) if total else 0.0,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "dangerous_skip_count": fn,
        "dangerous_skip_rate_among_should_keep": round(fn / (tp + fn), 4) if tp + fn else 0.0,
        "overkeep_count": fp,
        "overkeep_rate_among_should_skip": round(fp / (tn + fp), 4) if tn + fp else 0.0,
    }


def _save_usage_and_meta(
    *,
    out_dir: Path,
    usage: UsageTracker,
    run_started: float,
    model_key: str,
    n_articles: int,
    stages: list[str],
) -> None:
    usage.save(out_dir)
    cfg = MODEL_CONFIGS[model_key]
    save_json(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "model_key": model_key,
            "provider": cfg["provider"],
            "model": cfg["model"],
            "n_articles": n_articles,
            "stages": stages,
            "wall_clock_seconds": round(time.perf_counter() - run_started, 4),
            "pricing_note": "Prices are per 1M tokens, fixed in run_three_model_exp2.py for reproducible cost estimates.",
            "pricing": {
                "input_price_per_1m": cfg["input_price_per_1m"],
                "output_price_per_1m": cfg["output_price_per_1m"],
                "currency": cfg.get("currency", "USD"),
                "note": cfg.get("pricing_note", ""),
            },
            "prompts": {
                "article_af_extraction": str(ARTICLE_AF_PROMPT),
                "direct_coverage": str(DIRECT_COVERAGE_PROMPT),
                "keep_skip": str(KEEP_SKIP_PROMPT),
                "partc_summary_af": str(PARTC_SUMMARY_AF_PROMPT),
                "partc_selected_keep": str(PARTC_SELECTED_KEEP_PROMPT),
                "partc_eval": str(PARTC_EVAL_PROMPT),
            },
        },
        out_dir / "run_metadata.json",
    )


def extract_article_af(
    *,
    client: ProviderClient,
    articles: list[dict[str, Any]],
    out_dir: Path,
    chunk_words: int,
    overlap_words: int,
    resume: bool,
) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_af = out_dir / "af_article.jsonl"
    prompt = ARTICLE_AF_PROMPT.read_text(encoding="utf-8")
    rows = load_jsonl(out_af) if resume and out_af.exists() else []
    done = {str(r.get("article_id")) for r in rows}
    failed: list[dict[str, Any]] = []

    for art in tqdm(articles, desc=f"{client.model_key} | part_b article AF", total=len(articles)):
        article_id = str(art["id"])
        if resume and article_id in done:
            continue
        seen: set[str] = set()
        cnt = 0
        for ch in _chunk_words(str(art["article"]), chunk_words, overlap_words=overlap_words):
            raw = ""
            try:
                raw = client.chat(
                    stage="part_b.article_af_extraction",
                    item_id=f"{article_id}:chunk:{ch['chunk_idx']}",
                    messages=[{"role": "user", "content": prompt.replace("{source_text}", ch["text"])}],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                obj = json.loads(raw)
                facts = obj.get("atomic_facts", [])
                if not isinstance(facts, list):
                    facts = []
                for fact_obj in facts:
                    if not isinstance(fact_obj, dict):
                        continue
                    fact = str(fact_obj.get("fact", "")).strip()
                    source_span = str(fact_obj.get("source_span", "")).strip()
                    n = _norm(fact)
                    if not fact or n in seen:
                        continue
                    seen.add(n)
                    cnt += 1
                    rows.append(
                        {
                            "af_id": f"{article_id}_af_{cnt:04d}",
                            "article_id": article_id,
                            "source_dataset": art["source_dataset"],
                            "original_index": art["original_index"],
                            "fact": fact,
                            "source_span": source_span,
                            "article_chunk_idx": ch["chunk_idx"],
                            "chunk_start_word": ch["start_word"],
                            "chunk_end_word": ch["end_word"],
                            "lay_summary": art["lay_summary"],
                        }
                    )
            except Exception as exc:
                failed.append(
                    {
                        "article_id": article_id,
                        "chunk_idx": ch["chunk_idx"],
                        "error": str(exc),
                        "raw_preview": raw[:300],
                    }
                )
        done.add(article_id)
        save_jsonl(rows, out_af)

    save_jsonl(rows, out_af)
    save_json(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "model_key": client.model_key,
            "provider": client.provider,
            "model": client.model,
            "chunk_words": chunk_words,
            "overlap_words": overlap_words,
            "articles_total": len(articles),
            "articles_completed": len(done),
            "total_afs": len(rows),
            "afs_by_source": dict(Counter(str(r["source_dataset"]) for r in rows)),
            "afs_by_article": dict(Counter(str(r["article_id"]) for r in rows)),
            "failed_chunks": failed,
            "failed_chunk_count": len(failed),
        },
        out_dir / "af_extraction_metadata.json",
    )
    return rows


def part_b_direct_coverage(
    *,
    client: ProviderClient,
    af_rows: list[dict[str, Any]],
    out_dir: Path,
    resume: bool,
) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "coverage_direct_predictions.jsonl"
    prompt = DIRECT_COVERAGE_PROMPT.read_text(encoding="utf-8")
    rows = load_jsonl(out_path) if resume and out_path.exists() else []
    done = {str(r.get("af_id")) for r in rows}
    parse_errors = 0

    for af in tqdm(af_rows, desc=f"{client.model_key} | part_b direct coverage", total=len(af_rows)):
        af_id = str(af["af_id"])
        if resume and af_id in done:
            continue
        covered = False
        supporting_span = None
        reasoning = ""
        parse_error = False
        raw = ""
        try:
            raw = client.chat(
                stage="part_b.direct_coverage_gt",
                item_id=af_id,
                messages=[
                    {
                        "role": "user",
                        "content": prompt.replace("{atomic_fact}", str(af["fact"])).replace(
                            "{lay_summary}", str(af["lay_summary"])
                        ),
                    }
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            obj = json.loads(raw)
            covered = _coerce_bool(obj.get("covered", False))
            raw_span = obj.get("supporting_span", None)
            supporting_span = None if raw_span is None else str(raw_span).strip()
            if supporting_span in {"", "null", "None"}:
                supporting_span = None
            reasoning = str(obj.get("reasoning", "")).strip()
        except Exception as exc:
            parse_error = True
            parse_errors += 1
            reasoning = f"parse_or_runtime_error: {exc}"

        rows.append(
            {
                "af_id": af_id,
                "article_id": af["article_id"],
                "source_dataset": af["source_dataset"],
                "fact": af["fact"],
                "covered": covered,
                "supporting_span": supporting_span,
                "reasoning": reasoning,
                "parse_error": parse_error,
            }
        )
        done.add(af_id)
        save_jsonl(rows, out_path)

    save_json(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "model_key": client.model_key,
            "provider": client.provider,
            "model": client.model,
            "total_afs": len(af_rows),
            "coverage_rows": len(rows),
            "parse_error_count": sum(1 for r in rows if r.get("parse_error")),
            "parse_errors_this_run": parse_errors,
        },
        out_dir / "coverage_direct_metadata.json",
    )
    return rows


def part_b_keep_skip(
    *,
    client: ProviderClient,
    af_rows: list[dict[str, Any]],
    out_dir: Path,
    resume: bool,
) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "keep_skip_predictions.jsonl"
    prompt = KEEP_SKIP_PROMPT.read_text(encoding="utf-8")
    rows = load_jsonl(out_path) if resume and out_path.exists() else []
    done = {str(r.get("af_id")) for r in rows}
    parse_errors = 0

    for af in tqdm(af_rows, desc=f"{client.model_key} | part_b keep/skip", total=len(af_rows)):
        af_id = str(af["af_id"])
        if resume and af_id in done:
            continue
        keep = False
        confidence = "low"
        reasoning = ""
        parse_error = False
        raw = ""
        try:
            raw = client.chat(
                stage="part_b.keep_skip_ultra_recall",
                item_id=af_id,
                messages=[
                    {"role": "user", "content": prompt.replace("{atomic_fact}", str(af["fact"]))}
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            obj = json.loads(raw)
            keep = _coerce_bool(obj.get("keep", False))
            confidence = str(obj.get("confidence", "low")).strip().lower()
            if confidence not in {"high", "medium", "low"}:
                confidence = "low"
            reasoning = str(obj.get("reasoning", "")).strip()
        except Exception as exc:
            parse_error = True
            parse_errors += 1
            reasoning = f"parse_or_runtime_error: {exc}"

        rows.append(
            {
                "af_id": af_id,
                "article_id": af["article_id"],
                "source_dataset": af["source_dataset"],
                "fact": af["fact"],
                "module_keep": keep,
                "module_label": "keep" if keep else "skip",
                "confidence": confidence,
                "reasoning": reasoning,
                "parse_error": parse_error,
            }
        )
        done.add(af_id)
        save_jsonl(rows, out_path)

    save_json(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "model_key": client.model_key,
            "provider": client.provider,
            "model": client.model,
            "prompt_path": str(KEEP_SKIP_PROMPT),
            "total_afs": len(af_rows),
            "prediction_rows": len(rows),
            "parse_error_count": sum(1 for r in rows if r.get("parse_error")),
            "parse_errors_this_run": parse_errors,
        },
        out_dir / "keep_skip_metadata.json",
    )
    return rows


def part_b_eval(*, direct_rows: list[dict[str, Any]], keep_rows: list[dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cov_map = {str(r["af_id"]): r for r in direct_rows if not r.get("parse_error", False)}
    keep_map = {str(r["af_id"]): r for r in keep_rows if not r.get("parse_error", False)}
    rows: list[dict[str, Any]] = []
    for af_id in sorted(set(cov_map) & set(keep_map)):
        cov = cov_map[af_id]
        keep = keep_map[af_id]
        should_keep = bool(cov["covered"])
        pred_keep = bool(keep["module_keep"])
        rows.append(
            {
                "af_id": af_id,
                "article_id": cov["article_id"],
                "source_dataset": cov["source_dataset"],
                "fact": cov["fact"],
                "ground_truth_keep": should_keep,
                "module_keep": pred_keep,
                "module_label": "keep" if pred_keep else "skip",
                "covered_by_summary": should_keep,
                "supporting_span": cov.get("supporting_span", "Quote: NONE"),
                "confidence": keep.get("confidence", "low"),
                "module_reasoning": keep.get("reasoning", ""),
                "correct": should_keep == pred_keep,
                "error_type": (
                    "correct_keep"
                    if should_keep and pred_keep
                    else "correct_skip"
                    if (not should_keep and not pred_keep)
                    else "dangerous_skip"
                    if should_keep and not pred_keep
                    else "overkeep"
                ),
            }
        )
    save_jsonl(rows, out_dir / "keep_skip_vs_coverage.jsonl")

    def summarize(subrows: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(subrows)
        covered = sum(1 for r in subrows if r["ground_truth_keep"])
        not_covered = total - covered
        tp = sum(1 for r in subrows if r["ground_truth_keep"] and r["module_keep"])
        tn = sum(1 for r in subrows if (not r["ground_truth_keep"]) and (not r["module_keep"]))
        fp = sum(1 for r in subrows if (not r["ground_truth_keep"]) and r["module_keep"])
        fn = sum(1 for r in subrows if r["ground_truth_keep"] and (not r["module_keep"]))
        return {
            "n": total,
            "direct_coverage_ground_truth": {
                "coverage_recall": round(covered / total, 4) if total else 0.0,
                "covered_count": covered,
                "not_covered_count": not_covered,
            },
            "keep_skip": {
                "confusion_matrix": {
                    "module_keep__gt_keep": tp,
                    "module_keep__gt_skip": fp,
                    "module_skip__gt_keep_dangerous": fn,
                    "module_skip__gt_skip": tn,
                },
                **_binary_metrics(tp, tn, fp, fn),
            },
        }

    summary = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "overall": summarize(rows),
        "by_source": {
            source: summarize([r for r in rows if r["source_dataset"] == source])
            for source in sorted({r["source_dataset"] for r in rows})
        },
    }
    save_json(summary, out_dir / "metrics_summary.json")

    lines = ["source,tp,tn,fp,fn,accuracy,precision,recall,f1,dangerous_skip_rate_among_should_keep"]
    for source, data in {"overall": summary["overall"], **summary["by_source"]}.items():
        m = data["keep_skip"]
        cm = m["confusion_matrix"]
        lines.append(
            ",".join(
                [
                    source,
                    str(cm["module_keep__gt_keep"]),
                    str(cm["module_skip__gt_skip"]),
                    str(cm["module_keep__gt_skip"]),
                    str(cm["module_skip__gt_keep_dangerous"]),
                    str(m["accuracy"]),
                    str(m["precision"]),
                    str(m["recall"]),
                    str(m["f1"]),
                    str(m["dangerous_skip_rate_among_should_keep"]),
                ]
            )
        )
    (out_dir / "confusion_matrix.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def partc_summary_af(
    *,
    client: ProviderClient,
    articles: list[dict[str, Any]],
    out_dir: Path,
    resume: bool,
) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary_af.jsonl"
    prompt = PARTC_SUMMARY_AF_PROMPT.read_text(encoding="utf-8")
    rows = load_jsonl(out_path) if resume and out_path.exists() else []
    done = {str(r.get("article_id")) for r in rows}
    failed: list[dict[str, Any]] = []

    for art in tqdm(articles, desc=f"{client.model_key} | part_c summary AF", total=len(articles)):
        article_id = str(art["id"])
        if resume and article_id in done:
            continue
        raw = ""
        try:
            raw = client.chat(
                stage="part_c.summary_af_extraction",
                item_id=article_id,
                messages=[
                    {"role": "user", "content": prompt.replace("{lay_summary}", str(art["lay_summary"]))}
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            obj = json.loads(raw)
            facts = obj.get("summary_atomic_facts", [])
            if not isinstance(facts, list):
                facts = []
            seen: set[str] = set()
            cnt = 0
            for fact_obj in facts:
                if not isinstance(fact_obj, dict):
                    continue
                fact = str(fact_obj.get("fact", "")).strip()
                source_sentence = str(fact_obj.get("source_sentence", "")).strip()
                n = _norm(fact)
                if not fact or n in seen:
                    continue
                seen.add(n)
                cnt += 1
                rows.append(
                    {
                        "summary_af_id": f"{article_id}_summary_af_{cnt:04d}",
                        "article_id": article_id,
                        "source_dataset": art["source_dataset"],
                        "fact": fact,
                        "source_sentence": source_sentence,
                        "lay_summary": art["lay_summary"],
                    }
                )
        except Exception as exc:
            failed.append({"article_id": article_id, "error": str(exc), "raw_preview": raw[:300]})
        done.add(article_id)
        save_jsonl(rows, out_path)

    save_json(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "model_key": client.model_key,
            "provider": client.provider,
            "model": client.model,
            "articles_total": len(articles),
            "summary_af_total": len(rows),
            "summary_af_by_source": dict(Counter(str(r["source_dataset"]) for r in rows)),
            "summary_af_by_article": dict(Counter(str(r["article_id"]) for r in rows)),
            "failed_items": failed,
        },
        out_dir / "summary_af_metadata.json",
    )
    return rows


def partc_selected_keep_af(
    *,
    client: ProviderClient,
    articles: list[dict[str, Any]],
    out_dir: Path,
    chunk_words: int,
    overlap_words: int,
    resume: bool,
) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "selected_keep_af.jsonl"
    prompt = PARTC_SELECTED_KEEP_PROMPT.read_text(encoding="utf-8")
    rows = load_jsonl(out_path) if resume and out_path.exists() else []
    done = {str(r.get("article_id")) for r in rows}
    failed: list[dict[str, Any]] = []

    for art in tqdm(articles, desc=f"{client.model_key} | part_c selected keep AF", total=len(articles)):
        article_id = str(art["id"])
        if resume and article_id in done:
            continue
        seen: set[str] = set()
        cnt = 0
        for ch in _chunk_words(str(art["article"]), chunk_words, overlap_words=overlap_words):
            raw = ""
            try:
                raw = client.chat(
                    stage="part_c.selected_keep_af_extraction",
                    item_id=f"{article_id}:chunk:{ch['chunk_idx']}",
                    messages=[{"role": "user", "content": prompt.replace("{source_text}", ch["text"])}],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                obj = json.loads(raw)
                facts = obj.get("keep_atomic_facts", [])
                if not isinstance(facts, list):
                    facts = []
                for fact_obj in facts:
                    if not isinstance(fact_obj, dict):
                        continue
                    fact = str(fact_obj.get("fact", "")).strip()
                    source_span = str(fact_obj.get("source_span", "")).strip()
                    reasoning = str(fact_obj.get("reasoning", "")).strip()
                    n = _norm(fact)
                    if not fact or n in seen:
                        continue
                    seen.add(n)
                    cnt += 1
                    rows.append(
                        {
                            "selected_keep_af_id": f"{article_id}_selected_keep_af_{cnt:04d}",
                            "article_id": article_id,
                            "source_dataset": art["source_dataset"],
                            "fact": fact,
                            "source_span": source_span,
                            "keep_reasoning": reasoning,
                            "article_chunk_idx": ch["chunk_idx"],
                            "chunk_start_word": ch["start_word"],
                            "chunk_end_word": ch["end_word"],
                        }
                    )
            except Exception as exc:
                failed.append(
                    {
                        "article_id": article_id,
                        "chunk_idx": ch["chunk_idx"],
                        "error": str(exc),
                        "raw_preview": raw[:300],
                    }
                )
        done.add(article_id)
        save_jsonl(rows, out_path)

    save_json(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "model_key": client.model_key,
            "provider": client.provider,
            "model": client.model,
            "chunk_words": chunk_words,
            "overlap_words": overlap_words,
            "articles_total": len(articles),
            "selected_keep_af_total": len(rows),
            "selected_keep_af_by_source": dict(Counter(str(r["source_dataset"]) for r in rows)),
            "selected_keep_af_by_article": dict(Counter(str(r["article_id"]) for r in rows)),
            "failed_chunks": failed,
            "failed_chunk_count": len(failed),
        },
        out_dir / "selected_keep_af_metadata.json",
    )
    return rows


def partc_eval(
    *,
    client: ProviderClient,
    summary_af: list[dict[str, Any]],
    selected_keep: list[dict[str, Any]],
    out_dir: Path,
    resume: bool,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary_af_coverage_by_selected_keep.jsonl"
    prompt = PARTC_EVAL_PROMPT.read_text(encoding="utf-8")
    by_article: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in selected_keep:
        by_article[str(r["article_id"])].append(r)

    rows = load_jsonl(out_path) if resume and out_path.exists() else []
    done = {str(r.get("summary_af_id")) for r in rows}
    parse_errors = 0
    for af in tqdm(summary_af, desc=f"{client.model_key} | part_c eval", total=len(summary_af)):
        sid = str(af["summary_af_id"])
        if resume and sid in done:
            continue
        cands = by_article.get(str(af["article_id"]), [])
        selected_text = "\n".join(f"- {r['selected_keep_af_id']}: {r['fact']}" for r in cands)
        if not selected_text:
            selected_text = "NONE"

        covered = False
        support_ids: list[str] = []
        support_facts: list[str] = []
        reasoning = ""
        parse_error = False
        raw = ""
        try:
            raw = client.chat(
                stage="part_c.selected_keep_covers_summary_af",
                item_id=sid,
                messages=[
                    {
                        "role": "user",
                        "content": prompt.replace("{summary_fact}", str(af["fact"])).replace(
                            "{selected_keep_facts}", selected_text
                        ),
                    }
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            obj = json.loads(raw)
            covered = _coerce_bool(obj.get("covered", False))
            raw_ids = obj.get("supporting_keep_af_ids", obj.get("supporting_keep_af_id", []))
            raw_facts = obj.get("supporting_keep_facts", obj.get("supporting_keep_fact", []))
            if not isinstance(raw_ids, list):
                raw_ids = [raw_ids]
            if not isinstance(raw_facts, list):
                raw_facts = [raw_facts]
            support_ids = [
                str(x).strip()
                for x in raw_ids
                if x is not None and str(x).strip() not in {"", "null", "None"}
            ]
            support_facts = [
                str(x).strip()
                for x in raw_facts
                if x is not None and str(x).strip() not in {"", "null", "None"}
            ]
            reasoning = str(obj.get("reasoning", "")).strip()
        except Exception as exc:
            parse_error = True
            parse_errors += 1
            reasoning = f"parse_or_runtime_error: {exc}"

        rows.append(
            {
                "summary_af_id": sid,
                "article_id": af["article_id"],
                "source_dataset": af["source_dataset"],
                "summary_fact": af["fact"],
                "covered_by_selected_keep_af": covered,
                "supporting_keep_af_id": support_ids[0] if support_ids else None,
                "supporting_keep_fact": support_facts[0] if support_facts else None,
                "supporting_keep_af_ids": support_ids,
                "supporting_keep_facts": support_facts,
                "selected_keep_af_count_for_article": len(cands),
                "reasoning": reasoning,
                "parse_error": parse_error,
            }
        )
        done.add(sid)
        save_jsonl(rows, out_path)

    valid = [r for r in rows if not r.get("parse_error", False)]
    total = len(valid)
    covered_count = sum(1 for r in valid if r["covered_by_selected_keep_af"])
    selected_by_article = Counter(str(r["article_id"]) for r in selected_keep)
    summary_by_article = Counter(str(r["article_id"]) for r in summary_af)
    metrics = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "model_key": client.model_key,
        "provider": client.provider,
        "model": client.model,
        "summary_af_total": len(summary_af),
        "selected_keep_af_total": len(selected_keep),
        "valid_eval_rows": total,
        "parse_error_count": sum(1 for r in rows if r.get("parse_error")),
        "parse_errors_this_run": parse_errors,
        "summary_fact_recall": round(covered_count / total, 4) if total else 0.0,
        "covered_summary_af_count": covered_count,
        "missed_summary_af_count": total - covered_count,
        "selected_keep_af_by_source": dict(Counter(str(r["source_dataset"]) for r in selected_keep)),
        "summary_af_by_source": dict(Counter(str(r["source_dataset"]) for r in summary_af)),
        "by_source": {},
        "by_article": {},
    }
    for source in sorted({str(r["source_dataset"]) for r in valid}):
        sub = [r for r in valid if str(r["source_dataset"]) == source]
        cov = sum(1 for r in sub if r["covered_by_selected_keep_af"])
        metrics["by_source"][source] = {
            "summary_af_total": len(sub),
            "covered": cov,
            "summary_fact_recall": round(cov / len(sub), 4) if sub else 0.0,
        }
    for article_id in sorted(set(summary_by_article) | set(selected_by_article)):
        sub = [r for r in valid if str(r["article_id"]) == article_id]
        cov = sum(1 for r in sub if r["covered_by_selected_keep_af"])
        metrics["by_article"][article_id] = {
            "summary_af_total": len(sub),
            "selected_keep_af_total": selected_by_article.get(article_id, 0),
            "covered": cov,
            "summary_fact_recall": round(cov / len(sub), 4) if sub else 0.0,
        }
    save_json(metrics, out_dir / "partc_metrics_summary.json")
    (out_dir / "partc_summary.csv").write_text(
        "metric,value\n"
        f"summary_af_total,{metrics['summary_af_total']}\n"
        f"selected_keep_af_total,{metrics['selected_keep_af_total']}\n"
        f"summary_fact_recall,{metrics['summary_fact_recall']}\n"
        f"covered_summary_af_count,{metrics['covered_summary_af_count']}\n"
        f"missed_summary_af_count,{metrics['missed_summary_af_count']}\n",
        encoding="utf-8",
    )
    return metrics


def run_model(
    *,
    model_key: str,
    n_articles: int,
    chunk_words: int,
    overlap_words: int,
    resume: bool,
) -> dict[str, Any]:
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model key: {model_key}. Valid: {sorted(MODEL_CONFIGS)}")

    run_started = time.perf_counter()
    articles = _select_articles(n_articles)
    run_dir = RUNS_DIR / f"n{n_articles}" / model_key
    usage = UsageTracker.from_csv(run_dir / "api_usage_calls.csv") if resume else UsageTracker(rows=[])
    client = ProviderClient(model_key, usage)

    part_b_dir = run_dir / "part_b"
    part_c_dir = run_dir / "part_c"
    save_jsonl(articles, run_dir / "selected_articles.jsonl")
    save_json(
        {
            "n_articles": n_articles,
            "by_source": dict(Counter(str(r["source_dataset"]) for r in articles)),
            "article_ids": [str(r["id"]) for r in articles],
        },
        run_dir / "selected_articles_metadata.json",
    )

    af_rows = extract_article_af(
        client=client,
        articles=articles,
        out_dir=part_b_dir / "01_article_af",
        chunk_words=chunk_words,
        overlap_words=overlap_words,
        resume=resume,
    )
    direct_rows = part_b_direct_coverage(
        client=client,
        af_rows=af_rows,
        out_dir=part_b_dir / "02_direct_coverage_gt",
        resume=resume,
    )
    keep_rows = part_b_keep_skip(
        client=client,
        af_rows=af_rows,
        out_dir=part_b_dir / "03_keep_skip",
        resume=resume,
    )
    part_b_summary = part_b_eval(
        direct_rows=direct_rows,
        keep_rows=keep_rows,
        out_dir=part_b_dir / "04_eval",
    )

    summary_af = partc_summary_af(
        client=client,
        articles=articles,
        out_dir=part_c_dir / "01_summary_af",
        resume=resume,
    )
    selected_keep = partc_selected_keep_af(
        client=client,
        articles=articles,
        out_dir=part_c_dir / "02_selected_keep_af",
        chunk_words=chunk_words,
        overlap_words=overlap_words,
        resume=resume,
    )
    part_c_summary = partc_eval(
        client=client,
        summary_af=summary_af,
        selected_keep=selected_keep,
        out_dir=part_c_dir / "03_eval",
        resume=resume,
    )

    part_b_usage = usage.subset("part_b.")
    part_c_usage = usage.subset("part_c.")
    part_b_usage.save(part_b_dir)
    part_c_usage.save(part_c_dir)
    _save_usage_and_meta(
        out_dir=run_dir,
        usage=usage,
        run_started=run_started,
        model_key=model_key,
        n_articles=n_articles,
        stages=["part_b", "part_c"],
    )
    method_usage = usage.summarize()
    summary = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "model_key": model_key,
        "provider": client.provider,
        "model": client.model,
        "n_articles": n_articles,
        "part_b": part_b_summary,
        "part_c": part_c_summary,
        "part_b_usage": part_b_usage.summarize(),
        "part_c_usage": part_c_usage.summarize(),
        "usage": method_usage,
    }
    save_json(summary, run_dir / "model_summary.json")
    return summary


def write_comparison(n_articles: int) -> None:
    base = RUNS_DIR / f"n{n_articles}"
    rows: list[dict[str, Any]] = []
    for model_key in MODEL_CONFIGS:
        summary_path = base / model_key / "model_summary.json"
        usage_path = base / model_key / "api_usage_summary.json"
        if not summary_path.exists() or not usage_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        usage = json.loads(usage_path.read_text(encoding="utf-8"))
        part_b_overall = summary["part_b"]["overall"]
        part_b_keep = part_b_overall["keep_skip"]
        part_c = summary["part_c"]
        rows.append(
            {
                "model_key": model_key,
                "provider": summary["provider"],
                "model": summary["model"],
                "n_articles": n_articles,
                "part_b_article_af_total": part_b_overall["n"],
                "part_b_gt_keep": part_b_overall["direct_coverage_ground_truth"]["covered_count"],
                "part_b_gt_skip": part_b_overall["direct_coverage_ground_truth"]["not_covered_count"],
                "part_b_recall": part_b_keep["recall"],
                "part_b_dangerous_skip_rate": part_b_keep["dangerous_skip_rate_among_should_keep"],
                "part_b_precision": part_b_keep["precision"],
                "part_c_summary_af_total": part_c["summary_af_total"],
                "part_c_selected_keep_af_total": part_c["selected_keep_af_total"],
                "part_c_summary_fact_recall": part_c["summary_fact_recall"],
                "api_calls": usage["api_calls"],
                "total_tokens": usage["total_tokens"],
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
                "api_elapsed_seconds": usage["api_elapsed_seconds"],
                "cost_currencies": ";".join(usage.get("cost_currencies", [])),
                "estimated_cost": usage["estimated_cost"],
            }
        )

    out_csv = base / "comparison_summary.csv"
    out_json = base / "comparison_summary.json"
    if rows:
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    save_json(rows, out_json)
    print(f"[comparison] wrote {out_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Experiment 2 Part B/C with three models.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_CONFIGS.keys()),
        help=f"Model keys to run. Valid: {', '.join(MODEL_CONFIGS)}",
    )
    parser.add_argument("--n-articles", type=int, default=5)
    parser.add_argument("--chunk-words", type=int, default=1200)
    parser.add_argument("--overlap-words", type=int, default=120)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--comparison-only", action="store_true")
    args = parser.parse_args()

    if args.comparison_only:
        write_comparison(args.n_articles)
        return

    for model_key in args.models:
        print(f"[run] model={model_key} n_articles={args.n_articles}")
        run_model(
            model_key=model_key,
            n_articles=args.n_articles,
            chunk_words=args.chunk_words,
            overlap_words=args.overlap_words,
            resume=not args.no_resume,
        )
    write_comparison(args.n_articles)


if __name__ == "__main__":
    main()
