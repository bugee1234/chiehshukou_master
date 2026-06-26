from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.experiment_2.run_three_model_exp2 import MODEL_CONFIGS, ProviderClient, UsageTracker
from src.thesis_laysumm.llm_utils import ParallelExecutionError, run_parallel, sha256_text
from src.thesis_laysumm.paths import run_data_dir as atlas_run_data_dir
from src.thesis_laysumm.paths import run_results_dir as atlas_run_results_dir
from src.thesis_laysumm_ablation_m1_recall.paths import PROMPTS_DIR, run_data_dir, run_results_dir, usage_path
from src.thesis_laysumm_v11.common import find_mojibake, length_policy, safe_word_count
from src.utils import load_jsonl, save_json, save_jsonl


DEFAULT_ATLAS_RUN = "v11_constellation_validation284_fixed_m12_pilot20"
DEFAULT_PILOT_RUN = "pilot_n20_v11_factuality_chase"
DEFAULT_DIRECT_RUN = "direct_zero_shot_pilot20_raw_article_len220"
DEFAULT_RUN_NAME = "module1_only_ablation_and_expert_af_recall_pilot20"
DEFAULT_MODELS = ["gemini25_flash_non_thinking", "gemini3_flash_preview_minimal", "gpt41_mini"]
DEFAULT_RECALL_JUDGE = "gemini25_flash_non_thinking"
METRICS = ["ROUGE", "BLEU", "METEOR", "BERTScore", "FKGL", "DCRS", "CLI", "LENS", "AlignScore", "SummaC"]
MODEL_ABBR = {
    "gemini25_flash_non_thinking": "g25",
    "gemini3_flash_preview_minimal": "g3",
    "gpt41_mini": "g41m",
}
VARIANT_ABBR = {
    "direct": "dir",
    "module1": "m1",
    "module12_generated": "m12g",
    "module123_rewritten": "m123r",
}
VARIANTS = {
    "direct": "Direct (Zero-shot)",
    "module1": "Module 1",
    "module12_generated": "Module 1+2 (Generated)",
    "module123_rewritten": "Module 1+2+3 (Rewritten)",
}
MODEL_LABELS = {
    "gemini25_flash_non_thinking": "Gemini 2.5 Flash",
    "gemini3_flash_preview_minimal": "Gemini 3 Flash",
    "gpt41_mini": "GPT-4.1 Mini",
}


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _ensure_local_nltk_data() -> None:
    nltk_dir = ROOT_DIR / ".nltk_data"
    if not nltk_dir.exists():
        return
    current = os.environ.get("NLTK_DATA", "")
    paths = [p for p in current.split(os.pathsep) if p]
    nltk_text = str(nltk_dir)
    if nltk_text not in paths:
        os.environ["NLTK_DATA"] = os.pathsep.join([nltk_text, *paths])
    try:
        import nltk

        if nltk_text not in nltk.data.path:
            nltk.data.path.insert(0, nltk_text)
    except Exception:
        pass


def _loads_json_object(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Model response did not contain a JSON object: {raw[:300]}")
    obj = json.loads(match.group(0))
    if not isinstance(obj, dict):
        raise ValueError("Model response JSON was not an object")
    return obj


def _parse_recall_judge_response(raw: str) -> dict[str, Any]:
    try:
        obj = _loads_json_object(raw)
        obj["_parse_fallback"] = False
        return obj
    except Exception as exc:
        text = str(raw or "").strip()
        lower = text.lower()
        label = ""
        label_match = re.search(
            r'"?label"?\s*[:=]\s*"?\b(covered|partially_covered|not_covered)\b"?',
            lower,
        )
        if label_match:
            label = label_match.group(1)
        else:
            for candidate in ("not_covered", "partially_covered", "covered"):
                if re.search(rf"\b{candidate}\b", lower):
                    label = candidate
                    break
        if not label:
            raise ValueError(f"Could not parse recall judge response as JSON or label text: {text[:300]}") from exc
        rationale = ""
        rationale_match = re.search(r'"?rationale"?\s*[:=]\s*"?(.+?)"?\s*(?:\}|\n|$)', text, flags=re.IGNORECASE | re.DOTALL)
        if rationale_match:
            rationale = rationale_match.group(1).strip().strip('",')
        return {
            "label": label,
            "rationale": rationale or "Recovered label from malformed judge JSON.",
            "_parse_fallback": True,
            "_parse_error": str(exc),
        }


def _summary_from_model_response(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    try:
        obj = _loads_json_object(text)
        summary = str(obj.get("summary") or "").strip()
        if summary:
            return summary
    except Exception:
        pass
    if text.startswith("```"):
        text = re.sub(r"^```(?:json|text)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text).strip()
    return text


def _validate_english_summary(summary: str, *, article_id: str, model_key: str) -> list[str]:
    issues = find_mojibake(summary)
    chars = [ch for ch in str(summary or "") if not ch.isspace()]
    non_ascii = [ch for ch in chars if ord(ch) > 127]
    non_ascii_ratio = len(non_ascii) / max(len(chars), 1)
    if non_ascii_ratio > 0.08:
        issues.append(f"high_non_ascii_ratio:{non_ascii_ratio:.3f}")
    if issues:
        raise RuntimeError(
            f"Invalid Module 1-only summary for {article_id} ({model_key}); "
            f"expected English text, got issues={issues[:20]}"
        )
    return issues


def _load_usage(run_name: str, resume: bool) -> UsageTracker:
    path = usage_path(run_name)
    return UsageTracker.from_csv(path) if resume and path.exists() else UsageTracker(rows=[])


def _save_usage(run_name: str, usage: UsageTracker) -> None:
    usage.save(run_data_dir(run_name))


def _model_cfg_payload(model_key: str) -> dict[str, Any]:
    cfg = MODEL_CONFIGS[model_key]
    return {
        "model_key": model_key,
        "provider": cfg["provider"],
        "model": cfg["model"],
    }


def _judge_cfg_payload(judge_model_key: str) -> dict[str, Any]:
    cfg = MODEL_CONFIGS[judge_model_key]
    return {
        "judge_provider": cfg["provider"],
        "judge_model": cfg["model"],
    }


def _sample_path(run_name: str) -> Path:
    return run_data_dir(run_name) / "00_sample" / "articles.jsonl"


def _sample_metadata_path(run_name: str) -> Path:
    return run_data_dir(run_name) / "00_sample" / "sample_metadata.json"


def _module1_path(run_name: str, model_key: str) -> Path:
    return run_data_dir(run_name) / "01_module1_only_summaries" / model_key / "module1_summaries.jsonl"


def _variants_path(run_name: str, model_key: str) -> Path:
    return run_data_dir(run_name) / "02_variants" / model_key / "summary_variants.jsonl"


def _expert_af_path(run_name: str) -> Path:
    return run_data_dir(run_name) / "03_expert_summary_af" / "expert_summary_af.jsonl"


def _abstract_claim_path(run_name: str) -> Path:
    return run_data_dir(run_name) / "03b_abstract_grounded_reference" / "abstract_grounded_claims.jsonl"


def _recall_judgements_path(run_name: str, model_key: str) -> Path:
    return run_data_dir(run_name) / "04_expert_af_recall" / model_key / "recall_judgements.jsonl"


def _metrics_dir(run_name: str, model_key: str) -> Path:
    return run_results_dir(run_name) / model_key / "official_metrics"


def _recall_dir(run_name: str, model_key: str) -> Path:
    return run_results_dir(run_name) / model_key / "expert_af_recall"


def _load_sample(run_name: str) -> list[dict[str, Any]]:
    path = _sample_path(run_name)
    if not path.exists():
        raise FileNotFoundError(f"Missing prepared sample: {path}")
    return load_jsonl(path)


def _article_order(articles: list[dict[str, Any]]) -> dict[str, int]:
    return {str(row["id"]): idx for idx, row in enumerate(articles)}


def prepare_sample(*, run_name: str, atlas_run: str, pilot_run: str, force: bool = False) -> None:
    out_path = _sample_path(run_name)
    meta_path = _sample_metadata_path(run_name)
    if out_path.exists() and not force:
        print(f"[prepare-sample] exists -> {out_path}")
        return
    pilot_articles_path = atlas_run_data_dir(pilot_run) / "00_articles" / "articles.jsonl"
    atlas_articles_path = atlas_run_data_dir(atlas_run) / "00_articles" / "articles.jsonl"
    if not pilot_articles_path.exists():
        raise FileNotFoundError(f"Missing pilot articles: {pilot_articles_path}")
    if not atlas_articles_path.exists():
        raise FileNotFoundError(f"Missing ATLAS articles: {atlas_articles_path}")

    pilot_articles = load_jsonl(pilot_articles_path)
    atlas_by_id = {str(row["id"]): row for row in load_jsonl(atlas_articles_path)}
    sample: list[dict[str, Any]] = []
    missing: list[str] = []
    for row in pilot_articles:
        aid = str(row["id"])
        if aid not in atlas_by_id:
            missing.append(aid)
            continue
        sample.append(atlas_by_id[aid])
    if missing:
        raise ValueError(f"Pilot articles missing from ATLAS run: {missing}")
    if len(sample) != 20:
        raise ValueError(f"Expected 20 pilot articles, got {len(sample)}")
    save_jsonl(sample, out_path)
    save_json(
        {
            "time": _now(),
            "run_name": run_name,
            "atlas_run": atlas_run,
            "pilot_run": pilot_run,
            "article_count": len(sample),
            "by_source": dict(sorted({s: sum(1 for r in sample if r.get("source_dataset") == s) for s in {"PLOS", "eLife"}}.items())),
            "article_ids": [str(row["id"]) for row in sample],
            "note": "Pilot n=20 articles forced into the ATLAS validation-284 run; used for Module 1 ablation and Expert-AF Recall.",
        },
        meta_path,
    )
    print(f"[prepare-sample] articles={len(sample)} -> {out_path}")


def _candidate_facts_by_article(atlas_run: str, model_key: str) -> dict[str, list[dict[str, Any]]]:
    path = atlas_run_data_dir(atlas_run) / "01_module1" / model_key / "candidate_keep_af.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing Module 1 candidate AF file: {path}")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in load_jsonl(path):
        grouped[str(row["article_id"])].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda r: str(r.get("af_id", "")))
    return grouped


def _format_candidate_facts(rows: list[dict[str, Any]]) -> str:
    compact_rows = [
        {
            "af_id": row.get("af_id"),
            "fact": row.get("fact"),
        }
        for row in rows
        if str(row.get("fact") or "").strip()
    ]
    return json.dumps(compact_rows, ensure_ascii=False, separators=(",", ":"))


def generate_module1_summaries(
    *,
    run_name: str,
    atlas_run: str,
    model_key: str,
    max_workers: int,
    resume: bool,
) -> None:
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model_key: {model_key}")
    articles = _load_sample(run_name)
    order = _article_order(articles)
    candidates = _candidate_facts_by_article(atlas_run, model_key)
    out_path = _module1_path(run_name, model_key)
    template = (PROMPTS_DIR / "module1_only_summary.txt").read_text(encoding="utf-8")
    prompt_hash = sha256_text(template)
    existing = load_jsonl(out_path) if resume and out_path.exists() else []
    done = {str(row.get("article_id")) for row in existing if str(row.get("generated_summary") or "").strip() and not row.get("parse_error")}
    usage = _load_usage(run_name, resume=resume)
    client = ProviderClient(model_key, usage)

    def worker(article: dict[str, Any]) -> dict[str, Any]:
        aid = str(article["id"])
        fact_rows = candidates.get(aid, [])
        if not fact_rows:
            raise RuntimeError(f"No Module 1 candidate AFs for {aid} ({model_key})")
        prompt = (
            template.replace("{article}", str(article.get("article") or article.get("document") or "").strip())
            .replace("{candidate_facts}", _format_candidate_facts(fact_rows))
        )
        raw = client.chat(
            stage=f"m1_ablation.generate_module1_summary:{model_key}",
            item_id=aid,
            messages=[{"role": "user", "content": prompt}],
            response_format=None,
            temperature=0.0,
        )
        summary = _summary_from_model_response(raw)
        if not summary:
            raise RuntimeError(f"Empty Module 1-only summary for {aid}")
        output_quality_warnings = _validate_english_summary(summary, article_id=aid, model_key=model_key)
        return {
            "article_id": aid,
            "source_dataset": article.get("source_dataset"),
            "original_index": article.get("original_index"),
            "model_key": model_key,
            **_model_cfg_payload(model_key),
            "variant": "module1",
            "phase_label": VARIANTS["module1"],
            "atlas_source_run": atlas_run,
            "prompt_file": str(PROMPTS_DIR / "module1_only_summary.txt"),
            "prompt_sha256": prompt_hash,
            "visible_input_fields": ["article", "module1_candidate_atomic_facts"],
            "module1_candidate_af_count": len(fact_rows),
            "generated_summary": summary,
            "word_count": safe_word_count(summary),
            "output_quality_warnings": output_quality_warnings,
            "parse_error": False,
            "raw_response": raw,
        }

    todo = [article for article in articles if str(article["id"]) not in done]
    try:
        new_rows = run_parallel(todo, worker, max_workers=max_workers, desc=f"Module1-only summary | {model_key}")
    except ParallelExecutionError as exc:
        checkpoint = sorted(existing + exc.partial_results, key=lambda r: order[str(r["article_id"])])
        save_jsonl(checkpoint, out_path)
        _save_usage(run_name, usage)
        print(f"[generate-module1] checkpointed {len(checkpoint)} rows -> {out_path}")
        raise
    combined = sorted(existing + new_rows, key=lambda r: order[str(r["article_id"])])
    save_jsonl(combined, out_path)
    _save_usage(run_name, usage)
    save_json(
        {
            "time": _now(),
            "run_name": run_name,
            "atlas_run": atlas_run,
            "model_key": model_key,
            "article_count": len(combined),
            "new_article_count": len(new_rows),
            "prompt_sha256": prompt_hash,
            "usage_summary": usage.summarize(),
        },
        out_path.parent / "module1_summaries_metadata.json",
    )
    print(f"[generate-module1] {model_key}: rows={len(combined)} new={len(new_rows)} -> {out_path}")


def _load_atlas_variant(atlas_run: str, model_key: str, variant: str) -> dict[str, str]:
    if variant == "module12_generated":
        path = atlas_run_data_dir(atlas_run) / "04_summaries" / model_key / "generated_summaries.jsonl"
        field = "generated_summary"
    elif variant == "module123_rewritten":
        path = atlas_run_data_dir(atlas_run) / "06_rewritten" / model_key / "rewritten_summaries.jsonl"
        field = "rewritten_summary"
    else:
        raise ValueError(variant)
    if not path.exists():
        raise FileNotFoundError(f"Missing ATLAS variant file: {path}")
    return {
        str(row["article_id"]): str(row.get(field) or "").strip()
        for row in load_jsonl(path)
        if not row.get("parse_error") and str(row.get(field) or "").strip()
    }


def _load_direct_variant(direct_run: str, model_key: str) -> dict[str, str]:
    path = (
        ROOT_DIR
        / "data"
        / "laysumm_direct_baseline"
        / "runs"
        / direct_run
        / "01_direct_summaries"
        / model_key
        / "generated_summaries.jsonl"
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing direct baseline summaries: {path}")
    return {
        str(row["article_id"]): str(row.get("generated_summary") or "").strip()
        for row in load_jsonl(path)
        if not row.get("parse_error") and str(row.get("generated_summary") or "").strip()
    }


def build_variants(*, run_name: str, atlas_run: str, direct_run: str, model_key: str) -> None:
    articles = _load_sample(run_name)
    module1_rows = {str(row["article_id"]): row for row in load_jsonl(_module1_path(run_name, model_key))}
    direct = _load_direct_variant(direct_run, model_key)
    module12 = _load_atlas_variant(atlas_run, model_key, "module12_generated")
    module123 = _load_atlas_variant(atlas_run, model_key, "module123_rewritten")
    out_rows: list[dict[str, Any]] = []
    for article in articles:
        aid = str(article["id"])
        rows = {
            "direct": direct.get(aid, ""),
            "module1": str(module1_rows.get(aid, {}).get("generated_summary") or "").strip(),
            "module12_generated": module12.get(aid, ""),
            "module123_rewritten": module123.get(aid, ""),
        }
        missing = [name for name, summary in rows.items() if not summary]
        if missing:
            raise ValueError(f"Missing summaries for {model_key} {aid}: {missing}")
        for variant, summary in rows.items():
            out_rows.append(
                {
                    "article_id": aid,
                    "source_dataset": article.get("source_dataset"),
                    "original_index": article.get("original_index"),
                    "model_key": model_key,
                    "variant": variant,
                    "phase_label": VARIANTS[variant],
                    "prediction": summary,
                    "document": str(article.get("article") or ""),
                    "reference": str(article.get("expert_summary") or ""),
                    "word_count": safe_word_count(summary),
                }
            )
    out_path = _variants_path(run_name, model_key)
    save_jsonl(out_rows, out_path)
    save_json(
        {
            "time": _now(),
            "run_name": run_name,
            "atlas_run": atlas_run,
            "direct_run": direct_run,
            "model_key": model_key,
            "article_count": len(articles),
            "variant_count": len(VARIANTS),
            "rows": len(out_rows),
            "variants": VARIANTS,
        },
        out_path.parent / "summary_variants_metadata.json",
    )
    print(f"[build-variants] {model_key}: rows={len(out_rows)} -> {out_path}")


def legacy_eval_run_name(*, run_name: str, model_key: str, variant: str) -> str:
    readable_run = re.sub(r"[^A-Za-z0-9]+", "_", run_name).strip("_").lower()
    if len(readable_run) > 32:
        readable_run = f"{readable_run[:24]}_{sha256_text(run_name)[:7]}"
    model_abbr = MODEL_ABBR.get(model_key, re.sub(r"[^A-Za-z0-9]+", "", model_key)[:12] or "model")
    variant_abbr = VARIANT_ABBR.get(variant, re.sub(r"[^A-Za-z0-9]+", "", variant)[:12] or "variant")
    return f"eval_{readable_run}_{model_abbr}_{variant_abbr}"


def legacy_eval_run_name_long(*, run_name: str, model_key: str, variant: str) -> str:
    return f"{run_name}__legacy_eval__{model_key}__{variant}"


def export_legacy_eval_run(*, run_name: str, model_key: str, variant: str) -> str:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    articles = _load_sample(run_name)
    variant_rows = [
        row
        for row in load_jsonl(_variants_path(run_name, model_key))
        if str(row.get("variant")) == variant
    ]
    if len(variant_rows) != len(articles):
        raise ValueError(f"Expected {len(articles)} rows for {variant}, got {len(variant_rows)}")
    order = _article_order(articles)
    variant_rows.sort(key=lambda row: order[str(row["article_id"])])

    legacy_run = legacy_eval_run_name(run_name=run_name, model_key=model_key, variant=variant)
    legacy_dir = atlas_run_data_dir(legacy_run)
    save_jsonl(articles, legacy_dir / "00_articles" / "articles.jsonl")
    save_json(
        {
            "time": _now(),
            "source_run_name": run_name,
            "legacy_run_name": legacy_run,
            "model_key": model_key,
            "variant": variant,
            "phase_label": VARIANTS[variant],
            "purpose": "Compatibility wrapper for src.thesis_laysumm.run_evaluate.",
        },
        legacy_dir / "00_articles" / "legacy_eval_metadata.json",
    )
    generated_rows = [
        {
            "article_id": row["article_id"],
            "source_dataset": row["source_dataset"],
            "original_index": row["original_index"],
            "model_key": model_key,
            "generated_summary": row["prediction"],
            "parse_error": False,
        }
        for row in variant_rows
    ]
    save_jsonl(generated_rows, legacy_dir / "04_summaries" / model_key / "generated_summaries.jsonl")
    save_json(
        {
            "time": _now(),
            "legacy_run_name": legacy_run,
            "model_key": model_key,
            "variant": variant,
            "rows": len(generated_rows),
        },
        legacy_dir / "04_summaries" / model_key / "generated_summaries_metadata.json",
    )
    print(f"[export-legacy-eval] {model_key} {variant} -> {legacy_run}")
    return legacy_run


def import_legacy_eval_scores(*, run_name: str, model_key: str, variant: str) -> None:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    legacy_run = legacy_eval_run_name(run_name=run_name, model_key=model_key, variant=variant)
    score_path = atlas_run_results_dir(legacy_run) / model_key / "generated_scores.json"
    if not score_path.exists():
        legacy_run_long = legacy_eval_run_name_long(run_name=run_name, model_key=model_key, variant=variant)
        legacy_score_path_long = atlas_run_results_dir(legacy_run_long) / model_key / "generated_scores.json"
        if legacy_score_path_long.exists():
            legacy_run = legacy_run_long
            score_path = legacy_score_path_long
    if not score_path.exists():
        raise FileNotFoundError(f"Missing legacy generated_scores.json: {score_path}")
    payload = json.loads(score_path.read_text(encoding="utf-8"))
    payload["variant"] = variant
    payload["phase_label"] = VARIANTS[variant]
    payload["legacy_eval_run"] = legacy_run
    out_dir = _metrics_dir(run_name, model_key)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(payload, out_dir / f"{variant}_scores.json")
    (out_dir / f"{variant}_scores.txt").write_text(
        "\n".join(f"{metric}: {payload['overall'][metric]}" for metric in METRICS) + "\n",
        encoding="utf-8",
    )

    available: dict[str, Any] = {}
    for variant_name in VARIANTS:
        path = out_dir / f"{variant_name}_scores.json"
        if path.exists():
            available[variant_name] = json.loads(path.read_text(encoding="utf-8"))
    if set(available) == set(VARIANTS):
        save_json(
            {
                "time": _now(),
                "run_name": run_name,
                "model_key": model_key,
                "article_count": payload.get("article_count"),
                "variants": {k: v["overall"] for k, v in available.items()},
            },
            out_dir / "all_variant_scores.json",
        )
    _write_combined_metric_csv(run_name=run_name)
    print(f"[import-legacy-eval] {model_key} {variant} <- {score_path}")


def _evaluate_variant(rows: list[dict[str, Any]]) -> dict[str, Any]:
    _ensure_local_nltk_data()
    from evaluation.evaluation_final import evaluate_all

    source_scores: dict[str, dict[str, float]] = {}
    for source in ("PLOS", "eLife"):
        source_rows = [row for row in rows if row["source_dataset"] == source]
        if not source_rows:
            raise ValueError(f"No {source} rows for variant")
        preds = [str(row["prediction"]) for row in source_rows]
        refs = [{"document": str(row["document"]), "reference": str(row["reference"])} for row in source_rows]
        docs = [str(row["document"]) for row in source_rows]
        source_scores[source] = evaluate_all(preds, refs, "lay_summ", summac_docs=docs)
    overall = {
        metric: float(np.mean([source_scores["PLOS"][metric], source_scores["eLife"][metric]]))
        for metric in source_scores["PLOS"]
    }
    return {"PLOS": source_scores["PLOS"], "eLife": source_scores["eLife"], "overall": overall}


def evaluate_official_metrics(*, run_name: str, model_key: str) -> None:
    evaluate_official_metrics_variant(run_name=run_name, model_key=model_key, variant=None)


def evaluate_official_metrics_variant(*, run_name: str, model_key: str, variant: str | None) -> None:
    rows = load_jsonl(_variants_path(run_name, model_key))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["variant"])].append(row)
    out_dir = _metrics_dir(run_name, model_key)
    out_dir.mkdir(parents=True, exist_ok=True)
    combined: dict[str, Any] = {}
    variants_to_run = [variant] if variant else list(VARIANTS)
    for variant_name in variants_to_run:
        if variant_name not in VARIANTS:
            raise ValueError(f"Unknown variant: {variant_name}")
        variant_rows = grouped.get(variant_name, [])
        if not variant_rows:
            raise ValueError(f"No rows for variant={variant_name}, model={model_key}")
        result = _evaluate_variant(variant_rows)
        payload = {
            "time": _now(),
            "run_name": run_name,
            "model_key": model_key,
            "variant": variant_name,
            "phase_label": VARIANTS[variant_name],
            "article_count": len(variant_rows),
            **result,
        }
        save_json(payload, out_dir / f"{variant_name}_scores.json")
        (out_dir / f"{variant_name}_scores.txt").write_text(
            "\n".join(f"{metric}: {payload['overall'][metric]}" for metric in METRICS) + "\n",
            encoding="utf-8",
        )
        combined[variant_name] = payload
    available: dict[str, Any] = {}
    for variant_name in VARIANTS:
        path = out_dir / f"{variant_name}_scores.json"
        if path.exists():
            available[variant_name] = json.loads(path.read_text(encoding="utf-8"))
    if set(available) == set(VARIANTS):
        save_json(
            {
                "time": _now(),
                "run_name": run_name,
                "model_key": model_key,
                "article_count": len(grouped["module1"]),
                "variants": {k: v["overall"] for k, v in available.items()},
            },
            out_dir / "all_variant_scores.json",
        )
    _write_combined_metric_csv(run_name=run_name)
    print(f"[evaluate] {model_key}: variants={len(combined)} -> {out_dir}")


def extract_expert_afs(
    *,
    run_name: str,
    judge_model_key: str,
    max_workers: int,
    resume: bool,
) -> None:
    articles = _load_sample(run_name)
    order = _article_order(articles)
    out_path = _expert_af_path(run_name)
    existing = load_jsonl(out_path) if resume and out_path.exists() else []
    done = {str(row.get("article_id")) for row in existing}
    template = (PROMPTS_DIR / "expert_summary_af_extraction.txt").read_text(encoding="utf-8")
    prompt_hash = sha256_text(template)
    usage = _load_usage(run_name, resume=resume)
    client = ProviderClient(judge_model_key, usage)

    def worker(article: dict[str, Any]) -> dict[str, Any]:
        aid = str(article["id"])
        expert_summary = str(article.get("expert_summary") or "").strip()
        if not expert_summary:
            raise RuntimeError(f"Empty expert summary for {aid}")
        prompt = template.replace("{expert_summary}", expert_summary)
        raw = client.chat(
            stage=f"m1_ablation.extract_expert_af:{judge_model_key}",
            item_id=aid,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        obj = _loads_json_object(raw)
        facts: list[dict[str, Any]] = []
        for idx, fact_obj in enumerate(obj.get("atomic_facts") or [], start=1):
            if not isinstance(fact_obj, dict):
                continue
            fact = str(fact_obj.get("fact") or "").strip()
            source_sentence = str(fact_obj.get("source_sentence") or "").strip()
            if not fact:
                continue
            facts.append(
                {
                    "expert_af_id": f"{aid}_expert_af_{idx:04d}",
                    "fact": fact,
                    "source_sentence": source_sentence,
                }
            )
        if not facts:
            raise RuntimeError(f"No expert AFs extracted for {aid}")
        return {
            "article_id": aid,
            "source_dataset": article.get("source_dataset"),
            "original_index": article.get("original_index"),
            "judge_model_key": judge_model_key,
            **_judge_cfg_payload(judge_model_key),
            "prompt_file": str(PROMPTS_DIR / "expert_summary_af_extraction.txt"),
            "prompt_sha256": prompt_hash,
            "expert_summary": expert_summary,
            "expert_af_count": len(facts),
            "expert_afs": facts,
            "raw_response": raw,
            "parse_error": False,
        }

    todo = [article for article in articles if str(article["id"]) not in done]
    try:
        new_rows = run_parallel(todo, worker, max_workers=max_workers, desc=f"Expert AF extraction | {judge_model_key}")
    except ParallelExecutionError as exc:
        checkpoint = sorted(existing + exc.partial_results, key=lambda r: order[str(r["article_id"])])
        save_jsonl(checkpoint, out_path)
        _save_usage(run_name, usage)
        print(f"[extract-expert-af] checkpointed {len(checkpoint)} rows -> {out_path}")
        raise
    combined = sorted(existing + new_rows, key=lambda r: order[str(r["article_id"])])
    save_jsonl(combined, out_path)
    _save_usage(run_name, usage)
    save_json(
        {
            "time": _now(),
            "run_name": run_name,
            "judge_model_key": judge_model_key,
            "article_count": len(combined),
            "expert_af_total": sum(int(row["expert_af_count"]) for row in combined),
            "prompt_sha256": prompt_hash,
            "usage_summary": usage.summarize(),
        },
        out_path.parent / "expert_summary_af_metadata.json",
    )
    print(f"[extract-expert-af] rows={len(combined)} -> {out_path}")


def canonicalize_reference_claims(
    *,
    run_name: str,
    judge_model_key: str,
    max_workers: int,
    resume: bool,
) -> None:
    articles = _load_sample(run_name)
    order = _article_order(articles)
    out_path = _abstract_claim_path(run_name)
    existing = load_jsonl(out_path) if resume and out_path.exists() else []
    done = {str(row.get("article_id")) for row in existing}
    template = (PROMPTS_DIR / "expert_af_to_abstract_claim.txt").read_text(encoding="utf-8")
    prompt_hash = sha256_text(template)
    usage = _load_usage(run_name, resume=resume)
    client = ProviderClient(judge_model_key, usage)

    def worker(article: dict[str, Any]) -> dict[str, Any]:
        aid = str(article["id"])
        abstract = str(article.get("abstract") or "").strip()
        expert_summary = str(article.get("expert_summary") or "").strip()
        if not abstract:
            raise RuntimeError(f"Empty abstract for {aid}")
        if not expert_summary:
            raise RuntimeError(f"Empty expert summary for {aid}")
        prompt = (
            template.replace("{abstract}", abstract)
            .replace("{expert_summary}", expert_summary)
        )
        raw = client.chat(
            stage=f"m1_ablation.select_expert_mentioned_abstract_af:{judge_model_key}",
            item_id=aid,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        obj = _loads_json_object(raw)
        claims: list[dict[str, Any]] = []
        for idx, item in enumerate(obj.get("selected_abstract_facts") or [], start=1):
            if not isinstance(item, dict):
                continue
            claim = str(item.get("abstract_grounded_claim") or "").strip()
            if not claim:
                continue
            abstract_af_id = str(item.get("abstract_af_id") or f"A{idx}")
            claims.append(
                {
                    "expert_af_id": f"{aid}_selected_abstract_af_{idx:04d}",
                    "abstract_af_id": abstract_af_id,
                    "expert_fact": "",
                    "abstract_grounded_claim": claim,
                    "support_level": "selected_by_expert_summary",
                    "abstract_span": str(item.get("abstract_span") or "").strip(),
                    "expert_summary_span": str(item.get("expert_summary_span") or "").strip(),
                }
            )
        if not claims:
            raise RuntimeError(f"No expert-mentioned abstract AFs selected for {aid}")
        return {
            "article_id": aid,
            "source_dataset": article.get("source_dataset"),
            "original_index": article.get("original_index"),
            "judge_model_key": judge_model_key,
            **_judge_cfg_payload(judge_model_key),
            "prompt_file": str(PROMPTS_DIR / "expert_af_to_abstract_claim.txt"),
            "prompt_sha256": prompt_hash,
            "abstract": abstract,
            "expert_summary": expert_summary,
            "all_abstract_fact_count": int(obj.get("all_abstract_fact_count") or 0),
            "claim_count": len(claims),
            "claims": claims,
            "raw_response": raw,
            "parse_error": False,
        }

    todo = [row for row in articles if str(row["id"]) not in done]
    try:
        new_rows = run_parallel(
            todo,
            worker,
            max_workers=max_workers,
            desc=f"Abstract-grounded reference | {judge_model_key}",
        )
    except ParallelExecutionError as exc:
        checkpoint = sorted(existing + exc.partial_results, key=lambda r: order[str(r["article_id"])])
        save_jsonl(checkpoint, out_path)
        _save_usage(run_name, usage)
        print(f"[canonicalize-reference] checkpointed {len(checkpoint)} rows -> {out_path}")
        raise
    combined = sorted(existing + new_rows, key=lambda r: order[str(r["article_id"])])
    save_jsonl(combined, out_path)
    _save_usage(run_name, usage)
    save_json(
        {
            "time": _now(),
            "run_name": run_name,
            "judge_model_key": judge_model_key,
            "article_count": len(combined),
            "claim_total": sum(int(row["claim_count"]) for row in combined),
            "reference_definition": "abstract atomic facts selected when also expressed in the expert lay summary",
            "prompt_sha256": prompt_hash,
            "usage_summary": usage.summarize(),
        },
        out_path.parent / "abstract_grounded_claims_metadata.json",
    )
    print(f"[canonicalize-reference] rows={len(combined)} -> {out_path}")


def judge_recall(
    *,
    run_name: str,
    model_key: str,
    judge_model_key: str,
    max_workers: int,
    resume: bool,
) -> None:
    abstract_claim_rows = load_jsonl(_abstract_claim_path(run_name)) if _abstract_claim_path(run_name).exists() else []
    expert_rows = [] if abstract_claim_rows else load_jsonl(_expert_af_path(run_name))
    variants = load_jsonl(_variants_path(run_name, model_key))
    summary_by_key = {
        (str(row["article_id"]), str(row["variant"])): str(row["prediction"])
        for row in variants
    }
    tasks: list[dict[str, Any]] = []
    if abstract_claim_rows:
        for reference_row in abstract_claim_rows:
            aid = str(reference_row["article_id"])
            for claim in reference_row.get("claims") or []:
                for variant in VARIANTS:
                    key = (aid, variant)
                    if key not in summary_by_key:
                        raise ValueError(f"Missing summary for recall task: {model_key} {aid} {variant}")
                    tasks.append(
                        {
                            "article_id": aid,
                            "source_dataset": reference_row.get("source_dataset"),
                            "original_index": reference_row.get("original_index"),
                            "model_key": model_key,
                            "variant": variant,
                            "phase_label": VARIANTS[variant],
                            "expert_af_id": claim["expert_af_id"],
                            "abstract_af_id": claim.get("abstract_af_id"),
                            "expert_fact": "",
                            "reference_fact": str(claim["abstract_grounded_claim"]),
                            "reference_source": "expert_selected_abstract_af",
                            "abstract_support_level": "selected_by_expert_summary",
                            "abstract_span": str(claim.get("abstract_span") or ""),
                            "expert_summary_span": str(claim.get("expert_summary_span") or ""),
                            "candidate_summary": summary_by_key[key],
                        }
                    )
    else:
        for expert_row in expert_rows:
            aid = str(expert_row["article_id"])
            for af in expert_row.get("expert_afs") or []:
                for variant in VARIANTS:
                    key = (aid, variant)
                    if key not in summary_by_key:
                        raise ValueError(f"Missing summary for recall task: {model_key} {aid} {variant}")
                    tasks.append(
                        {
                            "article_id": aid,
                            "source_dataset": expert_row.get("source_dataset"),
                            "original_index": expert_row.get("original_index"),
                            "model_key": model_key,
                            "variant": variant,
                            "phase_label": VARIANTS[variant],
                            "expert_af_id": af["expert_af_id"],
                            "expert_fact": af["fact"],
                            "reference_fact": af["fact"],
                            "reference_source": "expert_fact",
                            "abstract_support_level": "",
                            "candidate_summary": summary_by_key[key],
                        }
                    )
    out_path = _recall_judgements_path(run_name, model_key)
    existing = load_jsonl(out_path) if resume and out_path.exists() else []
    done = {
        (str(row.get("article_id")), str(row.get("variant")), str(row.get("expert_af_id")))
        for row in existing
        if str(row.get("label") or "").strip()
    }
    template = (PROMPTS_DIR / "expert_af_recall_judge.txt").read_text(encoding="utf-8")
    prompt_hash = sha256_text(template)
    usage = _load_usage(run_name, resume=resume)
    client = ProviderClient(judge_model_key, usage)
    allowed = {"covered", "partially_covered", "not_covered"}

    def worker(task: dict[str, Any]) -> dict[str, Any]:
        prompt = (
            template.replace("{reference_fact}", str(task["reference_fact"]))
            .replace("{candidate_summary}", str(task["candidate_summary"]))
        )
        raw = client.chat(
            stage=f"m1_ablation.judge_expert_af_recall:{judge_model_key}",
            item_id=f"{task['model_key']}:{task['variant']}:{task['expert_af_id']}",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        obj = _parse_recall_judge_response(raw)
        label = str(obj.get("label") or "").strip().lower()
        if label not in allowed:
            raise RuntimeError(f"Invalid recall label {label!r}")
        weight = {"covered": 1.0, "partially_covered": 0.5, "not_covered": 0.0}[label]
        return {
            **{k: v for k, v in task.items() if k != "candidate_summary"},
            "judge_model_key": judge_model_key,
            **_judge_cfg_payload(judge_model_key),
            "prompt_file": str(PROMPTS_DIR / "expert_af_recall_judge.txt"),
            "prompt_sha256": prompt_hash,
            "label": label,
            "covered_strict": label == "covered",
            "covered_lenient": label in {"covered", "partially_covered"},
            "covered_weight": weight,
            "rationale": str(obj.get("rationale") or "").strip(),
            "parse_fallback": bool(obj.get("_parse_fallback")),
            "parse_error": str(obj.get("_parse_error") or ""),
            "raw_response": raw,
        }

    todo = [
        task
        for task in tasks
        if (str(task["article_id"]), str(task["variant"]), str(task["expert_af_id"])) not in done
    ]
    try:
        new_rows = run_parallel(todo, worker, max_workers=max_workers, desc=f"Expert-AF recall judge | {model_key}")
    except ParallelExecutionError as exc:
        save_jsonl(existing + exc.partial_results, out_path)
        _save_usage(run_name, usage)
        print(f"[judge-recall] checkpointed {len(existing) + len(exc.partial_results)} rows -> {out_path}")
        raise
    combined = existing + new_rows
    save_jsonl(combined, out_path)
    _save_usage(run_name, usage)
    _summarize_recall_for_model(run_name=run_name, model_key=model_key)
    save_json(
        {
            "time": _now(),
            "run_name": run_name,
            "model_key": model_key,
            "judge_model_key": judge_model_key,
            "task_count": len(tasks),
            "completed_count": len(combined),
            "prompt_sha256": prompt_hash,
            "usage_summary": usage.summarize(),
        },
        out_path.parent / "recall_judgements_metadata.json",
    )
    print(f"[judge-recall] {model_key}: rows={len(combined)} -> {out_path}")


def _summarize_recall_for_model(*, run_name: str, model_key: str) -> None:
    rows = load_jsonl(_recall_judgements_path(run_name, model_key))
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["variant"]), str(row["article_id"]))].append(row)

    per_article: list[dict[str, Any]] = []
    for (variant, aid), group in sorted(grouped.items()):
        denom = len(group)
        strict = sum(1 for row in group if row.get("covered_strict"))
        lenient = sum(1 for row in group if row.get("covered_lenient"))
        weighted_sum = sum(float(row.get("covered_weight", 1.0 if row.get("covered_strict") else 0.5 if row.get("covered_lenient") else 0.0)) for row in group)
        source = str(group[0].get("source_dataset"))
        per_article.append(
            {
                "model_key": model_key,
                "variant": variant,
                "phase_label": VARIANTS[variant],
                "article_id": aid,
                "source_dataset": source,
                "expert_af_count": denom,
                "covered_strict": strict,
                "covered_lenient": lenient,
                "expert_af_recall_strict": strict / denom if denom else 0.0,
                "expert_af_recall_lenient": lenient / denom if denom else 0.0,
                "expert_af_recall_weighted": weighted_sum / denom if denom else 0.0,
            }
        )

    overall: dict[str, Any] = {}
    for variant in VARIANTS:
        rows_v = [row for row in per_article if row["variant"] == variant]
        if not rows_v:
            continue
        macro_strict = float(np.mean([row["expert_af_recall_strict"] for row in rows_v]))
        macro_lenient = float(np.mean([row["expert_af_recall_lenient"] for row in rows_v]))
        macro_weighted = float(np.mean([row["expert_af_recall_weighted"] for row in rows_v]))
        micro_denom = sum(int(row["expert_af_count"]) for row in rows_v)
        micro_strict = sum(int(row["covered_strict"]) for row in rows_v)
        micro_lenient = sum(int(row["covered_lenient"]) for row in rows_v)
        micro_weighted_num = sum(float(row["expert_af_recall_weighted"]) * int(row["expert_af_count"]) for row in rows_v)
        by_source: dict[str, Any] = {}
        for source in ("PLOS", "eLife"):
            rows_s = [row for row in rows_v if row["source_dataset"] == source]
            if rows_s:
                by_source[source] = {
                    "macro_strict": float(np.mean([row["expert_af_recall_strict"] for row in rows_s])),
                    "macro_lenient": float(np.mean([row["expert_af_recall_lenient"] for row in rows_s])),
                    "macro_weighted": float(np.mean([row["expert_af_recall_weighted"] for row in rows_s])),
                    "article_count": len(rows_s),
                    "expert_af_total": sum(int(row["expert_af_count"]) for row in rows_s),
                }
        overall[variant] = {
            "phase_label": VARIANTS[variant],
            "article_count": len(rows_v),
            "expert_af_total": micro_denom,
            "micro_strict": micro_strict / micro_denom if micro_denom else 0.0,
            "micro_lenient": micro_lenient / micro_denom if micro_denom else 0.0,
            "micro_weighted": micro_weighted_num / micro_denom if micro_denom else 0.0,
            "macro_strict": macro_strict,
            "macro_lenient": macro_lenient,
            "macro_weighted": macro_weighted,
            "by_source": by_source,
        }
    out_dir = _recall_dir(run_name, model_key)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(per_article, out_dir / "expert_af_recall_per_article.jsonl")
    save_json(
        {
            "time": _now(),
            "run_name": run_name,
            "model_key": model_key,
            "primary_metric": "micro_weighted",
            "note": (
                "Strict recall: covered/total. "
                "Weighted recall: (covered*1.0 + partially_covered*0.5)/total [primary]. "
                "Lenient recall: (covered+partially_covered)/total."
            ),
            "variants": overall,
        },
        out_dir / "expert_af_recall_summary.json",
    )


def summarize_all(*, run_name: str, model_keys: list[str]) -> None:
    _write_combined_metric_csv(run_name=run_name)
    _write_recall_all_models_csv(run_name=run_name, model_keys=model_keys)
    rows: list[dict[str, Any]] = []
    for model_key in model_keys:
        recall_path = _recall_dir(run_name, model_key) / "expert_af_recall_summary.json"
        if not recall_path.exists():
            continue
        recall = json.loads(recall_path.read_text(encoding="utf-8"))
        for variant, label in VARIANTS.items():
            variant_score_path = _metrics_dir(run_name, model_key) / f"{variant}_scores.json"
            if not variant_score_path.exists():
                continue
            metric_values = json.loads(variant_score_path.read_text(encoding="utf-8"))["overall"]
            recall_values = recall["variants"][variant]
            rows.append(
                {
                    "model_key": model_key,
                    "Model": MODEL_LABELS.get(model_key, model_key),
                    "variant": variant,
                    "phase": label,
                    **{metric: metric_values[metric] for metric in METRICS},
                    "Expert-AF Recall (weighted)": recall_values.get("micro_weighted", ""),
                    "Expert-AF Recall (strict)": recall_values["micro_strict"],
                    "Expert-AF Recall (lenient)": recall_values["micro_lenient"],
                }
            )
    out_dir = run_results_dir(run_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        with (out_dir / "table3_module_ablation_with_recall.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        save_json({"time": _now(), "run_name": run_name, "rows": rows}, out_dir / "table3_module_ablation_with_recall.json")
        _write_paper_table_html(rows, out_dir / "table3_module_ablation_with_recall.html")
        _write_paper_table_markdown(rows, out_dir / "table3_module_ablation_with_recall.md")
    print(f"[summarize] rows={len(rows)} -> {out_dir}")


def _write_recall_all_models_csv(*, run_name: str, model_keys: list[str]) -> None:
    out_dir = run_results_dir(run_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for model_key in model_keys:
        recall_path = _recall_dir(run_name, model_key) / "expert_af_recall_summary.json"
        if not recall_path.exists():
            continue
        payload = json.loads(recall_path.read_text(encoding="utf-8"))
        for variant, label in VARIANTS.items():
            values = payload.get("variants", {}).get(variant)
            if not values:
                continue
            rows.append(
                {
                    "model_key": model_key,
                    "Model": MODEL_LABELS.get(model_key, model_key),
                    "variant": variant,
                    "phase": label,
                    "expert_af_total": values["expert_af_total"],
                    "Expert-AF Recall (weighted)": values.get("micro_weighted", ""),
                    "Expert-AF Recall (strict)": values["micro_strict"],
                    "Expert-AF Recall (lenient)": values["micro_lenient"],
                    "Expert-AF Recall macro weighted": values.get("macro_weighted", ""),
                    "Expert-AF Recall macro strict": values["macro_strict"],
                    "Expert-AF Recall macro lenient": values["macro_lenient"],
                }
            )
    if not rows:
        return
    rows = _table_rows_for_display(rows)
    with (out_dir / "expert_af_recall_all_models.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_json({"time": _now(), "run_name": run_name, "rows": rows}, out_dir / "expert_af_recall_all_models.json")


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _table_rows_for_display(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    model_order = {model_key: idx for idx, model_key in enumerate(DEFAULT_MODELS)}
    variant_order = {variant: idx for idx, variant in enumerate(VARIANTS)}
    return sorted(
        rows,
        key=lambda row: (
            model_order.get(str(row.get("model_key")), 999),
            variant_order.get(str(row.get("variant")), 999),
        ),
    )


def _write_paper_table_html(rows: list[dict[str, Any]], path: Path) -> None:
    display_rows = _table_rows_for_display(rows)
    title = "Table 3. ATLAS Pipeline Module Ablation with Expert-AF Recall"
    lines = [
        "<!doctype html>",
        "<html>",
        "<head>",
        '<meta charset="utf-8">',
        f"<title>{title}</title>",
        "<style>",
        "body { font-family: Arial, Helvetica, sans-serif; margin: 24px; background: #ffffff; }",
        ".wrap { width: max-content; }",
        "h3 { text-align: center; margin: 0 0 4px; font-size: 16px; font-weight: 700; }",
        "table { border-collapse: collapse; font-size: 13px; }",
        "th, td { border: 1px solid #9fa7ad; padding: 6px 9px; text-align: center; white-space: nowrap; }",
        "th.model, th.phase { background: #bdd7ee; }",
        "th.rel { background: #d9ead3; }",
        "th.read { background: #fce4d6; }",
        "th.fact { background: #d9e2f3; }",
        "th.recall { background: #d9ead3; }",
        "td.model, td.phase { text-align: left; }",
        "tr.final-row td { background: #fff2cc; }",
        "tr.final-row td.phase { font-weight: 700; }",
        "tr.final-row td.fact-cell { background: #ffd966; }",
        "tr.final-row td.recall-cell { background: #d9ead3; font-weight: 700; }",
        ".note { margin-top: 8px; font-size: 12px; color: #333; }",
        "</style>",
        "</head>",
        "<body>",
        '<div class="wrap">',
        f"<h3>{title}</h3>",
        "<table>",
        "<thead>",
        "<tr>",
        '<th class="model">Model</th>',
        '<th class="phase">Phase</th>',
        '<th class="rel" colspan="4">Relevance (up)</th>',
        '<th class="read" colspan="4">Readability</th>',
        '<th class="fact" colspan="2">Factuality (up)</th>',
        '<th class="recall" colspan="1">Coverage (up)</th>',
        "</tr>",
        "<tr>",
        '<th class="model"></th>',
        '<th class="phase"></th>',
        '<th class="rel">ROUGE</th>',
        '<th class="rel">BLEU</th>',
        '<th class="rel">METEOR</th>',
        '<th class="rel">BERTScore</th>',
        '<th class="read">FKGL (down)</th>',
        '<th class="read">DCRS (down)</th>',
        '<th class="read">CLI (down)</th>',
        '<th class="read">LENS (up)</th>',
        '<th class="fact">AlignScore</th>',
        '<th class="fact">SummaC</th>',
        '<th class="recall">Expert-AF Recall (weighted)</th>',
        "</tr>",
        "</thead>",
        "<tbody>",
    ]
    for row in display_rows:
        is_final = str(row.get("variant")) == "module123_rewritten"
        tr_class = ' class="final-row"' if is_final else ""
        phase_text = str(row.get("phase", ""))
        lines.extend(
            [
                f"<tr{tr_class}>",
                f'<td class="model">{row.get("Model", row.get("model_key"))}</td>',
                f'<td class="phase">{phase_text}</td>',
                f"<td>{_fmt(row.get('ROUGE'))}</td>",
                f"<td>{_fmt(row.get('BLEU'))}</td>",
                f"<td>{_fmt(row.get('METEOR'))}</td>",
                f"<td>{_fmt(row.get('BERTScore'))}</td>",
                f"<td>{_fmt(row.get('FKGL'))}</td>",
                f"<td>{_fmt(row.get('DCRS'))}</td>",
                f"<td>{_fmt(row.get('CLI'))}</td>",
                f"<td>{_fmt(row.get('LENS'))}</td>",
                f'<td class="fact-cell">{_fmt(row.get("AlignScore"))}</td>',
                f'<td class="fact-cell">{_fmt(row.get("SummaC"))}</td>',
                f'<td class="recall-cell">{_fmt(row.get("Expert-AF Recall (weighted)"))}</td>',
                "</tr>",
            ]
        )
    lines.extend(
        [
            "</tbody>",
            "</table>",
            '<div class="note">Expert-AF Recall is weighted micro recall: (covered*1.0 + partially_covered*0.5) / total expert-summary atomic facts.</div>',
            "</div>",
            "</body>",
            "</html>",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_paper_table_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    display_rows = _table_rows_for_display(rows)
    headers = [
        "Model",
        "Phase",
        "ROUGE",
        "BLEU",
        "METEOR",
        "BERTScore",
        "FKGL",
        "DCRS",
        "CLI",
        "LENS",
        "AlignScore",
        "SummaC",
        "Expert-AF Recall (weighted)",
    ]
    lines = [
        "# Table 3. ATLAS Pipeline Module Ablation with Expert-AF Recall",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in display_rows:
        values = [
            str(row.get("Model", row.get("model_key"))),
            str(row.get("phase")),
            _fmt(row.get("ROUGE")),
            _fmt(row.get("BLEU")),
            _fmt(row.get("METEOR")),
            _fmt(row.get("BERTScore")),
            _fmt(row.get("FKGL")),
            _fmt(row.get("DCRS")),
            _fmt(row.get("CLI")),
            _fmt(row.get("LENS")),
            _fmt(row.get("AlignScore")),
            _fmt(row.get("SummaC")),
            _fmt(row.get("Expert-AF Recall (weighted)")),
        ]
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    lines.append("Expert-AF Recall is weighted micro recall: (covered*1.0 + partially_covered*0.5) / total expert-summary atomic facts.")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_combined_metric_csv(*, run_name: str) -> None:
    out_dir = run_results_dir(run_name)
    rows: list[dict[str, Any]] = []
    for model_dir in out_dir.glob("*/official_metrics"):
        model_key = model_dir.parent.name
        path = model_dir / "all_variant_scores.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for variant, scores in payload.get("variants", {}).items():
            row = {"model_key": model_key, "variant": variant, "phase": VARIANTS.get(variant, variant)}
            row.update({metric: scores.get(metric) for metric in METRICS})
            rows.append(row)
    if not rows:
        return
    with (out_dir / "official_metrics_all_models.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_inputs(*, run_name: str, atlas_run: str, direct_run: str, model_keys: list[str]) -> None:
    articles = _load_sample(run_name)
    ids = {str(row["id"]) for row in articles}
    problems: list[str] = []
    for model_key in model_keys:
        direct = _load_direct_variant(direct_run, model_key)
        missing_direct = sorted(ids - set(direct))
        if missing_direct:
            problems.append(f"{model_key} missing direct summaries: {missing_direct}")
        cand = _candidate_facts_by_article(atlas_run, model_key)
        missing_cand = sorted(ids - set(cand))
        if missing_cand:
            problems.append(f"{model_key} missing candidate AFs: {missing_cand}")
        for variant in ("module12_generated", "module123_rewritten"):
            summaries = _load_atlas_variant(atlas_run, model_key, variant)
            missing = sorted(ids - set(summaries))
            if missing:
                problems.append(f"{model_key} missing {variant}: {missing}")
    if problems:
        raise ValueError("\n".join(problems))
    print(f"[validate-inputs] ok | articles={len(articles)} models={model_keys}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Module 1-only ablation + Expert-AF Recall for ATLAS pilot n=20.")
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--atlas-run", default=DEFAULT_ATLAS_RUN)
    parser.add_argument("--pilot-run", default=DEFAULT_PILOT_RUN)
    parser.add_argument("--direct-run", default=DEFAULT_DIRECT_RUN)
    parser.add_argument("--model-key")
    parser.add_argument("--model-keys", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--judge-model-key", default=DEFAULT_RECALL_JUDGE)
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare-sample")
    validate_parser = sub.add_parser("validate-inputs")
    validate_parser.add_argument("--model-keys", nargs="+", default=None)
    sub.add_parser("generate-module1")
    sub.add_parser("build-variants")
    export_legacy_parser = sub.add_parser("export-legacy-eval")
    export_legacy_parser.add_argument("--variant", choices=list(VARIANTS), required=True)
    legacy_name_parser = sub.add_parser("legacy-eval-name")
    legacy_name_parser.add_argument("--variant", choices=list(VARIANTS), required=True)
    import_legacy_parser = sub.add_parser("import-legacy-eval")
    import_legacy_parser.add_argument("--variant", choices=list(VARIANTS), required=True)
    evaluate_parser = sub.add_parser("evaluate")
    evaluate_parser.add_argument("--variant", choices=list(VARIANTS), default=None)
    sub.add_parser("extract-expert-af")
    sub.add_parser("canonicalize-reference")
    sub.add_parser("judge-recall")
    summarize_parser = sub.add_parser("summarize")
    summarize_parser.add_argument("--model-keys", nargs="+", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model_key = args.model_key
    resume = not args.no_resume
    if args.command == "prepare-sample":
        prepare_sample(run_name=args.run_name, atlas_run=args.atlas_run, pilot_run=args.pilot_run, force=args.force)
    elif args.command == "validate-inputs":
        validate_inputs(run_name=args.run_name, atlas_run=args.atlas_run, direct_run=args.direct_run, model_keys=args.model_keys)
    elif args.command == "generate-module1":
        if not model_key:
            raise ValueError("--model-key is required for generate-module1")
        generate_module1_summaries(
            run_name=args.run_name,
            atlas_run=args.atlas_run,
            model_key=model_key,
            max_workers=args.max_workers,
            resume=resume,
        )
    elif args.command == "build-variants":
        if not model_key:
            raise ValueError("--model-key is required for build-variants")
        build_variants(run_name=args.run_name, atlas_run=args.atlas_run, direct_run=args.direct_run, model_key=model_key)
    elif args.command == "export-legacy-eval":
        if not model_key:
            raise ValueError("--model-key is required for export-legacy-eval")
        export_legacy_eval_run(run_name=args.run_name, model_key=model_key, variant=args.variant)
    elif args.command == "legacy-eval-name":
        if not model_key:
            raise ValueError("--model-key is required for legacy-eval-name")
        print(legacy_eval_run_name(run_name=args.run_name, model_key=model_key, variant=args.variant))
    elif args.command == "import-legacy-eval":
        if not model_key:
            raise ValueError("--model-key is required for import-legacy-eval")
        import_legacy_eval_scores(run_name=args.run_name, model_key=model_key, variant=args.variant)
    elif args.command == "evaluate":
        if not model_key:
            raise ValueError("--model-key is required for evaluate")
        evaluate_official_metrics_variant(run_name=args.run_name, model_key=model_key, variant=args.variant)
    elif args.command == "extract-expert-af":
        extract_expert_afs(
            run_name=args.run_name,
            judge_model_key=args.judge_model_key,
            max_workers=args.max_workers,
            resume=resume,
        )
    elif args.command == "canonicalize-reference":
        canonicalize_reference_claims(
            run_name=args.run_name,
            judge_model_key=args.judge_model_key,
            max_workers=args.max_workers,
            resume=resume,
        )
    elif args.command == "judge-recall":
        if not model_key:
            raise ValueError("--model-key is required for judge-recall")
        judge_recall(
            run_name=args.run_name,
            model_key=model_key,
            judge_model_key=args.judge_model_key,
            max_workers=args.max_workers,
            resume=resume,
        )
    elif args.command == "summarize":
        summarize_all(run_name=args.run_name, model_keys=args.model_keys)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()

