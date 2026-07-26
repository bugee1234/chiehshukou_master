from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_RUNS = ROOT_DIR / "data" / "laysumm_pipeline" / "runs"
RESULT_RUNS = ROOT_DIR / "results" / "laysumm_pipeline" / "runs"
OUT_DIR = RESULT_RUNS / "atlas_v11_qwen25_llama31_openrouter_n10_w4_comparison"
METRICS = ["ROUGE", "BLEU", "METEOR", "BERTScore", "FKGL", "DCRS", "CLI", "LENS", "AlignScore", "SummaC"]
SYSTEMS = [
    (
        "ATLAS Phase B + Qwen2.5-7B-Instruct (OpenRouter, off-the-shelf)",
        "atlas_v11_qwen25_7b_openrouter_fixed_gemini25_phaseb_n10_w4",
        "qwen25_7b_instruct_openrouter",
    ),
    (
        "ATLAS Phase B + Meta-Llama-3.1-8B-Instruct (OpenRouter proxy, off-the-shelf)",
        "atlas_v11_llama31_8b_openrouter_fixed_gemini25_phaseb_n10_w4",
        "llama31_8b_instruct_openrouter",
    ),
]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _article_ids(run_name: str) -> set[str]:
    path = DATA_RUNS / run_name / "00_articles" / "articles.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    return {
        str(json.loads(line)["id"])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def compare() -> None:
    expected_ids: set[str] | None = None
    rows: list[dict[str, Any]] = []
    payload_rows: list[dict[str, Any]] = []
    for label, run_name, model_key in SYSTEMS:
        article_ids = _article_ids(run_name)
        if expected_ids is None:
            expected_ids = article_ids
        elif article_ids != expected_ids:
            raise ValueError(f"The article IDs differ for run {run_name}")
        if len(article_ids) != 10:
            raise ValueError(f"Expected 10 articles for {run_name}, found {len(article_ids)}")

        score_path = RESULT_RUNS / run_name / model_key / "rewritten_scores.json"
        if not score_path.exists():
            raise FileNotFoundError(score_path)
        scores = _load_json(score_path)
        if int(scores.get("article_count", 0)) != 10:
            raise ValueError(f"Expected 10 evaluated articles in {score_path}")
        overall = scores.get("overall") or {}
        missing = [metric for metric in METRICS if metric not in overall]
        if missing:
            raise ValueError(f"Missing metrics in {score_path}: {missing}")
        rows.append({"system": label, **{metric: overall[metric] for metric in METRICS}})
        payload_rows.append(
            {"system": label, "run_name": run_name, "model_key": model_key, "scores": overall}
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["system", *METRICS])
        writer.writeheader()
        writer.writerows(rows)

    method_note = (
        "Both systems use off-the-shelf Instruct checkpoints through OpenRouter without BioLaySumm-specific "
        "fine-tuning. Meta-Llama-3.1-8B-Instruct is an availability-driven proxy because the original "
        "Meta-Llama-3-8B-Instruct endpoint was unavailable at run time; it is not the exact official baseline "
        "backbone. The systems use the same 5 PLOS + 5 eLife articles and the same precomputed Gemini 2.5 "
        "evidence tables and verification questions. Only each backbone's Phase-B generation, question "
        "answering, and rewriting are model-specific. Gemini outputs are not evaluated in this run."
    )
    payload = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "article_count": len(expected_ids or set()),
        "subset": "BioLaySumm 2025 validation: 5 PLOS + 5 eLife, identical article IDs",
        "max_workers": 4,
        "method_note": method_note,
        "systems": payload_rows,
    }
    (OUT_DIR / "comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# ATLAS V11 OpenRouter Qwen 2.5 / Llama 3.1 proxy comparison (fixed upstream, n=10)",
        "",
        f"> {method_note}",
        "",
        "| System | ROUGE | BLEU | METEOR | BERTScore | FKGL | DCRS | CLI | LENS | AlignScore | SummaC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| " + str(row["system"]) + " | " + " | ".join(f"{float(row[m]):.3f}" for m in METRICS) + " |"
        )
    lines.append("")
    (OUT_DIR / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[comparison] wrote comparison.csv/json/md -> {OUT_DIR}")


if __name__ == "__main__":
    compare()
