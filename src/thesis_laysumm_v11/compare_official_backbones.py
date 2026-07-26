from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_RUNS = ROOT_DIR / "data" / "laysumm_pipeline" / "runs"
RESULT_RUNS = ROOT_DIR / "results" / "laysumm_pipeline" / "runs"
OUT_DIR = RESULT_RUNS / "atlas_v11_official_backbones_fixed_upstream_n10_comparison"
METRICS = ["ROUGE", "BLEU", "METEOR", "BERTScore", "FKGL", "DCRS", "CLI", "LENS", "AlignScore", "SummaC"]
SYSTEMS = [
    (
        "ATLAS Phase B + Qwen2.5-7B-Instruct (off-the-shelf, 8-bit)",
        "atlas_v11_qwen25_7b_fixed_gemini25_phaseb_n10_compact",
        "qwen25_7b_instruct_local_8bit",
    ),
    (
        "ATLAS Phase B + Meta-Llama-3-8B-Instruct (off-the-shelf, 8-bit)",
        "atlas_v11_llama3_8b_fixed_gemini25_phaseb_n10_compact",
        "llama3_8b_instruct_local_8bit",
    ),
    (
        "ATLAS Phase B + Gemini 2.5 Flash (thesis configuration)",
        "atlas_v11_gemini25_fixed_upstream_phaseb_n10_compact",
        "gemini25_flash_non_thinking",
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
            raise ValueError(f"The 20 article IDs differ for run {run_name}")
        score_path = RESULT_RUNS / run_name / model_key / "rewritten_scores.json"
        if not score_path.exists():
            raise FileNotFoundError(score_path)
        scores = _load_json(score_path)
        if int(scores.get("article_count", 0)) != 10:
            raise ValueError(f"Expected 10 evaluated articles in {score_path}")
        overall = scores["overall"]
        rows.append({"system": label, **{metric: overall[metric] for metric in METRICS}})
        payload_rows.append(
            {"system": label, "run_name": run_name, "model_key": model_key, "scores": overall}
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["system", *METRICS])
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "article_count": len(expected_ids or set()),
        "subset": "BioLaySumm 2025 validation: 5 PLOS + 5 eLife, identical article IDs for all systems",
        "method_note": (
            "Qwen and Llama use the original Instruct checkpoints without BioLaySumm task-specific SFT; "
            "all systems share Gemini 2.5 evidence tables and verification questions, while each model "
            "independently performs summary generation, question answering, and summary rewriting."
        ),
        "systems": payload_rows,
    }
    (OUT_DIR / "comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# ATLAS V11 Phase-B backbone comparison (fixed upstream, n=10)",
        "",
        f"> {payload['method_note']}",
        "",
        "| System | ROUGE | BLEU | METEOR | BERTScore | FKGL | DCRS | CLI | LENS | AlignScore | SummaC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append("| " + str(row["system"]) + " | " + " | ".join(f"{float(row[m]):.3f}" for m in METRICS) + " |")
    lines.append("")
    (OUT_DIR / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[comparison] wrote comparison.csv/json/md -> {OUT_DIR}")


if __name__ == "__main__":
    compare()
