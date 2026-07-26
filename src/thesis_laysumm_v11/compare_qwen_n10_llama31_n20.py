from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[2]
RESULT_RUNS = ROOT_DIR / "results" / "laysumm_pipeline" / "runs"
OUT_DIR = RESULT_RUNS / "atlas_v11_qwen_n10_llama31_n20_mixed_comparison"
METRICS = ["ROUGE", "BLEU", "METEOR", "BERTScore", "FKGL", "DCRS", "CLI", "LENS", "AlignScore", "SummaC"]
SYSTEMS = [
    {
        "system": "ATLAS Phase B + Qwen2.5-7B-Instruct (OpenRouter, off-the-shelf)",
        "run_name": "atlas_v11_qwen25_7b_openrouter_fixed_gemini25_phaseb_n10_w4",
        "model_key": "qwen25_7b_instruct_openrouter",
        "article_count": 10,
        "PLOS_count": 5,
        "eLife_count": 5,
    },
    {
        "system": "ATLAS Phase B + Meta-Llama-3.1-8B-Instruct (OpenRouter proxy, off-the-shelf)",
        "run_name": "atlas_v11_llama31_8b_openrouter_fixed_gemini25_phaseb_n20_w4",
        "model_key": "llama31_8b_instruct_openrouter",
        "article_count": 20,
        "PLOS_count": 10,
        "eLife_count": 10,
    },
]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compare() -> None:
    rows: list[dict[str, Any]] = []
    payload_systems: list[dict[str, Any]] = []
    for spec in SYSTEMS:
        score_path = RESULT_RUNS / spec["run_name"] / spec["model_key"] / "rewritten_scores.json"
        if not score_path.exists():
            raise FileNotFoundError(score_path)
        scores = _load_json(score_path)
        expected_count = int(spec["article_count"])
        if int(scores.get("article_count", 0)) != expected_count:
            raise ValueError(
                f"Expected {expected_count} articles in {score_path}, "
                f"found {scores.get('article_count')}"
            )
        overall = scores.get("overall") or {}
        missing = [metric for metric in METRICS if metric not in overall]
        if missing:
            raise ValueError(f"Missing metrics in {score_path}: {missing}")
        row = {
            "system": spec["system"],
            "article_count": expected_count,
            "PLOS_count": int(spec["PLOS_count"]),
            "eLife_count": int(spec["eLife_count"]),
            **{metric: overall[metric] for metric in METRICS},
        }
        rows.append(row)
        payload_systems.append(
            {
                **spec,
                "score_path": str(score_path),
                "scores": overall,
            }
        )

    note = (
        "This is a descriptive mixed-sample table, not a controlled head-to-head comparison: "
        "Qwen uses 5 PLOS + 5 eLife articles (n=10), whereas Llama 3.1 uses all 10 PLOS + "
        "10 eLife pilot articles (n=20). Meta-Llama-3.1-8B-Instruct is an availability-driven "
        "proxy rather than the exact Meta-Llama-3-8B-Instruct official baseline backbone."
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = ["system", "article_count", "PLOS_count", "eLife_count", *METRICS]
    with (OUT_DIR / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "comparison_type": "descriptive_mixed_sample_sizes",
        "method_note": note,
        "systems": payload_systems,
    }
    (OUT_DIR / "comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# ATLAS V11 mixed-sample backbone results",
        "",
        f"> {note}",
        "",
        "| System | n | PLOS | eLife | ROUGE | BLEU | METEOR | BERTScore | FKGL | DCRS | CLI | LENS | AlignScore | SummaC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        values = [
            str(row["system"]),
            str(row["article_count"]),
            str(row["PLOS_count"]),
            str(row["eLife_count"]),
            *[f"{float(row[metric]):.3f}" for metric in METRICS],
        ]
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    (OUT_DIR / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[comparison] wrote comparison.csv/json/md -> {OUT_DIR}")


if __name__ == "__main__":
    compare()
