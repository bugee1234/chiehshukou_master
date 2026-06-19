from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_NAME = "pilot_n20_v11_factuality_chase"
RUN_ROOT = ROOT / "results" / "laysumm_pipeline" / "runs" / RUN_NAME
DATA_ROOT = ROOT / "data" / "laysumm_pipeline" / "runs" / RUN_NAME
OUT_DIR = ROOT / "results" / "laysumm_pipeline" / "analysis_v11_three_model"
MODELS = [
    "gpt41_mini",
    "gemini3_flash_preview_minimal",
    "gemini25_flash_non_thinking",
]


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    missing = []
    for model in MODELS:
        for path in (
            RUN_ROOT / model / "official_style_rank.json",
            DATA_ROOT / "06_rewritten" / model / "rewritten_summaries_metadata.json",
        ):
            if not path.exists():
                missing.append(path)
    if missing:
        raise FileNotFoundError("Run V11 evaluation first. Missing:\n" + "\n".join(str(p) for p in missing))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for model in MODELS:
        official = load_json(RUN_ROOT / model / "official_style_rank.json")["our_systems"]["rewritten"]
        metadata = load_json(DATA_ROOT / "06_rewritten" / model / "rewritten_summaries_metadata.json")
        raw = official["raw_scores"]
        rows.append(
            {
                "model": model,
                "final_score": official["final_score"],
                "relevance": official["relevance"],
                "readability": official["readability"],
                "factuality": official["factuality"],
                "alignscore": raw["AlignScore"],
                "summac": raw["SummaC"],
                "selected_variant_counts": json.dumps(metadata.get("selected_variant_counts", {}), sort_keys=True),
                "rewrite_invoked_count": metadata.get("rewrite_invoked_count"),
                "skipped_no_errors_count": metadata.get("skipped_no_errors_count"),
                "parse_error_count": metadata.get("parse_error_count"),
                "below_min_length_count": metadata.get("below_min_length_count"),
                "above_max_length_count": metadata.get("above_max_length_count"),
            }
        )
    write_csv(OUT_DIR / "v11_three_model_summary.csv", rows, list(rows[0]))

    finals = [float(row["final_score"]) for row in rows]
    factuality = [float(row["factuality"]) for row in rows]
    stability = {
        "run_name": RUN_NAME,
        "models": MODELS,
        "mean_final_score": round(sum(finals) / len(finals), 4),
        "min_final_score": round(min(finals), 4),
        "max_final_score": round(max(finals), 4),
        "final_score_spread": round(max(finals) - min(finals), 4),
        "mean_factuality": round(sum(factuality) / len(factuality), 4),
        "factuality_spread": round(max(factuality) - min(factuality), 4),
        "objective_hint": "Prefer high mean_final_score with low final_score_spread.",
    }
    (OUT_DIR / "v11_stability_summary.json").write_text(
        json.dumps(stability, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(stability, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
