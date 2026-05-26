from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _extract_row(run_id: str, run_dir: Path) -> dict[str, Any] | None:
    detail_candidates = sorted(run_dir.glob("metrics_detailed_k*.json"))
    if not detail_candidates:
        return None
    detail_path = detail_candidates[-1]
    data = _read_json(detail_path)
    setups = data.get("setups", {})
    a = setups.get("A", {}).get("overall", {})
    c = setups.get("C", {}).get("overall", {})
    c_by_type = setups.get("C", {}).get("by_type", {})
    return {
        "run_id": run_id,
        "A_f1": a.get("f1"),
        "A_recall": a.get("recall"),
        "A_fp": a.get("fp"),
        "A_fn": a.get("fn"),
        "C_f1": c.get("f1"),
        "C_recall": c.get("recall"),
        "C_fp": c.get("fp"),
        "C_fn": c.get("fn"),
        "delta_f1_C_minus_A": (
            (c.get("f1") - a.get("f1"))
            if isinstance(c.get("f1"), (int, float)) and isinstance(a.get("f1"), (int, float))
            else None
        ),
        "C_Fdel_recall": c_by_type.get("F_del", {}).get("recall"),
        "C_Ferror_recall": c_by_type.get("F_error", {}).get("recall"),
        "C_Fkept_fp": c_by_type.get("F_kept", {}).get("fp"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Exp1-1A ablation run metrics")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("data/experiment_1/1a/ablation_runs"),
        help="Directory containing run subfolders",
    )
    parser.add_argument(
        "--run-ids",
        type=str,
        default="k3_off,k3_lenient,k3_strict,k5_off,k5_lenient,k5_strict",
        help="Comma-separated run folder names to summarize",
    )
    args = parser.parse_args()

    run_ids = [x.strip() for x in args.run_ids.split(",") if x.strip()]
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for run_id in run_ids:
        row = _extract_row(run_id, args.base_dir / run_id)
        if row is None:
            missing.append(run_id)
        else:
            rows.append(row)

    if missing:
        print("Missing runs:", ", ".join(missing))

    if not rows:
        print("No valid runs found.")
        return

    headers = [
        "run_id",
        "A_f1",
        "A_recall",
        "A_fp",
        "A_fn",
        "C_f1",
        "C_recall",
        "C_fp",
        "C_fn",
        "delta_f1_C_minus_A",
        "C_Fdel_recall",
        "C_Ferror_recall",
        "C_Fkept_fp",
    ]
    print("\t".join(headers))
    for row in rows:
        print("\t".join(_fmt(row.get(h)) for h in headers))


if __name__ == "__main__":
    main()
