from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, List


REQUIRED_ARTICLE_FIELDS = {"id", "source_dataset", "article", "abstract", "expert_summary"}


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("{}:{} must contain a JSON object".format(path, line_number))
            rows.append(value)
    return rows


def load_articles(path: Path) -> List[dict]:
    rows = load_jsonl(path)
    seen = set()
    for row in rows:
        missing = REQUIRED_ARTICLE_FIELDS.difference(row)
        if missing:
            raise ValueError("Article record is missing fields: {}".format(sorted(missing)))
        article_id = str(row["id"]).strip()
        if not article_id or article_id in seen:
            raise ValueError("Every article must have a unique, non-empty id")
        seen.add(article_id)
        for field in ("article", "abstract", "expert_summary"):
            if not str(row[field]).strip():
                raise ValueError("Article {} has an empty {}".format(article_id, field))
    return rows


def save_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_jsonl(rows: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def mean(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        raise ValueError("Cannot calculate a mean from an empty collection")
    return sum(materialized) / len(materialized)

