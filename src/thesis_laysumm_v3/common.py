from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.thesis_laysumm.biolaysumm_data import word_count

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
VALID_MODES = {"balanced", "factuality_chase"}


def require_mode(mode: str) -> str:
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown v3 mode {mode!r}; expected one of {sorted(VALID_MODES)}")
    return mode


def loads_json_object(raw: str) -> dict[str, Any]:
    text = str(raw or "").strip()
    if not text:
        raise ValueError("empty response")
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        start = text.find("{")
        if start < 0:
            raise
        obj, _ = decoder.raw_decode(text[start:])
    if not isinstance(obj, dict):
        raise ValueError("response JSON must be an object")
    return obj


def compact_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def format_numbered_sentences(sentences: list[str]) -> str:
    lines = []
    for idx, sentence in enumerate(sentences, start=1):
        sentence = str(sentence or "").strip()
        if sentence:
            lines.append(f"[S{idx}] {sentence}")
    return "\n".join(lines) if lines else "[S1] (no abstract sentences parsed)"


def split_sentences(text: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", str(text or "").strip())
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", cleaned)
    out = [p.strip() for p in parts if len(p.strip()) > 10]
    return out if out else [cleaned]


def length_policy(*, mode: str, expert_word_count: int) -> dict[str, int]:
    expert = max(int(expert_word_count or 0), 1)
    if mode == "balanced":
        return {
            "target_word_count": expert,
            "min_word_count": max(150, int(round(expert * 0.85))),
            "max_word_count": max(220, int(round(expert * 1.15))),
        }
    return {
        "target_word_count": min(max(170, int(round(expert * 0.80))), 230),
        "min_word_count": min(max(130, int(round(expert * 0.60))), 180),
        "max_word_count": min(max(180, int(round(expert * 0.95))), 260),
    }


def safe_word_count(text: str) -> int:
    return word_count(text)

