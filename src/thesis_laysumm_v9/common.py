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
        raise ValueError(f"Unknown V9 mode {mode!r}; expected one of {sorted(VALID_MODES)}")
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


def length_policy(*, mode: str, expert_word_count: int, source_dataset: str = "") -> dict[str, Any]:
    expert = max(int(expert_word_count or 0), 1)
    if mode == "balanced":
        return {
            "target_word_count": expert,
            "min_word_count": max(150, int(round(expert * 0.85))),
            "max_word_count": max(220, int(round(expert * 1.15))),
        }
    source = str(source_dataset or "").strip().lower()
    if source == "plos":
        target, minimum, maximum = 205, 175, 245
        style_profile = "compact_factual"
    elif source == "elife":
        target, minimum, maximum = 245, 215, 295
        style_profile = "mechanism_short_sentence"
    else:
        target, minimum, maximum = 235, 185, 290
        style_profile = "compact_factual"
    return {
        "target_word_count": target,
        "min_word_count": minimum,
        "max_word_count": maximum,
        "style_profile": style_profile,
        "length_basis": "dataset_metric_policy_V9",
        "expert_summary_word_count": expert,
    }


def readability_policy(*, source_dataset: str, expert_word_count: int) -> dict[str, Any]:
    source = str(source_dataset or "").strip().lower()
    if source == "plos":
        return {
            "target_sentence_count": "11-14",
            "target_words_per_sentence": "9-14",
            "max_words_per_sentence": 16,
            "target_slots": "6-8",
            "max_clarification_sentences": 3,
            "technical_term_policy": "keep only essential disease, intervention, organism, method, and main result terms; use short aliases after first mention",
        }
    if source == "elife":
        return {
            "target_sentence_count": "14-18",
            "target_words_per_sentence": "9-15",
            "max_words_per_sentence": 16,
            "target_slots": "8-10",
            "max_clarification_sentences": 6,
            "technical_term_policy": "keep central mechanism terms only when they anchor a checked fact; use short aliases after first mention",
        }
        return {
            "target_sentence_count": "10-13",
            "target_words_per_sentence": "9-14",
            "max_words_per_sentence": 16,
            "target_slots": "5-7",
        "max_clarification_sentences": 2,
        "technical_term_policy": "keep only terms needed for factual correctness",
    }


def readability_stats(text: str) -> dict[str, Any]:
    sentences = split_sentences(text)
    words = re.findall(r"\b[\w'-]+\b", str(text or ""))
    word_count_value = len(words)
    sentence_count = max(len(sentences), 1)
    clean_words = [re.sub(r"[^A-Za-z]", "", w) for w in words]
    long_words = [w for w in clean_words if len(w) >= 10]
    hard_words = [w for w in clean_words if _rough_syllable_count(w) >= 3]
    long_sentences = [s for s in sentences if safe_word_count(s) > 16]
    very_long_sentences = [s for s in sentences if safe_word_count(s) > 20]
    high_risk_sentences = [s for s in sentences if sentence_risk_score(s) >= 4]
    return {
        "word_count": word_count_value,
        "sentence_count": len(sentences),
        "avg_words_per_sentence": round(word_count_value / sentence_count, 2),
        "long_word_count": len(long_words),
        "long_word_ratio": round(len(long_words) / max(word_count_value, 1), 4),
        "hard_word_count": len(hard_words),
        "hard_word_ratio": round(len(hard_words) / max(word_count_value, 1), 4),
        "long_sentence_count": len(long_sentences),
        "very_long_sentence_count": len(very_long_sentences),
        "high_risk_sentence_count": len(high_risk_sentences),
        "max_sentence_words": max([safe_word_count(s) for s in sentences] or [0]),
    }


def readability_proxy_score(text: str) -> float:
    stats = readability_stats(text)
    return (
        float(stats["avg_words_per_sentence"])
        + float(stats["long_word_ratio"]) * 65.0
        + float(stats["hard_word_ratio"]) * 25.0
        + float(stats["long_sentence_count"]) * 0.75
        + float(stats["very_long_sentence_count"]) * 1.75
        + float(stats["high_risk_sentence_count"]) * 0.9
    )


def _rough_syllable_count(word: str) -> int:
    clean = re.sub(r"[^A-Za-z]", "", str(word or "")).lower()
    if not clean:
        return 0
    clean = re.sub(r"e$", "", clean)
    groups = re.findall(r"[aeiouy]+", clean)
    return max(1, len(groups))


def sentence_risk_score(sentence: str) -> int:
    text = str(sentence or "").strip()
    lower = text.lower()
    words = re.findall(r"\b[\w'-]+\b", text)
    clean_words = [re.sub(r"[^A-Za-z]", "", w) for w in words]
    long_ratio = sum(1 for w in clean_words if len(w) >= 10) / max(len(words), 1)
    hard_ratio = sum(1 for w in clean_words if _rough_syllable_count(w) >= 3) / max(len(words), 1)
    score = 0
    wc = safe_word_count(text)
    if wc > 16:
        score += 1
    if wc > 20:
        score += 1
    if long_ratio >= 0.18:
        score += 2
    if hard_ratio >= 0.30:
        score += 2
    if lower.startswith(("this ", "these ", "it ", "they ", "this change", "this finding", "this means")):
        score += 2
    if any(x in lower for x in ("means that", "shows that", "confirms", "ensures", "causes", "allows", "makes", "drives", "explains")):
        score += 1
    if any(x in lower for x in ("may ", "might ", "could ", "suggest", "potential", "likely")):
        score += 1
    if any(x in lower for x in ("not ", " no ", "without", "only", "but not", "however")):
        score += 1
    return score



def find_mojibake(text: str) -> list[str]:
    value = str(text or "")
    hits: list[str] = []
    if "\ufffd" in value:
        hits.append("replacement_character")
    if re.search(r"[\uE000-\uF8FF]", value):
        hits.append("private_use_character")
    if re.search(r"[\u4E00-\u9FFF\u3040-\u30FF\uAC00-\uD7AF]", value):
        hits.append("cjk_character")
    if re.search(r"[ĄąĆćĘęŁłŃńŚśŹźŻż]", value):
        hits.append("latin_extended_mojibake_like_character")
    if re.search(r"[\u02C7\u02D8-\u02DD]", value):
        hits.append("modifier_mark_mojibake_like_character")
    for match in re.finditer(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", value):
        hits.append(f"control_char_U+{ord(match.group(0)):04X}")
    suspicious_sequences = re.findall(r"(?:[?]{2,}|[A-Za-z][?][A-Za-z]|[?][\u0080-\uFFFF])", value)
    hits.extend(seq for seq in suspicious_sequences if seq not in hits)
    return hits[:20]


def safe_word_count(text: str) -> int:
    return word_count(text)


def dataset_profile(source_dataset: str) -> str:
    source = str(source_dataset or "").strip().lower()
    if source == "elife":
        return "elife_mechanism"
    if source == "plos":
        return "plos_concise"
    return "generic_laysumm"


BANNED_FREE_IMPLICATION_PHRASES = [
    "acted like",
    "call for help",
    "could help",
    "could facilitate",
    "important for",
    "may be able",
    "suggest a common mechanism",
    "suggests a common mechanism",
    "this means",
    "this shows",
    "acted as",
    "works like",
]

