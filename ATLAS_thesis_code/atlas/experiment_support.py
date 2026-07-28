from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Iterable, List, Optional, Tuple


def sentence_spans(text: str) -> List[Tuple[int, int, str]]:
    spans: List[Tuple[int, int, str]] = []
    for match in re.finditer(r"[^.!?]+(?:[.!?]+|$)", text, flags=re.DOTALL):
        sentence = match.group(0).strip()
        if sentence:
            spans.append((match.start(), match.end(), sentence))
    return spans


def locate_supporting_sentence(summary: str, source_span: str, atomic_fact: str) -> Optional[Tuple[int, int, str]]:
    candidates = sentence_spans(summary)
    if not candidates:
        return None
    normalized_source = " ".join(source_span.lower().split())
    normalized_fact = " ".join(atomic_fact.lower().split())
    for candidate in candidates:
        normalized_sentence = " ".join(candidate[2].lower().split())
        if normalized_source and (
            normalized_source in normalized_sentence or normalized_sentence in normalized_source
        ):
            return candidate
    scored = []
    for candidate in candidates:
        normalized_sentence = " ".join(candidate[2].lower().split())
        score = max(
            SequenceMatcher(None, normalized_sentence, normalized_source).ratio(),
            SequenceMatcher(None, normalized_sentence, normalized_fact).ratio(),
        )
        scored.append((score, candidate))
    best_score, best = max(scored, key=lambda item: item[0])
    return best if best_score >= 0.45 else None


def replace_supporting_sentence(summary: str, source_span: str, atomic_fact: str, replacement: str) -> str:
    located = locate_supporting_sentence(summary, source_span, atomic_fact)
    if located is None:
        raise ValueError("Could not localize the supporting sentence in the complete summary")
    start, end, _ = located
    revised = (summary[:start] + replacement.strip() + " " + summary[end:]).strip()
    return re.sub(r"\s+", " ", revised)


def false_option_letter(question: dict) -> Tuple[str, str]:
    correct = question["correct_letter"]
    for letter in "ABCD":
        if letter != correct:
            return letter, question["options"][letter]
    raise AssertionError("A 1T3F question must contain a false option")


def proportion(flags: Iterable[bool]) -> float:
    values = list(flags)
    if not values:
        raise ValueError("Cannot calculate a proportion without evaluated cases")
    return sum(1 for flag in values if flag) / len(values)
