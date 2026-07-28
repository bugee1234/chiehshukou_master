from __future__ import annotations

from typing import Any, Dict


CANDIDATE_FACTS: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "candidate_atomic_facts": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "atomic_fact": {"type": "string"},
                    "source_span": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["atomic_fact", "source_span", "reason"],
            },
        }
    },
    "required": ["candidate_atomic_facts"],
}

REFERENCE_ATOMIC_FACTS: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "reference_atomic_facts": {
            "type": "array",
            "items": {"type": "string"},
        }
    },
    "required": ["reference_atomic_facts"],
}

FILTER_DECISION: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "keep": {"type": "boolean"},
        "abstract_sentence": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "reason": {"type": "string"},
    },
    "required": ["keep", "abstract_sentence", "reason"],
}

SUMMARY: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
}

QUESTION: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "question": {"type": "string"},
        "true_option": {"type": "string"},
        "false_options": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "text": {"type": "string"},
                    "strategy": {"type": "string"},
                },
                "required": ["text", "strategy"],
            },
        },
        "none_of_the_above": {"type": "string"},
    },
    "required": ["question", "true_option", "false_options", "none_of_the_above"],
}

ANSWER_AND_REWRITE: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "selected_answer": {"type": "string", "enum": ["A", "B", "C", "D", "E"]},
        "correctly_preserved": {"type": "boolean"},
        "reason": {"type": "string"},
        "revised_summary": {"anyOf": [{"type": "string"}, {"type": "null"}]},
    },
    "required": ["selected_answer", "correctly_preserved", "reason", "revised_summary"],
}

ANSWER_WITH_REASONING: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "answer": {"type": "string", "enum": ["A", "B", "C", "D", "E"]},
        "reasoning": {"type": "string"},
    },
    "required": ["answer", "reasoning"],
}

ANSWER_ONLY: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "answer": {"type": "string", "enum": ["A", "B", "C", "D", "E"]},
    },
    "required": ["answer"],
}

DIRECT_SUPPORT_DECISION: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"supported": {"type": "string", "enum": ["yes", "no"]}},
    "required": ["supported"],
}

COVERAGE_DECISION: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"covered": {"type": "string", "enum": ["yes", "no"]}},
    "required": ["covered"],
}

LLM_JUDGEMENT: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        dimension: {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
                "reason": {"type": "string"},
            },
            "required": ["score", "reason"],
        }
        for dimension in ("relevance", "readability", "factuality")
    },
    "required": ["relevance", "readability", "factuality"],
}
