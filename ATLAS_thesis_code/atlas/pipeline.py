from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .clients import StructuredLLM
from .io import save_jsonl
from .schemas import (
    ANSWER_AND_REWRITE,
    ANSWER_ONLY,
    ANSWER_WITH_REASONING,
    CANDIDATE_FACTS,
    COVERAGE_DECISION,
    DIRECT_SUPPORT_DECISION,
    FILTER_DECISION,
    LLM_JUDGEMENT,
    QUESTION,
    REFERENCE_ATOMIC_FACTS,
    SUMMARY,
)


PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"
OPERATION_PROMPT_DIR = PROMPT_DIR / "experiment_operations"


def load_prompt(filename: str) -> str:
    """Load one prompt transcribed from the thesis appendix."""
    path = PROMPT_DIR / filename
    if not path.is_file():
        raise FileNotFoundError("Missing appendix prompt: {}".format(path))
    return path.read_text(encoding="utf-8").strip()


def load_operation_prompt(filename: str) -> str:
    """Load an operational prompt required by an experiment described in the thesis body."""
    path = OPERATION_PROMPT_DIR / filename
    if not path.is_file():
        raise FileNotFoundError("Missing experiment-operation prompt: {}".format(path))
    return path.read_text(encoding="utf-8").strip()


def _render_template(prompt: str, filename: str, replacements: Mapping[str, Any]) -> str:
    for field, value in replacements.items():
        placeholder = "{" + field + "}"
        if placeholder not in prompt:
            raise ValueError("Prompt {} has no placeholder {}".format(filename, placeholder))
        prompt = prompt.replace(placeholder, str(value))
    return prompt


def render_prompt(filename: str, replacements: Mapping[str, Any]) -> str:
    """Substitute only appendix placeholders without interpreting JSON braces."""
    return _render_template(load_prompt(filename), filename, replacements)


def render_operation_prompt(filename: str, replacements: Mapping[str, Any]) -> str:
    """Substitute placeholders in a thesis experiment-operation prompt."""
    return _render_template(load_operation_prompt(filename), filename, replacements)


def _clean_text(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError("Model returned an empty {}".format(field))
    return text


def _normalized(text: str) -> str:
    return " ".join(text.lower().split())


def _option_inputs(question: dict) -> Dict[str, str]:
    return {"option_" + letter: str(question["options"][letter]) for letter in "ABCD"}


class AtlasPipeline:
    """ATLAS Modules 1, 2, and 3 in the fixed order defined in the thesis."""

    def __init__(self, model_key: str, client: Optional[StructuredLLM] = None):
        self.model_key = model_key
        self.client = client or StructuredLLM(model_key)

    def extract_candidate_atomic_facts(self, article_id: str, article_text: str) -> List[dict]:
        response = self.client.generate(
            load_prompt("module_1_candidate_atomic_fact_extraction.txt"),
            {"full_article_text": article_text},
            CANDIDATE_FACTS,
        )
        raw_facts = response.get("candidate_atomic_facts")
        if not isinstance(raw_facts, list):
            raise ValueError("Module 1 did not return candidate_atomic_facts")
        facts: List[dict] = []
        seen = set()
        for item in raw_facts:
            if not isinstance(item, dict):
                raise ValueError("Every Module 1 item must be an object")
            fact = _clean_text(item.get("atomic_fact"), "atomic fact")
            key = _normalized(fact)
            if key in seen:
                continue
            seen.add(key)
            facts.append(
                {
                    "article_id": article_id,
                    "fact_id": "{}_af_{:04d}".format(article_id, len(facts) + 1),
                    "atomic_fact": fact,
                    "source_span": _clean_text(item.get("source_span"), "source span"),
                    "selection_reason": _clean_text(item.get("reason"), "selection reason"),
                }
            )
        if not facts:
            raise ValueError("Module 1 returned no candidate atomic facts for {}".format(article_id))
        return facts

    def filter_against_abstract(self, abstract: str, candidate_facts: Iterable[dict]) -> List[dict]:
        final_facts: List[dict] = []
        for candidate in candidate_facts:
            response = self.client.generate(
                load_prompt("module_2_abstract_based_atomic_fact_filtering.txt"),
                {
                    "candidate_atomic_fact": candidate["atomic_fact"],
                    "article_abstract": abstract,
                },
                FILTER_DECISION,
            )
            if not isinstance(response.get("keep"), bool):
                raise ValueError("Module 2 keep decision must be boolean")
            if response["keep"]:
                support = _clean_text(response.get("abstract_sentence"), "supporting abstract sentence")
                kept = dict(candidate)
                kept["abstract_support"] = support
                kept["filter_reason"] = _clean_text(response.get("reason"), "filter reason")
                final_facts.append(kept)
        if not final_facts:
            raise ValueError("Module 2 discarded every candidate atomic fact")
        return final_facts

    def generate_fact_driven_summary(self, abstract: str, facts: Iterable[dict]) -> str:
        fact_texts = [_clean_text(row.get("atomic_fact"), "atomic fact") for row in facts]
        if not fact_texts:
            raise ValueError("Fact-driven summary generation requires atomic facts")
        response = self.client.generate(
            load_prompt("fact_driven_summary_generation.txt"),
            {"article_abstract": abstract, "final_atomic_facts": fact_texts},
            SUMMARY,
        )
        return _clean_text(response.get("summary"), "fact-driven summary")

    def generate_module_1_only_summary(
        self, article_text: str, candidate_facts: Iterable[dict]
    ) -> str:
        fact_texts = [_clean_text(row.get("atomic_fact"), "candidate atomic fact") for row in candidate_facts]
        if not fact_texts:
            raise ValueError("Module 1-only summary generation requires candidate atomic facts")
        instruction = render_operation_prompt(
            "module_1_only_summary_generation.txt",
            {
                "article": article_text,
                "candidate_facts": "\n".join("- " + fact for fact in fact_texts),
            },
        )
        response = self.client.generate(instruction, {}, SUMMARY)
        return _clean_text(response.get("summary"), "Module 1-only summary")

    def generate_question(self, fact: dict) -> dict:
        response = self.client.generate(
            load_prompt("module_3_question_generation.txt"),
            {"final_atomic_fact": fact["atomic_fact"]},
            QUESTION,
        )
        false_options = response.get("false_options")
        if not isinstance(false_options, list) or len(false_options) != 3:
            raise ValueError("Module 3 must return exactly three false options")
        _clean_text(response.get("true_option"), "true option")
        true_text = _clean_text(fact.get("atomic_fact"), "final atomic fact")
        false_rows: List[dict] = []
        seen = {_normalized(true_text)}
        for item in false_options:
            text = _clean_text(item.get("text") if isinstance(item, dict) else None, "false option")
            key = _normalized(text)
            if key in seen:
                raise ValueError("Question options must be distinct")
            seen.add(key)
            false_rows.append(
                {"text": text, "strategy": _clean_text(item.get("strategy"), "distractor strategy")}
            )
        statements = [(True, true_text)] + [(False, row["text"]) for row in false_rows]
        seed = int(hashlib.md5(str(fact["fact_id"]).encode("utf-8")).hexdigest()[:8], 16)
        random.Random(seed).shuffle(statements)
        letters = "ABCD"
        options = {letters[index]: statement for index, (_, statement) in enumerate(statements)}
        options["E"] = "None of the above"
        correct_letter = next(letters[index] for index, (is_true, _) in enumerate(statements) if is_true)
        return {
            "article_id": fact["article_id"],
            "fact_id": fact["fact_id"],
            "question": _clean_text(response.get("question"), "question"),
            "options": options,
            "correct_letter": correct_letter,
            "correct_atomic_fact": fact["atomic_fact"],
            "source_evidence": fact["source_span"],
            "false_options": false_rows,
        }

    def answer_and_rewrite(self, summary: str, question: dict) -> dict:
        response = self.client.generate(
            load_prompt("module_3_answering_and_targeted_rewrite.txt"),
            {
                "generated_lay_summary": summary,
                "verification_question": {
                    "question": question["question"],
                    "options": question["options"],
                },
                "correct_atomic_fact": question["correct_atomic_fact"],
                "source_evidence": question["source_evidence"],
            },
            ANSWER_AND_REWRITE,
        )
        answer = str(response.get("selected_answer", "")).strip().upper()
        if answer not in set("ABCDE"):
            raise ValueError("Module 3 selected_answer must be A, B, C, D, or E")
        preserved = answer == question["correct_letter"]
        if response.get("correctly_preserved") is not preserved:
            raise ValueError("Module 3 preservation decision conflicts with its selected answer")
        revised: Optional[str] = None
        if not preserved:
            revised = _clean_text(response.get("revised_summary"), "minimally revised summary")
        return {
            "article_id": question["article_id"],
            "fact_id": question["fact_id"],
            "selected_answer": answer,
            "correct_letter": question["correct_letter"],
            "correctly_preserved": preserved,
            "reason": _clean_text(response.get("reason"), "answer reason"),
            "summary_before": summary,
            "summary_after": summary if preserved else revised,
        }

    def extract_reference_atomic_facts(self, article_id: str, expert_summary: str) -> List[dict]:
        instruction = render_prompt(
            "pre_experiment_3_expert_summary_atomic_fact_extraction.txt",
            {"expert_summary": expert_summary},
        )
        response = self.client.generate(instruction, {}, REFERENCE_ATOMIC_FACTS)
        raw_facts = response.get("reference_atomic_facts")
        if not isinstance(raw_facts, list):
            raise ValueError("Reference extraction did not return reference_atomic_facts")
        facts: List[dict] = []
        seen = set()
        for raw_fact in raw_facts:
            fact = _clean_text(raw_fact, "reference atomic fact")
            key = _normalized(fact)
            if key in seen:
                continue
            seen.add(key)
            facts.append(
                {
                    "article_id": article_id,
                    "fact_id": "{}_ref_{:04d}".format(article_id, len(facts) + 1),
                    "atomic_fact": fact,
                    "source_span": fact,
                }
            )
        if not facts:
            raise ValueError("No reference atomic facts were extracted for {}".format(article_id))
        return facts

    def answer_pre_experiment_question(
        self,
        prompt_filename: str,
        summary_field: str,
        summary: str,
        question: dict,
        include_reasoning: bool,
    ) -> dict:
        replacements = {summary_field: summary, **_option_inputs(question)}
        schema = ANSWER_WITH_REASONING if include_reasoning else ANSWER_ONLY
        response = self.client.generate(render_prompt(prompt_filename, replacements), {}, schema)
        response["answer"] = _clean_text(response.get("answer"), "answer").upper()
        return response

    def direct_support_judgment(self, summary: str, atomic_fact: str) -> bool:
        instruction = render_prompt(
            "pre_experiment_2_direct_judgment.txt",
            {"test_summary": summary, "atomic_fact": atomic_fact},
        )
        response = self.client.generate(instruction, {}, DIRECT_SUPPORT_DECISION)
        return str(response.get("supported", "")).strip().lower() == "yes"

    def rewrite_controlled_summary(
        self,
        expert_summary: str,
        atomic_fact: str,
        condition: str,
        altered_statement: Optional[str] = None,
    ) -> str:
        if condition not in {"F_keep", "F_delete", "F_alter"}:
            raise ValueError("Unknown controlled summary condition")
        if condition == "F_alter" and not str(altered_statement or "").strip():
            raise ValueError("F_alter requires an altered statement")
        instruction = render_operation_prompt(
            "pre_experiment_2_controlled_summary_rewrite.txt",
            {
                "expert_summary": expert_summary,
                "atomic_fact": atomic_fact,
                "condition": condition,
                "altered_statement": altered_statement or "Not applicable.",
            },
        )
        response = self.client.generate(instruction, {}, SUMMARY)
        rewritten = _clean_text(response.get("summary"), "controlled complete summary")
        original_fact_supported = self.direct_support_judgment(rewritten, atomic_fact)
        if condition == "F_keep" and not original_fact_supported:
            raise ValueError("F_keep summary does not preserve the original atomic fact")
        if condition != "F_keep" and original_fact_supported:
            raise ValueError("Controlled summary still supports the original atomic fact")
        if condition == "F_alter" and not self.direct_support_judgment(
            rewritten, str(altered_statement)
        ):
            raise ValueError("Controlled summary does not support the altered statement")
        return rewritten

    def candidate_set_covers(self, reference_fact: str, candidate_facts: Iterable[str]) -> bool:
        candidate_text = "\n".join("- " + str(fact) for fact in candidate_facts)
        if not candidate_text:
            raise ValueError("Coverage evaluation requires a non-empty candidate set")
        instruction = render_prompt(
            "pre_experiment_3_candidate_set_coverage_evaluation.txt",
            {
                "reference_atomic_fact": reference_fact,
                "candidate_atomic_facts": candidate_text,
            },
        )
        response = self.client.generate(instruction, {}, COVERAGE_DECISION)
        return str(response.get("covered", "")).strip().lower() == "yes"

    def target_text_covers(self, reference_fact: str, target_text: str) -> bool:
        instruction = render_operation_prompt(
            "expert_summary_recall_coverage_evaluation.txt",
            {"reference_atomic_fact": reference_fact, "target_text": target_text},
        )
        response = self.client.generate(instruction, {}, COVERAGE_DECISION)
        return str(response.get("covered", "")).strip().lower() == "yes"

    def judge_lay_summary(self, article: str, reference_summary: str, generated_summary: str) -> dict:
        instruction = render_prompt(
            "llm_as_a_judge_evaluation.txt",
            {
                "source_article": article,
                "reference_lay_summary": reference_summary,
                "generated_lay_summary": generated_summary,
            },
        )
        return self.client.generate(instruction, {}, LLM_JUDGEMENT)

    def run_article(self, article: dict) -> Dict[str, Any]:
        article_id = str(article["id"])
        candidates = self.extract_candidate_atomic_facts(article_id, str(article["article"]))
        final_facts = self.filter_against_abstract(str(article["abstract"]), candidates)
        initial_summary = self.generate_fact_driven_summary(str(article["abstract"]), final_facts)
        questions = [self.generate_question(fact) for fact in final_facts]
        current_summary = initial_summary
        checks: List[dict] = []
        for question in questions:
            check = self.answer_and_rewrite(current_summary, question)
            checks.append(check)
            current_summary = check["summary_after"]
        return {
            "article_id": article_id,
            "source_dataset": article["source_dataset"],
            "candidate_atomic_facts": candidates,
            "final_atomic_facts": final_facts,
            "initial_summary": initial_summary,
            "questions": questions,
            "verification_and_rewrites": checks,
            "final_summary": current_summary,
        }

    def run(self, articles: Iterable[dict], output_dir: Path) -> List[dict]:
        completed = [self.run_article(article) for article in articles]
        model_dir = output_dir / self.model_key
        save_jsonl(
            ({"article_id": row["article_id"], "facts": row["candidate_atomic_facts"]} for row in completed),
            model_dir / "module_1_candidate_atomic_facts.jsonl",
        )
        save_jsonl(
            ({"article_id": row["article_id"], "facts": row["final_atomic_facts"]} for row in completed),
            model_dir / "module_2_final_atomic_facts.jsonl",
        )
        save_jsonl(
            ({"article_id": row["article_id"], "summary": row["initial_summary"]} for row in completed),
            model_dir / "fact_driven_lay_summaries.jsonl",
        )
        save_jsonl(
            ({"article_id": row["article_id"], "questions": row["questions"]} for row in completed),
            model_dir / "module_3_questions.jsonl",
        )
        save_jsonl(
            ({"article_id": row["article_id"], "checks": row["verification_and_rewrites"]} for row in completed),
            model_dir / "module_3_verification_and_targeted_rewrites.jsonl",
        )
        save_jsonl(
            (
                {
                    "article_id": row["article_id"],
                    "source_dataset": row["source_dataset"],
                    "summary": row["final_summary"],
                }
                for row in completed
            ),
            model_dir / "final_lay_summaries.jsonl",
        )
        return completed


def run_direct_generation(
    articles: Iterable[dict],
    model_key: str,
    output_dir: Path,
    client: Optional[StructuredLLM] = None,
) -> List[dict]:
    llm = client or StructuredLLM(model_key)
    prompt = load_prompt("direct_generation.txt")
    rows: List[dict] = []
    for article in articles:
        response = llm.generate(prompt, {"full_article_text": article["article"]}, SUMMARY)
        rows.append(
            {
                "article_id": article["id"],
                "source_dataset": article["source_dataset"],
                "summary": _clean_text(response.get("summary"), "direct lay summary"),
            }
        )
    save_jsonl(rows, output_dir / model_key / "direct_lay_summaries.jsonl")
    return rows
