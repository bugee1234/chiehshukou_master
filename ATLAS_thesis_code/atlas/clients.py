from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    model: str
    reasoning_effort: Optional[str] = None


MODEL_SPECS: Dict[str, ModelSpec] = {
    "gpt41": ModelSpec("openai", "gpt-4.1"),
    "gpt41_mini": ModelSpec("openai", "gpt-4.1-mini"),
    "gpt4o_mini": ModelSpec("openai", "gpt-4o-mini"),
    "gpt54_mini": ModelSpec("openai_responses", "gpt-5.4-mini-2026-03-17", "medium"),
    "gemini25_flash": ModelSpec("gemini", "gemini-2.5-flash"),
    "gemini3_flash_preview": ModelSpec("gemini", "gemini-3-flash-preview"),
    "qwen25_7b_instruct": ModelSpec("local_hf", "Qwen/Qwen2.5-7B-Instruct"),
    "llama3_8b_instruct": ModelSpec("local_hf", "meta-llama/Meta-Llama-3-8B-Instruct"),
}


def _json_contract(schema: Dict[str, Any]) -> str:
    return (
        "Return only one JSON object that conforms to this machine-readable output schema:\n"
        + json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
    )


def _parse_json_object(text: str) -> Dict[str, Any]:
    value = text.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
        value = re.sub(r"\s*```$", "", value)
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("The model response must be a JSON object")
    return parsed


class StructuredLLM:
    """Provider-neutral structured generation without loading any dotenv file."""

    def __init__(self, model_key: str):
        if model_key not in MODEL_SPECS:
            raise ValueError("Unknown model key: {}".format(model_key))
        self.model_key = model_key
        self.spec = MODEL_SPECS[model_key]
        self._client: Any = None
        self._tokenizer: Any = None
        self._model: Any = None

    def generate(
        self,
        instruction: str,
        inputs: Dict[str, Any],
        schema: Dict[str, Any],
    ) -> Dict[str, Any]:
        user_content = "" if not inputs else json.dumps(inputs, ensure_ascii=False, indent=2)
        if self.spec.provider == "openai":
            return self._openai_chat(instruction, user_content, schema)
        if self.spec.provider == "openai_responses":
            return self._openai_responses(instruction, user_content, schema)
        if self.spec.provider == "gemini":
            return self._gemini(instruction, user_content, schema)
        if self.spec.provider == "local_hf":
            return self._local_hf(instruction, user_content, schema)
        raise AssertionError("Unreachable provider")

    def _openai_client(self) -> Any:
        if self._client is None:
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY is not set in the process environment")
            from openai import OpenAI

            self._client = OpenAI(api_key=api_key)
        return self._client

    def _openai_chat(self, instruction: str, user_content: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        response = self._openai_client().chat.completions.create(
            model=self.spec.model,
            temperature=0,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "atlas_response",
                    "strict": True,
                    "schema": schema,
                },
            },
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": user_content},
            ],
        )
        return _parse_json_object(response.choices[0].message.content or "")

    def _openai_responses(self, instruction: str, user_content: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        request: Dict[str, Any] = {
            "model": self.spec.model,
            "instructions": instruction,
            "input": user_content,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "atlas_response",
                    "strict": True,
                    "schema": schema,
                }
            },
        }
        if self.spec.reasoning_effort:
            request["reasoning"] = {"effort": self.spec.reasoning_effort}
        response = self._openai_client().responses.create(**request)
        return _parse_json_object(response.output_text or "")

    def _gemini(self, instruction: str, user_content: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        if self._client is None:
            api_key = os.environ.get("GEMINI_API_KEY", "")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY is not set in the process environment")
            from google import genai

            self._client = genai.Client(api_key=api_key)
        from google.genai import types

        response = self._client.models.generate_content(
            model=self.spec.model,
            contents=user_content,
            config=types.GenerateContentConfig(
                system_instruction=instruction,
                temperature=0,
                response_mime_type="application/json",
                response_json_schema=schema,
            ),
        )
        return _parse_json_object(response.text or "")

    def _local_hf(self, instruction: str, user_content: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        if self._model is None or self._tokenizer is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if not torch.cuda.is_available():
                raise RuntimeError("The public-backbone validation requires a CUDA-capable GPU")
            self._tokenizer = AutoTokenizer.from_pretrained(self.spec.model)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.spec.model,
                device_map="auto",
                torch_dtype="auto",
            )
            self._model.eval()
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": instruction + "\n\n" + _json_contract(schema)},
            {"role": "user", "content": user_content},
        ]
        prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        output = self._model.generate(**encoded, max_new_tokens=2048, do_sample=False)
        generated = output[0, encoded["input_ids"].shape[-1] :]
        return _parse_json_object(self._tokenizer.decode(generated, skip_special_tokens=True))
