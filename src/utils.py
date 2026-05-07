from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from src import config


def setup_logger(name: str, log_file: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_path = Path(log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def save_json(data: Any, path: str | Path) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> Any:
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_jsonl(records: Iterable[Any], path: str | Path) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> list[Any]:
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class OpenAIClient:
    _total_usage: dict[str, int] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    def __init__(self, api_key: str | None = None, logger: logging.Logger | None = None) -> None:
        key = api_key or config.OPENAI_API_KEY
        if not key:
            raise ValueError("OPENAI_API_KEY is missing. Please set it in .env.")
        self.client = OpenAI(api_key=key)
        self.logger = logger or setup_logger("openai_client")

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        reraise=True,
    )
    def chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        response_format: dict[str, Any] | None = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            **kwargs,
        }
        if response_format is not None:
            params["response_format"] = response_format

        response = self.client.chat.completions.create(**params)
        content = response.choices[0].message.content or ""
        usage_obj = response.usage
        usage = {
            "prompt_tokens": getattr(usage_obj, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage_obj, "completion_tokens", 0) or 0,
            "total_tokens": getattr(usage_obj, "total_tokens", 0) or 0,
        }

        for key, value in usage.items():
            OpenAIClient._total_usage[key] = OpenAIClient._total_usage.get(key, 0) + int(value)

        self.logger.info(
            "OpenAI usage | prompt=%s completion=%s total=%s | accumulated_total=%s",
            usage["prompt_tokens"],
            usage["completion_tokens"],
            usage["total_tokens"],
            OpenAIClient._total_usage["total_tokens"],
        )
        return {"content": content, "usage": usage}

    @classmethod
    def get_total_usage(cls) -> dict[str, int]:
        return dict(cls._total_usage)
