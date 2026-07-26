from __future__ import annotations

import unittest

import torch

from src.experiment_2.run_three_model_exp2 import ProviderClient
from src.thesis_laysumm_v11.run_local_official_backbone import _selected_articles


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __init__(self) -> None:
        self.messages: list[dict[str, str]] = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        self.messages = messages
        return "fake prompt"

    def __call__(self, text, *, return_tensors, add_special_tokens):
        return {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}

    def decode(self, token_ids, *, skip_special_tokens):
        return '{"ok": true}'


class _FakeModel:
    def __init__(self) -> None:
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.kwargs = {}

    def parameters(self):
        yield self.weight

    def generate(self, **kwargs):
        self.kwargs = kwargs
        return torch.tensor([[1, 2, 3, 7, 8]])


class LocalHFBackendTest(unittest.TestCase):
    def test_json_instruction_and_usage_counts(self) -> None:
        tokenizer = _FakeTokenizer()
        model = _FakeModel()
        client = ProviderClient.__new__(ProviderClient)
        client.model = "fake/model"
        client.context_window = 32
        client.max_new_tokens = 8
        client._load_local_hf = lambda: (tokenizer, model)

        content, prompt_tokens, completion_tokens, total_tokens = client._local_hf_chat(
            stage="test",
            messages=[{"role": "user", "content": "hello"}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )

        self.assertEqual(content, '{"ok": true}')
        self.assertEqual((prompt_tokens, completion_tokens, total_tokens), (3, 2, 5))
        self.assertEqual(tokenizer.messages[0]["role"], "system")
        self.assertIn("valid JSON", tokenizer.messages[0]["content"])
        self.assertFalse(model.kwargs["do_sample"])
        self.assertNotIn("eos_token_id", model.kwargs)

    def test_fixed_subset_is_5_plus_5_and_from_final_284(self) -> None:
        rows = _selected_articles("pilot_n20_v11_fixed_m12_gemini25_same_model_phaseb_factuality_chase")
        self.assertEqual(len(rows), 10)
        self.assertEqual(sum(row["source_dataset"] == "PLOS" for row in rows), 5)
        self.assertEqual(sum(row["source_dataset"] == "eLife" for row in rows), 5)

    def test_fixed_subset_can_use_all_10_plus_10_pilot_articles(self) -> None:
        rows = _selected_articles(
            "pilot_n20_v11_fixed_m12_gemini25_same_model_phaseb_factuality_chase",
            articles_per_source=10,
        )
        self.assertEqual(len(rows), 20)
        self.assertEqual(sum(row["source_dataset"] == "PLOS" for row in rows), 10)
        self.assertEqual(sum(row["source_dataset"] == "eLife" for row in rows), 10)


if __name__ == "__main__":
    unittest.main()
