from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.experiment_2.run_three_model_exp2 import MODEL_CONFIGS, ProviderClient, UsageTracker
from src.thesis_laysumm_v11.run_local_official_backbone import run_pipeline


class OpenRouterBackendTest(unittest.TestCase):
    def test_openrouter_model_configs_are_off_the_shelf_checkpoints(self) -> None:
        self.assertEqual(
            MODEL_CONFIGS["qwen25_7b_instruct_openrouter"]["model"],
            "qwen/qwen-2.5-7b-instruct",
        )
        self.assertEqual(
            MODEL_CONFIGS["llama31_8b_instruct_openrouter"]["model"],
            "meta-llama/llama-3.1-8b-instruct",
        )
        self.assertEqual(MODEL_CONFIGS["llama31_8b_instruct_openrouter"]["max_new_tokens"], 2048)
        self.assertEqual(MODEL_CONFIGS["llama31_8b_instruct_openrouter"]["context_window"], 131072)

    @patch("src.experiment_2.run_three_model_exp2.OpenAI")
    def test_openrouter_request_uses_speed_routing_json_and_fixed_seed(self, openai_cls: MagicMock) -> None:
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))],
            usage=SimpleNamespace(
                prompt_tokens=12,
                completion_tokens=3,
                total_tokens=15,
                prompt_tokens_details=None,
            ),
        )
        create = openai_cls.return_value.chat.completions.create
        create.return_value = response
        usage = UsageTracker(rows=[])
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            client = ProviderClient("qwen25_7b_instruct_openrouter", usage)
            content = client.chat(
                stage="test",
                item_id="item-1",
                messages=[{"role": "user", "content": "return JSON"}],
                response_format={"type": "json_object"},
                temperature=0.0,
            )

        self.assertEqual(content, '{"ok": true}')
        kwargs = create.call_args.kwargs
        self.assertEqual(kwargs["model"], "qwen/qwen-2.5-7b-instruct")
        self.assertEqual(kwargs["response_format"], {"type": "json_object"})
        self.assertEqual(kwargs["seed"], 42)
        self.assertEqual(kwargs["max_tokens"], 2048)
        self.assertEqual(kwargs["extra_body"]["provider"]["sort"], "throughput")
        self.assertTrue(kwargs["extra_body"]["provider"]["require_parameters"])
        self.assertEqual(usage.rows[0]["provider"], "openrouter")

    @patch("src.thesis_laysumm_v11.run_local_official_backbone.rewrite_summaries")
    @patch("src.thesis_laysumm_v11.run_local_official_backbone.answer_questions")
    @patch("src.thesis_laysumm_v11.run_local_official_backbone.generate_summaries")
    @patch("src.thesis_laysumm_v11.run_local_official_backbone.validate_run")
    @patch("src.thesis_laysumm_v11.run_local_official_backbone.prepare_fixed_gemini_upstream")
    def test_pipeline_forwards_four_workers_to_all_phase_b_steps(
        self,
        prepare: MagicMock,
        validate: MagicMock,
        generate: MagicMock,
        answer: MagicMock,
        rewrite: MagicMock,
    ) -> None:
        prepare.return_value = {"copied_counts": {"questions_1t3f_nota.jsonl": 101}}
        run_pipeline(
            run_name="test-openrouter-run",
            model_key="llama31_8b_instruct_openrouter",
            source_run="fixed-gemini-source",
            articles_per_source=5,
            max_workers=4,
            resume=True,
        )
        for phase in (generate, answer, rewrite):
            self.assertEqual(phase.call_args.kwargs["max_workers"], 4)
            self.assertEqual(phase.call_args.kwargs["model_key"], "llama31_8b_instruct_openrouter")
            self.assertTrue(phase.call_args.kwargs["resume"])
        self.assertEqual(validate.call_count, 4)
        prepare.assert_called_once()


if __name__ == "__main__":
    unittest.main()
