import unittest

from request_transforms import (
    TransformConfig,
    apply_provider_request_transforms,
    apply_system_injection_tag_transform,
    build_completion_request_kwargs,
    detect_completion_provider,
    normalize_completion_base_url,
)


class RequestTransformsTest(unittest.TestCase):
    def test_normalize_completion_base_url_leaves_plain_base_url_unchanged(self):
        self.assertEqual(
            normalize_completion_base_url("https://api.openai.com/v1"),
            "https://api.openai.com/v1",
        )

    def test_normalize_completion_base_url_strips_trailing_chat_completions_suffix(self):
        self.assertEqual(
            normalize_completion_base_url("https://openrouter.ai/api/v1/chat/completions"),
            "https://openrouter.ai/api/v1",
        )

    def test_normalize_completion_base_url_strips_suffix_after_nested_prefix(self):
        self.assertEqual(
            normalize_completion_base_url("https://example.com/openai/v1/chat/completions/"),
            "https://example.com/openai/v1",
        )

    def test_detect_completion_provider_prefers_openrouter_flag(self):
        provider = detect_completion_provider(
            base_url="https://api.anthropic.com/v1",
            model="anthropic/claude-sonnet-4",
            is_openrouter=True,
        )

        self.assertEqual(provider, "openrouter")

    def test_injects_openrouter_reasoning_when_enabled_and_model_matches(self):
        payload = {
            "model": "z-ai/glm-4.6:nitro",
            "messages": [{"role": "user", "content": "hi"}],
        }
        config = TransformConfig(
            force_reasoning_enabled=True,
            force_reasoning_effort=None,
            force_reasoning_model_patterns=("*",),
            force_reasoning_override=False,
        )

        out = apply_provider_request_transforms(payload, "openrouter", payload["model"], config)

        self.assertEqual(out["reasoning"]["enabled"], True)
        self.assertNotIn("effort", out["reasoning"])

    def test_no_change_when_feature_disabled(self):
        payload = {"model": "z-ai/glm-4.6:nitro", "messages": []}
        config = TransformConfig(force_reasoning_enabled=False)

        out = apply_provider_request_transforms(payload, "openrouter", payload["model"], config)

        self.assertEqual(out, payload)

    def test_no_override_when_reasoning_already_present_and_override_disabled(self):
        payload = {
            "model": "z-ai/glm-4.6:nitro",
            "reasoning": {"enabled": False, "effort": "low"},
        }
        config = TransformConfig(
            force_reasoning_enabled=True,
            force_reasoning_effort=None,
            force_reasoning_model_patterns=("*",),
            force_reasoning_override=False,
        )

        out = apply_provider_request_transforms(payload, "openrouter", payload["model"], config)

        self.assertEqual(out, payload)

    def test_no_override_when_openrouter_reasoning_is_already_present_in_extra_body(self):
        payload = {
            "model": "z-ai/glm-4.6:nitro",
            "extra_body": {"reasoning": {"enabled": False}},
        }
        config = TransformConfig(
            force_reasoning_enabled=True,
            force_reasoning_effort=None,
            force_reasoning_model_patterns=("*",),
            force_reasoning_override=False,
        )

        out = apply_provider_request_transforms(payload, "openrouter", payload["model"], config)

        self.assertEqual(out, payload)

    def test_no_change_when_model_does_not_match(self):
        payload = {"model": "openai/gpt-4o-mini", "messages": []}
        config = TransformConfig(
            force_reasoning_enabled=True,
            force_reasoning_effort=None,
            force_reasoning_model_patterns=("z-ai/glm-4.6:nitro",),
            force_reasoning_override=False,
        )

        out = apply_provider_request_transforms(payload, "openrouter", payload["model"], config)

        self.assertEqual(out, payload)

    def test_override_when_reasoning_present_and_override_enabled(self):
        payload = {
            "model": "z-ai/glm-4.6:nitro",
            "reasoning": {"enabled": False, "effort": "low"},
        }
        config = TransformConfig(
            force_reasoning_enabled=True,
            force_reasoning_effort="high",
            force_reasoning_model_patterns=("*",),
            force_reasoning_override=True,
        )

        out = apply_provider_request_transforms(payload, "openrouter", payload["model"], config)

        self.assertEqual(out["reasoning"]["enabled"], True)
        self.assertEqual(out["reasoning"]["effort"], "high")

    def test_no_change_for_non_openrouter_provider(self):
        payload = {"model": "z-ai/glm-4.6:nitro", "messages": []}
        config = TransformConfig(
            force_reasoning_enabled=True,
            force_reasoning_effort=None,
            force_reasoning_model_patterns=("*",),
            force_reasoning_override=False,
        )

        out = apply_provider_request_transforms(payload, "other-provider", payload["model"], config)

        self.assertEqual(out, payload)

    def test_injects_openai_reasoning_effort_for_reasoning_models(self):
        payload = {"model": "gpt-5-mini", "messages": []}
        config = TransformConfig(force_reasoning_enabled=True, force_reasoning_effort=None)

        out = apply_provider_request_transforms(payload, "openai", payload["model"], config)

        self.assertEqual(out["reasoning_effort"], "low")

    def test_injects_anthropic_thinking_budget_into_extra_body(self):
        payload = {"model": "claude-sonnet-4-20250514", "messages": []}
        config = TransformConfig(force_reasoning_enabled=True)

        out = apply_provider_request_transforms(payload, "anthropic", payload["model"], config)

        self.assertEqual(out["extra_body"]["thinking"]["type"], "enabled")
        self.assertEqual(out["extra_body"]["thinking"]["budget_tokens"], 1024)

    def test_injects_deepseek_thinking_when_not_using_reasoner_model(self):
        payload = {"model": "deepseek-chat", "messages": []}
        config = TransformConfig(force_reasoning_enabled=True)

        out = apply_provider_request_transforms(payload, "deepseek", payload["model"], config)

        self.assertEqual(out["extra_body"]["thinking"]["type"], "enabled")

    def test_build_completion_request_kwargs_moves_reasoning_to_extra_body_for_sdk_calls(self):
        payload = {"reasoning": {"enabled": True}}
        config = TransformConfig(force_reasoning_enabled=True)

        out = build_completion_request_kwargs(
            payload,
            provider="openrouter",
            model="z-ai/glm-4.6:nitro",
            config=config,
        )

        self.assertEqual(out["extra_body"]["reasoning"]["enabled"], True)
        self.assertNotIn("reasoning_effort", out)

    def test_build_completion_request_kwargs_preserves_explicit_extra_body(self):
        payload = {"extra_body": {"metadata": {"source": "test"}}}
        config = TransformConfig(force_reasoning_enabled=True)

        out = build_completion_request_kwargs(
            payload,
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            config=config,
        )

        self.assertEqual(out["extra_body"]["metadata"]["source"], "test")
        self.assertEqual(out["extra_body"]["thinking"]["budget_tokens"], 1024)

    def test_extracts_injection_tag_from_system_and_appends_to_last_message(self):
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": "Base system. <injection>Think step-by-step before answering.</injection>",
                },
                {"role": "user", "content": "What is 2+2?"},
            ]
        }
        config = TransformConfig(enable_system_injection_tag=True, system_injection_tag_name="injection")

        out = apply_system_injection_tag_transform(payload, config)

        self.assertNotIn("<injection>", out["messages"][0]["content"])
        self.assertIn("Base system.", out["messages"][0]["content"])
        self.assertIn("Think step-by-step before answering.", out["messages"][1]["content"])

    def test_no_injection_change_when_feature_disabled(self):
        payload = {
            "messages": [
                {"role": "system", "content": "<injection>secret</injection>"},
                {"role": "user", "content": "hi"},
            ]
        }
        config = TransformConfig(enable_system_injection_tag=False)

        out = apply_system_injection_tag_transform(payload, config)

        self.assertEqual(out, payload)

    def test_supports_multiple_system_tags(self):
        payload = {
            "messages": [
                {"role": "system", "content": "A <injection>first</injection> B"},
                {"role": "system", "content": "C <injection>second</injection> D"},
                {"role": "user", "content": "prompt"},
            ]
        }
        config = TransformConfig(enable_system_injection_tag=True, system_injection_tag_name="injection")

        out = apply_system_injection_tag_transform(payload, config)

        self.assertNotIn("<injection>", out["messages"][0]["content"])
        self.assertNotIn("<injection>", out["messages"][1]["content"])
        self.assertIn("first", out["messages"][2]["content"])
        self.assertIn("second", out["messages"][2]["content"])


if __name__ == "__main__":
    unittest.main()
