import unittest

from request_transforms import TransformConfig, build_completion_request_kwargs


class OpenRouterEffortTests(unittest.TestCase):
    def test_saved_effort_is_only_forwarded_to_openrouter(self):
        config = TransformConfig(force_reasoning_enabled=False)
        for provider, effort in [("openai", "high"), ("openrouter", "unknown")]:
            with self.subTest(provider=provider, effort=effort):
                self.assertEqual(
                    build_completion_request_kwargs(
                        payload={}, provider=provider, model="model", config=config, bot_effort=effort,
                    ),
                    {},
                )

    def test_explicit_controls_and_forced_overrides_take_precedence(self):
        explicit = build_completion_request_kwargs(
            payload={"_relay_thinking_control": {"enabled": True, "effort": "high"}},
            provider="openrouter", model="model", config=TransformConfig(), bot_effort="low",
        )
        self.assertEqual(explicit["extra_body"]["reasoning"]["effort"], "high")
        forced = build_completion_request_kwargs(
            payload={}, provider="openrouter", model="model", bot_effort="none",
            config=TransformConfig(
                force_reasoning_enabled=True, force_reasoning_override=True,
                force_reasoning_effort="high", force_reasoning_model_patterns=("*",),
            ),
        )
        self.assertEqual(forced["extra_body"]["reasoning"], {"enabled": True, "effort": "high"})
