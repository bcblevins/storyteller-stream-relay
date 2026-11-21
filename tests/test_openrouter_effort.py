import unittest
from unittest.mock import AsyncMock, patch

import app as relay_app
from openrouter_effort import parse_effort_options
from request_transforms import TransformConfig, build_completion_request_kwargs


class OpenRouterEffortTests(unittest.IsolatedAsyncioTestCase):
    def test_filters_optional_levels_for_mandatory_reasoning(self):
        self.assertEqual(
            parse_effort_options({"supported_efforts": ["high", "none", "low", "unknown"], "mandatory": True}),
            ("low", "high"),
        )
        self.assertEqual(parse_effort_options({"supported_efforts": None})[-2:], ("xhigh", "max"))
        self.assertEqual(parse_effort_options({"mandatory": True}), ())

    async def test_does_not_send_effort_when_model_support_changes(self):
        with patch("app.get_effort_options", AsyncMock(return_value=("low", "high"))):
            effort = await relay_app.resolve_bot_effort({"reasoning_effort": "none"}, "openrouter", "model")
        self.assertIsNone(effort)

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
