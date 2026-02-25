import unittest

from request_transforms import TransformConfig, apply_provider_request_transforms


class RequestTransformsTest(unittest.TestCase):
    def test_injects_reasoning_when_enabled_and_model_matches(self):
        payload = {
            "model": "z-ai/glm-4.6:nitro",
            "messages": [{"role": "user", "content": "hi"}],
        }
        config = TransformConfig(
            force_reasoning_enabled=True,
            force_reasoning_effort="high",
            force_reasoning_model_patterns=("z-ai/glm-4.6:nitro",),
            force_reasoning_override=False,
        )

        out = apply_provider_request_transforms(payload, "openrouter", payload["model"], config)

        self.assertEqual(out["reasoning"]["enabled"], True)
        self.assertEqual(out["reasoning"]["effort"], "high")

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
            force_reasoning_effort="high",
            force_reasoning_model_patterns=("z-ai/glm-4.6:nitro",),
            force_reasoning_override=False,
        )

        out = apply_provider_request_transforms(payload, "openrouter", payload["model"], config)

        self.assertEqual(out, payload)

    def test_no_change_when_model_does_not_match(self):
        payload = {"model": "openai/gpt-4o-mini", "messages": []}
        config = TransformConfig(
            force_reasoning_enabled=True,
            force_reasoning_effort="high",
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
            force_reasoning_model_patterns=("z-ai/glm-4.6:nitro",),
            force_reasoning_override=True,
        )

        out = apply_provider_request_transforms(payload, "openrouter", payload["model"], config)

        self.assertEqual(out["reasoning"]["enabled"], True)
        self.assertEqual(out["reasoning"]["effort"], "high")

    def test_no_change_for_non_openrouter_provider(self):
        payload = {"model": "z-ai/glm-4.6:nitro", "messages": []}
        config = TransformConfig(
            force_reasoning_enabled=True,
            force_reasoning_effort="high",
            force_reasoning_model_patterns=("z-ai/glm-4.6:nitro",),
            force_reasoning_override=False,
        )

        out = apply_provider_request_transforms(payload, "other-provider", payload["model"], config)

        self.assertEqual(out, payload)


if __name__ == "__main__":
    unittest.main()
