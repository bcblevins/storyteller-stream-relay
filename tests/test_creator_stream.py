import json
import unittest

from pydantic import ValidationError

from creator_stream import (
    CreatorContinuationRequest,
    CreatorStreamRequest,
    build_creator_continuation_messages,
)


class CreatorStreamRequestTests(unittest.TestCase):
    def test_native_tools_mode_requires_tools(self):
        with self.assertRaises(ValidationError):
            CreatorStreamRequest(messages=[], mode="native_tools")

    def test_approve_requires_tool_result(self):
        with self.assertRaises(ValidationError):
            CreatorContinuationRequest(
                messages=[{"role": "user", "content": "Patch the draft"}],
                mode="native_tools",
                tools=[{"type": "function", "function": {"name": "apply_patch", "parameters": {"type": "object"}}}],
                decision="approve",
                tool_call={"id": "call_1", "name": "apply_patch", "arguments": {"title": "New"}},
            )


class CreatorContinuationBuilderTests(unittest.TestCase):
    def test_approve_builds_assistant_and_tool_messages(self):
        request = CreatorContinuationRequest(
            messages=[{"role": "user", "content": "Patch the draft"}],
            mode="native_tools",
            tools=[{"type": "function", "function": {"name": "apply_patch", "parameters": {"type": "object"}}}],
            decision="approve",
            tool_call={"id": "call_1", "name": "apply_patch", "arguments": {"title": "New"}},
            tool_result={"ok": True, "draft_payload": {"title": "New"}},
        )

        messages = build_creator_continuation_messages(request)

        self.assertEqual(messages[1]["role"], "assistant")
        self.assertEqual(messages[1]["tool_calls"][0]["id"], "call_1")
        self.assertEqual(messages[2]["role"], "tool")
        self.assertEqual(json.loads(messages[2]["content"]), {"ok": True, "draft_payload": {"title": "New"}})

    def test_retry_adds_feedback_as_user_message(self):
        request = CreatorContinuationRequest(
            messages=[{"role": "user", "content": "Patch the draft"}],
            mode="native_tools",
            tools=[{"type": "function", "function": {"name": "apply_patch", "parameters": {"type": "object"}}}],
            decision="retry",
            tool_call={"id": "call_1", "name": "apply_patch", "arguments": {"title": "New"}},
            feedback="Please change the summary field instead.",
        )

        messages = build_creator_continuation_messages(request)

        self.assertEqual(messages[-1]["role"], "user")
        self.assertIn("Please try again with revised arguments", messages[-1]["content"])
        self.assertIn("Please change the summary field instead.", messages[-1]["content"])


if __name__ == "__main__":
    unittest.main()
