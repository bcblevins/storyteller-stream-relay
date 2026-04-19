import json
import unittest
from unittest.mock import AsyncMock, patch

from pydantic import ValidationError

from creator_stream import (
    CreatorContinuationRequest,
    CreatorStreamRequest,
    build_creator_continuation_messages,
    stream_creator_native_tool_turn,
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
    def test_accepts_flat_tool_call_payload_shape(self):
        request = CreatorContinuationRequest(
            messages=[{"role": "user", "content": "Patch the draft"}],
            mode="native_tools",
            tools=[{"type": "function", "function": {"name": "apply_patch", "parameters": {"type": "object"}}}],
            decision="retry",
            tool_call_id="call_1",
            tool_name="apply_patch",
            arguments={"title": "New"},
            raw_arguments='{"title":"New"}',
            feedback="Please change the summary field instead.",
        )

        self.assertEqual(request.tool_call.id, "call_1")
        self.assertEqual(request.tool_call.name, "apply_patch")
        self.assertEqual(request.tool_call.arguments, {"title": "New"})
        self.assertEqual(request.tool_call.raw_arguments, '{"title":"New"}')

    def test_approve_builds_assistant_and_tool_messages(self):
        request = CreatorContinuationRequest(
            messages=[{"role": "user", "content": "Patch the draft"}],
            mode="native_tools",
            tools=[{"type": "function", "function": {"name": "apply_patch", "parameters": {"type": "object"}}}],
            decision="approve",
            tool_call={"id": "call_1", "name": "apply_patch", "arguments": {"title": "New"}},
            assistant_content="I'll update the draft with the tool result.",
            tool_result={"ok": True, "draft_payload": {"title": "New"}},
        )

        messages = build_creator_continuation_messages(request)

        self.assertEqual(messages[1]["role"], "assistant")
        self.assertEqual(messages[1]["content"], "I'll update the draft with the tool result.")
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


class CreatorNativeToolStreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_tool_call_event_includes_nested_tool_call_shape(self):
        request = CreatorStreamRequest(
            messages=[{"role": "user", "content": "Patch the draft"}],
            mode="native_tools",
            stream_id="creator-stream-1",
            tools=[{"type": "function", "function": {"name": "apply_patch", "parameters": {"type": "object"}}}],
        )
        result = {
            "message": {
                "content": "I'll patch the draft now.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "apply_patch", "arguments": '{"title":"One"}'},
                    }
                ],
            },
            "finish_reason": "tool_calls",
            "usage": {"total_tokens": 42},
            "error": None,
        }

        with patch(
            "creator_stream.openai_service.create_chat_completion_response",
            AsyncMock(return_value=result),
        ):
            events = [
                event
                async for event in stream_creator_native_tool_turn(
                    request,
                    model="deepseek-chat",
                    temperature=0.1,
                    max_tokens=1000,
                    bot={},
                )
            ]

        self.assertEqual(events[0]["event"], "creator_tool_call")
        self.assertEqual(events[0]["data"]["mode"], "native_tools")
        self.assertEqual(events[0]["data"]["tool_call"]["id"], "call_1")
        self.assertEqual(events[0]["data"]["tool_call"]["name"], "apply_patch")
        self.assertEqual(events[0]["data"]["tool_call"]["arguments"], {"title": "One"})
        self.assertEqual(events[0]["data"]["assistant_content"], "I'll patch the draft now.")

    async def test_multiple_tool_calls_emit_error(self):
        request = CreatorStreamRequest(
            messages=[{"role": "user", "content": "Patch the draft"}],
            mode="native_tools",
            stream_id="creator-stream-1",
            tools=[{"type": "function", "function": {"name": "apply_patch", "parameters": {"type": "object"}}}],
        )
        result = {
            "message": {
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "apply_patch", "arguments": '{"title":"One"}'},
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "apply_patch", "arguments": '{"title":"Two"}'},
                    },
                ],
            },
            "finish_reason": "tool_calls",
            "usage": {"total_tokens": 42},
            "error": None,
        }

        with patch(
            "creator_stream.openai_service.create_chat_completion_response",
            AsyncMock(return_value=result),
        ):
            events = [
                event
                async for event in stream_creator_native_tool_turn(
                    request,
                    model="deepseek-chat",
                    temperature=0.1,
                    max_tokens=1000,
                    bot={},
                )
            ]

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event"], "error")
        self.assertEqual(events[0]["data"]["tool_call_count"], 2)
        self.assertIn("exactly one tool call", events[0]["data"]["error"])


if __name__ == "__main__":
    unittest.main()
