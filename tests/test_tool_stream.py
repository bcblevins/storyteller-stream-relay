import json
import unittest
from unittest.mock import AsyncMock, patch

from pydantic import ValidationError

import app as relay_app
from app import resolve_max_tokens
from tool_stream import (
    ToolStreamRequest,
    stream_tool_turn,
)


class ToolStreamRequestTests(unittest.TestCase):
    def test_resolve_max_tokens_prefers_payload_and_treats_zero_as_unset(self):
        self.assertIsNone(resolve_max_tokens({"max_tokens": 0}, {"max_tokens": 4096}))
        self.assertEqual(resolve_max_tokens({"max_tokens": 2048}, {"max_tokens": 4096}), 2048)

    def test_resolve_max_tokens_uses_bot_value_when_payload_omits_it(self):
        self.assertIsNone(resolve_max_tokens({}, {"max_tokens": 0}))
        self.assertEqual(resolve_max_tokens({}, {"max_tokens": 4096}), 4096)
        self.assertEqual(resolve_max_tokens({}, {}), 1000)

    def test_native_tools_mode_requires_tools(self):
        with self.assertRaises(ValidationError):
            ToolStreamRequest(messages=[], mode="native_tools")

    def test_accepts_structured_message_content_blocks(self):
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I'll patch the draft."},
                    {"type": "tool_use", "id": "call_1", "name": "apply_patch", "input": {"title": "New"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "call_1", "content": [{"type": "text", "text": "ok"}]},
                ],
            },
        ]

        request = ToolStreamRequest(
            messages=messages,
            mode="native_tools",
            tools=[{"type": "function", "function": {"name": "apply_patch", "parameters": {"type": "object"}}}],
        )

        self.assertEqual(request.messages, messages)


class _JsonRequest:
    def __init__(self, payload):
        self._payload = payload

    async def json(self):
        return self._payload


class _StreamRequest(_JsonRequest):
    async def is_disconnected(self):
        return False


class ToolStreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_text_response_streams_token_chunks(self):
        request = ToolStreamRequest(
            messages=[{"role": "user", "content": "Summarize the draft"}],
            mode="native_tools",
            stream_id="creator-stream-1",
            tools=[{"type": "function", "function": {"name": "apply_patch", "parameters": {"type": "object"}}}],
        )

        async def fake_stream(*args, **kwargs):
            yield {"content": "First ", "error": None}
            yield {"content": "second.", "finish_reason": "stop", "usage": {"total_tokens": 42}, "error": None}

        with patch(
            "tool_stream.openai_service.create_chat_completion_tool_stream",
            fake_stream,
        ):
            events = [
                event
                async for event in stream_tool_turn(
                    request,
                    model="deepseek-chat",
                    temperature=0.1,
                    max_tokens=1000,
                    bot={},
                )
            ]

        self.assertEqual([event["event"] for event in events], ["token", "token", "done"])
        self.assertEqual(events[0]["data"], "First ")
        self.assertEqual(events[1]["data"], "second.")
        self.assertEqual(events[2]["data"]["status"], "completed")
        self.assertEqual(events[2]["data"]["finish_reason"], "stop")

    async def test_tool_call_event_includes_nested_tool_call_shape(self):
        request = ToolStreamRequest(
            messages=[{"role": "user", "content": "Patch the draft"}],
            mode="native_tools",
            stream_id="creator-stream-1",
            tools=[{"type": "function", "function": {"name": "apply_patch", "parameters": {"type": "object"}}}],
        )
        async def fake_stream(*args, **kwargs):
            yield {"content": "I'll patch ", "error": None}
            yield {"content": "the draft now.", "error": None}
            yield {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "apply_patch", "arguments": '{"title"'},
                    }
                ],
                "error": None,
            }
            yield {
                "tool_calls": [
                    {
                        "index": 0,
                        "function": {"arguments": ':"One"}'},
                    }
                ],
                "finish_reason": "tool_calls",
                "usage": {"total_tokens": 42},
                "error": None,
            }

        with patch(
            "tool_stream.openai_service.create_chat_completion_tool_stream",
            fake_stream,
        ):
            events = [
                event
                async for event in stream_tool_turn(
                    request,
                    model="deepseek-chat",
                    temperature=0.1,
                    max_tokens=1000,
                    bot={},
                )
            ]

        self.assertEqual(
            [event["event"] for event in events],
            ["token", "token", "tool_call_start", "tool_call", "done"],
        )
        self.assertEqual(events[2]["data"], {"tool_name": "apply_patch"})
        self.assertEqual(events[3]["data"]["mode"], "native_tools")
        self.assertEqual(events[3]["data"]["tool_call"]["id"], "call_1")
        self.assertEqual(events[3]["data"]["tool_call"]["name"], "apply_patch")
        self.assertEqual(events[3]["data"]["tool_call"]["arguments"], {"title": "One"})
        self.assertEqual(events[3]["data"]["assistant_content"], "I'll patch the draft now.")

    async def test_reasoning_chunks_stream_separately_from_content_and_tool_calls(self):
        request = ToolStreamRequest(
            messages=[{"role": "user", "content": "Patch the draft"}],
            mode="native_tools",
            stream_id="creator-stream-1",
            tools=[{"type": "function", "function": {"name": "apply_patch", "parameters": {"type": "object"}}}],
        )

        async def fake_stream(*args, **kwargs):
            yield {"reasoning": "Considering the edit.", "error": None}
            yield {"content": "I'll patch it.", "error": None}
            yield {"reasoning": "Preparing tool call.", "error": None}
            yield {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "apply_patch", "arguments": '{"title":"One"}'},
                    }
                ],
                "finish_reason": "tool_calls",
                "usage": {"total_tokens": 42},
                "error": None,
            }

        with patch(
            "tool_stream.openai_service.create_chat_completion_tool_stream",
            fake_stream,
        ):
            events = [
                event
                async for event in stream_tool_turn(
                    request,
                    model="deepseek-chat",
                    temperature=0.1,
                    max_tokens=1000,
                    bot={},
                )
            ]

        self.assertEqual(
            [event["event"] for event in events],
            ["reasoning", "token", "reasoning", "tool_call_start", "tool_call", "done"],
        )
        self.assertEqual(events[0]["data"], "Considering the edit.")
        self.assertEqual(events[2]["data"], "Preparing tool call.")
        self.assertEqual(events[3]["data"], {"tool_name": "apply_patch"})
        self.assertEqual(events[4]["data"]["assistant_content"], "I'll patch it.")

    async def test_tool_call_start_event_allows_unknown_tool_name(self):
        request = ToolStreamRequest(
            messages=[{"role": "user", "content": "Patch the draft"}],
            mode="native_tools",
            stream_id="creator-stream-1",
            tools=[{"type": "function", "function": {"name": "apply_patch", "parameters": {"type": "object"}}}],
        )

        async def fake_stream(*args, **kwargs):
            yield {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {"arguments": '{"title"'},
                    }
                ],
                "error": None,
            }
            yield {
                "tool_calls": [
                    {
                        "index": 0,
                        "function": {"name": "apply_patch", "arguments": ':"One"}'},
                    }
                ],
                "finish_reason": "tool_calls",
                "error": None,
            }

        with patch(
            "tool_stream.openai_service.create_chat_completion_tool_stream",
            fake_stream,
        ):
            events = [
                event
                async for event in stream_tool_turn(
                    request,
                    model="deepseek-chat",
                    temperature=0.1,
                    max_tokens=1000,
                    bot={},
                )
            ]

        self.assertEqual(
            [event["event"] for event in events],
            ["tool_call_start", "tool_call", "done"],
        )
        self.assertEqual(events[0]["data"], {"tool_name": None})
        self.assertEqual(events[1]["data"]["tool_name"], "apply_patch")

    async def test_tool_call_start_event_for_normalized_provider_start(self):
        request = ToolStreamRequest(
            messages=[{"role": "user", "content": "Patch the draft"}],
            mode="native_tools",
            stream_id="creator-stream-1",
            tools=[{"type": "function", "function": {"name": "apply_patch", "parameters": {"type": "object"}}}],
        )

        async def fake_stream(*args, **kwargs):
            yield {"tool_call_start": {"tool_name": "apply_patch"}, "error": None}
            yield {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "apply_patch", "arguments": '{"title":"One"}'},
                    }
                ],
                "finish_reason": "tool_calls",
                "error": None,
            }

        with patch(
            "tool_stream.openai_service.create_chat_completion_tool_stream",
            fake_stream,
        ):
            events = [
                event
                async for event in stream_tool_turn(
                    request,
                    model="claude-sonnet",
                    temperature=0.1,
                    max_tokens=1000,
                    bot={},
                )
            ]

        self.assertEqual(
            [event["event"] for event in events],
            ["tool_call_start", "tool_call", "done"],
        )
        self.assertEqual(events[0]["data"], {"tool_name": "apply_patch"})
        self.assertEqual(events[1]["data"]["tool_name"], "apply_patch")

    async def test_multiple_tool_calls_emit_error(self):
        request = ToolStreamRequest(
            messages=[{"role": "user", "content": "Patch the draft"}],
            mode="native_tools",
            stream_id="creator-stream-1",
            tools=[{"type": "function", "function": {"name": "apply_patch", "parameters": {"type": "object"}}}],
        )
        async def fake_stream(*args, **kwargs):
            yield {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "apply_patch", "arguments": '{"title":"One"}'},
                    },
                    {
                        "index": 1,
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "apply_patch", "arguments": '{"title":"Two"}'},
                    },
                ],
                "finish_reason": "tool_calls",
                "usage": {"total_tokens": 42},
                "error": None,
            }

        with patch(
            "tool_stream.openai_service.create_chat_completion_tool_stream",
            fake_stream,
        ):
            events = [
                event
                async for event in stream_tool_turn(
                    request,
                    model="deepseek-chat",
                    temperature=0.1,
                    max_tokens=1000,
                    bot={},
                )
            ]

        self.assertEqual([event["event"] for event in events], ["tool_call_start", "error"])
        self.assertEqual(events[0]["data"], {"tool_name": "apply_patch"})
        self.assertEqual(events[1]["data"]["tool_call_count"], 2)
        self.assertIn("exactly one tool call", events[1]["data"]["error"])


if __name__ == "__main__":
    unittest.main()
