import json
import unittest
from unittest.mock import AsyncMock, patch

from pydantic import ValidationError

import app as relay_app
from app import _persist_creator_assistant_message_for_done
from creator_stream import (
    CreatorContinuationRequest,
    CreatorStreamRequest,
    stream_creator_native_tool_turn,
)


class CreatorStreamRequestTests(unittest.TestCase):
    def test_native_tools_mode_requires_tools(self):
        with self.assertRaises(ValidationError):
            CreatorStreamRequest(messages=[], mode="native_tools")

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

        request = CreatorStreamRequest(
            messages=messages,
            mode="native_tools",
            tools=[{"type": "function", "function": {"name": "apply_patch", "parameters": {"type": "object"}}}],
        )

        self.assertEqual(request.messages, messages)


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

    def test_continuation_allows_telemetry_without_tool_result(self):
        request = CreatorContinuationRequest(
            messages=[{"role": "user", "content": "Patch the draft"}],
            mode="native_tools",
            tools=[{"type": "function", "function": {"name": "apply_patch", "parameters": {"type": "object"}}}],
            decision="approve",
            tool_call={"id": "call_1", "name": "apply_patch", "arguments": {"title": "New"}},
        )

        self.assertEqual(request.decision, "approve")
        self.assertIsNone(request.tool_result)

    def test_continuation_allows_missing_legacy_telemetry_fields(self):
        request = CreatorContinuationRequest(
            messages=[{"role": "user", "content": "Patch the draft"}],
            mode="native_tools",
            tools=[{"type": "function", "function": {"name": "apply_patch", "parameters": {"type": "object"}}}],
        )

        self.assertIsNone(request.decision)
        self.assertIsNone(request.tool_call)


class _JsonRequest:
    def __init__(self, payload):
        self._payload = payload

    async def json(self):
        return self._payload


class _StreamRequest(_JsonRequest):
    async def is_disconnected(self):
        return False


class CreatorContinuationEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_continue_preserves_messages_without_synthesizing_tool_turns(self):
        messages = [
            {"role": "user", "content": "Patch the draft"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I'll use the patch tool."},
                    {"type": "tool_use", "id": "call_1", "name": "apply_patch", "input": {"title": "New"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "call_1", "content": [{"type": "text", "text": "Applied"}]},
                ],
            },
        ]
        payload = {
            "messages": messages,
            "mode": "native_tools",
            "creator_session_id": 123,
            "stream_id": "creator-stream-1",
            "tools": [{"type": "function", "function": {"name": "apply_patch", "parameters": {"type": "object"}}}],
            "decision": "approve",
            "tool_call": {"id": "call_1", "name": "apply_patch", "arguments": {"title": "New"}},
            "tool_result": {"ok": True},
            "feedback": "Telemetry only.",
        }

        with patch("app._stream_creator_native_tool_mode", AsyncMock(return_value={"ok": True})) as stream_mock:
            result = await relay_app.creator_stream_continue(_JsonRequest(payload))

        self.assertEqual(result, {"ok": True})
        forwarded_payload, forwarded_model = stream_mock.await_args.args[1:]
        self.assertEqual(forwarded_payload["messages"], messages)
        self.assertEqual(forwarded_model.messages, messages)
        self.assertEqual(len(forwarded_payload["messages"]), 3)
        self.assertEqual(forwarded_payload["decision"], "approve")
        self.assertEqual(forwarded_payload["tool_result"], {"ok": True})


class CreatorNativeToolPersistenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_tool_stream_persists_awaiting_approval_message_and_done_id(self):
        payload = {
            "messages": [{"role": "user", "content": "Patch the draft"}],
            "mode": "native_tools",
            "creator_session_id": 123,
            "stream_id": "creator-stream-1",
            "assistant_turn_id": "turn-1",
            "tools": [{"type": "function", "function": {"name": "apply_patch", "parameters": {"type": "object"}}}],
        }

        async def fake_native_turn(*args, **kwargs):
            yield {"event": "token", "data": "I'll patch the draft now."}
            yield {
                "event": "creator_tool_call",
                "data": {
                    "stream_id": "creator-stream-1",
                    "assistant_content": "I'll patch the draft now.",
                    "tool_call_id": "call-1",
                    "tool_name": "apply_patch",
                    "arguments": {"title": "One"},
                },
            }
            yield {
                "event": "done",
                "data": {
                    "stream_id": "creator-stream-1",
                    "status": "awaiting_tool_approval",
                    "mode": "native_tools",
                    "tool_call_count": 1,
                },
            }

        with (
            patch("app.verify_jwt", AsyncMock(return_value=("user-1", "token"))),
            patch("app.get_creator_session", AsyncMock(return_value={"id": 123, "entity_type": "character"})),
            patch("app.resolve_stream_bot", AsyncMock(return_value={"id": 9, "access_key": "key", "model": "model"})),
            patch("app.openai_service.initialize_with_config", AsyncMock()),
            patch("app.build_completion_request_kwargs", return_value={}),
            patch("app.stream_creator_native_tool_turn", fake_native_turn),
            patch("app.safe_post_creator_message", AsyncMock(return_value=[{"id": 777}])) as post_mock,
            patch("app.EventSourceResponse", side_effect=lambda generator, **kwargs: generator),
        ):
            generator = await relay_app._stream_creator_native_tool_mode(
                _StreamRequest(payload),
                payload,
                CreatorStreamRequest.model_validate(payload),
            )
            events = [event async for event in generator]

        self.assertEqual([event["event"] for event in events], ["token", "creator_tool_call", "done"])
        done_payload = json.loads(events[-1]["data"])
        self.assertEqual(done_payload["message_id"], 777)
        post_mock.assert_awaited_once()
        self.assertEqual(post_mock.await_args.args[0]["content"], "I'll patch the draft now.")
        self.assertEqual(post_mock.await_args.args[0]["creator_turn_id"], "turn-1")

    async def test_persists_awaiting_tool_approval_assistant_text_with_turn_id(self):
        done_payload = {
            "stream_id": "creator-stream-1",
            "status": "awaiting_tool_approval",
            "mode": "native_tools",
        }

        with patch("app.safe_post_creator_message", AsyncMock(return_value=[{"id": 777}])) as post_mock:
            result = await _persist_creator_assistant_message_for_done(
                done_payload,
                user_id="user-1",
                creator_session_id=123,
                content="I'll patch the draft now.",
                stream_id="creator-stream-1",
                auth_token="token",
                creator_turn_id="turn-1",
            )

        post_mock.assert_awaited_once_with({
            "user_id": "user-1",
            "creator_session_id": 123,
            "content": "I'll patch the draft now.",
            "is_user_author": False,
            "is_streaming": False,
            "is_complete": True,
            "stream_id": "creator-stream-1",
            "creator_turn_id": "turn-1",
        }, "token")
        self.assertEqual(result["message_id"], 777)

    async def test_does_not_persist_empty_awaiting_tool_approval_assistant_text(self):
        done_payload = {
            "stream_id": "creator-stream-1",
            "status": "awaiting_tool_approval",
            "mode": "native_tools",
        }

        with patch("app.safe_post_creator_message", AsyncMock()) as post_mock:
            result = await _persist_creator_assistant_message_for_done(
                done_payload,
                user_id="user-1",
                creator_session_id=123,
                content="   ",
                stream_id="creator-stream-1",
                auth_token="token",
                creator_turn_id="turn-1",
            )

        post_mock.assert_not_awaited()
        self.assertNotIn("message_id", result)


class CreatorNativeToolStreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_text_response_streams_token_chunks(self):
        request = CreatorStreamRequest(
            messages=[{"role": "user", "content": "Summarize the draft"}],
            mode="native_tools",
            stream_id="creator-stream-1",
            tools=[{"type": "function", "function": {"name": "apply_patch", "parameters": {"type": "object"}}}],
        )

        async def fake_stream(*args, **kwargs):
            yield {"content": "First ", "error": None}
            yield {"content": "second.", "finish_reason": "stop", "usage": {"total_tokens": 42}, "error": None}

        with patch(
            "creator_stream.openai_service.create_chat_completion_tool_stream",
            fake_stream,
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

        self.assertEqual([event["event"] for event in events], ["token", "token", "done"])
        self.assertEqual(events[0]["data"], "First ")
        self.assertEqual(events[1]["data"], "second.")
        self.assertEqual(events[2]["data"]["status"], "completed")
        self.assertEqual(events[2]["data"]["finish_reason"], "stop")

    async def test_tool_call_event_includes_nested_tool_call_shape(self):
        request = CreatorStreamRequest(
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
            "creator_stream.openai_service.create_chat_completion_tool_stream",
            fake_stream,
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

        self.assertEqual([event["event"] for event in events], ["token", "token", "creator_tool_call", "done"])
        self.assertEqual(events[2]["data"]["mode"], "native_tools")
        self.assertEqual(events[2]["data"]["tool_call"]["id"], "call_1")
        self.assertEqual(events[2]["data"]["tool_call"]["name"], "apply_patch")
        self.assertEqual(events[2]["data"]["tool_call"]["arguments"], {"title": "One"})
        self.assertEqual(events[2]["data"]["assistant_content"], "I'll patch the draft now.")

    async def test_reasoning_chunks_stream_separately_from_content_and_tool_calls(self):
        request = CreatorStreamRequest(
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
            "creator_stream.openai_service.create_chat_completion_tool_stream",
            fake_stream,
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

        self.assertEqual(
            [event["event"] for event in events],
            ["reasoning", "token", "reasoning", "creator_tool_call", "done"],
        )
        self.assertEqual(events[0]["data"], "Considering the edit.")
        self.assertEqual(events[2]["data"], "Preparing tool call.")
        self.assertEqual(events[3]["data"]["assistant_content"], "I'll patch it.")

    async def test_multiple_tool_calls_emit_error(self):
        request = CreatorStreamRequest(
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
            "creator_stream.openai_service.create_chat_completion_tool_stream",
            fake_stream,
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
