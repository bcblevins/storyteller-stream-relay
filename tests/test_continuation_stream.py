import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
from openai.types.chat import ChatCompletionChunk

import app as relay_app
from openai_service import OpenAIService
from tool_stream import ToolStreamRequest, stream_tool_turn


TOOLS = [{"type": "function", "function": {"name": "world_index", "parameters": {"type": "object"}}}]


def chunk(delta=None, finish=None, usage=None, model="provider-model"):
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=delta or {}, finish_reason=finish)],
        usage=usage, model=model,
    )


def call(index=0, call_id="call-1", name="world_index", arguments='{"operation":"list"}'):
    return {"index": index, "id": call_id, "type": "function",
            "function": {"name": name, "arguments": arguments}}


class FakeStream:
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for item in self.chunks:
            if isinstance(item, BaseException):
                raise item
            yield item

    async def close(self):
        self.closed = True


class Request:
    def __init__(self, payload, disconnect=False):
        self.payload = payload
        self.disconnect = disconnect

    async def json(self):
        return self.payload

    async def is_disconnected(self):
        return self.disconnect


class ContinuationTests(unittest.IsolatedAsyncioTestCase):
    async def test_route_structured_output_reaches_the_sdk_and_keeps_the_terminal_envelope(self):
        response_format = {"type": "json_schema", "json_schema": {
            "name": "entry", "strict": True, "schema": {"type": "object", "properties": {
                "title": {"type": "string"}, "tags": {"type": "array", "items": {"type": "string"}},
                "content": {"type": "string"}}, "required": ["title", "tags", "content"], "additionalProperties": False},
        }}
        for provider, base_url, model in (("openrouter", "https://openrouter.ai/api/v1", "deepseek/example"),
                                          ("deepseek", "https://api.deepseek.com", "deepseek-chat")):
            with self.subTest(provider=provider):
                raw = '{"title":"Mara","tags":[],"content":"She keeps the bell."}'
                upstream = FakeStream([chunk({"content": raw[:10]}), chunk({"content": raw[10:]}, "stop")])
                create = AsyncMock(return_value=upstream)
                service = OpenAIService()
                service.initialized = True
                service.initialize_with_config = AsyncMock()
                service.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)), close=AsyncMock())
                payload = {"bot_id": 9, "messages": [{"role": "user", "content": "Return JSON."}],
                           "tool_protocol": "continuation_v1", "response_format": response_format}
                with (
                    patch("app.verify_jwt", AsyncMock(return_value=("user-1", "test-token"))),
                    patch("app.resolve_stream_bot", AsyncMock(return_value={"id": 9, "access_key": "key", "model": model,
                                                                           "access_path": base_url})),
                    patch("app.OpenAIService", return_value=service),
                    patch("app.EventSourceResponse", side_effect=lambda generator, **kwargs: generator),
                ):
                    generator = await relay_app.stream(Request(payload))
                    events = [event async for event in generator]
                kwargs = create.call_args.kwargs
                self.assertEqual(kwargs["response_format"], response_format if provider == "openrouter" else {"type": "json_object"})
                if provider == "openrouter":
                    self.assertTrue(kwargs["extra_body"]["provider"]["require_parameters"])
                else:
                    self.assertNotIn("provider", kwargs.get("extra_body", {}))
                self.assertNotIn("tools", kwargs)
                self.assertEqual([event["event"] for event in events], ["token", "token", "done"])
                self.assertEqual(json.loads(events[-1]["data"])["assistant_message"], {"role": "assistant", "content": raw})
                self.assertTrue(upstream.closed)
                service.client.close.assert_awaited_once()

    async def test_route_rejects_invalid_response_format_in_legacy_and_continuation_modes(self):
        for protocol in (None, "continuation_v1"):
            with self.subTest(protocol=protocol), patch("app.resolve_stream_bot", AsyncMock()) as resolve:
                payload = {"messages": [{"role": "user", "content": "Hello"}], "response_format": {"type": "unknown"}}
                if protocol:
                    payload["tool_protocol"] = protocol
                with self.assertRaises(HTTPException) as error:
                    await relay_app.stream(Request(payload))
                self.assertEqual(error.exception.status_code, 400)
                resolve.assert_not_awaited()

    async def run_round(self, chunks, messages=None, tools=TOOLS):
        stream = FakeStream(chunks)
        create = AsyncMock(return_value=stream)
        service = OpenAIService()
        service.initialized = True
        service.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        request = ToolStreamRequest(
            messages=messages or [{"role": "user", "content": "Find the harbor."}],
            tools=tools, mode="native_tools" if tools else "text",
            tool_protocol="continuation_v1", stream_id="test-stream",
        )
        with patch("continuation_stream.openai_service", service):
            events = [event async for event in stream_tool_turn(
                request, model="configured-model", temperature=0.2, max_tokens=1000,
                bot={"access_key": "must-not-be-in-envelope"},
            )]
        self.assertTrue(stream.closed)
        return events, create.call_args.kwargs

    async def test_multiple_calls_preserve_order_ids_and_raw_json_across_fragments(self):
        raw = '  {"operation": "list", "query": "波"} '
        first = call(arguments=raw[:12])
        second = call(1, "call-2", "another_tool", '{"ids":[]}')
        events, _ = await self.run_round([
            chunk({"role": "assistant", "content": "Checking. ", "tool_calls": [second, first]}),
            chunk({"content": "One moment.", "tool_calls": [{"index": 0, "function": {"arguments": raw[12:]}}]}, "tool_calls"),
            SimpleNamespace(choices=[], usage={"total_tokens": 12}),
        ])
        self.assertEqual([event["event"] for event in events], ["token", "tool_call_start", "token", "done"])
        done = events[-1]["data"]
        self.assertEqual(done["tool_protocol"], "continuation_v1")
        self.assertEqual(done["status"], "tool_calls")
        self.assertEqual(done["model"], "configured-model")
        self.assertEqual(done["provider_model"], "provider-model")
        self.assertEqual(done["usage"], {"total_tokens": 12})
        self.assertEqual(done["assistant_message"], {
            "role": "assistant", "content": "Checking. One moment.", "tool_calls": [
                {"id": "call-1", "type": "function", "function": {"name": "world_index", "arguments": raw}},
                {"id": "call-2", "type": "function", "function": {"name": "another_tool", "arguments": '{"ids":[]}'}},
            ],
        })
        self.assertNotIn("must-not-be-in-envelope", json.dumps(events))

    async def test_reasoning_fields_and_signed_details_replay_without_modification(self):
        details1 = [{"type": "reasoning.text", "id": "r1", "index": 0, "text": "Think ", "signature": None}]
        details2 = [{"type": "reasoning.text", "id": "r1", "index": 0, "text": "more", "signature": "signed"},
                    {"type": "reasoning.encrypted", "id": "r2", "data": "opaque", "format": "provider-format"}]
        original = deepcopy(details1 + details2)
        sdk_chunk = ChatCompletionChunk.model_validate({
            "id": "completion", "object": "chat.completion.chunk", "created": 0, "model": "model",
            "choices": [{"index": 0, "delta": {"reasoning_details": details1}, "finish_reason": None}],
        })
        events, _ = await self.run_round([
            sdk_chunk,
            chunk({"reasoning_content": "Plan ", "reasoning": "Summary ", "reasoning_details": details2}),
            chunk({"reasoning_content": "done", "reasoning": "done", "tool_calls": [call()]}, "tool_calls"),
        ])
        assistant = events[-1]["data"]["assistant_message"]
        self.assertEqual(assistant["reasoning_content"], "Plan done")
        self.assertEqual(assistant["reasoning"], "Summary done")
        self.assertEqual(assistant["reasoning_details"], original)
        self.assertEqual(details1 + details2, original)
        messages = [{"role": "user", "content": "Find the harbor."}, assistant,
                    {"role": "tool", "tool_call_id": "call-1", "content": '{"entries":[]}'}]
        before = deepcopy(messages)
        final, forwarded = await self.run_round([chunk({"content": "No entries."}, "stop")], messages)
        self.assertEqual(forwarded["messages"], before)
        self.assertEqual(messages, before)
        self.assertEqual(forwarded["tools"], TOOLS)
        self.assertEqual(final[-1]["data"]["assistant_message"]["content"], "No entries.")

    async def test_text_only_opt_in_has_terminal_metadata_without_tool_parameters(self):
        events, forwarded = await self.run_round([chunk({"content": "Story"}, "stop", model=None)], tools=[])
        self.assertEqual([event["event"] for event in events], ["token", "done"])
        self.assertIsNone(events[-1]["data"]["usage"])
        self.assertIsNone(events[-1]["data"]["provider_model"])
        for field in ("tools", "tool_choice", "parallel_tool_calls"):
            self.assertNotIn(field, forwarded)

    async def test_role_only_usage_trailers_preserve_success_and_usage(self):
        trailer = {"role": "assistant", "content": None, "tool_calls": None,
                   "refusal": None, "reasoning": ""}
        for tools in ([], TOOLS):
            with self.subTest(tools=bool(tools)):
                delta = {"tool_calls": [call()]} if tools else {"content": "Story"}
                finish = "tool_calls" if tools else "stop"
                events, _ = await self.run_round([
                    chunk(delta, finish),
                    chunk(trailer, finish, usage={"completion_tokens": 25}),
                ], tools=tools)
                terminals = [event for event in events if event["event"] in ("done", "error")]
                self.assertEqual([event["event"] for event in terminals], ["done"])
                self.assertEqual(terminals[0]["data"]["usage"], {"completion_tokens": 25})
                self.assertEqual(terminals[0]["data"]["finish_reason"], finish)

    async def test_role_only_usage_trailer_does_not_obscure_reasoning_truncation(self):
        usage = {"completion_tokens": 1000, "completion_tokens_details": {"reasoning_tokens": 1000}}
        events, _ = await self.run_round([
            chunk({"reasoning": "Plan the story."}, "length"),
            chunk({"role": "assistant", "content": None}, "length", usage=usage),
        ], tools=[])
        terminals = [event for event in events if event["event"] in ("done", "error")]
        self.assertEqual([event["event"] for event in terminals], ["error"])
        error = terminals[0]["data"]
        self.assertEqual(error["error"], "Provider response ended without a complete assistant turn.")
        self.assertEqual(error["finish_reason"], "length")
        self.assertEqual(error["usage"], usage)
        self.assertEqual(error["partial_assistant_message"], {
            "role": "assistant", "content": None, "reasoning": "Plan the story.",
        })

    async def test_incomplete_invalid_and_conflicting_responses_have_one_error_no_done(self):
        scenarios = {
            "truncated": [chunk({"tool_calls": [call(arguments='{"unfinished":')]}, "length")],
            "missing_finish": [chunk({"content": "partial"})],
            "content_filter": [chunk({}, "content_filter")],
            "refusal": [chunk({"refusal": "Cannot comply"}, "stop")],
            "invalid_json": [chunk({"tool_calls": [call(arguments='{"broken":')]}, "tool_calls")],
            "nan_json": [chunk({"tool_calls": [call(arguments='{"value":NaN}')]}, "tool_calls")],
            "duplicate_ids": [chunk({"tool_calls": [call(), call(1)]}, "tool_calls")],
            "missing_id": [chunk({"tool_calls": [call(call_id=None)]}, "tool_calls")],
            "missing_name": [chunk({"tool_calls": [call(name=None)]}, "tool_calls")],
            "conflicting_id": [chunk({"tool_calls": [call()]}), chunk({"tool_calls": [call(call_id="other")]}, "tool_calls")],
            "conflicting_name": [chunk({"tool_calls": [call()]}), chunk({"tool_calls": [call(name="other")]}, "tool_calls")],
            "missing_index": [chunk({"tool_calls": [call(index=None)]}, "tool_calls")],
            "wrong_reason": [chunk({"tool_calls": [call()]}, "stop")],
            "late_content": [chunk({"content": "ready"}, "stop"), chunk({"content": "late"})],
            "late_reasoning": [chunk({"content": "ready"}, "stop"), chunk({"reasoning": "late"})],
            "late_tool_call": [chunk({"content": "ready"}, "stop"), chunk({"tool_calls": [call()]})],
            "conflicting_finish": [chunk({"content": "ready"}, "stop"), chunk({"role": "assistant"}, "length")],
            "invalid_trailing_role": [chunk({"content": "ready"}, "stop"), chunk({"role": "user"})],
            "bad_details": [chunk({"reasoning_details": "not a list"}, "stop")],
            "provider_error": [chunk({"content": "partial"}), RuntimeError("provider failed")],
        }
        for name, chunks in scenarios.items():
            with self.subTest(name=name):
                events, _ = await self.run_round(chunks)
                terminals = [event for event in events if event["event"] in ("done", "error")]
                self.assertEqual(len(terminals), 1)
                self.assertEqual(terminals[0]["event"], "error")
                self.assertNotIn("assistant_message", terminals[0]["data"])

    async def test_cancellation_closes_upstream_without_terminal_success(self):
        stream = FakeStream([chunk({"content": "Partial"}), asyncio.CancelledError()])
        service = OpenAIService()
        service.initialized = True
        service.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(return_value=stream))))
        events = []
        with patch("continuation_stream.openai_service", service):
            with self.assertRaises(asyncio.CancelledError):
                async for event in stream_tool_turn(
                    ToolStreamRequest(messages=[], tools=TOOLS, mode="native_tools", tool_protocol="continuation_v1"),
                    model="model", temperature=0.2, max_tokens=100, bot={},
                ):
                    events.append(event)
        self.assertEqual([event["event"] for event in events], ["token"])
        self.assertTrue(stream.closed)

    async def test_invalid_call_is_retained_only_as_nonexecutable_error_diagnostics(self):
        raw = '{"unfinished":'
        events, _ = await self.run_round([chunk({"tool_calls": [call(arguments=raw)]}, "tool_calls")])
        error = events[-1]
        self.assertEqual(error["event"], "error")
        self.assertNotIn("assistant_message", error["data"])
        self.assertEqual(
            error["data"]["partial_assistant_message"]["tool_calls"][0]["function"]["arguments"], raw,
        )

    async def test_route_rejects_unknown_protocol_instead_of_downgrading(self):
        with self.assertRaises(HTTPException) as error:
            await relay_app.stream(Request({"messages": [], "tool_protocol": "unknown"}))
        self.assertEqual(error.exception.status_code, 400)

    async def test_route_carries_opt_in_and_closes_stream_on_disconnect(self):
        for disconnected in (False, True):
            with self.subTest(disconnected=disconnected):
                stream = FakeStream([chunk({"content": "Text"}, "stop")])
                service = OpenAIService()
                service.initialized = True
                service.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(return_value=stream))))
                service.client.close = AsyncMock()
                service.initialize_with_config = AsyncMock()
                payload = {"bot_id": 9, "messages": [{"role": "user", "content": "Hello"}], "tool_protocol": "continuation_v1"}
                with (
                    patch("app.verify_jwt", AsyncMock(return_value=("user-1", "test-token"))),
                    patch("app.resolve_stream_bot", AsyncMock(return_value={"id": 9, "access_key": "key", "model": "model"})),
                    patch("app.OpenAIService", return_value=service),
                    patch("app.build_completion_request_kwargs", return_value={}),
                    patch("app.EventSourceResponse", side_effect=lambda generator, **kwargs: generator),
                    patch("continuation_stream.openai_service", service),
                ):
                    generator = await relay_app.stream(Request(payload, disconnect=disconnected))
                    events = [event async for event in generator]
                self.assertTrue(stream.closed)
                service.client.close.assert_awaited_once()
                self.assertEqual(service.initialize_with_config.call_args.kwargs["max_retries"], 0)
                self.assertEqual([event["event"] for event in events], [] if disconnected else ["token", "done"])
                if events:
                    self.assertEqual(json.loads(events[-1]["data"])["tool_protocol"], "continuation_v1")

    async def test_opt_in_requests_own_distinct_clients_even_when_streams_overlap(self):
        services = []
        streams = []

        def make_service():
            stream = FakeStream([chunk({"content": str(len(services))}), chunk({}, "stop")])
            service = OpenAIService()
            service.initialized = True
            service.initialize_with_config = AsyncMock()
            service.client = SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(return_value=stream))),
                close=AsyncMock(),
            )
            services.append(service)
            streams.append(stream)
            return service

        with (
            patch("app.verify_jwt", AsyncMock(return_value=("user-1", "test-token"))),
            patch("app.resolve_stream_bot", AsyncMock(side_effect=[
                {"access_key": "first-key", "model": "first-model"},
                {"access_key": "second-key", "model": "second-model"},
            ])),
            patch("app.OpenAIService", side_effect=make_service),
            patch("app.build_completion_request_kwargs", return_value={}),
            patch("app.EventSourceResponse", side_effect=lambda generator, **kwargs: generator),
        ):
            payload = {"messages": [{"role": "user", "content": "Hi"}], "tool_protocol": "continuation_v1"}
            first = await relay_app.stream(Request(payload))
            second = await relay_app.stream(Request(payload))
            self.assertEqual((await anext(first))["data"], "0")
            self.assertEqual((await anext(second))["data"], "1")
            first_done = [event async for event in first][-1]
            second_done = [event async for event in second][-1]
        for index, label in enumerate(("first", "second")):
            service = services[index]
            self.assertEqual(service.initialize_with_config.call_args.kwargs["api_key"], f"{label}-key")
            self.assertEqual(service.client.chat.completions.create.call_args.kwargs["model"], f"{label}-model")
            self.assertTrue(streams[index].closed)
            service.client.close.assert_awaited_once()
        self.assertEqual(json.loads(first_done["data"])["assistant_message"]["content"], "0")
        self.assertEqual(json.loads(second_done["data"])["assistant_message"]["content"], "1")

    async def test_sdk_retry_override_is_explicit_and_legacy_default_unchanged(self):
        with patch("openai_service.AsyncOpenAI") as constructor:
            await OpenAIService().initialize_with_config("test", max_retries=0)
            self.assertEqual(constructor.call_args.kwargs["max_retries"], 0)
            await OpenAIService().initialize_with_config("test")
            self.assertNotIn("max_retries", constructor.call_args.kwargs)
