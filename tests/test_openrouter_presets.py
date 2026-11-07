from copy import deepcopy
import json
import unittest
from unittest.mock import AsyncMock, patch

import httpx
from openai import AsyncOpenAI

import app as relay_app
from openai_service import OpenAIService
from request_transforms import (
    TransformConfig,
    apply_openrouter_preset_tag_transform,
    build_completion_request_kwargs,
)


CONFIG = TransformConfig(enable_openrouter_preset_tag=True)
TAG = "<openrouter-preset>storytelling</openrouter-preset>"
TOOLS = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]


class PresetTransformTests(unittest.TestCase):
    def transform(self, payload, **kwargs):
        return apply_openrouter_preset_tag_transform(payload, provider="openrouter", config=CONFIG, **kwargs)

    def test_normalizes_slug_and_reference_without_mutating_input_or_other_parameters(self):
        for value in ("storytelling", "  @preset/storytelling\n"):
            with self.subTest(value=value):
                payload = {"model": "vendor/model", "temperature": 0, "messages": [
                    {"role": "system", "content": f"Before\n<openrouter-preset>{value}</openrouter-preset>After", "name": "keep"},
                    {"role": "user", "content": TAG},
                    {"role": "assistant", "content": TAG},
                ]}
                original = deepcopy(payload)
                result = self.transform(payload)
                self.assertEqual(payload, original)
                self.assertEqual(result, {**original, "preset": "@preset/storytelling", "messages": [
                    {"role": "system", "content": "Before\nAfter", "name": "keep"},
                    *original["messages"][1:],
                ]})

    def test_last_valid_tag_wins_across_system_messages_and_text_parts(self):
        invalid = "<openrouter-preset>bad/slug</openrouter-preset>"
        payload = {"messages": [
            {"role": "system", "content": TAG},
            {"role": "system", "content": [
                {"type": "text", "text": "<OPENROUTER-PRESET>Second_2</OPENROUTER-PRESET>", "cache_control": {"type": "ephemeral"}},
                {"type": "image_url", "image_url": {"url": TAG}},
                {"type": "text", "text": "<openrouter-preset>@preset/last-one</openrouter-preset>" + invalid},
            ]},
            {"role": "tool", "content": TAG},
        ]}
        original = deepcopy(payload)
        result = self.transform(payload)
        self.assertEqual(payload, original)
        self.assertEqual(result["preset"], "@preset/last-one")
        self.assertEqual(result["messages"][0]["content"], "")
        self.assertEqual(result["messages"][1]["content"], [
            {"type": "text", "text": "", "cache_control": {"type": "ephemeral"}},
            original["messages"][1]["content"][1], {"type": "text", "text": invalid},
        ])
        self.assertEqual(result["messages"][2], original["messages"][2])

    def test_explicit_top_level_or_extra_body_preset_wins_but_tag_is_removed(self):
        for fields in ({"preset": "@preset/explicit"}, {"extra_body": {"preset": "@preset/explicit"}}):
            with self.subTest(fields=fields):
                result = self.transform({**fields, "messages": [{"role": "system", "content": TAG}]})
                self.assertEqual(result, {**fields, "messages": [{"role": "system", "content": ""}]})

    def test_invalid_empty_nested_unclosed_and_split_tags_are_unchanged(self):
        for content in (
            "<openrouter-preset></openrouter-preset>",
            "<openrouter-preset> </openrouter-preset>",
            "<openrouter-preset>bad slug</openrouter-preset>",
            "<openrouter-preset>bad/slug</openrouter-preset>",
            "<openrouter-preset>@preset/</openrouter-preset>",
            "<openrouter-preset>unclosed", "<openrouter-preset/>",
            '<openrouter-preset name="storytelling"/>',
            "<openrouter-preset>" + TAG + "</openrouter-preset>",
            [{"type": "text", "text": "<openrouter-preset>"},
             {"type": "text", "text": "storytelling</openrouter-preset>"}],
        ):
            with self.subTest(content=content):
                payload = {"messages": [{"role": "system", "content": content}]}
                self.assertEqual(self.transform(payload), payload)
        for payload in ({}, {"messages": None}, {"messages": [None, {}, {"role": "system", "content": None}]}):
            self.assertEqual(self.transform(payload), payload)

    def test_provider_and_feature_flag_gate_the_transform_and_tag_name_is_configurable(self):
        payload = {"messages": [{"role": "system", "content": TAG}]}
        for provider, config in ((None, CONFIG), ("deepseek", CONFIG), ("openrouter", TransformConfig())):
            self.assertEqual(apply_openrouter_preset_tag_transform(payload, provider=provider, config=config), payload)
        result = apply_openrouter_preset_tag_transform(
            {"messages": [{"role": "system", "content": "<preset>Custom_1</preset>"}]},
            provider="openrouter", config=TransformConfig(enable_openrouter_preset_tag=True, openrouter_preset_tag_name="preset"),
        )
        self.assertEqual(result["preset"], "@preset/Custom_1")

    def test_sdk_kwargs_merge_preset_with_reasoning_and_structured_output(self):
        payload = {"preset": "@preset/storytelling", "reasoning": {"enabled": False},
                   "response_format": {"type": "json_object"},
                   "extra_body": {"preset": "@preset/other", "provider": {"only": ["example"]}, "metadata": {"keep": True}}}
        original = deepcopy(payload)
        result = build_completion_request_kwargs(payload, provider="openrouter", model="vendor/model", config=CONFIG)
        self.assertEqual(result, {"response_format": {"type": "json_object"}, "extra_body": {
            "preset": "@preset/storytelling", "reasoning": {"enabled": False},
            "provider": {"only": ["example"], "require_parameters": True}, "metadata": {"keep": True},
        }})
        result["extra_body"]["metadata"]["keep"] = False
        self.assertEqual(payload, original)
        direct = build_completion_request_kwargs({"preset": "@preset/storytelling"}, provider="deepseek",
                                                 model="model", config=TransformConfig(force_reasoning_enabled=False))
        self.assertEqual(direct, {})


class Request:
    def __init__(self, payload):
        self.payload = payload

    async def json(self):
        return self.payload

    async def is_disconnected(self):
        return False


class PresetEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_all_stream_modes_serialize_preset_and_preserve_model_defaults_and_explicit_settings(self):
        for continuation, tools in ((False, []), (False, TOOLS), (True, []), (True, TOOLS)):
            for explicit in (False, True):
                with self.subTest(continuation=continuation, tools=bool(tools), explicit=explicit):
                    forwarded = []

                    def handle(request):
                        forwarded.append(json.loads(request.content))
                        event = {"id": "completion", "object": "chat.completion.chunk", "created": 0, "model": "vendor/model",
                                 "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": "stop"}]}
                        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                              content=f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n")

                    payload = {"messages": [{"role": "system", "content": "Base " + TAG},
                                            {"role": "user", "content": "Hello"}]}
                    if tools:
                        payload["tools"] = tools
                    if continuation:
                        payload["tool_protocol"] = "continuation_v1"
                    if explicit:
                        payload.update(preset="@preset/explicit", temperature=0, reasoning={"enabled": False})
                    original = deepcopy(payload)
                    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http_client:
                        async with AsyncOpenAI(api_key="test-only", base_url="https://openrouter.ai/api/v1",
                                               http_client=http_client, max_retries=0) as client:
                            service = OpenAIService()
                            service.client = client
                            service.initialized = True
                            service.initialize_with_config = AsyncMock()
                            with (
                                patch("app.verify_jwt", AsyncMock(return_value=("user", "test-only"))),
                                patch("app.resolve_stream_bot", AsyncMock(return_value={
                                    "access_key": "test-only", "access_path": "https://openrouter.ai/api/v1",
                                    "model": "vendor/model", "temperature": 0.4, "max_tokens": 200,
                                })),
                                patch("app.build_transform_config", return_value=CONFIG),
                                patch("app.openai_service", service),
                                patch("tool_stream.openai_service", service),
                                patch("app.OpenAIService", return_value=service),
                                patch("app.EventSourceResponse", side_effect=lambda generator, **kwargs: generator),
                            ):
                                generator = await relay_app.stream(Request(payload))
                                events = [event async for event in generator]
                    self.assertEqual(payload, original)
                    self.assertEqual([event["event"] for event in events], ["token", "done"])
                    self.assertEqual(len(forwarded), 1)
                    body = forwarded[0]
                    self.assertEqual(body["preset"], "@preset/explicit" if explicit else "@preset/storytelling")
                    self.assertEqual(body["model"], "vendor/model")
                    self.assertEqual(body["temperature"], 0 if explicit else 0.4)
                    self.assertEqual(body["max_tokens"], 200)
                    self.assertEqual(body["reasoning"]["enabled"], not explicit)
                    self.assertEqual(body["messages"], [{"role": "system", "content": "Base "}, original["messages"][1]])
                    self.assertNotIn("extra_body", body)
                    if tools:
                        self.assertEqual(body["tools"], TOOLS)

    async def test_proxy_applies_extension_then_capture_then_preset_for_json_and_sse(self):
        for stream in (False, True):
            for explicit in (False, True):
                with self.subTest(stream=stream, explicit=explicit):
                    forwarded = []

                    def handle(request):
                        forwarded.append(json.loads(request.content))
                        return httpx.Response(200, content=b"data: ok\n\n" if stream else b'{"answer":"ok"}',
                                              headers={"content-type": "text/event-stream" if stream else "application/json"})

                    extension = "Base " + TAG + "<thinking>off</thinking>"
                    payload = {"model": "vendor/model", "stream": stream, "messages": [
                        {"role": "user", "content": "<capture-content/><system-extension>" + extension + "</system-extension>Hello"},
                    ]}
                    if explicit:
                        payload["preset"] = "@preset/explicit"
                    upstream = httpx.AsyncClient(transport=httpx.MockTransport(handle))
                    try:
                        with (
                            patch("app.verify_proxy_api_key", return_value="test-only"),
                            patch("app.submit_prompt_capture", AsyncMock()) as capture,
                            patch("app.httpx.AsyncClient", return_value=upstream),
                            patch("app.build_transform_config", return_value=TransformConfig(
                                enable_openrouter_preset_tag=True, enable_system_thinking_tag=True)),
                        ):
                            response = await relay_app.chat_completions_proxy(Request(payload))
                            if stream:
                                self.assertEqual([part async for part in response.body_iterator], [b"data: ok\n\n"])
                            else:
                                self.assertEqual(json.loads(response.body), {"answer": "ok"})
                    finally:
                        await upstream.aclose()
                    capture.assert_awaited_once_with("vendor/model", [{"message_index": 0, "text_parts": [extension]}])
                    self.assertEqual(forwarded[0]["preset"], "@preset/explicit" if explicit else "@preset/storytelling")
                    self.assertEqual(forwarded[0]["model"], "vendor/model")
                    self.assertEqual(forwarded[0]["messages"], [{"role": "system", "content": "Base"},
                                                                {"role": "user", "content": "Hello"}])
                    self.assertFalse(forwarded[0]["reasoning"]["enabled"])
