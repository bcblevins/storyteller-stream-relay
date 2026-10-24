from copy import deepcopy
import json
import unittest
from unittest.mock import AsyncMock, patch

import httpx
from fastapi import HTTPException
from starlette.requests import Request

import app as relay_app
from request_transforms import TransformConfig, apply_system_extension_tag_transform


class SystemExtensionTransformTests(unittest.TestCase):
    def test_moves_first_block_from_any_role_to_last_system_without_mutating_input(self):
        for role in ("system", "user", "assistant", "tool", "developer"):
            with self.subTest(role=role):
                payload = {"messages": [
                    {"role": "system", "content": "first system", "name": "first"},
                    {"role": role, "content": "before <system-extension>  line one\nÉryndor\n</system-extension> after", "extra": 1},
                    {"role": "system", "content": "last system", "name": "last"},
                    {"role": "user", "content": "<system-extension>later</system-extension>"},
                ], "metadata": {"keep": True}}
                original = deepcopy(payload)
                result = apply_system_extension_tag_transform(payload)
                self.assertEqual(payload, original)
                self.assertEqual(result["messages"], [
                    original["messages"][0], {"role": role, "content": "before  after", "extra": 1},
                    {"role": "system", "content": "last system\n\n  line one\nÉryndor\n", "name": "last"},
                    original["messages"][3],
                ])
                self.assertEqual(result["metadata"], payload["metadata"])

    def test_only_first_block_in_a_message_is_consumed(self):
        result = apply_system_extension_tag_transform({"messages": [
            {"role": "system", "content": "base"},
            {"role": "user", "content": "<system-extension>first</system-extension><system-extension>second</system-extension>"},
        ]})
        self.assertEqual(result["messages"][0]["content"], "base\n\nfirst")
        self.assertEqual(result["messages"][1]["content"], "<system-extension>second</system-extension>")

    def test_scans_array_text_in_order_and_appends_after_all_system_parts(self):
        payload = {"messages": [
            {"role": "system", "content": [{"type": "text", "text": "one", "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "two"}, {"type": "image_url", "image_url": {"url": "image"}}]},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "<system-extension>ignore</system-extension>"}},
                {"type": "text", "text": "left<system-extension>append</system-extension>right", "extra": 1},
                {"type": "text", "text": "<system-extension>later</system-extension>"}]},
        ]}
        original = deepcopy(payload)
        result = apply_system_extension_tag_transform(payload)
        self.assertEqual(payload, original)
        self.assertEqual(result["messages"][0]["content"], [*original["messages"][0]["content"], {"type": "text", "text": "\n\nappend"}])
        self.assertEqual(result["messages"][1]["content"], [original["messages"][1]["content"][0],
            {"type": "text", "text": "leftright", "extra": 1}, original["messages"][1]["content"][2]])

    def test_prepends_system_when_absent(self):
        result = apply_system_extension_tag_transform({"messages": [
            {"role": "user", "content": "begin<system-extension>new system</system-extension>"},
        ]})
        self.assertEqual(result["messages"], [{"role": "system", "content": "new system"}, {"role": "user", "content": "begin"}])

    def test_moves_block_within_the_target_system_message(self):
        for content in ("before<system-extension>extra</system-extension>after",
                        [{"type": "text", "text": "before<system-extension>extra</system-extension>after"}, {"type": "text", "text": "last"}]):
            with self.subTest(content=content):
                result = apply_system_extension_tag_transform({"messages": [{"role": "system", "content": content}]})
                expected = "beforeafter\n\nextra" if isinstance(content, str) else [
                    {"type": "text", "text": "beforeafter"}, {"type": "text", "text": "last"}, {"type": "text", "text": "\n\nextra"}]
                self.assertEqual(result["messages"][0]["content"], expected)

    def test_empty_first_block_consumes_opportunity_without_creating_or_appending_system(self):
        for body in ("", " \n\t"):
            for system in ([], [{"role": "system", "content": "base"}]):
                with self.subTest(body=body, system=system):
                    result = apply_system_extension_tag_transform({"messages": [*system,
                        {"role": "user", "content": "before<system-extension>" + body + "</system-extension>after"},
                        {"role": "user", "content": "<system-extension>later</system-extension>"},
                    ]})
                    self.assertEqual(result["messages"], [*system, {"role": "user", "content": "beforeafter"},
                        {"role": "user", "content": "<system-extension>later</system-extension>"}])

    def test_appends_to_empty_system_text_without_leading_blank_lines(self):
        for content in ("", [], [{"type": "text", "text": ""}]):
            with self.subTest(content=content):
                result = apply_system_extension_tag_transform({"messages": [
                    {"role": "system", "content": content}, {"role": "user", "content": "<system-extension>extra</system-extension>"},
                ]})
                expected = "extra" if isinstance(content, str) else [*content, {"type": "text", "text": "extra"}]
                self.assertEqual(result["messages"][0]["content"], expected)

    def test_unmatched_unsupported_and_split_blocks_are_unchanged(self):
        for content in ("<system-extension>unclosed", "</system-extension>", "<system-extension/>",
                        '<system-extension extra="x">body</system-extension>', "<SYSTEM-EXTENSION>body</SYSTEM-EXTENSION>",
                        "<system-extension>outer<system-extension>inner</system-extension></system-extension>",
                        [{"type": "text", "text": "<system-extension>"}, {"type": "text", "text": "body</system-extension>"}]):
            with self.subTest(content=content):
                payload = {"messages": [{"role": "system", "content": "base"}, {"role": "user", "content": content}]}
                self.assertEqual(apply_system_extension_tag_transform(payload), payload)

    def test_no_match_or_unsupported_target_does_not_rewrite_payload(self):
        for payload in ({}, {"messages": None}, {"messages": []},
                        {"messages": [None, {"role": "user", "content": None}]},
                        {"messages": [{"role": "system", "content": None}, {"role": "user", "content": "<system-extension>extra</system-extension>"}]}):
            with self.subTest(payload=payload):
                self.assertEqual(apply_system_extension_tag_transform(payload), payload)


class SystemExtensionEndpointTests(unittest.IsolatedAsyncioTestCase):
    def request(self, payload):
        async def receive():
            return {"type": "http.request", "body": json.dumps(payload).encode(), "more_body": False}
        return Request({"type": "http", "method": "POST", "path": "/v1/chat/completions", "headers": []}, receive)

    async def forward(self, payload):
        forwarded = []
        class Client:
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
            async def aclose(self): pass
            async def post(self, url, **kwargs):
                forwarded.append(kwargs["json"])
                return httpx.Response(200, json={"answer": "ok"})
            def build_request(self, method, url, **kwargs):
                forwarded.append(kwargs["json"])
                return httpx.Request(method, url, json=kwargs["json"])
            async def send(self, request, **kwargs):
                return httpx.Response(200, content=b"data: ok\n\n")
        with (patch("app.verify_proxy_api_key", return_value="proxy-key"),
              patch("app.submit_prompt_capture", AsyncMock()) as submit,
              patch("app.httpx.AsyncClient", return_value=Client()),
              patch("app.build_transform_config", return_value=TransformConfig(
                  enable_system_injection_tag=True, enable_system_thinking_tag=True, force_reasoning_enabled=False))):
            response = await relay_app.chat_completions_proxy(self.request(payload))
            if payload.get("stream"):
                self.assertEqual([chunk async for chunk in response.body_iterator], [b"data: ok\n\n"])
            else:
                self.assertEqual(json.loads(response.body), {"answer": "ok"})
        return forwarded[0], submit

    async def test_extension_precedes_capture_macro_injection_thinking_and_provider_mapping(self):
        extension = "Eryndor speaks.<injection>extra Eryndor</injection><thinking>off</thinking>"
        for stream in (False, True):
            with self.subTest(stream=stream):
                payload = {"model": "model", "stream": stream, "messages": [
                    {"role": "system", "content": "Base Eryndor"},
                    {"role": "user", "content": '<capture-content user="Eryndor"/>begin<system-extension>' + extension + '</system-extension>end'},
                ]}
                original = deepcopy(payload)
                forwarded, submit = await self.forward(payload)
                self.assertEqual(payload, original)
                submit.assert_awaited_once_with("model", [{"message_index": 0, "text_parts": [
                    "Base {{user}}\n\n" + extension.replace("Eryndor", "{{user}}")]}])
                self.assertEqual(forwarded["messages"], [
                    {"role": "system", "content": "Base Eryndor\n\nEryndor speaks."},
                    {"role": "user", "content": "beginend\n\nextra Eryndor"},
                ])
                self.assertFalse(forwarded["reasoning"]["enabled"])

    async def test_extension_only_does_not_request_capture(self):
        forwarded, submit = await self.forward({"model": "model", "messages": [
            {"role": "user", "content": "begin<system-extension>new system</system-extension>"}]})
        self.assertEqual(forwarded["messages"], [{"role": "system", "content": "new system"}, {"role": "user", "content": "begin"}])
        submit.assert_not_awaited()

    async def test_capture_sees_new_system_message_when_original_request_has_none(self):
        _, submit = await self.forward({"model": "model", "messages": [
            {"role": "user", "content": "<capture-content/><system-extension>new system</system-extension>"}]})
        submit.assert_awaited_once_with("model", [{"message_index": 0, "text_parts": ["new system"]}])

    async def test_authentication_happens_before_extension_processing(self):
        with (patch("app.verify_proxy_api_key", side_effect=HTTPException(401)),
              patch("app.apply_system_extension_tag_transform") as transform):
            with self.assertRaises(HTTPException):
                await relay_app.chat_completions_proxy(self.request({"messages": []}))
            transform.assert_not_called()

    async def test_main_application_stream_leaves_system_extensions_untouched(self):
        from test_conversation_stream import ConversationStreamTests
        text = "base<system-extension>extra</system-extension>"
        with patch("app.apply_system_extension_tag_transform") as transform:
            events, forwarded, _, _ = await ConversationStreamTests()._run_stream(
                [{"content": "ok"}], payload={"conversation_id": 123, "messages": [{"role": "system", "content": text}]})
            transform.assert_not_called()
            self.assertEqual(forwarded["messages"][0]["content"], text)
            self.assertEqual(events[-1]["event"], "done")
