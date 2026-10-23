import asyncio
from copy import deepcopy
import json
import unittest
from unittest.mock import AsyncMock, patch

import httpx
from fastapi import HTTPException
from starlette.requests import Request

import app as relay_app
import prompt_capture
from request_transforms import extract_prompt_capture


class CaptureTransformTests(unittest.TestCase):
    def test_markers_in_any_role_capture_only_ordered_system_text(self):
        for role in ("system", "user", "assistant", "tool", "developer"):
            with self.subTest(role=role):
                payload = {"messages": [
                    {"role": "system", "content": "  original <injection>keep</injection> \n"},
                    {"role": role, "content": "before<capture-content/>middle<capture-content />after"},
                    {"role": "system", "content": [{"type": "text", "text": "part 1"},
                        {"type": "image_url", "image_url": {"url": "image"}}, {"type": "text", "text": "part 2"}]},
                ]}
                original = deepcopy(payload)
                cleaned, detected, system = extract_prompt_capture(payload)
                self.assertTrue(detected)
                self.assertEqual(payload, original)
                self.assertEqual(cleaned["messages"][1]["content"], "beforemiddleafter")
                self.assertEqual(system[0], {"message_index": 0, "text_parts": [payload["messages"][0]["content"]]})
                self.assertEqual(system[-1], {"message_index": 2, "text_parts": ["part 1", "part 2"]})
                self.assertEqual(len(system), 3 if role == "system" else 2)

    def test_array_markers_strip_only_text_and_leave_metadata_and_images(self):
        payload = {"messages": [{"role": "user", "content": [
            {"type": "text", "text": "a<capture-content/>b", "extra": 1},
            {"type": "image_url", "image_url": {"url": "<capture-content/>"}},
        ], "name": "<capture-content/>"}]}
        cleaned, detected, system = extract_prompt_capture(payload)
        self.assertTrue(detected)
        self.assertEqual(cleaned["messages"][0]["content"][0], {"type": "text", "text": "ab", "extra": 1})
        self.assertEqual(cleaned["messages"][0]["content"][1], payload["messages"][0]["content"][1])
        self.assertEqual(cleaned["messages"][0]["name"], "<capture-content/>")
        self.assertEqual(system, [])

    def test_near_matches_split_tags_and_unrelated_fields_do_not_trigger(self):
        for content in ("<capture-content>", "<capture-content-other/>", "<CAPTURE-CONTENT/>",
                        "<capture-content attribute='x'/>",
                        [{"type": "text", "text": "<capture-"}, {"type": "text", "text": "content/>"}], None):
            payload = {"messages": [{"role": "user", "content": content}], "metadata": "<capture-content/>"}
            cleaned, detected, _ = extract_prompt_capture(payload)
            self.assertFalse(detected)
            self.assertEqual(cleaned, payload)
        self.assertEqual(extract_prompt_capture({"messages": None}), ({"messages": None}, False, []))


class CaptureSubmissionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.messages = [{"message_index": 0, "text_parts": ["private prompt"]}]
        self.secret = patch.object(prompt_capture.settings, "PROMPT_CAPTURE_SECRET", "test-secret")
        self.secret.start()
        self.addCleanup(self.secret.stop)

    async def test_posts_only_capture_fields_and_secret(self):
        client = AsyncMock()
        client.post.return_value = httpx.Response(200, json={"captured": True})
        with patch("prompt_capture.httpx.AsyncClient") as factory:
            factory.return_value.__aenter__.return_value = client
            await prompt_capture.submit_prompt_capture("model", self.messages)
        args = client.post.call_args
        self.assertTrue(args.args[0].endswith("/functions/v1/capture-prompt"))
        self.assertEqual(args.kwargs["headers"]["X-Prompt-Capture-Secret"], "test-secret")
        body = json.loads(args.kwargs["content"])
        self.assertEqual(set(body), {"request_id", "model", "system_messages"})
        self.assertEqual(body["system_messages"], self.messages)

    async def test_empty_oversized_or_unconfigured_capture_does_not_post(self):
        with patch("prompt_capture.httpx.AsyncClient") as factory:
            await prompt_capture.submit_prompt_capture("model", [])
            await prompt_capture.submit_prompt_capture("model", [{"message_index": 0, "text_parts": [" \n"]}])
            await prompt_capture.submit_prompt_capture("model", [{"message_index": 0, "text_parts": ["😀" * (1024 * 1024)]}])
            with patch.object(prompt_capture.settings, "PROMPT_CAPTURE_SECRET", None):
                await prompt_capture.submit_prompt_capture("model", self.messages)
            factory.assert_not_called()

    async def test_busy_storage_errors_and_timeout_are_nonfatal_and_logs_are_safe(self):
        for response in (httpx.Response(200, json={"captured": False}), httpx.Response(500, text="private prompt"),
                         httpx.ConnectError("private prompt")):
            client = AsyncMock()
            if isinstance(response, Exception):
                client.post.side_effect = response
            else:
                client.post.return_value = response
            with patch("prompt_capture.httpx.AsyncClient") as factory, self.assertLogs("relay", level="INFO") as logs:
                factory.return_value.__aenter__.return_value = client
                await prompt_capture.submit_prompt_capture("model", self.messages)
            self.assertNotIn("private prompt", str(logs.output))
            self.assertNotIn("test-secret", str(logs.output))
        client = AsyncMock()
        async def slow_post(*args, **kwargs):
            await asyncio.sleep(10)
        client.post.side_effect = slow_post
        with patch("prompt_capture.httpx.AsyncClient") as factory, patch("prompt_capture.CAPTURE_DEADLINE_SECONDS", 0.01):
            factory.return_value.__aenter__.return_value = client
            await prompt_capture.submit_prompt_capture("model", self.messages)

    async def test_cancellation_propagates(self):
        client = AsyncMock()
        client.post.side_effect = asyncio.CancelledError()
        with patch("prompt_capture.httpx.AsyncClient") as factory:
            factory.return_value.__aenter__.return_value = client
            with self.assertRaises(asyncio.CancelledError):
                await prompt_capture.submit_prompt_capture("model", self.messages)


class ExternalCaptureEndpointTests(unittest.IsolatedAsyncioTestCase):
    def request(self, payload):
        async def receive():
            return {"type": "http.request", "body": json.dumps(payload).encode(), "more_body": False}
        return Request({"type": "http", "method": "POST", "path": "/v1/chat/completions", "headers": []}, receive)

    async def test_both_forwarding_modes_capture_original_system_and_strip_controls(self):
        for stream in (False, True):
            payload = {"model": "model", "stream": stream, "messages": [
                {"role": "system", "content": "original<injection>move</injection>"},
                {"role": "user", "content": "hello<capture-content/><capture-content/>"},
            ]}
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
                  patch("app.httpx.AsyncClient", return_value=Client())):
                response = await relay_app.chat_completions_proxy(self.request(payload))
                if stream:
                    chunks = [chunk async for chunk in response.body_iterator]
                    self.assertEqual(chunks, [b"data: ok\n\n"])
                else:
                    self.assertEqual(json.loads(response.body), {"answer": "ok"})
                submit.assert_awaited_once_with("model", [{"message_index": 0, "text_parts": ["original<injection>move</injection>"]}])
                self.assertNotIn("<capture-content", json.dumps(forwarded))

    async def test_unauthorized_request_does_not_capture(self):
        with patch("app.verify_proxy_api_key", side_effect=HTTPException(401)), patch("app.submit_prompt_capture", AsyncMock()) as submit:
            with self.assertRaises(HTTPException):
                await relay_app.chat_completions_proxy(self.request({"messages": [{"role": "user", "content": "<capture-content/>"}]}))
            submit.assert_not_awaited()

    async def test_unmarked_request_does_not_submit(self):
        with (patch("app.verify_proxy_api_key", return_value="proxy-key"),
              patch("app.submit_prompt_capture", AsyncMock()) as submit,
              patch("app.httpx.AsyncClient") as factory):
            client = factory.return_value.__aenter__.return_value
            client.post.return_value = httpx.Response(200, json={})
            await relay_app.chat_completions_proxy(self.request({"messages": [{"role": "system", "content": "ordinary"}]}))
            submit.assert_not_awaited()


    async def test_main_application_stream_never_captures_markers(self):
        from test_conversation_stream import ConversationStreamTests
        runner = ConversationStreamTests()
        with patch("app.submit_prompt_capture", AsyncMock()) as submit:
            events, forwarded, _, _ = await runner._run_stream(
                [{"content": "ok"}], payload={"conversation_id": 123,
                    "messages": [{"role": "system", "content": "system<capture-content/>"}]},
            )
            submit.assert_not_awaited()
            self.assertEqual(forwarded["messages"][0]["content"], "system<capture-content/>")
            self.assertEqual(events[-1]["event"], "done")
