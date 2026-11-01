import json
import unittest
from unittest.mock import AsyncMock, patch

import httpx
from openai import AsyncOpenAI
from starlette.requests import Request

import app as relay_app


MOCK_PATH = "/mock/v1/chat/completions"
EXPECTED_TEXT = "This is a mock streamed response."


class MockCompletionsTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        key_patch = patch.object(relay_app.settings, "GLM_PROXY_API_KEY", "test-proxy-key")
        key_patch.start()
        self.addCleanup(key_patch.stop)
        self.http = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=relay_app.app),
            base_url="http://relay.test",
            headers={"Authorization": "Bearer test-proxy-key"},
        )
        self.addAsyncCleanup(self.http.aclose)

    async def test_stream_has_standard_frames_and_one_successful_ending(self):
        response = await self.http.post(MOCK_PATH, json={
            "model": "test-model", "stream": True,
            "messages": [{"role": "user", "content": "Hello"}],
        })
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("text/event-stream"))
        self.assertEqual(response.headers["cache-control"], "no-cache")
        self.assertEqual(response.headers["x-accel-buffering"], "no")
        frames = response.text.split("\n\n")
        self.assertEqual(frames[-1], "")
        self.assertEqual(frames[-2], "data: [DONE]")
        self.assertEqual(frames.count("data: [DONE]"), 1)
        self.assertTrue(all(frame.startswith("data: ") for frame in frames[:-1]))
        chunks = [json.loads(frame.removeprefix("data: ")) for frame in frames[:-2]]
        self.assertGreater(len(chunks), 3)
        for chunk in chunks:
            self.assertEqual(chunk["id"], chunks[0]["id"])
            self.assertTrue(chunk["id"].startswith("chatcmpl-"))
            self.assertEqual(chunk["created"], chunks[0]["created"])
            self.assertIsInstance(chunk["created"], int)
            self.assertEqual(chunk["model"], "test-model")
            self.assertEqual(chunk["object"], "chat.completion.chunk")
            self.assertEqual(len(chunk["choices"]), 1)
            self.assertEqual(chunk["choices"][0]["index"], 0)
        choices = [chunk["choices"][0] for chunk in chunks]
        self.assertEqual(choices[0]["delta"], {"role": "assistant"})
        self.assertEqual("".join(choice["delta"].get("content", "") for choice in choices), EXPECTED_TEXT)
        self.assertTrue(all(choice["finish_reason"] is None for choice in choices[:-1]))
        self.assertEqual(choices[-1], {"index": 0, "delta": {}, "finish_reason": "stop"})

    async def test_openai_client_accepts_streaming_and_nonstreaming_responses(self):
        async with AsyncOpenAI(
            api_key="test-proxy-key", base_url="http://relay.test/mock/v1",
            http_client=self.http, max_retries=0,
        ) as client:
            stream = await client.chat.completions.create(
                model="sdk-test-model", messages=[{"role": "user", "content": "Hello"}], stream=True,
            )
            chunks = [chunk async for chunk in stream]
            self.assertEqual(chunks[0].choices[0].delta.role, "assistant")
            self.assertEqual("".join(chunk.choices[0].delta.content or "" for chunk in chunks), EXPECTED_TEXT)
            self.assertEqual(chunks[-1].choices[0].finish_reason, "stop")
            completion = await client.chat.completions.create(
                model="sdk-test-model", messages=[{"role": "user", "content": "Different input"}],
            )
            self.assertEqual(completion.object, "chat.completion")
            self.assertEqual(completion.model, "sdk-test-model")
            self.assertEqual(completion.choices[0].message.role, "assistant")
            self.assertEqual(completion.choices[0].message.content, EXPECTED_TEXT)
            self.assertEqual(completion.choices[0].finish_reason, "stop")

    async def test_default_and_explicit_nonstreaming_return_fixed_json(self):
        for payload in ({}, {"stream": False}):
            with self.subTest(payload=payload):
                response = await self.http.post(MOCK_PATH, json=payload)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.headers["content-type"], "application/json")
                body = response.json()
                self.assertEqual(body["model"], "mock-model")
                self.assertEqual(body["choices"][0]["message"]["content"], EXPECTED_TEXT)

    async def test_proxy_authentication_is_required_and_unconfigured_returns_503(self):
        for authorization in ("", "Basic test-proxy-key", "Bearer wrong-key"):
            with self.subTest(authorization=authorization):
                response = await self.http.post(
                    MOCK_PATH, json={"stream": True}, headers={"Authorization": authorization},
                )
                self.assertEqual(response.status_code, 401)
        with patch.object(relay_app.settings, "GLM_PROXY_API_KEY", None):
            response = await self.http.post(MOCK_PATH, json={"stream": True})
            self.assertEqual(response.status_code, 503)

    async def test_invalid_json_and_nonobject_bodies_return_400(self):
        for body in ("{", "null", "[]", '"text"'):
            with self.subTest(body=body):
                response = await self.http.post(MOCK_PATH, content=body, headers={"Content-Type": "application/json"})
                self.assertEqual(response.status_code, 400)

    async def test_messages_and_controls_never_trigger_outbound_calls_or_transforms(self):
        with (
            patch("app.httpx.AsyncClient") as outbound_client,
            patch("app.submit_prompt_capture", new_callable=AsyncMock) as capture,
            patch("app.build_transform_config") as transforms,
            patch("app.apply_system_extension_tag_transform") as extension,
        ):
            for stream in (False, True):
                response = await self.http.post(MOCK_PATH, json={
                    "stream": stream,
                    "messages": [{"role": "system", "content":
                        "private<capture-content/><system-extension>extra</system-extension>"}],
                    "temperature": 2, "max_tokens": 1,
                })
                self.assertEqual(response.status_code, 200)
            outbound_client.assert_not_called()
            capture.assert_not_awaited()
            transforms.assert_not_called()
            extension.assert_not_called()

    async def test_disconnect_stops_stream_without_terminal_frames(self):
        async def receive():
            return {"type": "http.request", "body": b'{"stream":true}', "more_body": False}

        request = Request({
            "type": "http", "method": "POST", "path": MOCK_PATH,
            "headers": [(b"authorization", b"Bearer test-proxy-key")],
        }, receive)
        for disconnect_checks in ([True], [False, False, True]):
            with self.subTest(disconnect_checks=disconnect_checks):
                request = Request(request.scope, receive)
                request.is_disconnected = AsyncMock(side_effect=disconnect_checks)
                response = await relay_app.mock_chat_completions(request)
                frames = [frame async for frame in response.body_iterator]
                self.assertEqual(len(frames), len(disconnect_checks) - 1)
                self.assertFalse(any('[DONE]' in frame or '"stop"' in frame for frame in frames))
