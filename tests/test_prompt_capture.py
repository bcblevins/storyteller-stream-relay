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


    def test_user_attribute_normalizes_all_system_text_without_changing_forwarded_names(self):
        for marker in ('<capture-content user="Eryndor"/>', "<capture-content user='Eryndor'/>",
                       '<capture-content user = "Eryndor" />'):
            for role in ("system", "user", "assistant", "tool", "developer"):
                with self.subTest(marker=marker, role=role):
                    payload = {"messages": [
                        {"role": "system", "content": "  Eryndor greets Eryndor. eryndor stays lowercase.\n"},
                        {"role": "system", "content": [{"type": "text", "text": "Eryndor's actions"},
                            {"type": "image_url", "image_url": {"url": "Eryndor"}},
                            {"type": "text", "text": "Eryndor again."}]},
                        {"role": role, "content": "Eryndor" + marker},
                    ]}
                    original = deepcopy(payload)
                    cleaned, detected, system = extract_prompt_capture(payload)
                    self.assertTrue(detected)
                    self.assertEqual(payload, original)
                    self.assertEqual(cleaned["messages"][0], payload["messages"][0])
                    self.assertEqual(cleaned["messages"][1], payload["messages"][1])
                    self.assertEqual(cleaned["messages"][2]["content"], "Eryndor")
                    self.assertEqual(system[0]["text_parts"], ["  {{user}} greets {{user}}. eryndor stays lowercase.\n"])
                    self.assertEqual(system[1]["text_parts"], ["{{user}}'s actions", "{{user}} again."])
                    self.assertEqual(len(system), 3 if role == "system" else 2)
                    if role == "system":
                        self.assertEqual(system[2]["text_parts"], ["{{user}}"])

    def test_user_attribute_in_array_text_uses_first_nonempty_value_request_wide(self):
        payload = {"messages": [
            {"role": "system", "content": "Eryndor and Lyra"},
            {"role": "user", "content": [
                {"type": "text", "text": '<capture-content/><capture-content user=""/>'},
                {"type": "text", "text": '<capture-content user="Eryndor"/>Eryndor'},
                {"type": "text", "text": "<capture-content user='Lyra'/>Lyra"},
            ]},
            {"role": "system", "content": "Lyra then Eryndor"},
        ]}
        cleaned, detected, system = extract_prompt_capture(payload)
        self.assertTrue(detected)
        self.assertEqual(system, [{"message_index": 0, "text_parts": ["{{user}} and Lyra"]},
                                  {"message_index": 2, "text_parts": ["Lyra then {{user}}"]}])
        self.assertEqual([part["text"] for part in cleaned["messages"][1]["content"]], ["", "Eryndor", "Lyra"])

    def test_user_replacement_is_literal_and_case_sensitive(self):
        payload = {"messages": [
            {"role": "system", "content": "A.* A.*'s a.* Another"},
            {"role": "user", "content": '<capture-content user="A.*"/>'},
        ]}
        cleaned, _, system = extract_prompt_capture(payload)
        self.assertEqual(system[0]["text_parts"], ["{{user}} {{user}}'s a.* Another"])
        self.assertEqual(cleaned["messages"][0]["content"], "A.* A.*'s a.* Another")

    def test_empty_user_or_plain_marker_does_not_replace_text(self):
        for marker in ('<capture-content/>', '<capture-content user=""/>', "<capture-content user=''/>"):
            with self.subTest(marker=marker):
                cleaned, detected, system = extract_prompt_capture({"messages": [
                    {"role": "system", "content": "Eryndor {{user}}" + marker},
                ]})
                self.assertTrue(detected)
                self.assertEqual(system[0]["text_parts"], ["Eryndor {{user}}"])
                self.assertEqual(cleaned["messages"][0]["content"], "Eryndor {{user}}")

    def test_malformed_or_unsupported_user_attributes_do_not_trigger(self):
        for marker in ('<capture-content user=Eryndor/>', '<capture-content user="Eryndor\'/>',
                       '<capture-content username="Eryndor"/>', '<capture-content user="Eryndor" extra="x"/>',
                       '<capture-content user="Eryndor" user="Lyra"/>'):
            with self.subTest(marker=marker):
                payload = {"messages": [{"role": "system", "content": "Eryndor"}, {"role": "user", "content": marker}]}
                cleaned, detected, system = extract_prompt_capture(payload)
                self.assertFalse(detected)
                self.assertEqual(cleaned, payload)
                self.assertEqual(system[0]["text_parts"], ["Eryndor"])

    def test_persona_mode_captures_only_the_first_body_and_combines_with_user(self):
        text = (
            "private preamble\n<Alex O'Neil's Persona>\n  Alex meets <place>Lyra</place>.\n"
            "</Alex O'Neil's Persona><UserPersona>private user text</UserPersona>"
            "<Lyra's Persona>another persona</Lyra's Persona>"
        )
        for marker in ('<capture-content mode="persona" user="Alex"/>',
                       """<capture-content user='Alex' mode = 'persona' />"""):
            with self.subTest(marker=marker):
                payload = {"messages": [
                    {"role": "system", "content": "other system instructions"},
                    {"role": "system", "content": [
                        {"type": "text", "text": text},
                        {"type": "text", "text": "<Later's Persona>later text</Later's Persona>"},
                    ]},
                    {"role": "user", "content": "continue<capture-content/>" + marker},
                ]}
                cleaned, detected, system = extract_prompt_capture(payload)
                self.assertTrue(detected)
                self.assertEqual(system, [{
                    "message_index": 1,
                    "text_parts": ["\n  {{user}} meets <place>Lyra</place>.\n"],
                }])
                self.assertEqual(cleaned["messages"][:2], payload["messages"][:2])
                self.assertEqual(cleaned["messages"][2]["content"], "continue")

    def test_persona_mode_requires_a_matching_close_in_the_same_system_text_part(self):
        payload = {"messages": [
            {"role": "system", "content": [
                {"type": "text", "text": "<Alex's Persona>one</Lyra's Persona>"},
                {"type": "text", "text": "</Alex's Persona><Alex's Persona>two<Alex's Persona>"},
            ]},
            {"role": "user", "content": "<Alex's Persona>user text</Alex's Persona><capture-content mode='persona'/>"},
        ]}
        cleaned, detected, system = extract_prompt_capture(payload)
        self.assertTrue(detected)
        self.assertEqual(system, [])
        self.assertEqual(cleaned["messages"][0], payload["messages"][0])
        self.assertEqual(cleaned["messages"][1]["content"], "<Alex's Persona>user text</Alex's Persona>")

    def test_unsupported_and_duplicate_capture_modes_do_not_trigger(self):
        for marker in ('<capture-content mode="other"/>', '<capture-content mode=persona/>',
                       '<capture-content mode="persona" mode="persona"/>'):
            with self.subTest(marker=marker):
                payload = {"messages": [{"role": "user", "content": marker}]}
                self.assertEqual(extract_prompt_capture(payload), (payload, False, []))


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

    async def test_named_capture_submits_macro_but_forwards_original_names_in_both_modes(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                payload = {"model": "model", "stream": stream, "messages": [
                    {"role": "system", "content": "Eryndor greets Eryndor."},
                    {"role": "user", "content": '<capture-content user="Eryndor"/>Eryndor enters.'},
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
                        self.assertEqual([chunk async for chunk in response.body_iterator], [b"data: ok\n\n"])
                    else:
                        self.assertEqual(json.loads(response.body), {"answer": "ok"})
                    submit.assert_awaited_once_with("model", [{"message_index": 0, "text_parts": ["{{user}} greets {{user}}."]}])
                    self.assertEqual(forwarded[0]["messages"], [
                        {"role": "system", "content": "Eryndor greets Eryndor."},
                        {"role": "user", "content": "Eryndor enters."},
                    ])

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
