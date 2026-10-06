import unittest
import json
from types import SimpleNamespace

import httpx
from openai import AsyncOpenAI
from openai_service import OpenAIService
from request_transforms import TransformConfig, build_completion_request_kwargs


class _AsyncStream:
    def __init__(self, chunks):
        self.chunks = chunks

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for chunk in self.chunks:
            yield chunk


class _FakeCompletions:
    def __init__(self, chunks):
        self.chunks = chunks
        self.last_kwargs = None

    async def create(self, **kwargs):
        self.last_kwargs = kwargs
        return _AsyncStream(self.chunks)


class _FakeClient:
    def __init__(self, chunks):
        self.completions = _FakeCompletions(chunks)
        self.chat = SimpleNamespace(completions=self.completions)


class OpenAIServiceReasoningTests(unittest.IsolatedAsyncioTestCase):
    async def test_sdk_serializes_structured_output_and_provider_preferences_into_the_http_body(self):
        response_format = {"type": "json_schema", "json_schema": {
            "name": "entry", "strict": True, "schema": {"type": "object"},
        }}
        for provider in ("openrouter", "deepseek"):
            with self.subTest(provider=provider):
                requests = []

                def handle(request):
                    requests.append(json.loads(request.content))
                    event = {"id": "completion", "object": "chat.completion.chunk", "created": 0, "model": "model",
                             "choices": [{"index": 0, "delta": {"content": "{}"}, "finish_reason": "stop"}]}
                    return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                          content=f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n")

                async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http_client:
                    async with AsyncOpenAI(api_key="test-only", base_url="https://example.invalid/v1", http_client=http_client,
                                           max_retries=0) as client:
                        service = OpenAIService()
                        service.initialized = True
                        service.client = client
                        kwargs = build_completion_request_kwargs({"response_format": response_format}, provider=provider,
                                                                 model="model", config=TransformConfig(force_reasoning_enabled=False))
                        events = [event async for event in service.create_chat_completion_tool_stream(
                            messages=[{"role": "user", "content": "Return JSON."}], model="model", temperature=0.1,
                            max_tokens=100, **kwargs,
                        )]
                self.assertEqual(len(requests), 1)
                body = requests[0]
                self.assertEqual(body["response_format"], response_format if provider == "openrouter" else {"type": "json_object"})
                if provider == "openrouter":
                    self.assertEqual(body["provider"], {"require_parameters": True})
                else:
                    self.assertNotIn("provider", body)
                self.assertNotIn("extra_body", body)
                self.assertTrue(body["stream"])
                self.assertEqual(events[0]["content"], "{}")

    async def test_tool_stream_yields_reasoning_separately_from_content(self):
        chunks = [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta={"reasoning_content": "Plan first.", "content": None},
                        finish_reason=None,
                    )
                ],
                usage=None,
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta={"content": "Visible answer."},
                        finish_reason="stop",
                    )
                ],
                usage=SimpleNamespace(model_dump=lambda exclude_none=True: {"total_tokens": 12}),
            ),
        ]
        service = OpenAIService()
        service.initialized = True
        service.client = _FakeClient(chunks)

        events = [
            event
            async for event in service.create_chat_completion_tool_stream(
                messages=[{"role": "user", "content": "Hi"}],
                model="deepseek-chat",
                temperature=0.1,
                max_tokens=100,
            )
        ]

        self.assertEqual(events[0]["reasoning"], "Plan first.")
        self.assertIsNone(events[0]["content"])
        self.assertEqual(events[1]["content"], "Visible answer.")
        self.assertIsNone(events[1]["reasoning"])
        self.assertEqual(events[1]["finish_reason"], "stop")

    async def test_tool_stream_yields_anthropic_tool_use_start(self):
        chunks = [
            SimpleNamespace(
                type="content_block_start",
                content_block={"type": "tool_use", "id": "toolu_1", "name": "apply_patch"},
            ),
        ]
        service = OpenAIService()
        service.initialized = True
        service.client = _FakeClient(chunks)

        events = [
            event
            async for event in service.create_chat_completion_tool_stream(
                messages=[{"role": "user", "content": "Patch"}],
                model="claude-sonnet",
                temperature=0.1,
                max_tokens=100,
            )
        ]

        self.assertEqual(events, [{"tool_call_start": {"tool_name": "apply_patch"}, "error": None}])

    async def test_tool_stream_omits_non_positive_max_tokens(self):
        service = OpenAIService()
        service.initialized = True
        service.client = _FakeClient([])

        events = [
            event
            async for event in service.create_chat_completion_tool_stream(
                messages=[{"role": "user", "content": "Patch"}],
                model="deepseek-chat",
                temperature=0.1,
                max_tokens=0,
            )
        ]

        self.assertEqual(events, [])
        self.assertNotIn("max_tokens", service.client.completions.last_kwargs)

    async def test_tool_stream_includes_positive_max_tokens(self):
        service = OpenAIService()
        service.initialized = True
        service.client = _FakeClient([])

        events = [
            event
            async for event in service.create_chat_completion_tool_stream(
                messages=[{"role": "user", "content": "Patch"}],
                model="deepseek-chat",
                temperature=0.1,
                max_tokens=2048,
            )
        ]

        self.assertEqual(events, [])
        self.assertEqual(service.client.completions.last_kwargs["max_tokens"], 2048)


if __name__ == "__main__":
    unittest.main()
