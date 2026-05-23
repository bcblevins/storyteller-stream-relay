import unittest
from types import SimpleNamespace

from openai_service import OpenAIService


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

    async def create(self, **kwargs):
        return _AsyncStream(self.chunks)


class _FakeClient:
    def __init__(self, chunks):
        self.chat = SimpleNamespace(completions=_FakeCompletions(chunks))


class OpenAIServiceReasoningTests(unittest.IsolatedAsyncioTestCase):
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


if __name__ == "__main__":
    unittest.main()
