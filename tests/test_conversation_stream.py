import json
import unittest
from unittest.mock import AsyncMock, patch

import app as relay_app


class _StreamRequest:
    def __init__(self, disconnect_after_checks=None):
        self.disconnect_after_checks = disconnect_after_checks
        self.disconnect_checks = 0

    async def is_disconnected(self):
        self.disconnect_checks += 1
        return (
            self.disconnect_after_checks is not None
            and self.disconnect_checks >= self.disconnect_after_checks
        )


class ConversationStreamTests(unittest.IsolatedAsyncioTestCase):
    async def _run_stream(self, chunks, *, payload=None, request=None):
        payload = payload or {
            "conversation_id": 123,
            "messages": [{"role": "user", "content": "Tell me a story."}],
        }
        request = request or _StreamRequest()
        forwarded = {}

        async def fake_stream(**kwargs):
            forwarded.update(kwargs)
            for chunk in chunks:
                if isinstance(chunk, Exception):
                    raise chunk
                yield chunk

        with (
            patch("app.verify_jwt", AsyncMock(return_value=("user-1", "token"))),
            patch("app.resolve_stream_bot", AsyncMock(return_value={
                "id": 9,
                "access_key": "key",
                "model": "model",
                "temperature": 0.2,
            })),
            patch("app.openai_service.initialize_with_config", AsyncMock()),
            patch("app.openai_service.create_chat_completion_stream", fake_stream),
            patch("app.build_completion_request_kwargs", return_value={}),
            patch("app.EventSourceResponse", side_effect=lambda generator, **kwargs: generator),
            patch("supabase._rest_post", AsyncMock()) as rest_post_mock,
        ):
            generator = await relay_app._stream_with_mode(request, payload, mode="conversation")
            events = [event async for event in generator]

        return events, forwarded, rest_post_mock

    async def test_success_streams_tokens_and_done_without_persistence_ids(self):
        events, forwarded, rest_post_mock = await self._run_stream([
            {"reasoning": "thinking"},
            {"content": "Once "},
            {"content": "upon a time"},
        ])

        self.assertEqual([event["event"] for event in events], ["reasoning", "token", "token", "done"])
        self.assertEqual(events[1]["data"], "Once ")
        done_payload = json.loads(events[-1]["data"])
        self.assertEqual(done_payload, {})
        self.assertEqual(forwarded["messages"], [{"role": "user", "content": "Tell me a story."}])
        rest_post_mock.assert_not_awaited()

    async def test_model_error_is_terminal_and_not_followed_by_done(self):
        events, _, rest_post_mock = await self._run_stream([
            {"content": "Partial"},
            {"error": "provider failed"},
            {"content": "ignored"},
        ])

        self.assertEqual([event["event"] for event in events], ["token", "error"])
        self.assertEqual(json.loads(events[-1]["data"]), {"error": "provider failed"})
        rest_post_mock.assert_not_awaited()

    async def test_model_exception_is_terminal_error_not_done(self):
        events, _, rest_post_mock = await self._run_stream([
            {"content": "Partial"},
            RuntimeError("provider exploded"),
        ])

        self.assertEqual([event["event"] for event in events], ["token", "error"])
        self.assertEqual(json.loads(events[-1]["data"]), {"error": "provider exploded"})
        rest_post_mock.assert_not_awaited()

    async def test_disconnect_emits_nothing_and_writes_nothing(self):
        events, _, rest_post_mock = await self._run_stream(
            [{"content": "Partial"}],
            request=_StreamRequest(disconnect_after_checks=1),
        )

        self.assertEqual(events, [])
        rest_post_mock.assert_not_awaited()

    async def test_ignores_legacy_persistence_fields(self):
        payload = {
            "conversation_id": 123,
            "stream_id": "legacy-stream",
            "is_alternative": True,
            "alternative_id": 456,
            "messages": [{"role": "user", "content": "Regenerate this."}],
        }

        events, forwarded, rest_post_mock = await self._run_stream(
            [{"content": "Replacement"}],
            payload=payload,
        )

        self.assertEqual([event["event"] for event in events], ["token", "done"])
        self.assertEqual(json.loads(events[-1]["data"]), {})
        self.assertEqual(forwarded["messages"], payload["messages"])
        rest_post_mock.assert_not_awaited()
