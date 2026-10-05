"""Opt-in Chat Completions continuation protocol; no application tool execution.

Assistant deltas are accumulated by the provider's wire rules. OpenRouter's
reasoning_details sequence is retained in received order, including opaque
blocks/signatures; DeepSeek reasoning_content and reasoning strings concatenate.
Only a successful terminal envelope is safe for the caller to act on.
"""
from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, AsyncGenerator

from openai_service import OpenAIService, openai_service


class ContinuationProtocolError(ValueError):
    pass


class AssistantAccumulator:
    def __init__(self):
        self.message: dict[str, Any] = {"role": "assistant", "content": None}
        self.calls: dict[int, dict[str, Any]] = {}

    def add(self, delta: dict[str, Any]) -> None:
        if not isinstance(delta, dict) or delta.get("role") not in (None, "assistant"):
            raise ContinuationProtocolError("Invalid assistant delta.")
        for field in ("content", "reasoning_content", "reasoning", "refusal"):
            value = delta.get(field)
            if value is not None:
                if not isinstance(value, str):
                    raise ContinuationProtocolError(f"Invalid assistant {field} delta.")
                self.message[field] = (self.message.get(field) or "") + value
        details = delta.get("reasoning_details")
        if details is not None:
            if not isinstance(details, list) or any(not isinstance(item, dict) for item in details):
                raise ContinuationProtocolError("Invalid reasoning_details sequence.")
            self.message.setdefault("reasoning_details", []).extend(deepcopy(details))

        calls = delta.get("tool_calls")
        if calls is not None:
            if not isinstance(calls, list):
                raise ContinuationProtocolError("Invalid tool_calls delta.")
            for call in calls:
                self._add_call(call)

    def _add_call(self, delta: dict[str, Any]) -> None:
        if not isinstance(delta, dict):
            raise ContinuationProtocolError("Invalid tool call delta.")
        index = delta.get("index")
        if type(index) is not int or index < 0:
            raise ContinuationProtocolError("Tool call delta requires a nonnegative index.")
        call = self.calls.setdefault(index, {"type": "function", "function": {"arguments": ""}})
        for field in ("id", "type"):
            value = delta.get(field)
            if value is not None:
                if not isinstance(value, str) or not value:
                    raise ContinuationProtocolError(f"Invalid tool call {field}.")
                if field in call and call[field] != value:
                    raise ContinuationProtocolError(f"Conflicting tool call {field}.")
                call[field] = value
        function = delta.get("function")
        if function is not None:
            if not isinstance(function, dict):
                raise ContinuationProtocolError("Invalid function delta.")
            name = function.get("name")
            if name is not None:
                if not isinstance(name, str) or not name:
                    raise ContinuationProtocolError("Invalid tool name.")
                if "name" in call["function"] and call["function"]["name"] != name:
                    raise ContinuationProtocolError("Conflicting tool name.")
                call["function"]["name"] = name
            arguments = function.get("arguments")
            if arguments is not None:
                if not isinstance(arguments, str):
                    raise ContinuationProtocolError("Tool arguments must be a JSON string.")
                call["function"]["arguments"] += arguments

    def complete(self, finish_reason: str | None) -> dict[str, Any]:
        expected = "tool_calls" if self.calls else "stop"
        if finish_reason != expected:
            raise ContinuationProtocolError("Provider response ended without a complete assistant turn.")
        if self.message.get("refusal"):
            raise ContinuationProtocolError("Provider refused the request.")
        if self.calls:
            calls = [self.calls[index] for index in sorted(self.calls)]
            ids: set[str] = set()
            for call in calls:
                call_id = call.get("id")
                if not call_id or call_id in ids or not call["function"].get("name"):
                    raise ContinuationProtocolError("Missing or duplicate tool call identity.")
                ids.add(call_id)
                try:
                    # Preserve the raw string. Parsing proves completeness only;
                    # argument schemas and tool permissions belong to the app.
                    json.loads(call["function"]["arguments"], parse_constant=_reject_json_constant)
                except (ValueError, TypeError):
                    raise ContinuationProtocolError("Tool call contains incomplete or invalid JSON.") from None
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        message = deepcopy(self.message)
        if self.calls:
            message["tool_calls"] = deepcopy([self.calls[index] for index in sorted(self.calls)])
        return message


def _reject_json_constant(value: str) -> None:
    raise ValueError("Non-JSON numeric constant.")


async def stream_continuation_turn(
    *, stream_id: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
    tool_choice: str | dict[str, Any] | None, model: str, temperature: float,
    max_tokens: int | None, bot: dict[str, Any], completion_kwargs: dict[str, Any],
    service: OpenAIService | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    accumulator = AssistantAccumulator()
    finish_reason = None
    usage = None
    provider_model = None
    started_tools = False
    upstream = (service or openai_service).create_chat_completion_tool_stream(
        messages=messages, model=model, temperature=temperature,
        max_tokens=max_tokens, bot_config=bot, tools=tools or None,
        tool_choice=(tool_choice or "auto") if tools else None,
        parallel_tool_calls=False if tools else None, **completion_kwargs,
    )
    try:
        async for chunk in upstream:
            if chunk.get("error"):
                raise ContinuationProtocolError(chunk["error"])
            if chunk.get("usage") is not None:
                usage = deepcopy(chunk["usage"])
            if chunk.get("provider_model"):
                provider_model = chunk["provider_model"]
            delta = chunk.get("assistant_delta") or {}
            # Some compatible providers repeat the assistant role on their final
            # usage chunk. Role-only metadata is not additional generated data.
            if finish_reason is not None and any(
                value for field, value in delta.items() if field != "role"
            ):
                raise ContinuationProtocolError("Assistant data arrived after the finish reason.")
            accumulator.add(delta)
            if chunk.get("finish_reason"):
                if finish_reason is not None and finish_reason != chunk["finish_reason"]:
                    raise ContinuationProtocolError("Conflicting finish reasons.")
                finish_reason = chunk["finish_reason"]
            if chunk.get("reasoning"):
                yield {"event": "reasoning", "data": chunk["reasoning"]}
            if delta.get("content"):
                yield {"event": "token", "data": delta["content"]}
            if accumulator.calls and not started_tools:
                started_tools = True
                first = accumulator.calls[min(accumulator.calls)]
                yield {"event": "tool_call_start", "data": {"tool_name": first["function"].get("name")}}
        message = accumulator.complete(finish_reason)
    except Exception as error:
        # CancelledError/GeneratorExit are not caught: disconnect emits nothing.
        yield {"event": "error", "data": {
            "error": str(error), "stream_id": stream_id,
            "tool_protocol": "continuation_v1", "finish_reason": finish_reason,
            "usage": usage, "partial_assistant_message": accumulator.snapshot(),
        }}
        return
    finally:
        await upstream.aclose()

    yield {"event": "done", "data": {
        "stream_id": stream_id, "tool_protocol": "continuation_v1",
        "status": "tool_calls" if accumulator.calls else "completed",
        "assistant_message": message, "finish_reason": finish_reason,
        "usage": usage, "model": model, "provider_model": provider_model,
    }}
