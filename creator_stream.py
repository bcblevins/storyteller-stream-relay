import json
import logging
import time
from typing import Any, AsyncGenerator, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from openai_service import openai_service

log = logging.getLogger("relay.creator")


class CreatorStreamRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    messages: list[dict[str, Any]] = Field(default_factory=list)
    creator_session_id: int | None = None
    bot_id: int | None = None
    stream_id: str | None = None
    mode: Literal["text", "native_tools"] = "text"
    tools: list[dict[str, Any]] = Field(default_factory=list)
    tool_choice: str | dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_native_tools(self):
        if self.mode == "native_tools" and not self.tools:
            raise ValueError("tools are required when mode is native_tools")
        return self


class CreatorToolCallInput(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    raw_arguments: str | None = None


class CreatorContinuationRequest(CreatorStreamRequest):
    mode: Literal["native_tools"] = "native_tools"
    decision: Literal["approve", "reject", "retry"] | None = None
    tool_call: CreatorToolCallInput | None = None
    assistant_content: str | None = None
    tool_result: Any | None = None
    feedback: str | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_tool_call_payload(cls, data: Any):
        if not isinstance(data, dict):
            return data

        if data.get("tool_call") is not None:
            return data

        tool_call_id = data.get("tool_call_id")
        tool_name = data.get("tool_name")
        if not tool_call_id or not tool_name:
            return data

        normalized = dict(data)
        normalized["tool_call"] = {
            "id": tool_call_id,
            "name": tool_name,
            "arguments": data.get("arguments") or {},
            "raw_arguments": data.get("raw_arguments"),
        }
        return normalized

def _json_dumps(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True)


def _tool_arguments_to_dict(raw_arguments: Any) -> tuple[dict[str, Any], str]:
    if isinstance(raw_arguments, dict):
        return raw_arguments, _json_dumps(raw_arguments)

    if raw_arguments is None:
        return {}, "{}"

    if not isinstance(raw_arguments, str):
        log.warning("Tool call arguments had unexpected type %s", type(raw_arguments).__name__)
        return {}, _json_dumps(raw_arguments)

    text = raw_arguments or "{}"
    try:
        parsed = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        log.warning("Failed to parse tool call arguments as JSON: %s", text)
        return {}, text

    if isinstance(parsed, dict):
        return parsed, text

    log.warning("Tool call arguments were not a JSON object: %s", text)
    return {}, text


def _coerce_message_text(content: Any) -> str | None:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                text_parts.append(part["text"])
        if text_parts:
            return "\n".join(text_parts)
    return None
def _build_tool_call_event(message: dict[str, Any], stream_id: str, finish_reason: str | None, usage: dict[str, Any] | None):
    tool_calls = message.get("tool_calls") or []
    assistant_content = _coerce_message_text(message.get("content"))
    for index, tool_call in enumerate(tool_calls):
        function = tool_call.get("function") or {}
        raw_arguments = function.get("arguments")
        arguments, raw_text = _tool_arguments_to_dict(raw_arguments)
        yield {
            "event": "creator_tool_call",
            "data": {
                "stream_id": stream_id,
                "status": "awaiting_tool_approval",
                "mode": "native_tools",
                "sequence": index,
                "finish_reason": finish_reason,
                "usage": usage,
                "assistant_content": assistant_content,
                "tool_call": {
                    "id": tool_call.get("id"),
                    "name": function.get("name"),
                    "arguments": arguments,
                    "raw_arguments": raw_text,
                },
                "tool_call_id": tool_call.get("id"),
                "tool_name": function.get("name"),
                "arguments": arguments,
                "raw_arguments": raw_text,
            },
        }


def _tool_name_from_delta(tool_call_delta: dict[str, Any]) -> str | None:
    function = tool_call_delta.get("function") or {}
    name = function.get("name")
    return name if isinstance(name, str) and name else None


def _build_tool_call_start_event(tool_name: str | None) -> dict[str, Any]:
    return {
        "event": "creator_tool_call_start",
        "data": {"tool_name": tool_name},
    }


def _accumulate_tool_call_delta(final_tool_calls: dict[int, dict[str, Any]], tool_call_delta: dict[str, Any]):
    index = tool_call_delta.get("index")
    if not isinstance(index, int):
        index = len(final_tool_calls)

    current = final_tool_calls.setdefault(
        index,
        {
            "id": None,
            "type": "function",
            "function": {"name": None, "arguments": ""},
        },
    )

    if tool_call_delta.get("id"):
        current["id"] = tool_call_delta["id"]
    if tool_call_delta.get("type"):
        current["type"] = tool_call_delta["type"]

    incoming_function = tool_call_delta.get("function") or {}
    current_function = current.setdefault("function", {"name": None, "arguments": ""})
    if incoming_function.get("name"):
        current_function["name"] = incoming_function["name"]
    if incoming_function.get("arguments"):
        current_function["arguments"] = (current_function.get("arguments") or "") + incoming_function["arguments"]


def _final_tool_call_list(final_tool_calls: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    return [final_tool_calls[index] for index in sorted(final_tool_calls)]


async def stream_creator_native_tool_turn(
    request_payload: CreatorStreamRequest,
    *,
    model: str,
    temperature: float,
    max_tokens: int,
    bot: dict[str, Any],
    completion_kwargs: dict[str, Any] | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    stream_id = request_payload.stream_id or f"creator-stream-{int(time.time() * 1000)}"
    completion_kwargs = completion_kwargs or {}
    content_parts: list[str] = []
    final_tool_calls: dict[int, dict[str, Any]] = {}
    has_emitted_tool_call_start = False
    usage = None
    finish_reason = None

    async for chunk in openai_service.create_chat_completion_tool_stream(
        messages=request_payload.messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        bot_config=bot,
        tools=request_payload.tools,
        tool_choice=request_payload.tool_choice or "auto",
        parallel_tool_calls=False,
        **completion_kwargs,
    ):
        if chunk.get("error"):
            yield {
                "event": "error",
                "data": {
                    "error": chunk["error"],
                    "stream_id": stream_id,
                    "mode": request_payload.mode,
                },
            }
            return

        if chunk.get("usage"):
            usage = chunk["usage"]
        if chunk.get("finish_reason"):
            finish_reason = chunk["finish_reason"]

        reasoning_chunk = chunk.get("reasoning")
        if reasoning_chunk:
            yield {"event": "reasoning", "data": reasoning_chunk}

        content_chunk = chunk.get("content")
        if content_chunk:
            content_parts.append(content_chunk)
            yield {"event": "token", "data": content_chunk}

        tool_call_start = chunk.get("tool_call_start") or {}
        if tool_call_start and not has_emitted_tool_call_start:
            yield _build_tool_call_start_event(tool_call_start.get("tool_name"))
            has_emitted_tool_call_start = True

        for tool_call_delta in chunk.get("tool_calls") or []:
            if not has_emitted_tool_call_start:
                yield _build_tool_call_start_event(_tool_name_from_delta(tool_call_delta))
                has_emitted_tool_call_start = True
            _accumulate_tool_call_delta(final_tool_calls, tool_call_delta)

    content = "".join(content_parts)
    tool_calls = _final_tool_call_list(final_tool_calls)
    message = {"content": content, "tool_calls": tool_calls}

    if tool_calls:
        if len(tool_calls) > 1:
            log.warning(
                "Creator native tool turn produced multiple tool calls; rejecting turn - stream_id: %s, tool_call_count: %d",
                stream_id,
                len(tool_calls),
            )
            yield {
                "event": "error",
                "data": {
                    "error": (
                        "Creator native tool turns must contain exactly one tool call. "
                        "Received multiple tool calls in a single assistant turn."
                    ),
                    "stream_id": stream_id,
                    "mode": request_payload.mode,
                    "tool_call_count": len(tool_calls),
                },
            }
            return
        for event in _build_tool_call_event(message, stream_id, finish_reason, usage):
            yield event
        yield {
            "event": "done",
            "data": {
                "stream_id": stream_id,
                "status": "awaiting_tool_approval",
                "mode": request_payload.mode,
                "finish_reason": finish_reason,
                "tool_call_count": len(tool_calls),
                "usage": usage,
            },
        }
        return

    yield {
        "event": "done",
        "data": {
            "stream_id": stream_id,
            "status": "completed",
            "mode": request_payload.mode,
            "finish_reason": finish_reason,
            "usage": usage,
        },
    }
