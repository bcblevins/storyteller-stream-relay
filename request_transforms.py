from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch
import re
from typing import Any


@dataclass(frozen=True)
class TransformConfig:
    force_reasoning_enabled: bool = False
    force_reasoning_effort: str = "high"
    force_reasoning_model_patterns: tuple[str, ...] = ("z-ai/glm-4.6:nitro",)
    force_reasoning_override: bool = False
    enable_system_injection_tag: bool = False
    system_injection_tag_name: str = "injection"


def _model_matches(model: str, patterns: tuple[str, ...]) -> bool:
    if not model:
        return False
    normalized_model = model.strip()
    return any(fnmatch(normalized_model, p.strip()) for p in patterns if p and p.strip())


def apply_provider_request_transforms(
    payload: dict[str, Any],
    provider: str,
    model: str,
    config: TransformConfig,
) -> dict[str, Any]:
    """Apply provider/model-specific request transforms without mutating input payload."""
    out = dict(payload)

    if provider != "openrouter":
        return out
    if not config.force_reasoning_enabled:
        return out
    if not _model_matches(model, config.force_reasoning_model_patterns):
        return out

    has_reasoning = isinstance(out.get("reasoning"), dict)
    if has_reasoning and not config.force_reasoning_override:
        return out

    reasoning: dict[str, Any] = {}
    if has_reasoning:
        reasoning = dict(out["reasoning"])

    reasoning["enabled"] = True
    if config.force_reasoning_effort:
        reasoning["effort"] = config.force_reasoning_effort

    out["reasoning"] = reasoning
    return out


def _extract_injection_blocks(text: str, tag_name: str) -> tuple[str, list[str]]:
    pattern = re.compile(
        rf"<\s*{re.escape(tag_name)}\s*>(.*?)<\s*/\s*{re.escape(tag_name)}\s*>",
        re.IGNORECASE | re.DOTALL,
    )
    captured: list[str] = [m.strip() for m in pattern.findall(text) if m and m.strip()]
    cleaned = pattern.sub("", text)
    return cleaned, captured


def _append_text_to_message_content(content: Any, appended: str) -> Any:
    if isinstance(content, str):
        joiner = "\n\n" if content.strip() else ""
        return f"{content}{joiner}{appended}"

    if isinstance(content, list):
        updated = []
        appended_to_text_part = False
        for part in content:
            if (
                not appended_to_text_part
                and isinstance(part, dict)
                and part.get("type") == "text"
                and isinstance(part.get("text"), str)
            ):
                text = part["text"]
                joiner = "\n\n" if text.strip() else ""
                updated.append({**part, "text": f"{text}{joiner}{appended}"})
                appended_to_text_part = True
            else:
                updated.append(part)

        if not appended_to_text_part:
            updated.append({"type": "text", "text": appended})
        return updated

    return content


def apply_system_injection_tag_transform(payload: dict[str, Any], config: TransformConfig) -> dict[str, Any]:
    """Extract <tag>...</tag> blocks from system messages and append to the latest message."""
    out = dict(payload)
    if not config.enable_system_injection_tag:
        return out

    tag_name = (config.system_injection_tag_name or "injection").strip()
    if not tag_name:
        return out

    messages = out.get("messages")
    if not isinstance(messages, list) or not messages:
        return out

    extracted_chunks: list[str] = []
    updated_messages = list(messages)

    for idx, msg in enumerate(updated_messages):
        if not isinstance(msg, dict) or msg.get("role") != "system":
            continue
        content = msg.get("content")
        if not isinstance(content, str):
            continue

        cleaned, captured = _extract_injection_blocks(content, tag_name)
        if captured:
            extracted_chunks.extend(captured)
            updated_messages[idx] = {**msg, "content": cleaned.strip()}

    if not extracted_chunks:
        return out

    injection_text = "\n\n".join(extracted_chunks)
    last_idx = len(updated_messages) - 1
    last_msg = updated_messages[last_idx]
    if not isinstance(last_msg, dict):
        return out

    updated_messages[last_idx] = {
        **last_msg,
        "content": _append_text_to_message_content(last_msg.get("content"), injection_text),
    }
    out["messages"] = updated_messages
    return out
