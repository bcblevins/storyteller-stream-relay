from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from fnmatch import fnmatch
import re
from typing import Any
from urllib.parse import urlsplit, urlunsplit


@dataclass(frozen=True)
class TransformConfig:
    force_reasoning_enabled: bool = True
    force_reasoning_effort: str | None = None
    force_reasoning_model_patterns: tuple[str, ...] = ("*",)
    force_reasoning_override: bool = False
    enable_system_injection_tag: bool = False
    system_injection_tag_name: str = "injection"


_OPENAI_REASONING_MODEL_PREFIXES = ("o1", "o3", "o4", "gpt-5")
_REASONING_REQUEST_FIELDS = ("reasoning", "reasoning_effort", "thinking", "extra_body")


def _model_matches(model: str, patterns: tuple[str, ...]) -> bool:
    if not model:
        return False
    normalized_model = model.strip()
    return any(fnmatch(normalized_model, p.strip()) for p in patterns if p and p.strip())


def normalize_completion_base_url(base_url: str | None) -> str | None:
    if not isinstance(base_url, str):
        return base_url

    candidate = base_url.strip()
    if not candidate:
        return candidate

    parsed = urlsplit(candidate)
    path = (parsed.path or "").rstrip("/")
    if path.lower().endswith("/chat/completions"):
        path = path[: -len("/chat/completions")]

    return urlunsplit((parsed.scheme, parsed.netloc, path, parsed.query, parsed.fragment))


def detect_completion_provider(
    *,
    provider: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    is_openrouter: bool = False,
) -> str | None:
    normalized_provider = (provider or "").strip().lower()
    if normalized_provider:
        return normalized_provider

    if is_openrouter:
        return "openrouter"

    normalized_base_url = normalize_completion_base_url(base_url)
    parsed = urlsplit(normalized_base_url or "")
    host = parsed.netloc.lower()

    if "openrouter.ai" in host:
        return "openrouter"
    if "anthropic.com" in host:
        return "anthropic"
    if "deepseek.com" in host:
        return "deepseek"
    if "openai.com" in host:
        return "openai"

    normalized_model = _normalized_model_name(model)
    if normalized_model.startswith(("anthropic/", "claude")):
        return "anthropic"
    if normalized_model.startswith(("deepseek/", "deepseek")):
        return "deepseek"
    if normalized_model.startswith(("openai/", *_OPENAI_REASONING_MODEL_PREFIXES, "gpt-")):
        return "openai"

    return None


def build_completion_request_kwargs(
    payload: dict[str, Any],
    *,
    provider: str | None,
    model: str,
    config: TransformConfig,
) -> dict[str, Any]:
    transformed = apply_provider_request_transforms(
        payload=_copy_reasoning_fields(payload),
        provider=provider or "",
        model=model,
        config=config,
    )

    kwargs: dict[str, Any] = {}
    reasoning_effort = transformed.get("reasoning_effort")
    if isinstance(reasoning_effort, str) and reasoning_effort.strip():
        kwargs["reasoning_effort"] = reasoning_effort.strip()

    extra_body = deepcopy(transformed.get("extra_body")) if isinstance(transformed.get("extra_body"), dict) else {}
    for key in ("reasoning", "thinking"):
        value = transformed.get(key)
        if isinstance(value, dict):
            extra_body[key] = deepcopy(value)

    if extra_body:
        kwargs["extra_body"] = extra_body

    return kwargs


def _copy_reasoning_fields(payload: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in _REASONING_REQUEST_FIELDS:
        if key in payload:
            out[key] = deepcopy(payload[key])
    return out


def _normalized_model_name(model: str | None) -> str:
    return (model or "").strip().lower()


def _openai_reasoning_effort(config: TransformConfig) -> str:
    effort = (config.force_reasoning_effort or "").strip().lower()
    return effort or "low"


def _has_explicit_reasoning_controls(payload: dict[str, Any], *keys: str) -> bool:
    if any(key in payload for key in keys):
        return True

    extra_body = payload.get("extra_body")
    if not isinstance(extra_body, dict):
        return False

    return any(key in extra_body for key in keys)


def _apply_openrouter_reasoning(
    payload: dict[str, Any],
    model: str,
    config: TransformConfig,
) -> dict[str, Any]:
    out = dict(payload)
    if not _model_matches(model, config.force_reasoning_model_patterns):
        return out

    has_reasoning = _has_explicit_reasoning_controls(out, "reasoning")
    if has_reasoning and not config.force_reasoning_override:
        return out

    reasoning: dict[str, Any] = {}
    if isinstance(out.get("reasoning"), dict):
        reasoning = dict(out["reasoning"])
    elif isinstance(out.get("extra_body"), dict) and isinstance(out["extra_body"].get("reasoning"), dict):
        reasoning = dict(out["extra_body"]["reasoning"])

    reasoning["enabled"] = True
    if config.force_reasoning_effort:
        reasoning["effort"] = config.force_reasoning_effort

    out["reasoning"] = reasoning
    return out


def _apply_openai_reasoning(
    payload: dict[str, Any],
    model: str,
    config: TransformConfig,
) -> dict[str, Any]:
    out = dict(payload)
    if not _is_openai_reasoning_model(model):
        return out
    if "reasoning_effort" in out and not config.force_reasoning_override:
        return out

    out["reasoning_effort"] = _openai_reasoning_effort(config)
    return out


def _apply_thinking_extra_body(
    payload: dict[str, Any],
    *,
    default_thinking: dict[str, Any],
    config: TransformConfig,
) -> dict[str, Any]:
    out = dict(payload)
    has_top_level_thinking = isinstance(out.get("thinking"), dict)
    has_extra_body_thinking = isinstance(out.get("extra_body"), dict) and isinstance(out["extra_body"].get("thinking"), dict)
    if (has_top_level_thinking or has_extra_body_thinking) and not config.force_reasoning_override:
        return out

    thinking_value = dict(default_thinking)
    if has_top_level_thinking:
        merged = dict(out["thinking"])
        merged.update(thinking_value)
        out["thinking"] = merged
        return out

    extra_body = dict(out.get("extra_body") or {})
    existing_thinking = extra_body.get("thinking")
    if isinstance(existing_thinking, dict):
        merged = dict(existing_thinking)
        merged.update(thinking_value)
        extra_body["thinking"] = merged
    else:
        extra_body["thinking"] = thinking_value
    out["extra_body"] = extra_body
    return out


def _apply_anthropic_thinking(payload: dict[str, Any], config: TransformConfig) -> dict[str, Any]:
    return _apply_thinking_extra_body(
        payload,
        default_thinking={"type": "enabled", "budget_tokens": 1024},
        config=config,
    )


def _apply_deepseek_thinking(payload: dict[str, Any], model: str, config: TransformConfig) -> dict[str, Any]:
    if "reasoner" in _normalized_model_name(model):
        return dict(payload)
    return _apply_thinking_extra_body(
        payload,
        default_thinking={"type": "enabled"},
        config=config,
    )


def _is_openai_reasoning_model(model: str) -> bool:
    normalized_model = _normalized_model_name(model)
    if "/" in normalized_model:
        _, normalized_model = normalized_model.split("/", 1)
    return normalized_model.startswith(_OPENAI_REASONING_MODEL_PREFIXES)


def apply_provider_request_transforms(
    payload: dict[str, Any],
    provider: str,
    model: str,
    config: TransformConfig,
) -> dict[str, Any]:
    """Apply provider/model-specific request transforms without mutating input payload."""
    out = dict(payload)

    if not config.force_reasoning_enabled:
        return out

    normalized_provider = (provider or "").strip().lower()
    if normalized_provider == "openrouter":
        return _apply_openrouter_reasoning(out, model, config)
    if normalized_provider == "openai":
        return _apply_openai_reasoning(out, model, config)
    if normalized_provider == "anthropic":
        return _apply_anthropic_thinking(out, config)
    if normalized_provider == "deepseek":
        return _apply_deepseek_thinking(out, model, config)

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
