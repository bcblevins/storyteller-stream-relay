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
    enable_system_thinking_tag: bool = False
    system_thinking_tag_name: str = "thinking"
    enable_openrouter_preset_tag: bool = False
    openrouter_preset_tag_name: str = "openrouter-preset"


_OPENAI_REASONING_MODEL_PREFIXES = ("o1", "o3", "o4", "gpt-5")
_THINKING_CONTROL_KEY = "_relay_thinking_control"
_REASONING_REQUEST_FIELDS = (
    "reasoning",
    "reasoning_effort",
    "thinking",
    "extra_body",
    _THINKING_CONTROL_KEY,
)
_THINKING_ENABLED_VALUES = {"enabled", "enable", "on", "true", "yes", "1"}
_THINKING_DISABLED_VALUES = {"disabled", "disable", "off", "false", "no", "0", "none"}
_THINKING_EFFORT_VALUES = {"minimal", "low", "medium", "high", "xhigh", "max"}


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

    if provider == "openrouter" and "preset" in payload:
        extra_body["preset"] = deepcopy(payload["preset"])

    if "response_format" in payload:
        response_format = validate_response_format(payload["response_format"])
        # Direct DeepSeek supports JSON mode, not JSON Schema. The app remains
        # responsible for validating its domain schema after generation.
        if provider == "deepseek" and response_format["type"] == "json_schema":
            response_format = {"type": "json_object"}
        kwargs["response_format"] = response_format
        extra_body.pop("response_format", None)
        if provider == "openrouter":
            preferences = extra_body.get("provider")
            preferences = deepcopy(preferences) if isinstance(preferences, dict) else {}
            preferences["require_parameters"] = True
            extra_body["provider"] = preferences

    if extra_body:
        kwargs["extra_body"] = extra_body

    return kwargs


def validate_response_format(value: Any) -> dict[str, Any]:
    """Validate the optional /v1/stream structured-output contract, not its schema."""
    if not isinstance(value, dict):
        raise ValueError("response_format must be an object")
    if value.get("type") == "json_object" and set(value) == {"type"}:
        return deepcopy(value)
    if value.get("type") == "json_schema" and set(value) == {"type", "json_schema"}:
        specification = value["json_schema"]
        if (
            isinstance(specification, dict)
            and set(specification) == {"name", "strict", "schema"}
            and isinstance(specification["name"], str)
            and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", specification["name"])
            and isinstance(specification["strict"], bool)
            and isinstance(specification["schema"], dict)
        ):
            return deepcopy(value)
    raise ValueError("response_format must be json_object or a named json_schema with boolean strict and object schema")


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


def _apply_openrouter_thinking_control(
    payload: dict[str, Any],
    control: dict[str, Any],
    config: TransformConfig,
) -> dict[str, Any]:
    out = dict(payload)
    has_reasoning = _has_explicit_reasoning_controls(out, "reasoning")
    if has_reasoning and not config.force_reasoning_override:
        return out

    reasoning: dict[str, Any] = {}
    if isinstance(out.get("reasoning"), dict):
        reasoning = dict(out["reasoning"])
    elif isinstance(out.get("extra_body"), dict) and isinstance(out["extra_body"].get("reasoning"), dict):
        reasoning = dict(out["extra_body"]["reasoning"])

    enabled = bool(control.get("enabled"))
    reasoning["enabled"] = enabled
    effort = control.get("effort")
    if isinstance(effort, str) and effort:
        reasoning["effort"] = effort
    elif not enabled:
        reasoning["effort"] = "none"

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


def _apply_deepseek_thinking_control(
    payload: dict[str, Any],
    control: dict[str, Any],
    config: TransformConfig,
) -> dict[str, Any]:
    out = dict(payload)
    has_thinking = _has_explicit_reasoning_controls(out, "thinking", "reasoning_effort")
    if has_thinking and not config.force_reasoning_override:
        return out

    enabled = bool(control.get("enabled"))
    extra_body = dict(out.get("extra_body") or {})
    thinking = dict(extra_body.get("thinking") or {})
    thinking["type"] = "enabled" if enabled else "disabled"
    extra_body["thinking"] = thinking
    out["extra_body"] = extra_body

    effort = control.get("effort")
    if enabled and isinstance(effort, str) and effort:
        out["reasoning_effort"] = effort
    elif not enabled:
        out.pop("reasoning_effort", None)

    return out


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
    thinking_control = out.pop(_THINKING_CONTROL_KEY, None)

    if isinstance(thinking_control, dict):
        normalized_provider = (provider or "").strip().lower()
        if normalized_provider == "openrouter":
            return _apply_openrouter_thinking_control(out, thinking_control, config)
        if normalized_provider == "deepseek":
            return _apply_deepseek_thinking_control(out, thinking_control, config)
        return out

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


def apply_openrouter_preset_tag_transform(
    payload: dict[str, Any],
    *,
    provider: str | None,
    config: TransformConfig,
) -> dict[str, Any]:
    """Consume valid preset tags in system text, keeping explicit request fields."""
    out = dict(payload)
    if (provider or "").strip().lower() != "openrouter" or not config.enable_openrouter_preset_tag:
        return out

    tag_name = config.openrouter_preset_tag_name.strip()
    messages = out.get("messages")
    if not tag_name or not isinstance(messages, list):
        return out

    pattern = re.compile(
        rf"<\s*{re.escape(tag_name)}\s*>(.*?)<\s*/\s*{re.escape(tag_name)}\s*>",
        re.IGNORECASE | re.DOTALL,
    )
    selected_preset = None

    def consume(match: re.Match[str]) -> str:
        nonlocal selected_preset
        slug = match.group(1).strip()
        if slug.startswith("@preset/"):
            slug = slug[len("@preset/"):]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", slug):
            return match.group(0)
        selected_preset = f"@preset/{slug}"
        return ""

    updated_messages = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "system":
            updated_messages.append(message)
            continue
        content = message.get("content")
        if isinstance(content, str):
            content = pattern.sub(consume, content)
        elif isinstance(content, list):
            content = [
                {**part, "text": pattern.sub(consume, part["text"])}
                if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str)
                else part
                for part in content
            ]
        updated_messages.append({**message, "content": content} if "content" in message else message)

    if selected_preset is not None:
        out["messages"] = updated_messages
        extra_body = out.get("extra_body")
        if "preset" not in out and not (isinstance(extra_body, dict) and "preset" in extra_body):
            out["preset"] = selected_preset
    return out


def _parse_thinking_control(raw_value: str) -> dict[str, Any] | None:
    normalized = raw_value.strip().lower()
    if not normalized:
        return None

    parts = [part.strip() for part in re.split(r"[:=,\s]+", normalized) if part.strip()]
    if not parts:
        return None

    enabled: bool | None = None
    effort: str | None = None
    for part in parts:
        if part in _THINKING_ENABLED_VALUES:
            enabled = True
            continue
        if part in _THINKING_DISABLED_VALUES:
            enabled = False
            continue
        if part in _THINKING_EFFORT_VALUES:
            effort = part
            enabled = True if enabled is None else enabled

    if enabled is None:
        return None

    return {"enabled": enabled, "effort": effort}


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


def apply_system_thinking_tag_transform(payload: dict[str, Any], config: TransformConfig) -> dict[str, Any]:
    """Extract <thinking>...</thinking> controls from system messages for provider mapping."""
    out = dict(payload)
    if not config.enable_system_thinking_tag:
        return out

    tag_name = (config.system_thinking_tag_name or "thinking").strip()
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

    control = None
    for chunk in extracted_chunks:
        parsed = _parse_thinking_control(chunk)
        if parsed:
            control = parsed

    out["messages"] = updated_messages
    if control:
        out[_THINKING_CONTROL_KEY] = control
    return out


_SYSTEM_EXTENSION_BLOCK = re.compile(r"<system-extension>(.*?)</system-extension>", re.DOTALL)


def apply_system_extension_tag_transform(payload: dict[str, Any]) -> dict[str, Any]:
    """Move the first complete external extension block to the end of system text."""
    out = dict(payload)
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return out

    selected = None
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            candidates = [(None, content)]
        elif isinstance(content, list):
            candidates = [
                (index, part["text"]) for index, part in enumerate(content)
                if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str)
            ]
        else:
            continue
        for part_index, text in candidates:
            match = _SYSTEM_EXTENSION_BLOCK.search(text)
            if match:
                selected = (message_index, part_index, text, match)
                break
        if selected is not None:
            break

    if selected is None:
        return out
    message_index, part_index, text, match = selected
    extension = match.group(1)
    if "<system-extension>" in extension:
        return out
    system_index = next((index for index in range(len(messages) - 1, -1, -1)
                         if isinstance(messages[index], dict) and messages[index].get("role") == "system"), None)
    if extension.strip() and system_index is not None and not isinstance(messages[system_index].get("content"), (str, list)):
        # Do not discard the extension or overwrite unsupported system content.
        return out

    updated_messages = list(messages)
    source_message = messages[message_index]
    cleaned = text[:match.start()] + text[match.end():]
    if part_index is None:
        cleaned_content = cleaned
    else:
        cleaned_content = list(source_message["content"])
        cleaned_content[part_index] = {**cleaned_content[part_index], "text": cleaned}
    updated_messages[message_index] = {**source_message, "content": cleaned_content}

    if extension.strip():
        if system_index is None:
            updated_messages.insert(0, {"role": "system", "content": extension})
        else:
            system_message = updated_messages[system_index]
            content = system_message["content"]
            if isinstance(content, str):
                content = content + ("\n\n" if content else "") + extension
            else:
                has_text = any(isinstance(part, dict) and part.get("type") == "text"
                               and isinstance(part.get("text"), str) and part["text"] for part in content)
                content = [*content, {"type": "text", "text": ("\n\n" if has_text else "") + extension}]
            updated_messages[system_index] = {**system_message, "content": content}
    out["messages"] = updated_messages
    return out


_CAPTURE_ATTRIBUTE = re.compile(r"""\s+(user|mode)\s*=\s*(?:"([^"<]*)"|'([^'<]*)')""")
_CAPTURE_MARKER = re.compile(rf"<capture-content(?P<attributes>(?:{_CAPTURE_ATTRIBUTE.pattern}){{0,2}})\s*/>")
_PERSONA_TAG = re.compile(r"<(?P<closing>/)?(?P<name>[^/<>][^<>]*)'s Persona>")


def _first_persona_capture(system_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for message in system_messages:
        for part in message["text_parts"]:
            openings: dict[str, list[int]] = {}
            first_span = None
            for tag in _PERSONA_TAG.finditer(part):
                name = tag["name"]
                if not tag["closing"]:
                    openings.setdefault(name, []).append(tag.end())
                elif openings.get(name):
                    start = openings[name].pop()
                    if first_span is None or start < first_span[0]:
                        first_span = (start, tag.start())
            if first_span is not None:
                start, end = first_span
                return [{"message_index": message["message_index"], "text_parts": [part[start:end]]}]
    return []


def extract_prompt_capture(payload: dict[str, Any]) -> tuple[dict[str, Any], bool, list[dict[str, Any]]]:
    """Strip capture controls and apply capture options to the detached system text."""
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return dict(payload), False, []
    detected = False
    user_name = None
    persona_only = False
    cleaned_messages = []
    system_messages = []

    def remove_marker(match):
        nonlocal detected, user_name, persona_only
        attributes = {}
        for key, double, single in _CAPTURE_ATTRIBUTE.findall(match["attributes"]):
            value = double or single
            if key in attributes or (key == "mode" and value != "persona"):
                return match[0]
            attributes[key] = value
        detected = True
        persona_only = persona_only or "mode" in attributes
        if user_name is None:
            user_name = attributes.get("user") or None
        return ""

    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            cleaned_messages.append(message)
            continue
        text_parts = []

        def clean(text):
            stripped = _CAPTURE_MARKER.sub(remove_marker, text)
            text_parts.append(stripped)
            return stripped

        content = message.get("content")
        if isinstance(content, str):
            content = clean(content)
        elif isinstance(content, list):
            content = [
                {**part, "text": clean(part["text"])}
                if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str)
                else part for part in content
            ]
        cleaned_messages.append({**message, "content": content} if "content" in message else dict(message))
        if message.get("role") == "system" and text_parts:
            system_messages.append({"message_index": index, "text_parts": text_parts})
    if persona_only:
        system_messages = _first_persona_capture(system_messages)
    if user_name is not None:
        # Normalize only the detached capture copy, after scanning every message:
        # the marker can occur after the system messages it controls.
        system_messages = [
            {**message, "text_parts": [part.replace(user_name, "{{user}}") for part in message["text_parts"]]}
            for message in system_messages
        ]
    return {**payload, "messages": cleaned_messages}, detected, system_messages
