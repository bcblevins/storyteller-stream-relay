from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Any


@dataclass(frozen=True)
class TransformConfig:
    force_reasoning_enabled: bool = False
    force_reasoning_effort: str = "high"
    force_reasoning_model_patterns: tuple[str, ...] = ("z-ai/glm-4.6:nitro",)
    force_reasoning_override: bool = False


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
