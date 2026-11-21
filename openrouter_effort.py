"""Read the public OpenRouter catalog for adjustable reasoning effort."""
import asyncio
import time

import httpx

EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
MODELS_URL = "https://openrouter.ai/api/v1/models"

_catalog: dict[str, tuple[str, ...]] = {}
_expires_at = 0.0
_catalog_unavailable = False
_catalog_lock = asyncio.Lock()


def parse_effort_options(reasoning: object) -> tuple[str, ...]:
    if not isinstance(reasoning, dict) or "supported_efforts" not in reasoning:
        return ()
    supported = reasoning["supported_efforts"]
    if supported is not None and not isinstance(supported, list):
        return ()
    return tuple(
        effort for effort in EFFORTS
        if (supported is None or effort in supported)
        and (effort != "none" or reasoning.get("mandatory") is not True)
    )


async def get_effort_options(model: str) -> tuple[str, ...] | None:
    """An empty tuple means unsupported. None means the catalog is unavailable."""
    global _catalog, _expires_at, _catalog_unavailable
    if time.monotonic() >= _expires_at:
        async with _catalog_lock:
            if time.monotonic() >= _expires_at:
                try:
                    async with httpx.AsyncClient(timeout=3) as client:
                        response = await client.get(MODELS_URL)
                        response.raise_for_status()
                    models = response.json()["data"]
                    if not isinstance(models, list):
                        raise ValueError("Invalid OpenRouter model catalog")
                    _catalog = {
                        entry["id"]: parse_effort_options(entry.get("reasoning"))
                        for entry in models
                        if isinstance(entry, dict) and isinstance(entry.get("id"), str)
                    }
                    _catalog_unavailable = False
                    _expires_at = time.monotonic() + 300
                except (httpx.HTTPError, ValueError, KeyError, TypeError):
                    _catalog_unavailable = True
                    _expires_at = time.monotonic() + 15
    if _catalog_unavailable:
        return None
    return _catalog.get(model, ())
