"""Best-effort diagnostics for the external completion proxy only."""
import asyncio
import json
import logging
from uuid import uuid4

import httpx

from settings import settings

log = logging.getLogger("relay")
MAX_CAPTURE_BYTES = 1024 * 1024
CAPTURE_DEADLINE_SECONDS = 3.0


async def submit_prompt_capture(model: str, system_messages: list[dict]) -> None:
    if not any(part.strip() for message in system_messages for part in message["text_parts"]):
        return
    if not settings.PROMPT_CAPTURE_SECRET:
        log.warning("Prompt capture skipped: not configured")
        return
    if len(model) > 1024:
        log.warning("Prompt capture skipped: model too long")
        return
    try:
        body = json.dumps({"request_id": str(uuid4()), "model": model, "system_messages": system_messages},
                          ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if len(body) > MAX_CAPTURE_BYTES:
            log.warning("Prompt capture skipped: body too large")
            return
        async def post_capture():
            async with httpx.AsyncClient(timeout=CAPTURE_DEADLINE_SECONDS) as client:
                response = await client.post(
                    f"{settings.SUPABASE_URL.rstrip('/')}/functions/v1/capture-prompt",
                    headers={"X-Prompt-Capture-Secret": settings.PROMPT_CAPTURE_SECRET,
                             "Content-Type": "application/json"}, content=body,
                )
                if response.status_code != 200:
                    log.warning("Prompt capture failed: HTTP %d", response.status_code)
                else:
                    result = response.json()
                    log.info("Prompt capture submitted: captured=%s", result.get("captured") is True)
        await asyncio.wait_for(post_capture(), timeout=CAPTURE_DEADLINE_SECONDS)
    except Exception:
        # Never log provider/database bodies, prompt content, or secrets. Cancellation
        # inherits BaseException and continues to propagate to the request owner.
        log.warning("Prompt capture failed; continuing completion")
