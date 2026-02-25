# app.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette import EventSourceResponse
import json, time, logging, asyncio, os
import httpx
import hmac

from openai_service import openai_service
from auth import verify_jwt
from settings import settings
from request_transforms import TransformConfig, apply_provider_request_transforms
from supabase import (
    get_bot,
    get_default_bot,
    get_conversation_bot,
    post_message,
    post_message_alternative,
    get_message,
    update_message_alternative,
    get_message_by_stream_id,
    get_openrouter_demo_bot,
    create_demo_openrouter_bot,
)

app = FastAPI(title="Storyteller Relay", version="0.1")

# --- CORS configuration (mirrors legacy Flask server defaults) ---
default_cors_origins = [
    "http://localhost:5173",
    "http://localhost:5174",
    "https://storytellr.me",
    "https://storytellr.me/",
    "https://dev.storytellr.me",
    "https://dev.storytellr.me/",
    "https://storyteller-elv.pages.dev"
]

extra_origins_raw = os.getenv("CORS_EXTRA_ORIGINS", "")
extra_origins = [origin.strip() for origin in extra_origins_raw.split(",") if origin.strip()]
allowed_origins = default_cors_origins + extra_origins


def _normalize_origin(origin: str | None) -> str | None:
    if not origin:
        return None
    return origin.rstrip("/")


allowed_origin_set = {_normalize_origin(o) for o in allowed_origins if _normalize_origin(o)}


log = logging.getLogger("relay")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def apply_cors_headers(response: Response, request: Request) -> Response:
    origin = request.headers.get("origin")
    normalized = _normalize_origin(origin)
    if normalized and normalized in allowed_origin_set:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers.setdefault("Vary", "Origin")

        if request.method == "OPTIONS":
            request_headers = request.headers.get("access-control-request-headers")
            if request_headers:
                response.headers["Access-Control-Allow-Headers"] = request_headers
            response.headers["Access-Control-Allow-Methods"] = "GET,POST,PUT,DELETE,OPTIONS"

    else:
        log.warning("CORS origin blocked or missing: %s", origin)
    return response

origin_regex = os.getenv("CORS_ALLOW_ORIGIN_REGEX", "").strip() or None

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

user_buckets: dict[str, dict] = {}

OPENROUTER_BASE_URL = settings.OPENROUTER_BASE_URL


async def safe_post(msg, token: str):
    for attempt in range(3):
        try:
            return await post_message(msg, token)
        except Exception as e:
            if attempt == 2:
                raise
            log.warning(f"Retrying Supabase write ({attempt+1}/3): {e}")
            await asyncio.sleep(1)
    return None


async def safe_post_alternative(alternative, token: str):
    """Safe alternative message posting with retry logic"""
    for attempt in range(3):
        try:
            return await post_message_alternative(alternative, token)
        except Exception as e:
            if attempt == 2:
                raise
            log.warning(f"Retrying alternative write ({attempt+1}/3): {e}")
            await asyncio.sleep(1)
    return None


def verify_proxy_api_key(request: Request) -> str:
    """Validate Bearer token for addon passthrough endpoints."""
    configured_key = settings.GLM_PROXY_API_KEY
    if not configured_key:
        raise HTTPException(503, "GLM passthrough is not configured")

    auth_header = request.headers.get("authorization") or request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(401, "Missing or invalid Authorization header")

    supplied_key = auth_header.split(" ", 1)[1].strip()
    if not supplied_key or not hmac.compare_digest(supplied_key, configured_key):
        raise HTTPException(401, "Unauthorized")

    return configured_key


def build_transform_config() -> TransformConfig:
    return TransformConfig(
        force_reasoning_enabled=settings.FORCE_REASONING_ENABLED,
        force_reasoning_effort=settings.FORCE_REASONING_EFFORT,
        force_reasoning_model_patterns=settings.force_reasoning_model_patterns_list,
        force_reasoning_override=settings.FORCE_REASONING_OVERRIDE,
    )


def check_rate_limit(user_id: str, limit=5, window=60):
    now = time.time()
    bucket = user_buckets.setdefault(user_id, {"count": 0, "reset": now + window})
    if now > bucket["reset"]:
        bucket.update({"count": 0, "reset": now + window})
    bucket["count"] += 1
    if bucket["count"] > limit:
        raise HTTPException(429, f"Rate limit exceeded ({limit}/min)")

# --- Health endpoint ---
@app.get("/healthz")
async def health_check():
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def chat_completions_proxy(request: Request):
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON")

    if not isinstance(payload, dict):
        raise HTTPException(400, "JSON body must be an object")

    upstream_key = verify_proxy_api_key(request)
    model = str(payload.get("model") or "")
    transformed_payload = apply_provider_request_transforms(
        payload=payload,
        provider="openrouter",
        model=model,
        config=build_transform_config(),
    )

    headers = {
        "Authorization": f"Bearer {upstream_key}",
        "Content-Type": "application/json",
    }
    upstream_url = f"{OPENROUTER_BASE_URL}/chat/completions"

    is_stream = bool(transformed_payload.get("stream"))
    timeout = None if is_stream else 60.0

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            if not is_stream:
                upstream_response = await client.post(
                    upstream_url,
                    headers=headers,
                    json=transformed_payload,
                )
                return Response(
                    content=upstream_response.content,
                    status_code=upstream_response.status_code,
                    media_type=upstream_response.headers.get("content-type"),
                )

            upstream_request = client.build_request(
                "POST",
                upstream_url,
                headers=headers,
                json=transformed_payload,
            )
            upstream_response = await client.send(upstream_request, stream=True)

            async def stream_bytes():
                try:
                    try:
                        async for chunk in upstream_response.aiter_bytes():
                            if chunk:
                                yield chunk
                    except httpx.ReadError:
                        # Upstream SSE providers may terminate chunked responses abruptly.
                        # Treat as end-of-stream so we don't crash the ASGI response task.
                        log.warning("Upstream stream closed early for /v1/chat/completions")
                finally:
                    await upstream_response.aclose()

            return StreamingResponse(
                stream_bytes(),
                status_code=upstream_response.status_code,
                media_type=upstream_response.headers.get("content-type", "text/event-stream"),
            )
    except httpx.HTTPError as e:
        log.error("Chat completions passthrough failed: %s", e)
        raise HTTPException(502, "Upstream provider request failed")


@app.get("/auth/test")
async def auth_test(request: Request, bot_id: int):
    user_id, auth_token = await verify_jwt(request)
    bot = await get_bot(user_id, bot_id, auth_token)
    return {"user_id": user_id, "bot": bot["name"] if "name" in bot else bot["id"]}


@app.get("/v1/message-by-stream-id")
async def message_by_stream_id(request: Request, stream_id: str):
    """
    Lookup a persisted message by its stream_id.
    Useful for clients that started streaming with a temporary ID and need
    the real message id after persistence.
    """
    user_id, auth_token = await verify_jwt(request)
    if not stream_id:
        raise HTTPException(400, "stream_id is required")
    try:
        message = await get_message_by_stream_id(stream_id, user_id, auth_token)
        return {"message": message}
    except HTTPException as e:
        # Pass through known errors (e.g., not found)
        raise e
    except Exception as e:
        log.error("Failed to get message by stream_id: %s", e)
        raise HTTPException(500, "Failed to get message by stream_id")


@app.post("/v1/stream")
async def stream(request: Request):
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON")

    user_id, auth_token = await verify_jwt(request)
    log.info("Stream request - User: %s, conversation_id: %s, messages_count: %d", 
             user_id, payload.get("conversation_id"), len(payload.get("messages", [])))

    # ---- Resolve bot: explicit -> conversation -> default ----
    bot = None
    desired_bot_id = payload.get("bot_id")
    conversation_id = payload.get("conversation_id")
    messages = payload.get("messages", [])
    log.debug("Bot resolution - desired_bot_id: %s, conversation_id: %s, messages_count: %d", 
             desired_bot_id, conversation_id, len(messages))

    if isinstance(desired_bot_id, int):
        log.debug("Using explicit bot_id: %s", desired_bot_id)
        bot = await get_bot(user_id, desired_bot_id, auth_token)
    elif isinstance(conversation_id, int):
        log.debug("Using conversation bot for conversation_id: %s", conversation_id)
        bot = await get_conversation_bot(user_id, conversation_id, auth_token)
        if bot is None:
            log.debug("No conversation bot found, falling back to default bot")
            bot = await get_default_bot(user_id, auth_token)
    else:
        log.debug("Using default bot")
        bot = await get_default_bot(user_id, auth_token)
    
    log.debug("Selected bot - id: %s, name: %s, model: %s", bot.get("id"), bot.get("name"), bot.get("model"))

    # ---- Initialize provider client from bot config ----
    api_key = bot.get("access_key")
    if bot.get("is_openrouter") and bot.get("openrouter_key"):
        api_key = bot.get("openrouter_key")

    try:
        await openai_service.initialize_with_config(
            api_key=api_key,
            base_url=bot.get("access_path")
        )
    except Exception as e:
        raise HTTPException(500, f"Failed to init OpenAI client: {e}")

    log.debug("Messages to send to OpenAI: %s", messages)

    model = bot.get("model", "deepseek-chat")
    if bot.get("is_openrouter") and bot.get("openrouter_key"):
        model = settings.OPENROUTER_DEMO_MODEL
    # Prefer determinism/safety over creativity when temperature is absent.
    temperature = bot.get("temperature", 0.1)
    max_tokens = bot.get("max_tokens", 1000)
    stream_id = payload.get("stream_id") or f"s-{int(time.time() * 1000)}"
    
    # Check if this is an alternative message stream
    is_alternative = payload.get("is_alternative", False)
    alternative_id = payload.get("alternative_id")
    
    log.info("Stream configuration - model: %s, temperature: %s, max_tokens: %s, stream_id: %s, is_alternative: %s, alternative_id: %s",
             model, temperature, max_tokens, stream_id, is_alternative, alternative_id)

    async def event_gen():
        buffer = []
        ping_interval = 15
        last_ping = time.time()
        aborted = False

        try:
            async for chunk in openai_service.create_chat_completion_stream(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                bot_config=bot
            ):
                if await request.is_disconnected():
                    aborted = True
                    log.info("Client disconnected early - stream_id: %s", stream_id)
                    break

                if chunk.get("error"):
                    log.error("OpenAI streaming error: %s", chunk["error"])
                    yield {"event": "error", "data": json.dumps({
                        "error": chunk["error"],
                        "stream_id": stream_id
                    })}
                    break

                content_chunk = chunk.get("content")
                if content_chunk:
                    buffer.append(content_chunk)
                    yield {"event": "token", "data": content_chunk}

                if time.time() - last_ping > ping_interval:
                    yield {"event": "ping", "data": ""}
                    last_ping = time.time()
        finally:
            log.info("Stream complete - stream_id: %s, tokens: %d, aborted: %s", stream_id, len(buffer), aborted)

            # Persist message before sending final 'done' so the client can reconcile IDs.
            # Shield the persistence step so it still runs even if the client disconnects
            # and Starlette cancels the streaming task.
            final_text = "".join(buffer)

            async def persist_and_build_done_payload():
                done_payload = {"stream_id": stream_id}
                if is_alternative and alternative_id:
                    # Update existing alternative message
                    updates = {
                        "content": final_text,
                        "is_streaming": False,
                        "is_complete": (not aborted),
                        "stream_id": stream_id
                    }
                    log.debug("Updating alternative message - alternative_id: %s", alternative_id)
                    await update_message_alternative(alternative_id, updates, auth_token)
                    done_payload.update({"alternative_id": alternative_id})
                    log.info("Alternative message updated - alternative_id: %s, stream_id: %s", alternative_id, stream_id)
                else:
                    # Create new regular message
                    msg_record = {
                        "user_id": user_id,
                        "conversation_id": conversation_id,
                        "content": final_text,
                        "is_user_author": False,
                        "is_streaming": False,
                        "is_complete": (not aborted),
                        "stream_id": stream_id,
                    }

                    if aborted:
                        log.info("Stream aborted - stream_id: %s", stream_id)
                    log.debug("Persisting message to Supabase - stream_id: %s", stream_id)
                    persisted = await safe_post(msg_record, auth_token)
                    # Supabase REST returns a list when Prefer return=representation
                    if isinstance(persisted, list) and persisted:
                        persisted_id = persisted[0].get("id")
                    elif isinstance(persisted, dict):
                        persisted_id = persisted.get("id")
                    else:
                        persisted_id = None
                    if persisted_id:
                        done_payload.update({"message_id": persisted_id})
                    log.info("Message persisted - stream_id: %s, message_id: %s", stream_id, persisted_id)

                return done_payload

            try:
                done_payload = await asyncio.shield(persist_and_build_done_payload())
            except asyncio.CancelledError:
                log.warning("Persistence shield cancelled - stream_id: %s", stream_id)
                done_payload = {"stream_id": stream_id, "error": "persist_cancelled"}
            except Exception as e:
                log.error("Failed to persist message - stream_id: %s, error: %s", stream_id, e)
                done_payload = {"stream_id": stream_id, "error": "persist_failed"}

            # Always send 'done' so the adapter completes cleanly (include IDs when available)
            yield {"event": "done", "data": json.dumps(done_payload)}

    return EventSourceResponse(event_gen(), ping=10, media_type="text/event-stream")

async def _provision_openrouter_key(user_id: str) -> str:
    payload = {
        "name": f"storytellr-demo-{user_id}",
        "limit": settings.OPENROUTER_DEMO_LIMIT,
    }
    if settings.OPENROUTER_DEMO_LIMIT_RESET in {"daily", "weekly", "monthly"}:
        payload["limit_reset"] = settings.OPENROUTER_DEMO_LIMIT_RESET
    elif settings.OPENROUTER_DEMO_LIMIT_RESET:
        log.warning(
            "Invalid OPENROUTER_DEMO_LIMIT_RESET value; omitting from payload"
        )
    headers = {
        "Authorization": f"Bearer {settings.OPENROUTER_PROVISIONING_KEY}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{OPENROUTER_BASE_URL}/keys",
            headers=headers,
            json=payload,
            timeout=15,
        )

    if resp.status_code not in (200, 201):
        log.error("OpenRouter provisioning failed - status: %s, body: %s", resp.status_code, resp.text)
        raise HTTPException(502, "OpenRouter provisioning failed")

    data = resp.json()
    # OpenRouter returns the usable API key in the top-level "key" field (shown once).
    # "data.hash" is an identifier and is not usable for Authorization.
    or_key = data.get("key")
    if not or_key:
        log.error("OpenRouter provisioning response missing key")
        raise HTTPException(502, "OpenRouter provisioning returned no key")

    return or_key.strip()

@app.post("/v1/openrouter/demo")
async def provision_openrouter_demo(request: Request):
    user_id, auth_token = await verify_jwt(request)

    existing = await get_openrouter_demo_bot(user_id, auth_token)
    if existing:
        raise HTTPException(409, "Demo bot already exists")

    or_key = await _provision_openrouter_key(user_id)

    bot_id = await create_demo_openrouter_bot(
        user_id=user_id,
        or_key=or_key,
        model=settings.OPENROUTER_DEMO_MODEL,
        access_path=OPENROUTER_BASE_URL,
        name="Storyteller Demo",
        token=auth_token,
    )

    if not bot_id:
        log.error("OpenRouter provisioning succeeded but bot creation failed")
        raise HTTPException(502, "Failed to create demo bot")

    return {"success": True, "bot_id": bot_id}


@app.post("/v1/reroll")
async def reroll_message(request: Request):
    """
    Create a new alternative message for reroll functionality.
    Returns the new alternative message ID and stream ID for frontend streaming.
    """
    try:
        payload = await request.json()
        log.info("Reroll endpoint - Request received")
    except Exception as e:
        log.error("Reroll endpoint - Failed to parse JSON payload: %s", e)
        raise HTTPException(400, "Invalid JSON")

    user_id, auth_token = await verify_jwt(request)
    
    parent_message_id = payload.get("parent_message_id")
    conversation_id = payload.get("conversation_id")
    
    log.info("Reroll endpoint - user_id: %s, parent_message_id: %s, conversation_id: %s", 
             user_id, parent_message_id, conversation_id)
    
    # Validate required fields
    if not parent_message_id:
        log.error("Reroll endpoint - Missing required field: parent_message_id")
        raise HTTPException(400, "parent_message_id is required")
    if not conversation_id:
        log.error("Reroll endpoint - Missing required field: conversation_id")
        raise HTTPException(400, "conversation_id is required")
    
    # Check if this is a temporary message ID (from streaming)
    # Convert to string for temporary ID check, but keep original type for database operations
    is_temporary_id = str(parent_message_id).startswith('temp-')
    actual_parent_message_id = parent_message_id
    
    # If it's a temporary ID, we need to find the actual message by stream_id
    if is_temporary_id:
        log.debug("Reroll endpoint - Temporary message ID detected: %s", parent_message_id)
        # Extract stream_id from temporary message ID (format: temp-stream-{timestamp}-{random})
        if str(parent_message_id).startswith('temp-stream-'):
            # The temporary ID itself is the stream_id used during streaming
            stream_id = parent_message_id
            log.debug("Reroll endpoint - Looking up message by stream_id: %s", stream_id)
            try:
                parent_message = await get_message_by_stream_id(stream_id, user_id, auth_token)
                actual_parent_message_id = parent_message['id']
                log.debug("Reroll endpoint - Found actual message ID: %s for stream_id: %s", actual_parent_message_id, stream_id)
            except HTTPException as e:
                log.error("Reroll endpoint - Failed to find message by stream_id: %s, error: %s", stream_id, e.detail)
                raise HTTPException(404, f"Temporary message not yet persisted. Please wait a moment and try again.")
        else:
            log.error("Reroll endpoint - Unsupported temporary message ID format: %s", parent_message_id)
            raise HTTPException(400, "Unsupported temporary message ID format")
    else:
        # Normal message ID lookup
        try:
            log.debug("Reroll endpoint - Calling get_message with parent_message_id: %s, user_id: %s", parent_message_id, user_id)
            parent_message = await get_message(parent_message_id, user_id, auth_token)
            log.debug("Reroll endpoint - get_message result: %s", parent_message)
        except HTTPException as e:
            log.error("Reroll endpoint - get_message failed with HTTPException: %s, status_code: %s", e.detail, e.status_code)
            raise HTTPException(404, "Parent message not found or unauthorized")
        except Exception as e:
            log.error("Reroll endpoint - get_message failed with unexpected error: %s", e)
            raise HTTPException(500, f"Internal server error: {str(e)}")
    
    # Verify this is an AI message (only AI messages can be rerolled)
    if parent_message.get("is_user_author"):
        raise HTTPException(400, "Cannot reroll user messages")
    
    # Generate a unique stream ID
    stream_id = f"reroll-{int(time.time() * 1000)}"
    
    # Create a new alternative message for streaming
    alternative_record = {
        "user_id": user_id,
        "conversation_id": conversation_id,
        "parent_message_id": actual_parent_message_id,
        "content": "",  # Start with empty content for streaming
        "is_user_author": False,  # AI messages
        "is_streaming": True,
        "is_complete": False,
        "stream_id": stream_id,
        "is_active": True
    }
    
    try:
        # Create the alternative message in Supabase
        alternative_response = await safe_post_alternative(alternative_record, auth_token)
        if not alternative_response:
            raise HTTPException(500, "Failed to create alternative message after retries")
        
        alternative_message = alternative_response[0] if isinstance(alternative_response, list) else alternative_response
        
        log.info("Created alternative message - id: %s, parent_message_id: %s", alternative_message['id'], actual_parent_message_id)
        
        return {
            "alternative_message": {
                "id": alternative_message["id"],
                "parent_message_id": actual_parent_message_id,
                "conversation_id": conversation_id,
                "content": "",
                "is_user_author": False,
                "is_streaming": True,
                "is_complete": False,
                "stream_id": stream_id,
                "is_active": True
            },
            "stream_id": stream_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        log.error("Failed to create alternative message: %s", e)
        raise HTTPException(500, f"Failed to create alternative message: {str(e)}")
