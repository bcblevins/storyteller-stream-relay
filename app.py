# app.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette import EventSourceResponse
import json, time, logging, asyncio

from openai_service import openai_service
from auth import verify_jwt
from supabase import get_bot, post_message

app = FastAPI(title="Storytellr Relay", version="0.1")

# --- CORS configuration ---
origins = [
    "http://localhost:5173",   # Vite dev server
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
    "https://storytellr.me",   # production frontend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

log = logging.getLogger("relay")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

user_buckets: dict[str, dict] = {}


async def safe_post(msg):
    for attempt in range(3):
        try:
            return await post_message(msg)
        except Exception as e:
            if attempt == 2:
                raise
            log.warning(f"Retrying Supabase write ({attempt+1}/3): {e}")
            await asyncio.sleep(1)
    return None


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


@app.get("/auth/test")
async def auth_test(request: Request, bot_id: int):
    user_id = await verify_jwt(request)
    bot = await get_bot(user_id, bot_id)
    return {"user_id": user_id, "bot": bot["name"] if "name" in bot else bot["id"]}


@app.post("/v1/stream")
async def stream(request: Request):
    """Stream LLM output through the OpenAIService."""
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON")

    user_id = await verify_jwt(request)

    bot = await get_bot(user_id, payload["bot_id"])

    try:
        await openai_service.initialize_with_config(
            api_key=bot["access_key"],
            base_url=bot.get("access_path")
        )
    except Exception as e:
        raise HTTPException(500, f"Failed to init OpenAI client: {e}")

    # Build chat messages
    messages = [
        {"role": "system", "content": payload.get("system", "")},
        {"role": "user", "content": payload.get("prompt", "")},
    ]

    model = bot.get("model", "deepseek-chat")
    temperature = bot.get("temperature", 0.7)
    max_tokens = bot.get("max_tokens", 1000)
    stream_id = payload.get("stream_id") or f"s-{int(time.time() * 1000)}"

    # Create SSE generator
    async def event_gen():
        buffer = []
        ping_interval = 15
        last_ping = time.time()

        try:
            async for chunk in openai_service.create_chat_completion_stream(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                bot_config=bot
            ):
                if await request.is_disconnected():
                    log.info(f"Client disconnected early: {stream_id}")
                    break

                if chunk.get("error"):
                    yield {"event": "error", "data": json.dumps({
                        "error": chunk["error"],
                        "stream_id": stream_id
                    })}
                    break

                token = chunk.get("content")
                if token:
                    buffer.append(token)
                    yield {"event": "token", "data": token}

                # periodic ping (good for Cloudflare Tunnel stability)
                if time.time() - last_ping > ping_interval:
                    yield {"event": "ping", "data": ""}
                    last_ping = time.time()
        finally:
            yield {"event": "done", "data": json.dumps({"stream_id": stream_id})}
            log.info(f"Stream complete ({stream_id}) - {len(buffer)} tokens")

            # after final yield, persist message
            final_text = "".join(buffer)
            msg_record = {
                "user_id": user_id,
                "conversation_id": payload.get("conversation_id"),
                "content": final_text,
                "is_user_author": False,
                "is_streaming": False,
                "is_complete": True,
                "stream_id": stream_id,
            }

            try:
                if await request.is_disconnected():
                    log.info(f"Stream aborted {stream_id}")
                await safe_post(msg_record)
                log.info(f"Message persisted for stream {stream_id}")
            except Exception as e:
                log.error(f"Failed to persist message {stream_id}: {e}")


    return EventSourceResponse(event_gen(), ping=10, media_type="text/event-stream")