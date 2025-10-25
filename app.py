# app.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette import EventSourceResponse
import httpx, asyncio, json, time, logging
from openai import AsyncOpenAI

from openai_service import openai_service
from auth import verify_jwt
from supabase import get_bot

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

    # Verify JWT
    user_id = await verify_jwt(request)

    # Fetch bot credentials from Supabase
    bot = await get_bot(user_id, payload["bot_id"])

    # Initialize the OpenAI client
    try:
        openai_service.initialize_with_config(
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

        # Run sync generator in a threadpool to not block FastAPI’s event loop
        loop = asyncio.get_event_loop()
        stream_gen = await loop.run_in_executor(
            None,
            lambda: openai_service.create_chat_completion_stream(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                bot_config=bot
            )
        )

        # The above returns a *generator*, not a list, so we’ll iterate directly.
        # To avoid blocking the loop, we’ll handle each chunk asynchronously.

        for chunk in stream_gen:
            if await request.is_disconnected():
                log.info(f"Client disconnected: {stream_id}")
                break

            if chunk.get("error"):
                yield {
                    "event": "error",
                    "data": json.dumps({
                        "stream_id": stream_id,
                        "error": chunk["error"]
                    }),
                }
                break

            token = chunk.get("content")
            if token:
                buffer.append(token)
                yield {"event": "token", "data": token}

            # keepalive ping
            if time.time() - last_ping > ping_interval:
                yield {"event": "ping", "data": ""}
                last_ping = time.time()

        yield {"event": "done", "data": json.dumps({"stream_id": stream_id})}
        log.info(f"Stream complete ({stream_id}) - {len(buffer)} tokens")

    return EventSourceResponse(event_gen(), ping=10, media_type="text/event-stream")