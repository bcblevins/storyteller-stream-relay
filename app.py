# app.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette import EventSourceResponse
import json, time, logging, asyncio

from openai_service import openai_service
from auth import verify_jwt
from supabase import get_bot, get_default_bot, get_conversation_bot, post_message, post_message_alternative, get_message, update_message_alternative

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


async def safe_post_alternative(alternative):
    """Safe alternative message posting with retry logic"""
    for attempt in range(3):
        try:
            return await post_message_alternative(alternative)
        except Exception as e:
            if attempt == 2:
                raise
            log.warning(f"Retrying alternative write ({attempt+1}/3): {e}")
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
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON")

    user_id = await verify_jwt(request)

    # ---- Resolve bot: explicit -> conversation -> default ----
    bot = None
    desired_bot_id = payload.get("bot_id")
    conversation_id = payload.get("conversation_id")
    messages = payload.get("messages", [])
    print(f"RECIEVED Conversation ID: {conversation_id}")

    if isinstance(desired_bot_id, int):
        bot = await get_bot(user_id, desired_bot_id)
    elif isinstance(conversation_id, int):
        bot = await get_conversation_bot(user_id, conversation_id)
        if bot is None:
            bot = await get_default_bot(user_id)
    else:
        bot = await get_default_bot(user_id)

    # ---- Initialize provider client from bot config ----
    try:
        await openai_service.initialize_with_config(
            api_key=bot["access_key"],
            base_url=bot.get("access_path")
        )
    except Exception as e:
        raise HTTPException(500, f"Failed to init OpenAI client: {e}")

    # What is this. Why would we do this.
    # messages = [
    #     {"role": "system", "content": payload.get("system", "")},
    #     {"role": "user", "content": payload.get("prompt", "")},
    # ]

    model = bot.get("model", "deepseek-chat")
    temperature = bot.get("temperature", 0.7)
    max_tokens = bot.get("max_tokens", 1000)
    stream_id = payload.get("stream_id") or f"s-{int(time.time() * 1000)}"
    
    # Check if this is an alternative message stream
    is_alternative = payload.get("is_alternative", False)
    alternative_id = payload.get("alternative_id")

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

                if time.time() - last_ping > ping_interval:
                    yield {"event": "ping", "data": ""}
                    last_ping = time.time()
        finally:
            # Always send 'done' so the adapter completes cleanly
            yield {"event": "done", "data": json.dumps({"stream_id": stream_id})}
            log.info(f"Stream complete ({stream_id}) - {len(buffer)} tokens")

            # Persist message after completion/abort
            final_text = "".join(buffer)
            
            if is_alternative and alternative_id:
                # Update existing alternative message
                try:
                    updates = {
                        "content": final_text,
                        "is_streaming": False,
                        "is_complete": (not aborted),
                        "stream_id": stream_id
                    }
                    await update_message_alternative(alternative_id, updates)
                    log.info(f"Alternative message {alternative_id} updated for stream {stream_id}")
                except Exception as e:
                    log.error(f"Failed to update alternative message {alternative_id}: {e}")
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

                try:
                    if aborted:
                        log.info(f"Stream aborted {stream_id}")
                    await safe_post(msg_record)
                    log.info(f"Message persisted for stream {stream_id}")
                except Exception as e:
                    log.error(f"Failed to persist message {stream_id}: {e}")

    return EventSourceResponse(event_gen(), ping=10, media_type="text/event-stream")


@app.post("/v1/reroll")
async def reroll_message(request: Request):
    """
    Create a new alternative message for reroll functionality.
    Returns the new alternative message ID and stream ID for frontend streaming.
    """
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON")

    user_id = await verify_jwt(request)
    
    parent_message_id = payload.get("parent_message_id")
    conversation_id = payload.get("conversation_id")
    
    # Validate required fields
    if not parent_message_id:
        raise HTTPException(400, "parent_message_id is required")
    if not conversation_id:
        raise HTTPException(400, "conversation_id is required")
    
    # Verify user owns the parent message
    try:
        parent_message = await get_message(parent_message_id, user_id)
    except HTTPException:
        raise HTTPException(404, "Parent message not found or unauthorized")
    
    # Verify this is an AI message (only AI messages can be rerolled)
    if parent_message.get("is_user_author"):
        raise HTTPException(400, "Cannot reroll user messages")
    
    # Generate a unique stream ID
    stream_id = f"reroll-{int(time.time() * 1000)}"
    
    # Create a new alternative message for streaming
    alternative_record = {
        "user_id": user_id,
        "conversation_id": conversation_id,
        "parent_message_id": parent_message_id,
        "content": "",  # Start with empty content for streaming
        "is_user_author": False,  # AI messages
        "is_streaming": True,
        "is_complete": False,
        "stream_id": stream_id,
        "is_active": True
    }
    
    try:
        # Create the alternative message in Supabase
        alternative_response = await safe_post_alternative(alternative_record)
        alternative_message = alternative_response[0] if isinstance(alternative_response, list) else alternative_response
        
        log.info(f"Created alternative message {alternative_message['id']} for parent {parent_message_id}")
        
        return {
            "alternative_message": {
                "id": alternative_message["id"],
                "parent_message_id": parent_message_id,
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
        
    except Exception as e:
        log.error(f"Failed to create alternative message: {e}")
        raise HTTPException(500, f"Failed to create alternative message: {str(e)}")
