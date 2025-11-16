# supabase.py
import httpx
from fastapi import HTTPException, status
from settings import settings

COMMON_HEADERS = {
    "apikey": settings.SUPABASE_SERVICE_ROLE_KEY,
    "Authorization": f"Bearer {settings.SUPABASE_SERVICE_ROLE_KEY}",
}

async def _rest_get(path: str, params: dict):
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{settings.SUPABASE_REST_URL}/{path}", headers=COMMON_HEADERS, params=params, timeout=10)
    if r.status_code != 200:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Supabase fetch failed ({path})")
    return r.json()

async def get_bot(user_id: str, bot_id: int):
    """Fetch a specific bot by id, owned by user."""
    data = await _rest_get("bots", {"id": f"eq.{bot_id}", "user_id": f"eq.{user_id}", "limit": 1})
    if not data:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Bot not found or unauthorized")
    return data[0]

async def get_default_bot(user_id: str):
    """
    Fetch the user's default bot.
    If multiple defaults exist, prefer the newest by updated_at/created_at.
    """
    # Prefer updated_at if present; fall back to created_at
    params = {
        "user_id": f"eq.{user_id}",
        "is_default": "eq.true",
        "limit": 1,
        "order": "updated_at.desc,created_at.desc",
    }
    data = await _rest_get("bots", params)
    if data:
        return data[0]

    # Fallback: pick any bot owned by the user (most recent)
    params = {
        "user_id": f"eq.{user_id}",
        "limit": 1,
        "order": "updated_at.desc,created_at.desc",
    }
    data = await _rest_get("bots", params)
    if not data:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "No bot configured for this user")
    return data[0]

async def get_conversation_bot(user_id: str, conversation_id: int):
    """
    Optional: fetch conversation->bot binding if your schema stores it.
    Expects a `bot_id` column on `conversations`. If missing, returns None.
    """
    try:
        data = await _rest_get("conversations", {
            "id": f"eq.{conversation_id}",
            "user_id": f"eq.{user_id}",
            "select": "id,bot_id",
            "limit": 1,
        })
    except HTTPException:
        # Table/column may not exist yet during migration; gracefully ignore.
        return None

    if not data:
        return None
    conv = data[0]
    bot_id = conv.get("bot_id")
    if bot_id is None:
        return None
    # ensure ownership
    return await get_bot(user_id, bot_id)

async def post_message(message: dict):
    """Write a message record to Supabase REST endpoint."""
    headers = {
        **COMMON_HEADERS,
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates,return=representation",
    }
    # print each key value pair in message
    print("Posting message to Supabase:")
    for k, v in message.items():
        print(f"{k} = {v}")

    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{settings.SUPABASE_REST_URL}/messages", headers=headers, json=message)
    if resp.status_code not in (200, 201):
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Supabase insert failed: {resp.text}")
    return resp.json()


async def post_message_alternative(alternative: dict):
    """Write a message alternative record to Supabase REST endpoint."""
    headers = {
        **COMMON_HEADERS,
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates,return=representation",
    }
    print("Posting message alternative to Supabase:")
    for k, v in alternative.items():
        print(f"{k} = {v}")

    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{settings.SUPABASE_REST_URL}/message_alternatives", headers=headers, json=alternative)
    if resp.status_code not in (200, 201):
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Supabase insert failed: {resp.text}")
    return resp.json()


async def update_message_alternative(alternative_id: int, updates: dict):
    """Update an existing message alternative record."""
    headers = {
        **COMMON_HEADERS,
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates,return=representation",
    }
    
    async with httpx.AsyncClient() as client:
        resp = await client.patch(
            f"{settings.SUPABASE_REST_URL}/message_alternatives?id=eq.{alternative_id}",
            headers=headers,
            json=updates
        )
    if resp.status_code != 200:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Supabase update failed: {resp.text}")
    return resp.json()


async def get_message_alternatives(parent_message_id: int, user_id: str):
    """Get all alternatives for a parent message."""
    params = {
        "parent_message_id": f"eq.{parent_message_id}",
        "user_id": f"eq.{user_id}",
        "order": "t.asc"
    }
    
    data = await _rest_get("message_alternatives", params)
    return data


async def get_message(message_id: int, user_id: str):
    """Get a specific message by ID, verifying user ownership."""
    params = {
        "id": f"eq.{message_id}",
        "user_id": f"eq.{user_id}",
        "limit": 1
    }
    
    data = await _rest_get("messages", params)
    if not data:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Message not found or unauthorized")
    return data[0]
