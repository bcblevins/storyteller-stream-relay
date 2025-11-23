# supabase.py
import httpx
import logging
from fastapi import HTTPException, status
from settings import settings

log = logging.getLogger("relay")

COMMON_HEADERS = {
    "apikey": settings.SUPABASE_SERVICE_ROLE_KEY,
    "Authorization": f"Bearer {settings.SUPABASE_SERVICE_ROLE_KEY}",
}

async def _rest_get(path: str, params: dict):
    url = f"{settings.SUPABASE_REST_URL}/{path}"
    log.debug("Supabase GET - url: %s, params: %s", url, params)
    async with httpx.AsyncClient() as client:
        r = await client.get(url, headers=COMMON_HEADERS, params=params, timeout=10)
    log.debug("Supabase GET response - status: %s, body: %s", r.status_code, r.text)
    if r.status_code != 200:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Supabase fetch failed ({path})")
    return r.json()

async def get_bot(user_id: str, bot_id: int):
    """Fetch a specific bot by id, owned by user."""
    log.info("Getting bot - user_id: %s, bot_id: %s", user_id, bot_id)
    data = await _rest_get("bots", {"id": f"eq.{bot_id}", "user_id": f"eq.{user_id}", "limit": 1})
    if not data:
        log.warning("Bot not found or unauthorized - user_id: %s, bot_id: %s", user_id, bot_id)
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Bot not found or unauthorized")
    log.info("Bot found - id: %s, name: %s", data[0].get("id"), data[0].get("name"))
    return data[0]

async def get_default_bot(user_id: str):
    """
    Fetch the user's default bot.
    If multiple defaults exist, prefer the newest by updated_at/created_at.
    """
    log.info("Getting default bot - user_id: %s", user_id)
    # Prefer updated_at if present; fall back to created_at
    params = {
        "user_id": f"eq.{user_id}",
        "is_default": "eq.true",
        "limit": 1,
        "order": "updated_at.desc,created_at.desc",
    }
    data = await _rest_get("bots", params)
    if data:
        log.info("Default bot found - id: %s, name: %s", data[0].get("id"), data[0].get("name"))
        return data[0]

    # Fallback: pick any bot owned by the user (most recent)
    log.info("No default bot found, falling back to most recent bot")
    params = {
        "user_id": f"eq.{user_id}",
        "limit": 1,
        "order": "updated_at.desc,created_at.desc",
    }
    data = await _rest_get("bots", params)
    if not data:
        log.error("No bot configured for user - user_id: %s", user_id)
        raise HTTPException(status.HTTP_403_FORBIDDEN, "No bot configured for this user")
    log.info("Fallback bot found - id: %s, name: %s", data[0].get("id"), data[0].get("name"))
    return data[0]

async def get_conversation_bot(user_id: str, conversation_id: int):
    """
    Optional: fetch conversation->bot binding if your schema stores it.
    Expects a `bot_id` column on `conversations`. If missing, returns None.
    """
    log.info("Getting conversation bot - user_id: %s, conversation_id: %s", user_id, conversation_id)
    try:
        data = await _rest_get("conversations", {
            "id": f"eq.{conversation_id}",
            "user_id": f"eq.{user_id}",
            "select": "id,bot_id",
            "limit": 1,
        })
    except HTTPException:
        # Table/column may not exist yet during migration; gracefully ignore.
        log.info("Conversation bot lookup failed (table/column may not exist)")
        return None

    if not data:
        log.info("No conversation found - user_id: %s, conversation_id: %s", user_id, conversation_id)
        return None
    conv = data[0]
    bot_id = conv.get("bot_id")
    if bot_id is None:
        log.info("Conversation has no bot_id - conversation_id: %s", conversation_id)
        return None
    # ensure ownership
    log.info("Conversation bot found - bot_id: %s", bot_id)
    return await get_bot(user_id, bot_id)

async def post_message(message: dict):
    """Write a message record to Supabase REST endpoint."""
    headers = {
        **COMMON_HEADERS,
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates,return=representation",
    }
    url = f"{settings.SUPABASE_REST_URL}/messages"
    log.info("Posting message to Supabase - url: %s, message: %s", url, message)

    async with httpx.AsyncClient() as client:
        resp = await client.post(url, headers=headers, json=message)
    log.debug("Supabase POST response - status: %s, body: %s", resp.status_code, resp.text)
    if resp.status_code not in (200, 201):
        log.error("Supabase insert failed - status: %s, error: %s", resp.status_code, resp.text)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Supabase insert failed: {resp.text}")
    log.info("Message posted successfully")
    return resp.json()


async def post_message_alternative(alternative: dict):
    """Write a message alternative record to Supabase REST endpoint."""
    headers = {
        **COMMON_HEADERS,
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates,return=representation",
    }
    url = f"{settings.SUPABASE_REST_URL}/message_alternatives"
    log.info("Posting message alternative to Supabase - url: %s, alternative: %s", url, alternative)

    async with httpx.AsyncClient() as client:
        resp = await client.post(url, headers=headers, json=alternative)
    log.debug("Supabase POST alternative response - status: %s, body: %s", resp.status_code, resp.text)
    if resp.status_code not in (200, 201):
        log.error("Supabase alternative insert failed - status: %s, error: %s", resp.status_code, resp.text)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Supabase insert failed: {resp.text}")
    log.info("Message alternative posted successfully")
    return resp.json()


async def update_message_alternative(alternative_id: int, updates: dict):
    """Update an existing message alternative record."""
    headers = {
        **COMMON_HEADERS,
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates,return=representation",
    }
    url = f"{settings.SUPABASE_REST_URL}/message_alternatives?id=eq.{alternative_id}"
    log.info("Updating message alternative - url: %s, alternative_id: %s, updates: %s", url, alternative_id, updates)
    
    async with httpx.AsyncClient() as client:
        resp = await client.patch(
            url,
            headers=headers,
            json=updates
        )
    log.debug("Supabase PATCH response - status: %s, body: %s", resp.status_code, resp.text)
    if resp.status_code != 200:
        log.error("Supabase update failed - status: %s, error: %s", resp.status_code, resp.text)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Supabase update failed: {resp.text}")
    log.info("Message alternative updated successfully")
    return resp.json()


async def get_message_alternatives(parent_message_id: int, user_id: str):
    """Get all alternatives for a parent message."""
    log.info("Getting message alternatives - parent_message_id: %s, user_id: %s", parent_message_id, user_id)
    params = {
        "parent_message_id": f"eq.{parent_message_id}",
        "user_id": f"eq.{user_id}",
        "order": "t.asc"
    }
    
    data = await _rest_get("message_alternatives", params)
    log.info("Found %d message alternatives", len(data))
    return data


async def get_message_by_stream_id(stream_id: str, user_id: str):
    """Get a specific message by stream_id, verifying user ownership."""
    log.info("get_message_by_stream_id - Starting lookup - stream_id: %s, user_id: %s", stream_id, user_id)
    
    params = {
        "stream_id": f"eq.{stream_id}",
        "user_id": f"eq.{user_id}",
        "limit": 1
    }
    
    log.debug("get_message_by_stream_id - REST API parameters (SQL equivalent: SELECT * FROM messages WHERE stream_id = %s AND user_id = %s LIMIT 1): %s", 
              stream_id, user_id, params)
    
    try:
        data = await _rest_get("messages", params)
        
        if not data:
            log.warning("get_message_by_stream_id - No message found - stream_id: %s, user_id: %s", stream_id, user_id)
            log.debug("get_message_by_stream_id - Supabase returned empty result set")
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Message not found or unauthorized")
        
        log.info("get_message_by_stream_id - Message found successfully - id: %s", data[0].get("id"))
        log.debug("get_message_by_stream_id - Full message data: %s", data[0])
        return data[0]
        
    except HTTPException as e:
        log.error("get_message_by_stream_id - HTTPException during message lookup - status_code: %s, detail: %s", 
                  e.status_code, e.detail)
        log.debug("get_message_by_stream_id - Full HTTPException: %s", e)
        raise
    except Exception as e:
        log.error("get_message_by_stream_id - Unexpected error during message lookup - error: %s, type: %s", 
                  e, type(e).__name__)
        log.debug("get_message_by_stream_id - Full exception details: %s", e)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"Internal server error: {str(e)}")


async def get_message(message_id: int, user_id: str):
    """Get a specific message by ID, verifying user ownership."""
    log.info("get_message - Starting lookup - message_id: %s, user_id: %s", message_id, user_id)
    
    params = {
        "id": f"eq.{message_id}",
        "user_id": f"eq.{user_id}",
        "limit": 1
    }
    
    log.debug("get_message - REST API parameters (SQL equivalent: SELECT * FROM messages WHERE id = %s AND user_id = %s LIMIT 1): %s", 
              message_id, user_id, params)
    
    try:
        data = await _rest_get("messages", params)
        
        if not data:
            log.warning("get_message - No message found - message_id: %s, user_id: %s", message_id, user_id)
            log.debug("get_message - Supabase returned empty result set")
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Message not found or unauthorized")
        
        log.info("get_message - Message found successfully - id: %s", data[0].get("id"))
        log.debug("get_message - Full message data: %s", data[0])
        return data[0]
        
    except HTTPException as e:
        log.error("get_message - HTTPException during message lookup - status_code: %s, detail: %s", 
                  e.status_code, e.detail)
        log.debug("get_message - Full HTTPException: %s", e)
        raise
    except Exception as e:
        log.error("get_message - Unexpected error during message lookup - error: %s, type: %s", 
                  e, type(e).__name__)
        log.debug("get_message - Full exception details: %s", e)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"Internal server error: {str(e)}")
