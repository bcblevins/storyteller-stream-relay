# supabase.py
import httpx
import logging
from fastapi import HTTPException, status
from settings import settings

log = logging.getLogger("relay")

_SENSITIVE_FIELDS = {"access_key", "openrouter_key", "or_key"}

def _sanitize_for_log(data):
    if isinstance(data, dict):
        return {
            key: ("***" if key in _SENSITIVE_FIELDS else value)
            for key, value in data.items()
        }
    if isinstance(data, list):
        return [_sanitize_for_log(item) for item in data]
    return data

def _build_headers(token: str):
    """Build headers for Supabase REST API using user JWT and anon key."""
    return {
        "apikey": settings.SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {token}",
    }

async def _rest_get(path: str, params: dict, token: str):
    url = f"{settings.SUPABASE_REST_URL}/{path}"
    log.debug("Supabase GET - url: %s, params: %s", url, params)
    headers = _build_headers(token)
    async with httpx.AsyncClient() as client:
        r = await client.get(url, headers=headers, params=params, timeout=10)
    log.debug("Supabase GET response - status: %s, body: %s", r.status_code, r.text)
    if r.status_code != 200:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Supabase fetch failed ({path})")
    return r.json()

async def get_bot(user_id: str, bot_id: int, token: str):
    """Fetch a specific bot by id, owned by user."""
    log.info("Getting bot - user_id: %s, bot_id: %s", user_id, bot_id)
    data = await _rest_get("bots", {"id": f"eq.{bot_id}", "user_id": f"eq.{user_id}", "limit": 1}, token)
    if not data:
        log.warning("Bot not found or unauthorized - user_id: %s, bot_id: %s", user_id, bot_id)
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Bot not found or unauthorized")
    log.info("Bot found - id: %s, name: %s", data[0].get("id"), data[0].get("name"))
    return data[0]

async def get_default_bot(user_id: str, token: str):
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
    data = await _rest_get("bots", params, token)
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
    data = await _rest_get("bots", params, token)
    if not data:
        log.error("No bot configured for user - user_id: %s", user_id)
        raise HTTPException(status.HTTP_403_FORBIDDEN, "No bot configured for this user")
    log.info("Fallback bot found - id: %s, name: %s", data[0].get("id"), data[0].get("name"))
    return data[0]

async def get_conversation_bot(user_id: str, conversation_id: int, token: str):
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
        }, token)
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
    return await get_bot(user_id, bot_id, token)

async def _rest_post(path: str, data: dict, token: str):
    """POST data to Supabase REST endpoint."""
    headers = {
        **_build_headers(token),
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates,return=representation",
    }
    url = f"{settings.SUPABASE_REST_URL}/{path}"
    log.info("Supabase POST - url: %s, data: %s", url, _sanitize_for_log(data))
    
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, headers=headers, json=data)
    
    log.debug("Supabase POST response - status: %s, body: %s", resp.status_code, resp.text)
    if resp.status_code not in (200, 201):
        log.error("Supabase insert failed - status: %s, error: %s", resp.status_code, resp.text)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Supabase insert failed: {resp.text}")
    
    log.info("Supabase POST successful")
    return resp.json()

async def _rest_patch(path: str, data: dict, token: str):
    """PATCH data to Supabase REST endpoint."""
    headers = {
        **_build_headers(token),
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates,return=representation",
    }
    url = f"{settings.SUPABASE_REST_URL}/{path}"
    log.info("Supabase PATCH - url: %s, data: %s", url, _sanitize_for_log(data))
    
    async with httpx.AsyncClient() as client:
        resp = await client.patch(url, headers=headers, json=data)
    
    log.debug("Supabase PATCH response - status: %s, body: %s", resp.status_code, resp.text)
    if resp.status_code != 200:
        log.error("Supabase update failed - status: %s, error: %s", resp.status_code, resp.text)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Supabase update failed: {resp.text}")
    
    log.info("Supabase PATCH successful")
    return resp.json()

async def _rest_rpc(function_name: str, data: dict, token: str):
    """POST data to Supabase RPC endpoint."""
    headers = {
        **_build_headers(token),
        "Content-Type": "application/json",
    }
    url = f"{settings.SUPABASE_REST_URL}/rpc/{function_name}"
    log.info("Supabase RPC - url: %s, data: %s", url, _sanitize_for_log(data))

    async with httpx.AsyncClient() as client:
        resp = await client.post(url, headers=headers, json=data)

    log.debug("Supabase RPC response - status: %s, body: %s", resp.status_code, resp.text)
    if resp.status_code != 200:
        log.error("Supabase RPC failed - status: %s, error: %s", resp.status_code, resp.text)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Supabase RPC failed: {resp.text}")

    log.info("Supabase RPC successful")
    return resp.json()

async def post_message(message: dict, token: str):
    """Write a message record to Supabase REST endpoint."""
    return await _rest_post("messages", message, token)


async def post_message_alternative(alternative: dict, token: str):
    """Write a message alternative record to Supabase REST endpoint."""
    return await _rest_post("message_alternatives", alternative, token)

async def update_message_alternative(alternative_id: int, updates: dict, token: str):
    """Update an existing message alternative record."""
    return await _rest_patch(f"message_alternatives?id=eq.{alternative_id}", updates, token)

async def get_message_alternatives(parent_message_id: int, user_id: str, token: str):
    """Get all alternatives for a parent message."""
    log.info("Getting message alternatives - parent_message_id: %s, user_id: %s", parent_message_id, user_id)
    params = {
        "parent_message_id": f"eq.{parent_message_id}",
        "user_id": f"eq.{user_id}",
        "order": "t.asc"
    }
    
    data = await _rest_get("message_alternatives", params, token)
    log.info("Found %d message alternatives", len(data))
    return data

async def get_openrouter_demo_bot(user_id: str, token: str):
    """Get existing OpenRouter demo bot for a user, if any."""
    log.info("Getting OpenRouter demo bot - user_id: %s", user_id)
    params = {
        "user_id": f"eq.{user_id}",
        "is_openrouter": "eq.true",
        "openrouter_key": "is.not.null",
        "limit": 1,
    }
    data = await _rest_get("bots", params, token)
    return data[0] if data else None

async def create_demo_openrouter_bot(
    user_id: str,
    or_key: str,
    model: str,
    access_path: str,
    name: str,
    token: str
):
    """Create OpenRouter demo key + bot via RPC (atomic)."""
    payload = {
        "p_user_id": user_id,
        "p_or_key": or_key,
        "p_model": model,
        "p_access_path": access_path,
        "p_name": name,
    }
    result = await _rest_rpc("create_demo_openrouter_bot", payload, token)
    if isinstance(result, list) and result:
        return result[0].get("bot_id")
    if isinstance(result, dict):
        return result.get("bot_id")
    return None

async def get_message_by_stream_id(stream_id: str, user_id: str, token: str):
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
        data = await _rest_get("messages", params, token)
        
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

async def get_message(message_id: int, user_id: str, token: str):
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
        data = await _rest_get("messages", params, token)
        
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
