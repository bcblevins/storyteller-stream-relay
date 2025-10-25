# supabase.py
import httpx
from fastapi import HTTPException, status
from settings import settings

async def get_bot(user_id: str, bot_id: int):
    """Fetch bot credentials for a given user & bot."""
    url = f"{settings.supabase_rest_url}/bots"
    params = {"id": f"eq.{bot_id}", "user_id": f"eq.{user_id}"}
    headers = {
        "apikey": settings.supabase_service_role_key,
        "Authorization": f"Bearer {settings.supabase_service_role_key}",
    }
    async with httpx.AsyncClient() as client:
        r = await client.get(url, headers=headers, params=params, timeout=10)
    if r.status_code != 200:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Supabase fetch failed")
    data = r.json()
    if not data:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Bot not found or unauthorized")
    return data[0]



async def post_message(message: dict):
    """Write a message record to Supabase REST endpoint."""
    url = f"{settings.supabase_rest_url}/messages"
    headers = {
        "apikey": settings.supabase_service_role_key,
        "Authorization": f"Bearer {settings.supabase_service_role_key}",
        "Content-Type": "application/json",
        # merge duplicates on conflict if unique index exists
        "Prefer": "resolution=merge-duplicates,return=representation",
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(url, headers=headers, json=message)
        if resp.status_code not in (200, 201):
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Supabase insert failed: {resp.text}")
        return resp.json()