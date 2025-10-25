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
