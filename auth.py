# auth.py
import httpx, time, json
from jose import jwt
from fastapi import Request, HTTPException, status
from settings import settings

# simple in-memory JWKS cache
_jwks_cache = {"keys": None, "exp": 0}

async def get_jwks():
    now = time.time()
    if _jwks_cache["keys"] and now < _jwks_cache["exp"]:
        return _jwks_cache["keys"]
    async with httpx.AsyncClient() as client:
        r = await client.get(settings.supabase_jwks_url, timeout=10)
        r.raise_for_status()
        data = r.json()
    _jwks_cache["keys"] = data
    _jwks_cache["exp"] = now + 3600  # cache 1 hour
    return data

async def verify_jwt(request: Request) -> str:
    """Return user_id (sub) if valid, raise HTTPException otherwise."""
    auth = request.headers.get("authorization")
    if not auth or not auth.lower().startswith("bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing Bearer token")

    token = auth.split(" ")[1]
    jwks = await get_jwks()

    try:
        # auto-select correct key via 'kid'
        user = jwt.decode(token, jwks, algorithms=["RS256"], options={"verify_aud": False})
        return user["sub"]   # Supabase user ID
    except Exception:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid JWT")
