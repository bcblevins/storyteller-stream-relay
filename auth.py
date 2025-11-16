# auth.py
import httpx, time, json
from jose import jwt, JWTError
from fastapi import Request, HTTPException, status
from settings import settings

# simple in-memory JWKS cache

_jwks_cache = None
_jwks_cache_time = 0

# Optional: expected audience (Supabase typically uses 'authenticated')
EXPECTED_AUD = "authenticated"

async def verify_jwt(request: Request) -> str:
    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    if not auth or not auth.lower().startswith("bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing Bearer token")

    token = auth.split(" ", 1)[1].strip()
    try:
        # HS256 with your project's JWT secret
        payload = jwt.decode(
            token,
            settings.SUPABASE_JWT_SECRET,
            algorithms=["HS256"],
            options={"verify_aud": False}  # or True with audience=EXPECTED_AUD
            # audience=EXPECTED_AUD,
        )
    except JWTError as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, f"Invalid JWT: {e}")

    sub = payload.get("sub")
    if not sub:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "JWT missing sub")

    return sub  # user_id

