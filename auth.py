# auth.py
import httpx, time, json
import logging
import traceback
from jose import jwt, JWTError
from fastapi import Request, HTTPException, status
from settings import settings

# Configure logger
logger = logging.getLogger(__name__)

# simple in-memory JWKS cache

_jwks_cache = None
_jwks_cache_time = 0

# Optional: expected audience (Supabase typically uses 'authenticated')
EXPECTED_AUD = "authenticated"

async def verify_jwt(request: Request) -> str:
    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    
    # Debug logging for authorization header (without exposing token)
    logger.debug("Authorization header present: %s", auth is not None)
    if auth:
        logger.debug("Authorization header type: Bearer")
    
    if not auth or not auth.lower().startswith("bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing Bearer token")

    token = auth.split(" ", 1)[1].strip()
    logger.debug("Token extracted successfully")
    
    try:
        # Parse JWT header without verification to extract algorithm info
        try:
            header = jwt.get_unverified_header(token)
            logger.debug(f"JWT Header: {header}")
        except Exception as header_error:
            logger.warning(f"Failed to parse JWT header: {header_error}")
        
        # Log obfuscated JWT secret for debugging
        secret = settings.SUPABASE_JWT_SECRET
        if secret:
            obfuscated_secret = f"{secret[:4]}***{secret[-4:]}" if len(secret) > 8 else "***"
            logger.debug(f"JWT Secret (obfuscated): {obfuscated_secret}")
        else:
            logger.error("SUPABASE_JWT_SECRET is empty or not set")
        
        # HS256 with your project's JWT secret
        payload = jwt.decode(
            token,
            settings.SUPABASE_JWT_SECRET,
            algorithms=["HS256"],
            options={"verify_aud": False}  # or True with audience=EXPECTED_AUD
            # audience=EXPECTED_AUD,
        )
    except JWTError as e:
        # Secure error logging without exposing tokens
        logger.error("JWT Verification Failed: %s: %s", type(e).__name__, str(e))
        
        # Log full traceback for debugging
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, f"Invalid JWT: {e}")

    sub = payload.get("sub")
    if not sub:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "JWT missing sub")

    return sub  # user_id