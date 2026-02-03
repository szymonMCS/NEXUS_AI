# api/auth.py
"""
Clerk Authentication Module for FastAPI
Verifies JWT tokens from Clerk frontend
"""

import os
import jwt
from typing import Optional, Dict, Any
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import requests
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

# Security scheme for JWT
token_auth_scheme = HTTPBearer(auto_error=False)


@lru_cache(maxsize=1)
def get_clerk_jwks() -> Dict[str, Any]:
    """Fetch Clerk JWKS (JSON Web Key Set) for token verification."""
    jwks_url = os.getenv("CLERK_JWKS_URL")
    if not jwks_url:
        # Default fallback for development
        logger.warning("CLERK_JWKS_URL not set, using development mode")
        return {}
    
    try:
        response = requests.get(jwks_url, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Failed to fetch Clerk JWKS: {e}")
        return {}


def verify_clerk_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Verify a Clerk JWT token.
    
    Args:
        token: JWT token from Clerk frontend
        
    Returns:
        Decoded token payload or None if invalid
    """
    jwks = get_clerk_jwks()
    if not jwks or "keys" not in jwks:
        logger.warning("No JWKS available, skipping token verification")
        # In development mode, return a mock payload
        if os.getenv("APP_MODE") == "development":
            return {"sub": "dev-user", "email": "dev@example.com"}
        return None
    
    try:
        # Get the key ID from token header
        unverified_header = jwt.get_unverified_header(token)
        kid = unverified_header.get("kid")
        
        if not kid:
            logger.error("Token missing 'kid' header")
            return None
        
        # Find matching key
        matching_key = None
        for key in jwks["keys"]:
            if key.get("kid") == kid:
                matching_key = key
                break
        
        if not matching_key:
            logger.error(f"No matching key found for kid: {kid}")
            return None
        
        # Convert JWK to PEM
        from jwt.algorithms import RSAAlgorithm
        public_key = RSAAlgorithm.from_jwk(matching_key)
        
        # Verify token
        issuer = os.getenv("CLERK_ISSUER")
        payload = jwt.decode(
            token,
            public_key,
            algorithms=["RS256"],
            issuer=issuer,
            options={"verify_aud": False}  # Clerk tokens don't always have aud
        )
        
        return payload
        
    except jwt.ExpiredSignatureError:
        logger.error("Token has expired")
        return None
    except jwt.InvalidTokenError as e:
        logger.error(f"Invalid token: {e}")
        return None
    except Exception as e:
        logger.error(f"Token verification error: {e}")
        return None


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(token_auth_scheme)
) -> Optional[Dict[str, Any]]:
    """
    Dependency to get current authenticated user.
    
    Usage:
        @app.get("/protected")
        async def protected_route(user: dict = Depends(get_current_user)):
            return {"user": user}
    """
    # Allow unauthenticated access in development mode
    if os.getenv("APP_MODE") == "development" and not credentials:
        return {"sub": "dev-user", "email": "dev@example.com", "mode": "development"}
    
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = credentials.credentials
    payload = verify_clerk_token(token)
    
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return payload


async def require_auth(
    credentials: HTTPAuthorizationCredentials = Depends(token_auth_scheme)
) -> Dict[str, Any]:
    """
    Strict dependency that requires authentication.
    
    Usage:
        @app.post("/api/analysis")
        async def run_analysis(user: dict = Depends(require_auth)):
            # Only authenticated users can access
            pass
    """
    return await get_current_user(credentials)


# Optional auth - returns user if authenticated, None otherwise
async def optional_auth(
    credentials: HTTPAuthorizationCredentials = Depends(token_auth_scheme)
) -> Optional[Dict[str, Any]]:
    """
    Optional dependency that returns user if authenticated, None otherwise.
    
    Usage:
        @app.get("/api/status")
        async def get_status(user: Optional[dict] = Depends(optional_auth)):
            # Works for both authenticated and unauthenticated users
            return {"status": "ok", "user": user}
    """
    if not credentials:
        return None
    
    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None
