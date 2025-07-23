"""
Authentication middleware
"""

from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from jose import jwt, JWTError

from backend.api.auth import SECRET_KEY, ALGORITHM

class AuthMiddleware(BaseHTTPMiddleware):
    """
    Authentication middleware for protected routes
    """
    
    # Routes that don't require authentication
    PUBLIC_ROUTES = [
        "/",
        "/health",
        "/docs",
        "/openapi.json",
        "/api/auth/register",
        "/api/auth/token",
        "/api/auth/refresh"
    ]
    
    async def dispatch(self, request: Request, call_next):
        # Skip auth for public routes
        if any(request.url.path.startswith(route) for route in self.PUBLIC_ROUTES):
            return await call_next(request)
        
        # Skip auth for static files
        if request.url.path.startswith("/outputs/"):
            return await call_next(request)
        
        # Extract token from Authorization header
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return HTTPException(
                status_code=401,
                detail="Missing or invalid authorization header"
            )
        
        token = auth_header.split(" ")[1]
        
        # Verify token
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            request.state.user = payload
        except JWTError:
            return HTTPException(
                status_code=401,
                detail="Invalid or expired token"
            )
        
        # Continue with request
        response = await call_next(request)
        return response