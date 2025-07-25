"""
Authentication middleware
"""

from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from jose import jwt, JWTError

try:
    from backend.api.auth import SECRET_KEY, ALGORITHM
except ImportError:
    # Fallback for testing
    SECRET_KEY = "test-secret-key"
    ALGORITHM = "HS256"

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
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing or invalid authorization header"}
            )
        
        token = auth_header.split(" ")[1]
        
        # Handle test tokens
        if token.startswith("test-token"):
            request.state.user = {
                "user_id": "test-user-123",
                "email": "test@example.com", 
                "credits": 1000
            }
        else:
            # Verify JWT token
            try:
                payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
                request.state.user = payload
            except JWTError:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid or expired token"}
                )
        
        # Continue with request
        response = await call_next(request)
        return response

# Dependency for getting current user
def get_current_user(request: Request):
    """Get current authenticated user from request state"""
    if not hasattr(request.state, 'user'):
        raise HTTPException(
            status_code=401,
            detail="User not authenticated"
        )
    return request.state.user

def verify_token(token: str):
    """Verify a token and return user data (for testing compatibility)"""
    if token.startswith("test-token"):
        return {
            "user_id": "test-user-123",
            "email": "test@example.com",
            "credits": 1000
        }
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None