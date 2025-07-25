"""
Rate limiting middleware
"""

import time
from collections import defaultdict
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response, JSONResponse

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware to prevent abuse
    """
    
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests = defaultdict(list)
        
    async def dispatch(self, request: Request, call_next) -> Response:
        # Skip rate limiting for certain paths
        if request.url.path in ["/", "/health", "/docs", "/openapi.json"]:
            return await call_next(request)
        
        # Get client identifier (IP or user ID)
        client_id = request.client.host if request.client else "unknown"
        if hasattr(request.state, "user"):
            client_id = request.state.user.get("sub", client_id)
        
        # Clean old requests
        current_time = time.time()
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if current_time - req_time < 60
        ]
        
        # Check rate limit  
        if len(self.requests[client_id]) >= self.requests_per_minute:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Please try again later."}
            )
        
        # Record request
        self.requests[client_id].append(current_time)
        
        # Add rate limit headers
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(
            self.requests_per_minute - len(self.requests[client_id])
        )
        response.headers["X-RateLimit-Reset"] = str(int(current_time + 60))
        
        return response