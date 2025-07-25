"""
NeuralForge Studio Web Backend - Main Application
"""

import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.api import auth, generation, models, queue, websocket
from backend.api.v1.main import router as v1_router
from backend.api.middleware.auth import AuthMiddleware
from backend.api.middleware.rate_limit import RateLimitMiddleware
from backend.services import init_services

try:
    from core.config import OUTPUT_DIR, MODELS_DIR
except ImportError:
    from pathlib import Path
    OUTPUT_DIR = Path("outputs")
    MODELS_DIR = Path("models")

# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    print("🚀 Starting NeuralForge Studio API...")
    await init_services()
    
    yield
    
    # Shutdown
    print("👋 Shutting down NeuralForge Studio API...")

# Create FastAPI app
app = FastAPI(
    title="NeuralForge Studio API",
    description="""
    NeuralForge Studio is a comprehensive AI creative suite for generating high-quality images, 3D models, and videos.
    
    ## Features
    
    * **Image Generation** - Create stunning images using FLUX models
    * **3D Model Generation** - Generate 3D models with HunyuanVideo 3D
    * **Prompt Enhancement** - Enhance prompts using local LLM
    * **Real-time Updates** - WebSocket support for progress tracking
    * **Credit System** - Usage-based billing with credit management
    
    ## Authentication
    
    The API uses JWT tokens for authentication:
    - Access Token: Short-lived (30 min), used for API requests
    - Refresh Token: Long-lived (7 days), used to get new access tokens
    
    Include the access token in the Authorization header:
    ```
    Authorization: Bearer <access_token>
    ```
    
    ## Rate Limiting
    
    Default: 60 requests per minute per user/IP
    """,
    version="1.0.0",
    lifespan=lifespan,
    openapi_tags=[
        {"name": "Authentication", "description": "User authentication and registration"},
        {"name": "Generation", "description": "Content generation endpoints"},
        {"name": "Models", "description": "Model management and downloads"},
        {"name": "Queue", "description": "Job queue and status management"},
        {"name": "WebSocket", "description": "Real-time progress updates"}
    ],
    servers=[
        {"url": "http://localhost:8000", "description": "Development server"},
        {"url": "https://api.neuralforge.studio", "description": "Production server"}
    ]
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom middleware
app.add_middleware(RateLimitMiddleware, requests_per_minute=10)  # Lower limit for testing
app.add_middleware(AuthMiddleware)

# Mount static files for generated content
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

# Include v1 API router
app.include_router(v1_router)

# Include legacy API routers for backward compatibility
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(generation.router, prefix="/api/generate", tags=["Generation"])
app.include_router(models.router, prefix="/api/models", tags=["Models"])
app.include_router(queue.router, prefix="/api/queue", tags=["Queue"])

# Include WebSocket endpoint
app.include_router(websocket.router, prefix="/ws", tags=["WebSocket"])

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "NeuralForge Studio API",
        "version": "1.0.0",
        "status": "online",
        "docs": "/docs",
        "health": "/health"
    }

# Health check endpoint
@app.get("/health")
async def health():
    """Health check endpoint"""
    from datetime import datetime
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "api": "online",
            "gpu": "available",  # Check actual GPU status
            "models": "loaded"   # Check model status
        }
    }

if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )