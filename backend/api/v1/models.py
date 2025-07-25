"""
Models API v1 endpoints
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.api.middleware.auth import get_current_user
from backend.models.generation import ModelInfo

router = APIRouter()

# Mock model data for testing
MOCK_MODELS = [
    {
        "id": "flux-1-dev",
        "name": "FLUX.1 [dev]",
        "type": "image",
        "status": "available",
        "description": "High-quality text-to-image model with excellent prompt following",
        "version": "1.0",
        "size_mb": 5200,
        "gpu_memory_gb": 8.0,
        "credits_per_generation": 1,
        "supported_features": ["text_to_image", "image_to_image"],
        "requirements": {
            "min_vram_gb": 8,
            "compute_capability": "7.0"
        }
    },
    {
        "id": "flux-1-schnell",
        "name": "FLUX.1 [schnell]", 
        "type": "image",
        "status": "available",
        "description": "Fast text-to-image model optimized for speed",
        "version": "1.0",
        "size_mb": 5200,
        "gpu_memory_gb": 6.0,
        "credits_per_generation": 1,
        "supported_features": ["text_to_image"],
        "requirements": {
            "min_vram_gb": 6,
            "compute_capability": "7.0"
        }
    },
    {
        "id": "hunyuan3d-2.1",
        "name": "HunyuanVideo 3D v2.1",
        "type": "3d",
        "status": "available", 
        "description": "Latest 3D model generation with improved quality",
        "version": "2.1",
        "size_mb": 8500,
        "gpu_memory_gb": 16.0,
        "credits_per_generation": 5,
        "supported_features": ["image_to_3d", "text_to_3d"],
        "requirements": {
            "min_vram_gb": 16,
            "compute_capability": "8.0"
        }
    },
    {
        "id": "hunyuan3d-mini",
        "name": "HunyuanVideo 3D Mini",
        "type": "3d", 
        "status": "available",
        "description": "Lightweight 3D model for faster generation",
        "version": "1.0",
        "size_mb": 3200,
        "gpu_memory_gb": 8.0,
        "credits_per_generation": 3,
        "supported_features": ["image_to_3d"],
        "requirements": {
            "min_vram_gb": 8,
            "compute_capability": "7.0"
        }
    },
    {
        "id": "ltx-video",
        "name": "LTX Video",
        "type": "video",
        "status": "available",
        "description": "High-quality video generation model",
        "version": "1.0", 
        "size_mb": 12000,
        "gpu_memory_gb": 20.0,
        "credits_per_generation": 10,
        "supported_features": ["text_to_video", "image_to_video"],
        "requirements": {
            "min_vram_gb": 20,
            "compute_capability": "8.0"
        }
    }
]

@router.get("", response_model=dict)
async def list_models(
    model_type: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """List all available models"""
    models = MOCK_MODELS
    
    # Filter by type if specified
    if model_type:
        models = [m for m in models if m["type"] == model_type]
    
    return {"models": models}

@router.get("/{model_id}")
async def get_model_details(
    model_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get detailed information about a specific model"""
    model = next((m for m in MOCK_MODELS if m["id"] == model_id), None)
    
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return model