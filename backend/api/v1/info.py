"""
API info endpoints
"""

from fastapi import APIRouter
from datetime import datetime

router = APIRouter()

@router.get("/info")
async def get_api_info():
    """Get API information"""
    return {
        "name": "NeuralForge Studio API",
        "version": "1.0.0",
        "description": "AI creative suite for generating images, 3D models, and videos",
        "features": [
            "text_to_image",
            "image_to_image", 
            "image_to_3d",
            "text_to_3d",
            "text_to_video",
            "image_to_video",
            "face_swap",
            "batch_generation"
        ],
        "models": {
            "image": ["flux-1-dev", "flux-1-schnell", "sdxl-turbo"],
            "3d": ["hunyuan3d-2.1", "hunyuan3d-2.0", "hunyuan3d-mini"],
            "video": ["ltx-video", "cogvideox-5b"]
        },
        "limits": {
            "max_image_size": "2048x2048",
            "max_video_duration": 10,
            "max_batch_size": 50,
            "max_file_size_mb": 10
        },
        "timestamp": datetime.utcnow().isoformat()
    }