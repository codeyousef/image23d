"""
Upload API v1 endpoints
"""

import io
import uuid
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from PIL import Image

from backend.api.middleware.auth import get_current_user

router = APIRouter()

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

@router.post("/image")
async def upload_image(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """Upload an image file"""
    # Check file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )
    
    # Read file content
    content = await file.read()
    
    # Check file size
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File too large. Maximum size is 10MB"
        )
    
    try:
        # Validate and get image info
        image = Image.open(io.BytesIO(content))
        width, height = image.size
        
        # Generate file ID
        file_id = f"img-{uuid.uuid4().hex[:8]}"
        
        # Mock URL for testing
        url = f"https://api.example.com/uploads/{file_id}.{image.format.lower()}"
        
        return {
            "file_id": file_id,
            "url": url,
            "size": len(content),
            "dimensions": {
                "width": width,
                "height": height
            },
            "format": image.format.lower()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file"
        )