"""
Model management endpoints
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel

from backend.api.auth import get_current_user
from backend.services.model_service import ModelService

router = APIRouter()

# Response models
class ModelInfo(BaseModel):
    id: str
    name: str
    type: str  # image, 3d, video
    size: str
    vram_required: str
    description: str
    is_downloaded: bool
    download_progress: Optional[float] = None
    is_downloading: bool = False

class ModelDownloadRequest(BaseModel):
    model_id: str
    model_type: str

class ModelDownloadResponse(BaseModel):
    job_id: str
    message: str

# Endpoints
@router.get("/", response_model=List[ModelInfo])
async def list_models(
    model_type: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
    model_service: ModelService = Depends()
):
    """List all available models"""
    
    models = await model_service.list_models(model_type)
    
    return [
        ModelInfo(
            id=model.id,
            name=model.name,
            type=model.type,
            size=model.size,
            vram_required=model.vram_required,
            description=model.description,
            is_downloaded=model.is_downloaded,
            download_progress=model.download_progress,
            is_downloading=model.is_downloading
        )
        for model in models
    ]

@router.get("/{model_id}")
async def get_model_info(
    model_id: str,
    current_user: dict = Depends(get_current_user),
    model_service: ModelService = Depends()
):
    """Get detailed information about a specific model"""
    
    model = await model_service.get_model(model_id)
    
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return {
        "id": model.id,
        "name": model.name,
        "type": model.type,
        "size": model.size,
        "vram_required": model.vram_required,
        "description": model.description,
        "is_downloaded": model.is_downloaded,
        "download_progress": model.download_progress,
        "is_downloading": model.is_downloading,
        "repo_id": model.repo_id,
        "files": model.files if model.is_downloaded else [],
        "requirements": model.requirements
    }

@router.post("/download", response_model=ModelDownloadResponse)
async def download_model(
    request: ModelDownloadRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    model_service: ModelService = Depends()
):
    """Start downloading a model"""
    
    # Check if model exists
    model = await model_service.get_model(request.model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Check if already downloaded
    if model.is_downloaded:
        return ModelDownloadResponse(
            job_id="",
            message="Model already downloaded"
        )
    
    # Check if already downloading
    if model.is_downloading:
        return ModelDownloadResponse(
            job_id=model.download_job_id,
            message="Model download already in progress"
        )
    
    # Start download in background
    job_id = await model_service.start_download(
        model_id=request.model_id,
        user_id=current_user["id"]
    )
    
    background_tasks.add_task(
        model_service.download_model,
        model_id=request.model_id,
        job_id=job_id
    )
    
    return ModelDownloadResponse(
        job_id=job_id,
        message=f"Started downloading {model.name}"
    )

@router.delete("/{model_id}")
async def delete_model(
    model_id: str,
    current_user: dict = Depends(get_current_user),
    model_service: ModelService = Depends()
):
    """Delete a downloaded model"""
    
    # Check if model exists
    model = await model_service.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Check if downloaded
    if not model.is_downloaded:
        raise HTTPException(
            status_code=400,
            detail="Model is not downloaded"
        )
    
    # Delete model files
    success = await model_service.delete_model(model_id)
    
    if not success:
        raise HTTPException(
            status_code=500,
            detail="Failed to delete model"
        )
    
    return {"message": f"Successfully deleted {model.name}"}

@router.get("/download/progress/{job_id}")
async def get_download_progress(
    job_id: str,
    current_user: dict = Depends(get_current_user),
    model_service: ModelService = Depends()
):
    """Get download progress for a job"""
    
    progress = await model_service.get_download_progress(job_id)
    
    if progress is None:
        raise HTTPException(
            status_code=404,
            detail="Download job not found"
        )
    
    return progress

@router.post("/validate/{model_id}")
async def validate_model(
    model_id: str,
    current_user: dict = Depends(get_current_user),
    model_service: ModelService = Depends()
):
    """Validate a downloaded model"""
    
    # Check if model exists and is downloaded
    model = await model_service.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    if not model.is_downloaded:
        raise HTTPException(
            status_code=400,
            detail="Model is not downloaded"
        )
    
    # Validate model files
    validation_result = await model_service.validate_model(model_id)
    
    return {
        "model_id": model_id,
        "is_valid": validation_result["is_valid"],
        "missing_files": validation_result.get("missing_files", []),
        "message": validation_result.get("message", "Validation complete")
    }