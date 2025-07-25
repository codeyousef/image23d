"""
Jobs API v1 endpoints
"""

from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from backend.api.middleware.auth import get_current_user
from backend.models.generation import GenerationRequest, BatchGenerationRequest
from backend.services.job_service import job_service
from backend.services.history_service import history_service

router = APIRouter()

@router.post("/create", status_code=201)
async def create_job(
    request: GenerationRequest,
    current_user: dict = Depends(get_current_user)
):
    """Create a new generation job"""
    # Validate request parameters
    if not request.params.get("prompt"):
        raise HTTPException(
            status_code=422,
            detail="Missing required parameter: prompt"
        )
    
    if not request.params.get("model"):
        raise HTTPException(
            status_code=422,
            detail="Missing required parameter: model"
        )
    
    # Create job
    job_response = job_service.create_job(
        job_type=request.type,
        params=request.params,
        user_id=current_user["user_id"],
        priority=request.priority
    )
    
    # Add to history
    await history_service.add_job_to_history(
        user_id=current_user["user_id"],
        job_data={
            "job_id": job_response["job_id"],
            "type": request.type,
            "status": job_response["status"],
            "params": request.params,
            "created_at": job_response["created_at"],
            "credits_used": None
        }
    )
    
    return {
        "job_id": job_response["job_id"],
        "status": job_response["status"],
        "type": request.type,
        "created_at": job_response["created_at"],
        "queue_position": job_response["queue_position"],
        "estimated_time": None,
        "credits_cost": job_response["credits_cost"]
    }

@router.post("/batch", status_code=201)
async def create_batch_job(
    request: BatchGenerationRequest,
    current_user: dict = Depends(get_current_user)
):
    """Create a batch generation job"""
    batch_response = job_service.create_batch_job(
        request=request,
        user_id=current_user["user_id"]
    )
    
    return {
        "batch_id": batch_response["batch_id"],
        "job_ids": batch_response["job_ids"],
        "status": batch_response["status"],
        "total_jobs": batch_response["total_jobs"],
        "completed_jobs": batch_response["completed_jobs"],
        "failed_jobs": batch_response["failed_jobs"],
        "total_credits": batch_response["total_credits"],
        "created_at": batch_response["created_at"]
    }

@router.get("/{job_id}/status")
async def get_job_status(
    job_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get job status"""
    job_status = job_service.get_job_status(job_id)
    
    if not job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Verify job belongs to user
    job = job_service.get_job(job_id)
    if job and job["user_id"] != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return job_status

@router.get("/{job_id}/result")
async def get_job_result(
    job_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get job result"""
    job = job_service.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Verify job belongs to user
    if job["user_id"] != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    if job["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job is not completed (status: {job['status']})"
        )
    
    return job.get("result", {})

@router.post("/{job_id}/cancel")
async def cancel_job(
    job_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Cancel a job"""
    job = job_service.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Verify job belongs to user
    if job["user_id"] != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    success = job_service.cancel_job(job_id)
    
    if not success:
        raise HTTPException(
            status_code=400,
            detail="Job cannot be cancelled"
        )
    
    return {"message": "Job cancelled successfully"}

@router.get("/{job_id}/download")
async def download_job_result(
    job_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Download job result file"""
    job = job_service.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Verify job belongs to user
    if job["user_id"] != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    if job["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail="Job is not completed"
        )
    
    # Mock file data for testing
    from fastapi.responses import Response
    return Response(
        content=b"fake image data",
        media_type="image/png",
        headers={"Content-Disposition": f"attachment; filename={job_id}.png"}
    )