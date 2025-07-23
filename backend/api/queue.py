"""
Queue management endpoints
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from datetime import datetime

from backend.api.auth import get_current_user
from backend.services.queue_service import QueueService
from backend.models.generation import JobStatus

router = APIRouter()

# Response models
class QueueJob(BaseModel):
    id: str
    type: str  # image, 3d, video
    status: JobStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    message: Optional[str] = None
    result_url: Optional[str] = None
    error: Optional[str] = None
    credits_used: Optional[int] = None
    cost_usd: Optional[float] = None

class QueuePosition(BaseModel):
    job_id: str
    position: int
    estimated_wait_seconds: int
    ahead_in_queue: int

# Endpoints
@router.get("/jobs", response_model=List[QueueJob])
async def get_user_jobs(
    status: Optional[JobStatus] = None,
    limit: int = 20,
    offset: int = 0,
    current_user: dict = Depends(get_current_user),
    queue_service: QueueService = Depends()
):
    """Get user's generation jobs"""
    
    jobs = await queue_service.get_user_jobs(
        user_id=current_user["id"],
        status=status,
        limit=limit,
        offset=offset
    )
    
    return [
        QueueJob(
            id=job.id,
            type=job.type,
            status=job.status,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            progress=job.progress,
            message=job.message,
            result_url=job.result_url,
            error=job.error,
            credits_used=job.credits_used,
            cost_usd=job.cost_usd
        )
        for job in jobs
    ]

@router.get("/job/{job_id}", response_model=QueueJob)
async def get_job_status(
    job_id: str,
    current_user: dict = Depends(get_current_user),
    queue_service: QueueService = Depends()
):
    """Get status of a specific job"""
    
    job = await queue_service.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Verify job belongs to user
    if job.user_id != current_user["id"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return QueueJob(
        id=job.id,
        type=job.type,
        status=job.status,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        progress=job.progress,
        message=job.message,
        result_url=job.result_url,
        error=job.error,
        credits_used=job.credits_used,
        cost_usd=job.cost_usd
    )

@router.delete("/job/{job_id}")
async def cancel_job(
    job_id: str,
    current_user: dict = Depends(get_current_user),
    queue_service: QueueService = Depends()
):
    """Cancel a pending or running job"""
    
    job = await queue_service.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Verify job belongs to user
    if job.user_id != current_user["id"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Check if job can be cancelled
    if job.status not in [JobStatus.PENDING, JobStatus.RUNNING]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job with status: {job.status}"
        )
    
    # Cancel the job
    success = await queue_service.cancel_job(job_id)
    
    if not success:
        raise HTTPException(
            status_code=500,
            detail="Failed to cancel job"
        )
    
    return {"message": "Job cancelled successfully"}

@router.get("/position/{job_id}", response_model=QueuePosition)
async def get_queue_position(
    job_id: str,
    current_user: dict = Depends(get_current_user),
    queue_service: QueueService = Depends()
):
    """Get queue position for a pending job"""
    
    job = await queue_service.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Verify job belongs to user
    if job.user_id != current_user["id"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Get queue position
    position_info = await queue_service.get_queue_position(job_id)
    
    if not position_info:
        raise HTTPException(
            status_code=400,
            detail="Job is not in queue"
        )
    
    return QueuePosition(
        job_id=job_id,
        position=position_info["position"],
        estimated_wait_seconds=position_info["estimated_wait"],
        ahead_in_queue=position_info["ahead_in_queue"]
    )

@router.get("/stats")
async def get_queue_stats(
    current_user: dict = Depends(get_current_user),
    queue_service: QueueService = Depends()
):
    """Get queue statistics"""
    
    stats = await queue_service.get_queue_stats()
    
    return {
        "total_pending": stats["total_pending"],
        "total_running": stats["total_running"],
        "average_wait_time": stats["average_wait_time"],
        "average_processing_time": stats["average_processing_time"],
        "gpu_utilization": stats["gpu_utilization"],
        "user_stats": {
            "total_jobs": stats["user_stats"][current_user["id"]]["total"],
            "completed_jobs": stats["user_stats"][current_user["id"]]["completed"],
            "failed_jobs": stats["user_stats"][current_user["id"]]["failed"],
            "total_credits_used": stats["user_stats"][current_user["id"]]["credits_used"]
        }
    }

@router.post("/retry/{job_id}")
async def retry_job(
    job_id: str,
    current_user: dict = Depends(get_current_user),
    queue_service: QueueService = Depends()
):
    """Retry a failed job"""
    
    job = await queue_service.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Verify job belongs to user
    if job.user_id != current_user["id"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Check if job can be retried
    if job.status != JobStatus.FAILED:
        raise HTTPException(
            status_code=400,
            detail="Only failed jobs can be retried"
        )
    
    # Check user credits
    if current_user["credits"] < job.credits_required:
        raise HTTPException(
            status_code=402,
            detail=f"Insufficient credits. Required: {job.credits_required}"
        )
    
    # Retry the job
    new_job_id = await queue_service.retry_job(job_id)
    
    return {
        "message": "Job queued for retry",
        "new_job_id": new_job_id
    }