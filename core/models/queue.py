"""
Models for job queue management
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Union
from enum import Enum
from datetime import datetime
import uuid

class JobPriority(str, Enum):
    """Job priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"

class JobType(str, Enum):
    """Types of jobs"""
    IMAGE = "image"
    THREE_D = "3d"
    VIDEO = "video"
    FACE_SWAP = "face_swap"
    BATCH = "batch"

class JobStatus(str, Enum):
    """Job execution status"""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class JobRequest(BaseModel):
    """Job request model"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique job ID")
    type: JobType = Field(..., description="Type of job")
    priority: JobPriority = Field(JobPriority.NORMAL, description="Job priority")
    params: Dict[str, Any] = Field(..., description="Job parameters")
    user_id: Optional[str] = Field(None, description="User ID for web app")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation time")
    scheduled_at: Optional[datetime] = Field(None, description="Scheduled execution time")
    
class JobProgress(BaseModel):
    """Job progress update"""
    job_id: str = Field(..., description="Job ID")
    status: JobStatus = Field(..., description="Current status")
    progress: float = Field(0.0, ge=0.0, le=100.0, description="Progress percentage")
    message: Optional[str] = Field(None, description="Progress message")
    current_step: Optional[int] = Field(None, description="Current step number")
    total_steps: Optional[int] = Field(None, description="Total number of steps")
    eta_seconds: Optional[int] = Field(None, description="Estimated time remaining")
    
class JobResult(BaseModel):
    """Job execution result"""
    job_id: str = Field(..., description="Job ID")
    status: JobStatus = Field(..., description="Final status")
    result: Optional[Union[Dict[str, Any], str]] = Field(None, description="Job result data")
    error: Optional[str] = Field(None, description="Error message if failed")
    started_at: datetime = Field(..., description="Start time")
    completed_at: datetime = Field(..., description="Completion time")
    duration_seconds: float = Field(..., description="Execution duration")
    cost_credits: Optional[int] = Field(None, description="Credits consumed (web app)")
    
class QueueStats(BaseModel):
    """Queue statistics"""
    total_queued: int = Field(0, description="Total jobs in queue")
    total_running: int = Field(0, description="Total running jobs")
    by_priority: Dict[JobPriority, int] = Field(default_factory=dict, description="Jobs by priority")
    by_type: Dict[JobType, int] = Field(default_factory=dict, description="Jobs by type")
    average_wait_time: float = Field(0.0, description="Average wait time in seconds")
    average_execution_time: float = Field(0.0, description="Average execution time in seconds")