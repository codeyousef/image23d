"""
Generation job models
"""

from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field

from core.services.gpu_orchestrator import ExecutionMode

class JobStatus(str, Enum):
    """Job status enum"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class GenerationJob(BaseModel):
    """Generation job model"""
    id: str
    user_id: str
    type: str  # image, 3d, video
    request: Dict[str, Any]
    status: JobStatus = JobStatus.PENDING
    execution_mode: ExecutionMode = ExecutionMode.AUTO
    credits_required: int
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    message: Optional[str] = None
    result_url: Optional[str] = None
    error: Optional[str] = None
    credits_used: Optional[int] = None
    cost_usd: Optional[float] = None
    
    class Config:
        use_enum_values = True