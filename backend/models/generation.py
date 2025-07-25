"""
Generation job models and API types
"""

from typing import Optional, Dict, Any, List, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field

try:
    from core.services.gpu_orchestrator import ExecutionMode
except ImportError:
    # Fallback if gpu_orchestrator is not available
    class ExecutionMode(str, Enum):
        AUTO = "auto"
        LOCAL = "local"
        CLOUD = "cloud"

class JobStatus(str, Enum):
    """Job status enum"""
    PENDING = "pending"
    RUNNING = "running"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class JobPriority(str, Enum):
    """Job priority enumeration"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"

class GenerationType(str, Enum):
    """Generation type enumeration"""
    TEXT_TO_IMAGE = "text_to_image"
    IMAGE_TO_IMAGE = "image_to_image"
    IMAGE_TO_3D = "image_to_3d"
    TEXT_TO_3D = "text_to_3d"  
    TEXT_TO_VIDEO = "text_to_video"
    IMAGE_TO_VIDEO = "image_to_video"
    FACE_SWAP = "face_swap"
    BATCH = "batch"

class GenerationRequest(BaseModel):
    """Base generation request model"""
    type: GenerationType
    params: Dict[str, Any]
    priority: JobPriority = JobPriority.NORMAL
    user_id: Optional[str] = None
    webhook_url: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

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

class JobResponse(BaseModel):
    """Job creation response"""
    job_id: str
    status: JobStatus
    type: Optional[str] = None
    created_at: datetime
    queue_position: Optional[int] = None
    estimated_time: Optional[int] = None  # seconds
    credits_cost: Optional[int] = None

class JobStatusResponse(BaseModel):
    """Job status response"""
    job_id: str
    status: JobStatus
    progress: float = Field(0.0, ge=0.0, le=1.0)
    message: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    eta: Optional[int] = None  # seconds remaining
    result: Optional[Dict[str, Any]] = None

class ModelInfo(BaseModel):
    """Model information"""
    id: str
    name: str
    type: str  # image, 3d, video
    status: str  # available, loading, unavailable
    description: Optional[str] = None
    version: Optional[str] = None
    size_mb: Optional[int] = None
    gpu_memory_gb: Optional[float] = None
    credits_per_generation: Optional[int] = None
    supported_features: List[str] = []
    requirements: Dict[str, Any] = {}

class BatchGenerationRequest(BaseModel):
    """Batch generation request"""
    type: GenerationType
    items: List[Dict[str, Any]] = Field(..., min_items=1, max_items=50)
    shared_params: Dict[str, Any] = {}
    priority: JobPriority = JobPriority.NORMAL
    max_parallel: int = Field(3, ge=1, le=10)

class BatchJobResponse(BaseModel):
    """Batch job response"""
    batch_id: str
    job_ids: List[str]
    status: JobStatus
    total_jobs: int
    completed_jobs: int = 0
    failed_jobs: int = 0
    total_credits: Optional[int] = None
    created_at: datetime

# Utility functions for backward compatibility and tests
def estimate_credits_cost(gen_type: str, params: Dict[str, Any]) -> int:
    """Estimate credits cost for generation"""
    base_costs = {
        "text_to_image": 1,
        "image_to_image": 1,
        "image_to_3d": 5,
        "text_to_3d": 7,
        "text_to_video": 10,
        "image_to_video": 8,
        "face_swap": 2,
    }
    
    base_cost = base_costs.get(gen_type, 1)
    
    # Apply multipliers based on parameters
    if gen_type in ["text_to_image", "image_to_image"]:
        # Higher resolution costs more
        width = params.get('width', 512)
        height = params.get('height', 512)
        pixels = width * height
        if pixels > 512 * 512:
            base_cost = int(base_cost * (pixels / (512 * 512)))
            
        # More steps cost more
        steps = params.get('num_inference_steps', 20)
        if steps > 20:
            base_cost = int(base_cost * (steps / 20))
            
    elif gen_type == "image_to_3d":
        # Higher quality presets cost more
        quality = params.get('quality_preset', 'standard')
        quality_multipliers = {'draft': 0.5, 'standard': 1.0, 'high': 2.0, 'ultra': 4.0}
        base_cost = int(base_cost * quality_multipliers.get(quality, 1.0))
        
    elif gen_type in ["text_to_video", "image_to_video"]:
        # Longer videos cost more
        duration = params.get('duration', 3.0)
        base_cost = int(base_cost * duration / 3.0)
    
    return max(1, base_cost)