"""Generation-related type definitions."""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Callable
from pathlib import Path


class GenerationType(Enum):
    """Types of generation tasks."""
    IMAGE = "image"
    THREED = "3d"
    VIDEO = "video"
    TEXTURE = "texture"
    MULTIVIEW = "multiview"


class GenerationStatus(Enum):
    """Status of a generation task."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class GenerationRequest:
    """Base class for generation requests."""
    prompt: str
    negative_prompt: Optional[str] = None
    seed: Optional[int] = None
    model_id: Optional[str] = None
    quality_preset: str = "standard"
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class GenerationResult:
    """Base class for generation results."""
    request_id: str
    status: GenerationStatus
    output_path: Optional[Path] = None
    preview_paths: Optional[List[Path]] = None
    error: Optional[str] = None
    generation_time: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


# Type alias for progress callbacks
ProgressCallback = Callable[[float, str], None]