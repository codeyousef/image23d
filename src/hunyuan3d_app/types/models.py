"""Model-related type definitions."""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any


class ModelType(Enum):
    """Types of models in the system."""
    IMAGE = "image"
    THREED = "3d"
    VIDEO = "video"
    LORA = "lora"
    CONTROLNET = "controlnet"
    TEXTURE = "texture"


class ModelStatus(Enum):
    """Status of a model."""
    NOT_DOWNLOADED = "not_downloaded"
    DOWNLOADING = "downloading"
    DOWNLOADED = "downloaded"
    LOADING = "loading"
    LOADED = "loaded"
    ERROR = "error"


class QuantizationLevel(Enum):
    """GGUF quantization levels."""
    Q8_0 = "Q8_0"      # Best quality
    Q6_K = "Q6_K"      # Balanced
    Q5_K_M = "Q5_K_M"  # Efficient
    Q4_K_S = "Q4_K_S"  # Memory saver
    Q3_K_M = "Q3_K_M"  # Low VRAM
    Q2_K = "Q2_K"      # Ultra low VRAM
    FP8 = "FP8"        # FP8 quantization
    FP16 = "FP16"      # No quantization


@dataclass
class ModelInfo:
    """Information about a model."""
    id: str
    name: str
    type: ModelType
    repo_id: str
    size: str
    vram_required: str
    description: str
    status: ModelStatus = ModelStatus.NOT_DOWNLOADED
    is_gated: bool = False
    is_gguf: bool = False
    quantization: Optional[QuantizationLevel] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class DownloadProgress:
    """Progress information for model downloads."""
    model_id: str
    downloaded_size: int
    total_size: int
    speed: float  # bytes per second
    eta_seconds: Optional[int] = None
    status: str = "downloading"
    error: Optional[str] = None
    
    @property
    def progress_percentage(self) -> float:
        """Calculate progress percentage."""
        if self.total_size == 0:
            return 0.0
        return (self.downloaded_size / self.total_size) * 100