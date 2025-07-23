"""Type definitions for hunyuan3d_app."""

from .models import (
    ModelType,
    ModelStatus,
    QuantizationLevel,
    ModelInfo,
    DownloadProgress,
)

from .generation import (
    GenerationType,
    GenerationStatus,
    GenerationRequest,
    GenerationResult,
    ProgressCallback,
)

from .ui import (
    TabType,
    UIState,
    NotificationType,
    ComponentVisibility,
)

__all__ = [
    # Model types
    "ModelType",
    "ModelStatus",
    "QuantizationLevel",
    "ModelInfo",
    "DownloadProgress",
    # Generation types
    "GenerationType",
    "GenerationStatus",
    "GenerationRequest",
    "GenerationResult",
    "ProgressCallback",
    # UI types
    "TabType",
    "UIState",
    "NotificationType",
    "ComponentVisibility",
]