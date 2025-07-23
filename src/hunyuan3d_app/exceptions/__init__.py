"""Custom exceptions for hunyuan3d_app."""

from .models import (
    ModelNotFoundError,
    ModelLoadError,
    ModelDownloadError,
    InvalidModelFormatError,
    InsufficientVRAMError,
)

from .generation import (
    GenerationError,
    GenerationTimeoutError,
    GenerationCancelledError,
    InvalidInputError,
    PipelineError,
)

from .validation import (
    ValidationError,
    InvalidImageFormatError,
    InvalidVideoFormatError,
    Invalid3DFormatError,
    FileSizeExceededError,
    DimensionExceededError,
)

__all__ = [
    # Model exceptions
    "ModelNotFoundError",
    "ModelLoadError",
    "ModelDownloadError",
    "InvalidModelFormatError",
    "InsufficientVRAMError",
    # Generation exceptions
    "GenerationError",
    "GenerationTimeoutError",
    "GenerationCancelledError",
    "InvalidInputError",
    "PipelineError",
    # Validation exceptions
    "ValidationError",
    "InvalidImageFormatError",
    "InvalidVideoFormatError",
    "Invalid3DFormatError",
    "FileSizeExceededError",
    "DimensionExceededError",
]