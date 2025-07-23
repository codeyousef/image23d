"""Validation-related exceptions."""

from ..config.constants import (
    MAX_IMAGE_SIZE_MB,
    MAX_VIDEO_SIZE_MB,
    MAX_3D_SIZE_MB,
    MAX_IMAGE_WIDTH,
    MAX_IMAGE_HEIGHT,
)


class ValidationError(Exception):
    """Base exception for validation errors."""
    pass


class InvalidImageFormatError(ValidationError):
    """Raised when image format is invalid."""
    def __init__(self, format: str, supported_formats: set):
        self.format = format
        self.supported_formats = supported_formats
        super().__init__(
            f"Invalid image format: {format}. "
            f"Supported formats: {', '.join(sorted(supported_formats))}"
        )


class InvalidVideoFormatError(ValidationError):
    """Raised when video format is invalid."""
    def __init__(self, format: str, supported_formats: set):
        self.format = format
        self.supported_formats = supported_formats
        super().__init__(
            f"Invalid video format: {format}. "
            f"Supported formats: {', '.join(sorted(supported_formats))}"
        )


class Invalid3DFormatError(ValidationError):
    """Raised when 3D format is invalid."""
    def __init__(self, format: str, supported_formats: set):
        self.format = format
        self.supported_formats = supported_formats
        super().__init__(
            f"Invalid 3D format: {format}. "
            f"Supported formats: {', '.join(sorted(supported_formats))}"
        )


class FileSizeExceededError(ValidationError):
    """Raised when file size exceeds limit."""
    def __init__(self, size_mb: float, limit_mb: int, file_type: str):
        self.size_mb = size_mb
        self.limit_mb = limit_mb
        self.file_type = file_type
        super().__init__(
            f"{file_type} file size ({size_mb:.1f}MB) exceeds "
            f"limit of {limit_mb}MB"
        )


class DimensionExceededError(ValidationError):
    """Raised when image dimensions exceed limits."""
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        super().__init__(
            f"Image dimensions ({width}x{height}) exceed maximum "
            f"allowed ({MAX_IMAGE_WIDTH}x{MAX_IMAGE_HEIGHT})"
        )