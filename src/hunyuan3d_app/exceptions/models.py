"""Model-related exceptions."""


class ModelError(Exception):
    """Base exception for model-related errors."""
    pass


class ModelNotFoundError(ModelError):
    """Raised when a requested model is not found."""
    pass


class ModelLoadError(ModelError):
    """Raised when a model fails to load."""
    pass


class ModelDownloadError(ModelError):
    """Raised when a model download fails."""
    pass


class InvalidModelFormatError(ModelError):
    """Raised when a model has an invalid format."""
    pass


class InsufficientVRAMError(ModelError):
    """Raised when there is insufficient VRAM for a model."""
    def __init__(self, required_vram: str, available_vram: float):
        self.required_vram = required_vram
        self.available_vram = available_vram
        super().__init__(
            f"Insufficient VRAM. Required: {required_vram}, "
            f"Available: {available_vram:.1f}GB"
        )