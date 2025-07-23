"""Generation-related exceptions."""


class GenerationError(Exception):
    """Base exception for generation-related errors."""
    pass


class GenerationTimeoutError(GenerationError):
    """Raised when generation times out."""
    pass


class GenerationCancelledError(GenerationError):
    """Raised when generation is cancelled by user."""
    pass


class InvalidInputError(GenerationError):
    """Raised when generation input is invalid."""
    pass


class PipelineError(GenerationError):
    """Raised when pipeline execution fails."""
    pass