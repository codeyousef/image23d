"""Input validation utilities."""

import os
from pathlib import Path
from typing import Union, Optional, Tuple

from ..config.constants import (
    SUPPORTED_IMAGE_FORMATS,
    SUPPORTED_VIDEO_FORMATS,
    SUPPORTED_3D_FORMATS,
    MAX_IMAGE_SIZE_MB,
    MAX_VIDEO_SIZE_MB,
    MAX_3D_SIZE_MB,
    MAX_IMAGE_WIDTH,
    MAX_IMAGE_HEIGHT,
    MIN_IMAGE_SIZE,
)
from ..exceptions.validation import (
    InvalidImageFormatError,
    InvalidVideoFormatError,
    Invalid3DFormatError,
    FileSizeExceededError,
    DimensionExceededError,
)


def validate_image_file(file_path: Union[str, Path]) -> Path:
    """Validate image file format and size.
    
    Args:
        file_path: Path to the image file
        
    Returns:
        Path object for the validated file
        
    Raises:
        InvalidImageFormatError: If format is not supported
        FileSizeExceededError: If file size exceeds limit
        FileNotFoundError: If file doesn't exist
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    
    # Check format
    if path.suffix.lower() not in SUPPORTED_IMAGE_FORMATS:
        raise InvalidImageFormatError(path.suffix, SUPPORTED_IMAGE_FORMATS)
    
    # Check size
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_IMAGE_SIZE_MB:
        raise FileSizeExceededError(size_mb, MAX_IMAGE_SIZE_MB, "Image")
    
    return path


def validate_video_file(file_path: Union[str, Path]) -> Path:
    """Validate video file format and size.
    
    Args:
        file_path: Path to the video file
        
    Returns:
        Path object for the validated file
        
    Raises:
        InvalidVideoFormatError: If format is not supported
        FileSizeExceededError: If file size exceeds limit
        FileNotFoundError: If file doesn't exist
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")
    
    # Check format
    if path.suffix.lower() not in SUPPORTED_VIDEO_FORMATS:
        raise InvalidVideoFormatError(path.suffix, SUPPORTED_VIDEO_FORMATS)
    
    # Check size
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_VIDEO_SIZE_MB:
        raise FileSizeExceededError(size_mb, MAX_VIDEO_SIZE_MB, "Video")
    
    return path


def validate_3d_file(file_path: Union[str, Path]) -> Path:
    """Validate 3D file format and size.
    
    Args:
        file_path: Path to the 3D file
        
    Returns:
        Path object for the validated file
        
    Raises:
        Invalid3DFormatError: If format is not supported
        FileSizeExceededError: If file size exceeds limit
        FileNotFoundError: If file doesn't exist
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"3D file not found: {path}")
    
    # Check format
    if path.suffix.lower() not in SUPPORTED_3D_FORMATS:
        raise Invalid3DFormatError(path.suffix, SUPPORTED_3D_FORMATS)
    
    # Check size
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_3D_SIZE_MB:
        raise FileSizeExceededError(size_mb, MAX_3D_SIZE_MB, "3D")
    
    return path


def validate_image_dimensions(
    width: int, 
    height: int,
    allow_upscale: bool = False
) -> Tuple[int, int]:
    """Validate and optionally adjust image dimensions.
    
    Args:
        width: Image width
        height: Image height
        allow_upscale: Whether to allow upscaling
        
    Returns:
        Tuple of (width, height) possibly adjusted
        
    Raises:
        DimensionExceededError: If dimensions exceed limits
        ValueError: If dimensions are too small
    """
    # Check minimum
    if width < MIN_IMAGE_SIZE or height < MIN_IMAGE_SIZE:
        raise ValueError(
            f"Image dimensions too small. Minimum: {MIN_IMAGE_SIZE}x{MIN_IMAGE_SIZE}"
        )
    
    # Check maximum
    if width > MAX_IMAGE_WIDTH or height > MAX_IMAGE_HEIGHT:
        if not allow_upscale:
            raise DimensionExceededError(width, height)
        else:
            # Scale down to fit
            scale = min(MAX_IMAGE_WIDTH / width, MAX_IMAGE_HEIGHT / height)
            width = int(width * scale)
            height = int(height * scale)
    
    return width, height


def validate_prompt(prompt: str, max_length: int = 1000) -> str:
    """Validate and clean a text prompt.
    
    Args:
        prompt: The prompt text
        max_length: Maximum allowed length
        
    Returns:
        Cleaned prompt
        
    Raises:
        ValueError: If prompt is invalid
    """
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty")
    
    prompt = prompt.strip()
    
    if len(prompt) > max_length:
        raise ValueError(f"Prompt too long. Maximum {max_length} characters")
    
    return prompt


def validate_seed(seed: Optional[Union[int, str]]) -> Optional[int]:
    """Validate and convert seed value.
    
    Args:
        seed: Seed value (int, string, or None)
        
    Returns:
        Valid seed integer or None
        
    Raises:
        ValueError: If seed is invalid
    """
    if seed is None or seed == "":
        return None
    
    try:
        seed_int = int(seed)
        if seed_int < 0:
            raise ValueError("Seed must be non-negative")
        return seed_int
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid seed value: {seed}") from e