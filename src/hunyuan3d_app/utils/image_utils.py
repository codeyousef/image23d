"""Image processing utilities for consistent format handling"""

import hashlib
import logging
from PIL import Image
from typing import Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)


def ensure_image_format(
    image: Image.Image, 
    target_mode: str = "RGB",
    preserve_alpha: bool = False
) -> Image.Image:
    """Ensure image is in the target format while preserving quality
    
    Args:
        image: Input PIL Image
        target_mode: Target image mode (RGB, RGBA, L, etc.)
        preserve_alpha: Whether to preserve alpha channel when converting
        
    Returns:
        Image in target format
    """
    if image.mode == target_mode:
        return image.copy()
    
    original_mode = image.mode
    logger.debug(f"Converting image from {original_mode} to {target_mode}")
    
    # Handle RGBA → RGB conversion carefully
    if original_mode == "RGBA" and target_mode == "RGB":
        if preserve_alpha:
            # Create white background and composite
            background = Image.new("RGB", image.size, (255, 255, 255))
            # Use alpha channel for proper compositing
            converted = Image.alpha_composite(background.convert("RGBA"), image).convert("RGB")
        else:
            # Simple conversion (may lose transparency)
            converted = image.convert("RGB")
    else:
        # Standard conversion
        converted = image.convert(target_mode)
    
    # Log conversion for debugging
    logger.debug(f"Image format conversion: {original_mode} → {target_mode}")
    return converted


def get_image_hash(image: Image.Image) -> str:
    """Get MD5 hash of image for tracking
    
    Args:
        image: PIL Image
        
    Returns:
        8-character hex hash string
    """
    if not hasattr(image, 'tobytes'):
        logger.warning(f"Cannot hash image of type {type(image)}")
        return "unknown"
    
    try:
        return hashlib.md5(image.tobytes()).hexdigest()[:8]
    except Exception as e:
        logger.error(f"Failed to hash image: {e}")
        return "error"


def validate_image_consistency(
    image: Image.Image,
    expected_mode: Optional[str] = None,
    expected_size: Optional[Tuple[int, int]] = None,
    context: str = "unknown"
) -> bool:
    """Validate image format consistency
    
    Args:
        image: PIL Image to validate
        expected_mode: Expected image mode
        expected_size: Expected image size
        context: Context for logging
        
    Returns:
        True if image meets expectations
    """
    issues = []
    
    if expected_mode and image.mode != expected_mode:
        issues.append(f"mode mismatch: {image.mode} != {expected_mode}")
    
    if expected_size and image.size != expected_size:
        issues.append(f"size mismatch: {image.size} != {expected_size}")
    
    if issues:
        logger.warning(f"Image validation failed in {context}: {', '.join(issues)}")
        return False
    
    logger.debug(f"Image validation passed in {context}: {image.mode} {image.size}")
    return True


def log_image_pipeline_step(
    image: Image.Image,
    step_name: str,
    expected_hash: Optional[str] = None
) -> str:
    """Log image details at pipeline step for debugging
    
    Args:
        image: PIL Image
        step_name: Name of pipeline step
        expected_hash: Expected hash for consistency check
        
    Returns:
        Image hash for tracking
    """
    image_hash = get_image_hash(image)
    
    # Check for hash consistency
    consistency_status = "✅" if expected_hash is None or image_hash == expected_hash else "❌"
    
    logger.info(f"{consistency_status} [{step_name}] Image: {image.mode} {image.size}, hash: {image_hash}")
    
    if expected_hash and image_hash != expected_hash:
        logger.warning(f"🚨 Image hash mismatch at {step_name}: {image_hash} != {expected_hash}")
    
    return image_hash


def create_debug_image_copy(
    image: Image.Image,
    output_path: str,
    step_name: str
) -> None:
    """Save debug copy of image for troubleshooting
    
    Args:
        image: PIL Image
        output_path: Path to save debug image
        step_name: Pipeline step name for filename
    """
    try:
        from pathlib import Path
        debug_path = Path(output_path) / f"DEBUG_{step_name}_{get_image_hash(image)}.png"
        image.save(debug_path)
        logger.debug(f"Saved debug image: {debug_path}")
    except Exception as e:
        logger.error(f"Failed to save debug image: {e}")


def optimize_image_for_conditioning(
    image: Image.Image,
    target_size: Optional[Tuple[int, int]] = None,
    target_mode: str = "RGB"
) -> Image.Image:
    """Optimize image for best conditioning quality
    
    Args:
        image: Input PIL Image
        target_size: Target size (width, height)
        target_mode: Target color mode
        
    Returns:
        Optimized image
    """
    # Start with format conversion
    optimized = ensure_image_format(image, target_mode, preserve_alpha=True)
    
    # Resize if needed
    if target_size and optimized.size != target_size:
        # Use high-quality resampling
        optimized = optimized.resize(target_size, Image.Resampling.LANCZOS)
        logger.debug(f"Resized image to {target_size}")
    
    # Ensure image quality
    if optimized.mode == "RGB":
        # Ensure no pixel values are exactly 0 or 255 (can cause conditioning issues)
        img_array = np.array(optimized)
        img_array = np.clip(img_array, 1, 254)  # Clamp to [1, 254]
        optimized = Image.fromarray(img_array)
        logger.debug("Applied pixel value clamping for better conditioning")
    
    return optimized