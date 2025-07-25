"""Image Generation Tab - Backward Compatibility Wrapper

This file maintains backward compatibility by re-exporting the refactored
image generation components.
"""

# Re-export the main function from the refactored module
from .image_generation import create_image_generation_tab

__all__ = ["create_image_generation_tab"]