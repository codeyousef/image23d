"""Core Studio - Backward Compatibility Wrapper

This file maintains backward compatibility by re-exporting the refactored
studio base class.
"""

# Re-export the main class from the refactored module
from .studio.base import Hunyuan3DStudio

__all__ = ["Hunyuan3DStudio"]