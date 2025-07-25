"""Enhanced Hunyuan3D Studio - Backward Compatibility Wrapper

This file maintains backward compatibility by re-exporting the refactored
studio components.
"""

# Re-export the main class from the refactored module
from .studio.enhanced import Hunyuan3DStudioEnhanced

__all__ = ["Hunyuan3DStudioEnhanced"]