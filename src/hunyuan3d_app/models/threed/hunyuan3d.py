"""
HunYuan3D model - backward compatibility wrapper.

This file maintains backward compatibility by re-exporting the refactored
components from the hunyuan3d package.
"""

# Re-export all components from the refactored package
from .hunyuan3d import (
    HunYuan3DModel,
    HunYuan3DConfig,
    HunYuan3DPipeline,
    HunYuan3DMultiView,
    HunYuan3DReconstruction,
    HunYuan3DTexture,
    MODEL_VARIANTS,
    export_mesh,
    save_texture_maps,
)

# For complete backward compatibility, also import and expose internal classes
from .hunyuan3d.setup import setup_hunyuan3d_paths, get_hunyuan3d_path
from .hunyuan3d.utils import (
    estimate_memory_usage,
    create_view_angles,
    prepare_image_for_multiview,
)

__all__ = [
    # Main classes
    "HunYuan3DModel",
    "HunYuan3DConfig", 
    "HunYuan3DPipeline",
    "HunYuan3DMultiView",
    "HunYuan3DReconstruction",
    "HunYuan3DTexture",
    # Configuration
    "MODEL_VARIANTS",
    # Utilities
    "export_mesh",
    "save_texture_maps",
    "setup_hunyuan3d_paths",
    "get_hunyuan3d_path",
    "estimate_memory_usage",
    "create_view_angles",
    "prepare_image_for_multiview",
]

# Maintain version info
__version__ = "0.1.0"