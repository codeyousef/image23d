"""
Core 3D processing logic - backward compatibility wrapper

This module maintains backward compatibility by re-exporting
the ThreeDProcessor from the refactored threed package.
"""

# Re-export main class for backward compatibility
from .threed.processor import ThreeDProcessor

# Re-export any additional classes/functions that might be used elsewhere
from .threed.validation import RequestValidator
from .threed.image_processing import ImageProcessor
from .threed.depth_processing import DepthProcessor
from .threed.normal_processing import NormalProcessor
from .threed.mesh_reconstruction import MeshReconstructor
from .threed.texture_generation import TextureGenerator
from .threed.export import ModelExporter

__all__ = [
    'ThreeDProcessor',
    'RequestValidator',
    'ImageProcessor',
    'DepthProcessor',
    'NormalProcessor',
    'MeshReconstructor',
    'TextureGenerator',
    'ModelExporter'
]