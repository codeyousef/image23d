"""3D Model Generation Module

This module implements HunYuan3D 2.1, Hi3DGen, and Sparc3D models
following the architecture defined in the 3D Implementation Guide.
"""

from .base import (
    Base3DModel,
    Base3DPipeline,
    IntermediateProcessor,
    MultiViewModel,
    ReconstructionModel,
    TextureModel,
    QualityPreset3D
)

from .hunyuan3d import (
    HunYuan3DPipeline,
    HunYuan3DModel,
    HunYuan3DConfig
)

from .memory import (
    ThreeDMemoryManager,
    ModelSwapper,
    IntermediateCache
)

from .intermediate import (
    DepthEstimator,
    NormalEstimator,
    UVUnwrapper,
    TextureSynthesizer,
    PBRMaterialGenerator
)

__all__ = [
    # Base classes
    'Base3DModel',
    'Base3DPipeline',
    'IntermediateProcessor',
    'MultiViewModel',
    'ReconstructionModel',
    'TextureModel',
    'QualityPreset3D',
    
    # HunYuan3D
    'HunYuan3DPipeline',
    'HunYuan3DModel',
    'HunYuan3DConfig',
    
    # Memory management
    'ThreeDMemoryManager',
    'ModelSwapper',
    'IntermediateCache',
    
    # Intermediate processors
    'DepthEstimator',
    'NormalEstimator',
    'UVUnwrapper',
    'TextureSynthesizer',
    'PBRMaterialGenerator'
]