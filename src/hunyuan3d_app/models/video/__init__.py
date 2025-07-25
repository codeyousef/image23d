"""Video generation models package

This package implements state-of-the-art video generation models:
- Wan2.1: 1.3B/14B models with visual text generation
- HunyuanVideo: 13B cinema-quality generation
- LTX-Video: Real-time video generation
- Mochi-1: 10B model with 30fps smooth motion
- CogVideoX-5B: Superior image-to-video quality
"""

from .base import (
    BaseVideoModel,
    VideoModelType,
    VideoFormat,
    QuantizationType,
    VideoQualityPreset,
    VIDEO_QUALITY_PRESETS,
    VideoGenerationResult,
    VideoModelConfig
)

from .wan21 import Wan21VideoModel
from .hunyuanvideo import HunyuanVideoModel
from .ltxvideo import LTXVideoModel
from .mochi import Mochi1Model
from .cogvideox import CogVideoX5BModel

from .memory_optimizer import (
    VideoMemoryOptimizer,
    MemoryProfile,
    OptimizationLevel,
    auto_optimize_for_hardware,
    estimate_max_video_length
)

# Model registry
VIDEO_MODEL_REGISTRY = {
    VideoModelType.WAN2_1_1_3B: Wan21VideoModel,
    VideoModelType.WAN2_1_14B: Wan21VideoModel,
    VideoModelType.HUNYUANVIDEO: HunyuanVideoModel,
    VideoModelType.LTX_VIDEO: LTXVideoModel,
    VideoModelType.MOCHI_1: Mochi1Model,
    VideoModelType.COGVIDEOX_5B: CogVideoX5BModel
}


def create_video_model(
    model_type: VideoModelType,
    device: str = "cuda",
    dtype: str = "fp16",
    **kwargs
) -> BaseVideoModel:
    """Factory function to create video models
    
    Args:
        model_type: Type of video model to create
        device: Device to run on (cuda/cpu)
        dtype: Model precision (fp32/fp16/bf16)
        **kwargs: Additional model-specific arguments
        
    Returns:
        Initialized video model
    """
    # Convert dtype string to torch dtype
    import torch
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16
    }
    torch_dtype = dtype_map.get(dtype, torch.float16)
    
    # Create config
    config = VideoModelConfig(
        model_type=model_type,
        device=device,
        dtype=torch_dtype,
        **kwargs
    )
    
    # Get model class
    model_class = VIDEO_MODEL_REGISTRY.get(model_type)
    if not model_class:
        raise ValueError(f"Unknown video model type: {model_type}")
        
    # Create and return model
    return model_class(config)


__all__ = [
    # Base classes
    'BaseVideoModel',
    'VideoModelType', 
    'VideoFormat',
    'QuantizationType',
    'VideoQualityPreset',
    'VIDEO_QUALITY_PRESETS',
    'VideoGenerationResult',
    'VideoModelConfig',
    
    # Model implementations
    'Wan21VideoModel',
    'HunyuanVideoModel',
    'LTXVideoModel',
    'Mochi1Model',
    'CogVideoX5BModel',
    
    # Memory optimization
    'VideoMemoryOptimizer',
    'MemoryProfile',
    'OptimizationLevel',
    'auto_optimize_for_hardware',
    'estimate_max_video_length',
    
    # Factory
    'create_video_model',
    'VIDEO_MODEL_REGISTRY'
]