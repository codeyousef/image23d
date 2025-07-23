"""
Main configuration file for backward compatibility.
This file re-exports configurations from the new modular structure.
"""

# Re-export everything from the new config package
from .config import *

# Import additional model configurations that were in the original file
from .config.models import ThreeDModelConfig, VideoModelConfig

# Advanced 3D Models that weren't moved yet
SPARC3D_MODELS = {
    "sparc3d-v1": {
        "name": "Sparc3D v1 (High-Res)",
        "repo_id": "ilcve21/Sparc3D",
        "size": "~12 GB",
        "vram_required": "16GB+",
        "description": "Ultra high-resolution 1024³ reconstruction with sparse representation",
        "optimal_views": 1,
        "features": ["sparse_cubes", "high_resolution", "arbitrary_topology"],
        "space_url": "https://huggingface.co/spaces/ilcve21/Sparc3D"
    }
}

HI3DGEN_MODELS = {
    "hi3dgen-v1": {
        "name": "Hi3DGen v1 (Normal Bridge)",
        "repo_id": "Stable-X/trellis-normal-v0-1",
        "size": "~10 GB",
        "vram_required": "12GB+",
        "description": "High-fidelity geometry via normal map bridging - preserves fine details",
        "optimal_views": 1,
        "features": ["normal_mapping", "high_fidelity", "texture_preservation"],
        "space_url": "https://huggingface.co/spaces/Stable-X/Hi3DGen"
    }
}

# Combine all 3D models
ALL_3D_MODELS = {
    **HUNYUAN3D_MODELS_DICT,
    **SPARC3D_MODELS,
    **HI3DGEN_MODELS
}

# Component models for FLUX
FLUX_COMPONENTS = {
    "vae": {
        "name": "FLUX VAE",
        "repo_id": "black-forest-labs/FLUX.1-dev",
        "filename": "ae.safetensors",
        "size": "~335 MB",
        "description": "FLUX Autoencoder for image encoding/decoding"
    },
    "text_encoder_clip": {
        "name": "CLIP Text Encoder",
        "repo_id": "comfyanonymous/flux_text_encoders",
        "filename": "clip_l.safetensors", 
        "size": "~246 MB",
        "description": "CLIP text encoder for prompt processing"
    },
    "text_encoder_t5": {
        "name": "T5 Text Encoder",
        "repo_id": "comfyanonymous/flux_text_encoders",
        "filename": "t5xxl_fp8_e4m3fn_scaled.safetensors",
        "size": "~4.5 GB", 
        "description": "T5 text encoder (FP8 optimized) for advanced prompt understanding"
    }
}

# Video Model Configurations
VIDEO_MODELS = {
    "hunyuanvideo": {
        "name": "HunyuanVideo",
        "repo_id": "tencent/HunyuanVideo",
        "model_file": "hunyuanvideo_diffusion_pytorch_model.safetensors",
        "size": "~50 GB",
        "vram_required": "24GB+",
        "description": "High-quality text-to-video generation",
        "max_frames": 129,
        "fps": 24
    },
    "cogvideox": {
        "name": "CogVideoX",
        "repo_id": "THUDM/CogVideoX-2b",
        "size": "~10 GB",
        "vram_required": "12GB+",
        "description": "Efficient video generation model",
        "max_frames": 49,
        "fps": 8
    },
    "zeroscope": {
        "name": "Zeroscope V2",
        "repo_id": "cerspense/zeroscope_v2_576w",
        "size": "~7 GB",
        "vram_required": "10GB+",
        "description": "Open-source text-to-video model",
        "max_frames": 36,
        "fps": 8
    }
}

# Character-related models
CHARACTER_MODELS = {
    "faceswap": {
        "insightface": {
            "name": "InsightFace",
            "repo_id": "deepinsight/insightface",
            "size": "~300 MB",
            "description": "Face detection and recognition"
        },
        "facefusion": {
            "name": "FaceFusion Models",
            "repo_id": "facefusion/models",
            "size": "~2 GB",
            "description": "Complete face swapping toolkit"
        }
    },
    "pulid": {
        "name": "PuLID",
        "repo_id": "ByteDance/PuLID",
        "size": "~1 GB",
        "description": "Identity-preserving generation"
    }
}

# Texture component models
TEXTURE_COMPONENTS = {
    "background_remover": {
        "name": "RMBG-1.4",
        "repo_id": "briaai/RMBG-1.4",
        "size": "~176 MB",
        "description": "AI background removal"
    },
    "upscaler": {
        "realesrgan": {
            "name": "Real-ESRGAN",
            "repo_id": "ai-forever/Real-ESRGAN",
            "size": "~64 MB",
            "description": "4x image upscaling"
        },
        "gfpgan": {
            "name": "GFPGAN",
            "repo_id": "TencentARC/GFPGAN",
            "size": "~340 MB",
            "description": "Face restoration and enhancement"
        }
    }
}

# Export the rest of the original configurations that aren't modularized yet
FLUX_MODELS = {
    **IMAGE_MODELS,
    **GGUF_IMAGE_MODELS
}

# For backward compatibility with desktop app
from .config.models import ImageModelConfig as _ImageModelConfig
# Create a compatible class
class ImageModelConfig(_ImageModelConfig):
    def __init__(self, name, repo_id, pipeline_class, size, vram_required, 
                 description, optimal_resolution, supports_refiner=False, 
                 is_gguf=False, gguf_file=""):
        super().__init__(
            name=name,
            repo_id=repo_id,
            pipeline_class=pipeline_class,
            size=size,
            vram_required=vram_required,
            description=description,
            optimal_resolution=optimal_resolution,
            supports_refiner=supports_refiner,
            is_gguf=is_gguf,
            gguf_file=gguf_file
        )