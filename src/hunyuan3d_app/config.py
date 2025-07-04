from dataclasses import dataclass
from typing import Tuple, Dict, Any
from pathlib import Path
import os
import tempfile

# --- Paths ---
# Use absolute paths based on the project directory
PROJECT_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
MODELS_DIR = PROJECT_DIR / "cache/models"
OUTPUT_DIR = PROJECT_DIR / "outputs"
CACHE_DIR = PROJECT_DIR / "cache/data"
TEMP_DIR = PROJECT_DIR / "cache/temp"

# Create the temp directory if it doesn't exist
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Configure tempfile to use our custom temp directory
tempfile.tempdir = str(TEMP_DIR)


class ImageModelConfig:
    """Configuration for image generation models"""
    def __init__(self, name, repo_id, pipeline_class, size, vram_required, description, optimal_resolution, supports_refiner=False, is_gguf=False, gguf_file=""):
        self.name = name
        self.repo_id = repo_id
        self.pipeline_class = pipeline_class
        self.size = size
        self.vram_required = vram_required
        self.description = description
        self.optimal_resolution = optimal_resolution
        self.supports_refiner = supports_refiner
        self.is_gguf = is_gguf
        self.gguf_file = gguf_file


class QualityPreset:
    """Quality preset configurations"""
    def __init__(self, name, image_steps, image_guidance, use_refiner, num_3d_views, mesh_resolution, texture_resolution):
        self.name = name
        self.image_steps = image_steps
        self.image_guidance = image_guidance
        self.use_refiner = use_refiner
        self.num_3d_views = num_3d_views
        self.mesh_resolution = mesh_resolution
        self.texture_resolution = texture_resolution


# --- Model Configurations ---
IMAGE_MODELS = {
    "SDXL-Turbo": ImageModelConfig(
        name="SDXL Turbo",
        repo_id="stabilityai/sdxl-turbo",
        pipeline_class="AutoPipelineForText2Image",
        size="~7 GB",
        vram_required="8GB+",
        description="Fast SDXL variant, good quality with fewer steps",
        optimal_resolution=(1024, 1024),
        supports_refiner=False
    ),
    "Dreamshaper-XL-Turbo": ImageModelConfig(
        name="Dreamshaper XL Turbo",
        repo_id="Lykon/dreamshaper-xl-turbo",
        pipeline_class="DiffusionPipeline",
        size="~7 GB",
        vram_required="8GB+",
        description="Fast artistic model, great for creative objects",
        optimal_resolution=(1024, 1024),
        supports_refiner=False
    ),
    "RealVisXL-V4": ImageModelConfig(
        name="RealVisXL V4.0",
        repo_id="SG161222/RealVisXL_V4.0",
        pipeline_class="DiffusionPipeline",
        size="~7 GB",
        vram_required="10GB+",
        description="Photorealistic model, excellent for objects",
        optimal_resolution=(1024, 1024),
        supports_refiner=False
    ),
    "Juggernaut-XL": ImageModelConfig(
        name="Juggernaut XL V9",
        repo_id="RunDiffusion/Juggernaut-XL-v9",
        pipeline_class="DiffusionPipeline",
        size="~7 GB",
        vram_required="10GB+",
        description="High quality photorealistic model",
        optimal_resolution=(1024, 1024),
        supports_refiner=False
    ),
    "SDXL-Lightning": ImageModelConfig(
        name="SDXL Lightning 4-step",
        repo_id="ByteDance/SDXL-Lightning",
        pipeline_class="DiffusionPipeline",
        size="~7 GB",
        vram_required="8GB+",
        description="Ultra-fast 4-step SDXL model",
        optimal_resolution=(1024, 1024),
        supports_refiner=False
    ),
    "Playground-v2.5": ImageModelConfig(
        name="Playground v2.5",
        repo_id="playgroundai/playground-v2.5-1024px-aesthetic",
        pipeline_class="DiffusionPipeline",
        size="~7 GB",
        vram_required="12GB+",
        description="High aesthetic quality, great for products",
        optimal_resolution=(1024, 1024),
        supports_refiner=False
    )
}
# GGUF Models - Complete quantization spectrum
GGUF_IMAGE_MODELS = {
    # FLUX.1-dev quantized models
    "FLUX.1-dev-Q8": ImageModelConfig(
        name="FLUX.1-dev Q8 GGUF (Best Quality)",
        repo_id="city96/FLUX.1-dev-gguf",
        pipeline_class="FluxPipeline",
        size="~12.5 GB",
        vram_required="14GB+",
        description="Highest quality GGUF - 98% of original quality, 50% less VRAM",
        optimal_resolution=(1024, 1024),
        supports_refiner=False,
        is_gguf=True,
        gguf_file="flux1-dev-Q8_0.gguf"
    ),
    "FLUX.1-dev-Q6": ImageModelConfig(
        name="FLUX.1-dev Q6_K GGUF (Balanced)",
        repo_id="city96/FLUX.1-dev-gguf",
        pipeline_class="FluxPipeline",
        size="~9.8 GB",
        vram_required="11.5GB+",
        description="Great balance - 96% quality, 60% less VRAM",
        optimal_resolution=(1024, 1024),
        supports_refiner=False,
        is_gguf=True,
        gguf_file="flux1-dev-Q6_K.gguf"
    ),
    "FLUX.1-dev-Q5": ImageModelConfig(
        name="FLUX.1-dev Q5_K_M GGUF (Efficient)",
        repo_id="city96/FLUX.1-dev-gguf",
        pipeline_class="FluxPipeline",
        size="~8.5 GB",
        vram_required="10.5GB+",
        description="Good quality - 95% quality, 65% less VRAM",
        optimal_resolution=(1024, 1024),
        supports_refiner=False,
        is_gguf=True,
        gguf_file="flux1-dev-Q5_K_M.gguf"
    ),
    "FLUX.1-dev-Q4": ImageModelConfig(
        name="FLUX.1-dev Q4_K_S GGUF (Memory Saver)",
        repo_id="city96/FLUX.1-dev-gguf",
        pipeline_class="FluxPipeline",
        size="~6.5 GB",
        vram_required="8GB+",
        description="Memory efficient - 90% quality, 70% less VRAM",
        optimal_resolution=(1024, 1024),
        supports_refiner=False,
        is_gguf=True,
        gguf_file="flux1-dev-Q4_K_S.gguf"
    ),
    "FLUX.1-dev-Q3": ImageModelConfig(
        name="FLUX.1-dev Q3_K_M GGUF (Low VRAM)",
        repo_id="city96/FLUX.1-dev-gguf",
        pipeline_class="FluxPipeline",
        size="~5.0 GB",
        vram_required="6.5GB+",
        description="Low VRAM - 82% quality, works on 8GB GPUs",
        optimal_resolution=(1024, 1024),
        supports_refiner=False,
        is_gguf=True,
        gguf_file="flux1-dev-Q3_K_M.gguf"
    ),
    "FLUX.1-dev-Q2": ImageModelConfig(
        name="FLUX.1-dev Q2_K GGUF (Ultra Low VRAM)",
        repo_id="city96/FLUX.1-dev-gguf",
        pipeline_class="FluxPipeline",
        size="~3.5 GB",
        vram_required="5GB+",
        description="Extreme compression - 70% quality, works on 6GB GPUs",
        optimal_resolution=(1024, 1024),
        supports_refiner=False,
        is_gguf=True,
        gguf_file="flux1-dev-Q2_K.gguf"
    ),
    # FLUX.1-schnell quantized models
    "FLUX.1-schnell-Q8": ImageModelConfig(
        name="FLUX.1-schnell Q8 GGUF (Fast + Quality)",
        repo_id="city96/FLUX.1-schnell-gguf",
        pipeline_class="FluxPipeline",
        size="~12.5 GB",
        vram_required="14GB+",
        description="Fast generation with best GGUF quality - 4x faster than dev",
        optimal_resolution=(1024, 1024),
        supports_refiner=False,
        is_gguf=True,
        gguf_file="flux1-schnell-Q8_0.gguf"
    ),
    "FLUX.1-schnell-Q6": ImageModelConfig(
        name="FLUX.1-schnell Q6_K GGUF (Fast + Efficient)",
        repo_id="city96/FLUX.1-schnell-gguf",
        pipeline_class="FluxPipeline",
        size="~9.8 GB",
        vram_required="11.5GB+",
        description="Fast with good efficiency - great for quick iterations",
        optimal_resolution=(1024, 1024),
        supports_refiner=False,
        is_gguf=True,
        gguf_file="flux1-schnell-Q6_K.gguf"
    ),
    "FLUX.1-schnell-Q4": ImageModelConfig(
        name="FLUX.1-schnell Q4_K_S GGUF (Fastest)",
        repo_id="city96/FLUX.1-schnell-gguf",
        pipeline_class="FluxPipeline",
        size="~6.5 GB",
        vram_required="8GB+",
        description="Ultra-fast generation on low VRAM - ideal for prototyping",
        optimal_resolution=(1024, 1024),
        supports_refiner=False,
        is_gguf=True,
        gguf_file="flux1-schnell-Q4_K_S.gguf"
    ),
    # FP8 optimized models
    "FLUX.1-dev-FP8": ImageModelConfig(
        name="FLUX.1-dev FP8 (Near-Lossless)",
        repo_id="Kijai/flux-fp8",
        pipeline_class="FluxPipeline",
        size="~12 GB",
        vram_required="14GB+",
        description="FP8 quantization - 99% quality with 50% less VRAM than FP16",
        optimal_resolution=(1024, 1024),
        supports_refiner=False,
        is_gguf=True,
        gguf_file="flux1-dev-fp8-e4m3fn.safetensors"
    ),
}

GATED_IMAGE_MODELS = {
    "FLUX.1-schnell": ImageModelConfig(
        name="FLUX.1-schnell (Requires HF Login)",
        repo_id="black-forest-labs/FLUX.1-schnell",
        pipeline_class="FluxPipeline",
        size="~24 GB",
        vram_required="16GB+",
        description="Latest & fastest high-quality model (GATED - Requires Hugging Face login)",
        optimal_resolution=(1024, 1024),
        supports_refiner=False
    ),
    "FLUX.1-dev": ImageModelConfig(
        name="FLUX.1-dev (Requires HF Login)",
        repo_id="black-forest-labs/FLUX.1-dev",
        pipeline_class="FluxPipeline",
        size="~24 GB",
        vram_required="24GB+",
        description="Best quality FLUX model (GATED - Requires Hugging Face login)",
        optimal_resolution=(1024, 1024),
        supports_refiner=False
    )
}
ALL_IMAGE_MODELS = {**IMAGE_MODELS, **GGUF_IMAGE_MODELS, **GATED_IMAGE_MODELS}
HUNYUAN3D_MODELS = {
    "hunyuan3d-21": {
        "name": "Hunyuan3D 2.1 (Latest)",
        "repo_id": "tencent/Hunyuan3D-2.1",
        "size": "~30 GB",
        "vram_required": "16GB+",
        "description": "Latest model with PBR material synthesis - production ready",
        "optimal_views": 8
    },
    "hunyuan3d-2mini": {
        "name": "Hunyuan3D 2.0 Mini",
        "repo_id": "tencent/Hunyuan3D-2mini",
        "size": "~8 GB",  # 0.6B model
        "vram_required": "8GB+",
        "description": "Smallest, fastest model (0.6B) - good for quick tests",
        "optimal_views": 6
    },
    "hunyuan3d-2mv": {
        "name": "Hunyuan3D 2.0 Multiview",
        "repo_id": "tencent/Hunyuan3D-2mv",
        "size": "~25 GB",
        "vram_required": "12GB+",
        "description": "Multiview controlled shape generation",
        "optimal_views": 4
    },
    "hunyuan3d-2standard": {
        "name": "Hunyuan3D 2.0 Standard (Legacy)",
        "repo_id": "tencent/Hunyuan3D-2",
        "size": "~110 GB",  # Actual size based on user experience
        "vram_required": "16GB+",
        "description": "Older large model - consider using 2.1 instead",
        "optimal_views": 8
    }
}
# Component models for FLUX (VAE, Text Encoders)
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

QUALITY_PRESETS = {
    "draft": QualityPreset(
        name="Draft",
        image_steps=20,
        image_guidance=7.5,
        use_refiner=False,
        num_3d_views=4,
        mesh_resolution=128,
        texture_resolution=512
    ),
    "standard": QualityPreset(
        name="Standard",
        image_steps=35,
        image_guidance=8.0,
        use_refiner=False,
        num_3d_views=6,
        mesh_resolution=256,
        texture_resolution=1024
    ),
    "high": QualityPreset(
        name="High Quality",
        image_steps=50,
        image_guidance=8.5,
        use_refiner=True,
        num_3d_views=8,
        mesh_resolution=512,
        texture_resolution=2048
    )
}
