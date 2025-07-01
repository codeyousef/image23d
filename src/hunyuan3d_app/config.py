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


@dataclass
class ImageModelConfig:
    """Configuration for image generation models"""
    name: str
    repo_id: str
    pipeline_class: str
    size: str
    vram_required: str
    description: str
    optimal_resolution: Tuple[int, int]
    supports_refiner: bool = False


@dataclass
class QualityPreset:
    """Quality preset configurations"""
    name: str
    image_steps: int
    image_guidance: float
    use_refiner: bool
    num_3d_views: int
    mesh_resolution: int
    texture_resolution: int


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
ALL_IMAGE_MODELS = {**IMAGE_MODELS, **GATED_IMAGE_MODELS}
HUNYUAN3D_MODELS = {
    "hunyuan3d-2mini": {
        "name": "Hunyuan3D 2.0 Mini",
        "repo_id": "tencent/Hunyuan3D-2",
        "size": "~15 GB",
        "vram_required": "12GB+",
        "description": "Smaller, faster model suitable for quick previews",
        "optimal_views": 6
    },
    "hunyuan3d-2standard": {
        "name": "Hunyuan3D 2.0 Standard",
        "repo_id": "tencent/Hunyuan3D-2",
        "size": "~25 GB",
        "vram_required": "16GB+",
        "description": "Balanced model for high-quality results",
        "optimal_views": 8
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
