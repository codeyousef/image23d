"""Model definitions and configurations."""

from .models import ImageModelConfig, ThreeDModelConfig, QualityPreset

# --- Standard Image Models ---
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

# --- GGUF Quantized Models ---
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

# --- Gated Models (Require HF Login) ---
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

# Combined image models
ALL_IMAGE_MODELS = {**IMAGE_MODELS, **GGUF_IMAGE_MODELS, **GATED_IMAGE_MODELS}

# --- 3D Models ---
HUNYUAN3D_MODELS = {
    "hunyuan3d-21": ThreeDModelConfig(
        name="Hunyuan3D 2.1 (Latest)",
        repo_id="tencent/Hunyuan3D-2.1",
        size="~30 GB",
        vram_required="16GB+",
        description="Latest model with PBR material synthesis - production ready",
        optimal_views=8,
        supports_pbr=True,
        supports_texture=True
    ),
    "hunyuan3d-2mini": ThreeDModelConfig(
        name="Hunyuan3D 2.0 Mini",
        repo_id="tencent/Hunyuan3D-2mini",
        size="~8 GB",
        vram_required="8GB+",
        description="Smallest, fastest model (0.6B) - good for quick tests",
        optimal_views=6,
        supports_pbr=False,
        supports_texture=True
    ),
    "hunyuan3d-2mv": ThreeDModelConfig(
        name="Hunyuan3D 2.0 Multiview",
        repo_id="tencent/Hunyuan3D-2mv",
        size="~25 GB",
        vram_required="12GB+",
        description="Multiview controlled shape generation",
        optimal_views=4,
        supports_pbr=False,
        supports_texture=True
    ),
    "hunyuan3d-2standard": ThreeDModelConfig(
        name="Hunyuan3D 2.0 Standard (Legacy)",
        repo_id="tencent/Hunyuan3D-2",
        size="~110 GB",
        vram_required="16GB+",
        description="Older large model - consider using 2.1 instead",
        optimal_views=8,
        supports_pbr=False,
        supports_texture=True
    )
}

# Convert to dict format for backward compatibility
HUNYUAN3D_MODELS_DICT = {
    k: {
        "name": v.name,
        "repo_id": v.repo_id,
        "size": v.size,
        "vram_required": v.vram_required,
        "description": v.description,
        "optimal_views": v.optimal_views
    }
    for k, v in HUNYUAN3D_MODELS.items()
}

# --- Quality Presets ---
QUALITY_PRESETS = {
    "draft": QualityPreset(
        name="Draft (Fast)",
        image_steps=4,
        image_guidance=3.5,
        use_refiner=False,
        num_3d_views=4,
        mesh_resolution=256,
        texture_resolution=1024
    ),
    "standard": QualityPreset(
        name="Standard (Balanced)",
        image_steps=20,
        image_guidance=7.5,
        use_refiner=False,
        num_3d_views=6,
        mesh_resolution=512,
        texture_resolution=2048
    ),
    "high": QualityPreset(
        name="High (Quality)",
        image_steps=30,
        image_guidance=10.0,
        use_refiner=True,
        num_3d_views=8,
        mesh_resolution=768,
        texture_resolution=4096
    ),
    "ultra": QualityPreset(
        name="Ultra (Maximum)",
        image_steps=50,
        image_guidance=12.0,
        use_refiner=True,
        num_3d_views=12,
        mesh_resolution=1024,
        texture_resolution=8192
    )
}

# --- Video Models ---
VIDEO_MODELS = {
    "LTX-Video": {
        "name": "LTX Video",
        "repo_id": "Lightricks/LTX-Video",
        "pipeline_class": "LTXVideoPipeline",
        "size": "~10 GB",
        "vram_required": "12GB+",
        "description": "Text-to-video generation model"
    }
}

# --- Texture Pipeline Components ---
TEXTURE_PIPELINE_COMPONENTS = {
    "unet": {
        "name": "UNet Model",
        "repo_id": "stabilityai/stable-diffusion-2-1",
        "subfolder": "unet",
        "description": "UNet for texture generation"
    },
    "vae": {
        "name": "VAE Model",
        "repo_id": "stabilityai/stable-diffusion-2-1",
        "subfolder": "vae",
        "description": "VAE for texture generation"
    },
    "text_encoder": {
        "name": "Text Encoder",
        "repo_id": "stabilityai/stable-diffusion-2-1",
        "subfolder": "text_encoder",
        "description": "Text encoder for texture generation"
    },
    "tokenizer": {
        "name": "Tokenizer",
        "repo_id": "stabilityai/stable-diffusion-2-1",
        "subfolder": "tokenizer",
        "description": "Tokenizer for texture generation"
    }
}

# --- IP Adapter Models ---
IP_ADAPTER_MODELS = {
    "ip-adapter-plus": {
        "name": "IP-Adapter Plus",
        "repo_id": "h94/IP-Adapter",
        "file": "models/ip-adapter-plus_sd15.bin",
        "description": "IP-Adapter for style transfer"
    },
    "ip-adapter-plus-face": {
        "name": "IP-Adapter Plus Face",
        "repo_id": "h94/IP-Adapter",
        "file": "models/ip-adapter-plus-face_sd15.bin",
        "description": "IP-Adapter for face style transfer"
    }
}

# --- Face Swap Models ---
FACE_SWAP_MODELS = {
    "inswapper_128": {
        "name": "Inswapper 128",
        "repo_id": "deepinsight/inswapper",
        "file": "inswapper_128.onnx",
        "description": "Face swapping model"
    }
}

# --- Face Restore Models ---
FACE_RESTORE_MODELS = {
    "gfpgan": {
        "name": "GFPGAN v1.4",
        "repo_id": "tencentarc/GFPGAN",
        "file": "GFPGANv1.4.pth",
        "description": "Face restoration model"
    },
    "codeformer": {
        "name": "CodeFormer",
        "repo_id": "sczhou/CodeFormer",
        "file": "codeformer.pth",
        "description": "Advanced face restoration"
    }
}