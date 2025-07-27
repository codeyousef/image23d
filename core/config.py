"""
Core configuration shared between desktop and web apps
"""

from pathlib import Path
from typing import Dict, Any
import os

# Paths - inherit from existing project structure
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CACHE_DIR = PROJECT_ROOT / "cache"

# Ensure directories exist
for dir_path in [MODELS_DIR, OUTPUT_DIR, CACHE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Model configurations (extracted from existing config)
FLUX_MODELS = {
    "FLUX.1-dev": {
        "name": "FLUX.1 Dev",
        "repo_id": "black-forest-labs/FLUX.1-dev",
        "size": "~24GB",
        "vram_required": "24GB+",
        "description": "Best quality FLUX model for development",
        "optimal_resolution": (1024, 1024)
    },
    "FLUX.1-schnell": {
        "name": "FLUX.1 Schnell",
        "repo_id": "black-forest-labs/FLUX.1-schnell",
        "size": "~24GB", 
        "vram_required": "16GB+",
        "description": "Fast FLUX model for quick generation",
        "optimal_resolution": (1024, 1024)
    },
    "FLUX.1-dev-Q8": {
        "name": "FLUX.1 Dev Q8",
        "repo_id": "city96/FLUX.1-dev-gguf",
        "size": "~12.5GB",
        "vram_required": "14GB+",
        "description": "High quality quantized FLUX model",
        "optimal_resolution": (1024, 1024)
    },
    "FLUX.1-dev-Q6": {
        "name": "FLUX.1 Dev Q6",
        "repo_id": "city96/FLUX.1-dev-gguf",
        "size": "~9.8GB",
        "vram_required": "11.5GB+",
        "description": "Balanced quantized FLUX model",
        "optimal_resolution": (1024, 1024)
    },
    "FLUX.1-dev-FP8": {
        "name": "FLUX.1 Dev FP8",
        "repo_id": "Kijai/flux-fp8",
        "size": "~12GB",
        "vram_required": "14GB+",
        "description": "FP8 optimized FLUX model",
        "optimal_resolution": (1024, 1024)
    }
}

HUNYUAN3D_MODELS = {
    "hunyuan3d-21": {
        "name": "HunYuan3D 2.1",
        "repo_id": "Tencent/Hunyuan3D-2.1",
        "size": "~15GB",
        "vram_required": "16GB+",
        "description": "Latest HunYuan3D with improved quality"
    },
    "hunyuan3d-20": {
        "name": "HunYuan3D 2.0", 
        "repo_id": "Tencent/Hunyuan3D-2.0",
        "size": "~15GB",
        "vram_required": "16GB+",
        "description": "Stable HunYuan3D version"
    },
    "hunyuan3d-2mini": {
        "name": "HunYuan3D Mini",
        "repo_id": "Tencent/Hunyuan3D-mini",
        "size": "~8GB",
        "vram_required": "8GB+",
        "description": "Lightweight version for lower VRAM (4× faster)"
    }
}

# Advanced 3D Models
SPARC3D_MODELS = {
    "sparc3d-v1": {
        "name": "Sparc3D v1",
        "repo_id": "ilcve21/Sparc3D",  # HuggingFace Space - actual model repo TBD
        "size": "~12GB",
        "vram_required": "16GB+",
        "description": "High-resolution 1024³ 3D reconstruction from single images",
        "features": ["sparse_representation", "high_resolution", "arbitrary_topology"],
        "optimal_resolution": 1024,
        "space_url": "https://huggingface.co/spaces/ilcve21/Sparc3D"
    }
}

HI3DGEN_MODELS = {
    "hi3dgen-v1": {
        "name": "Hi3DGen v1",
        "repo_id": "Stable-X/trellis-normal-v0-1",  # Base model - full pipeline TBD
        "size": "~10GB",
        "vram_required": "12GB+",
        "description": "High-fidelity 3D geometry via normal bridging",
        "features": ["normal_mapping", "high_fidelity", "texture_preservation"],
        "optimal_resolution": 512,
        "space_url": "https://huggingface.co/spaces/Stable-X/Hi3DGen"
    }
}

# Combine all 3D models
ALL_3D_MODELS = {
    **HUNYUAN3D_MODELS,
    **SPARC3D_MODELS,
    **HI3DGEN_MODELS
}

# Enhancement field definitions
FLUX_ENHANCEMENT_FIELDS = {
    'art_style': {
        'label': '🎨 Art Style',
        'options': {
            'Photorealistic': 'photorealistic, hyperrealistic photography',
            'Digital Art': 'digital art, digital painting',
            'Oil Painting': 'oil painting on canvas, classical painting style',
            'Watercolor': 'watercolor illustration, watercolor painting',
            'Anime/Manga': 'anime style, manga illustration',
            '3D Render': '3D rendered, CGI, octane render',
            'Concept Art': 'concept art, production art',
            'Comic Book': 'comic book style, graphic novel art',
            'Pencil Sketch': 'pencil drawing, graphite sketch',
            'Vector Art': 'vector illustration, flat design'
        }
    },
    'lighting': {
        'label': '💡 Lighting',
        'options': {
            'Natural Light': 'natural lighting, sunlight',
            'Golden Hour': 'golden hour lighting, warm sunset light',
            'Studio': 'professional studio lighting, softbox lighting',
            'Dramatic': 'dramatic lighting, chiaroscuro, high contrast',
            'Cinematic': 'cinematic lighting, movie-like illumination',
            'Neon/Cyberpunk': 'neon lighting, cyberpunk glow, LED lights',
            'Candlelight': 'candlelight, warm flickering light',
            'Moonlight': 'moonlight, cool blue night lighting',
            'Volumetric': 'volumetric lighting, god rays, light shafts',
            'Rim Lighting': 'rim lighting, backlit, silhouette lighting'
        }
    },
    'camera_angle': {
        'label': '📷 Camera Angle',
        'options': {
            'Eye Level': 'eye level shot, straight on view',
            'Low Angle': 'low angle shot, looking up, hero shot',
            'High Angle': 'high angle shot, looking down',
            "Bird's Eye": "bird's eye view, aerial view, top down",
            'Dutch Angle': 'dutch angle, tilted camera',
            'Close-up': 'close-up shot, detailed view',
            'Wide Shot': 'wide angle shot, full view',
            'Portrait': 'portrait shot, head and shoulders',
            'Macro': 'macro photography, extreme close-up',
            'Isometric': 'isometric view, 3/4 perspective'
        }
    },
    'mood': {
        'label': '🎭 Mood/Atmosphere',
        'options': {
            'Epic': 'epic, grand, awe-inspiring',
            'Mysterious': 'mysterious, enigmatic, intriguing',
            'Peaceful': 'peaceful, serene, calm',
            'Energetic': 'energetic, dynamic, vibrant',
            'Dark/Gothic': 'dark, gothic, moody',
            'Whimsical': 'whimsical, playful, fantastical',
            'Romantic': 'romantic, dreamy, soft',
            'Futuristic': 'futuristic, sci-fi, advanced',
            'Nostalgic': 'nostalgic, retro, vintage',
            'Ethereal': 'ethereal, otherworldly, magical'
        }
    },
    'color_palette': {
        'label': '🎨 Color Palette',
        'options': {
            'Vibrant': 'vibrant colors, saturated, bold palette',
            'Muted': 'muted colors, desaturated, subtle tones',
            'Monochrome': 'monochromatic, black and white',
            'Warm': 'warm color palette, reds, oranges, yellows',
            'Cool': 'cool color palette, blues, greens, purples',
            'Pastel': 'pastel colors, soft hues',
            'Neon': 'neon colors, fluorescent, glowing',
            'Earth Tones': 'earth tones, natural colors, browns and greens',
            'High Contrast': 'high contrast, dramatic color differences',
            'Complementary': 'complementary colors, color theory'
        }
    },
    'quality_settings': {
        'label': '🔧 Quality Settings',
        'type': 'multi_checkbox',
        'options': {
            'High Resolution': '8k resolution, ultra HD',
            'Detailed': 'highly detailed, intricate details',
            'Sharp Focus': 'sharp focus, crystal clear',
            'Ray Tracing': 'ray traced, realistic reflections',
            'HDR': 'HDR, high dynamic range',
            'Award Winning': 'award winning photography/art',
            'Trending': 'trending on artstation',
            'Masterpiece': 'masterpiece, best quality'
        }
    }
}

HUNYUAN_ENHANCEMENT_FIELDS = {
    'model_style': {
        'label': '🎯 Model Style',
        'options': {
            'Photorealistic': 'photorealistic 3D model, lifelike',
            'Stylized': 'stylized 3D art, artistic interpretation',
            'Low Poly': 'low poly style, geometric simplification',
            'High Poly': 'high poly, detailed mesh',
            'Cartoon': 'cartoon style, toon shaded',
            'Realistic Game Asset': 'AAA game asset quality',
            'Architectural': 'architectural visualization quality',
            'Sculptural': 'digital sculpture, artistic'
        }
    },
    'material_type': {
        'label': '🎨 Material Type',
        'type': 'multi_checkbox',
        'options': {
            'Metal': 'metallic surface, brushed metal, chrome',
            'Wood': 'wood material, grain texture',
            'Stone': 'stone texture, marble, granite',
            'Fabric': 'fabric material, cloth texture',
            'Plastic': 'plastic material, glossy surface',
            'Glass': 'glass material, transparent',
            'Leather': 'leather texture, worn surface',
            'Organic': 'organic material, skin, flesh'
        }
    },
    'technical_specs': {
        'label': '🏗️ Technical Specs',
        'options': {
            'Game Ready (Low)': 'optimized for games, 5-10k triangles',
            'Game Ready (Mid)': 'game asset, 10-25k triangles',
            'Game Ready (High)': 'hero asset, 25-50k triangles',
            'Film/VFX': 'film quality, subdivision ready',
            '3D Printing': 'watertight mesh, printable',
            'Mobile Optimized': 'mobile ready, under 5k triangles',
            'VR/AR Ready': 'optimized for VR/AR, LODs included'
        }
    },
    'surface_detail': {
        'label': '🔍 Surface Detail',
        'options': {
            'Clean': 'clean surface, pristine condition',
            'Weathered': 'weathered, aged, worn',
            'Damaged': 'damaged, broken, distressed',
            'Ornate': 'ornate details, decorative elements',
            'Minimalist': 'minimalist, simple geometry',
            'Textured': 'heavy surface texture, detailed normal maps'
        }
    },
    'optimization': {
        'label': '⚡ Optimization',
        'type': 'multi_checkbox',
        'options': {
            'UV Unwrapped': 'proper UV mapping, texture ready',
            'Rigged': 'rigged for animation',
            'LODs': 'multiple LOD levels',
            'Collision Mesh': 'collision mesh included',
            'PBR Materials': 'PBR material setup'
        }
    }
}

# Sparc3D Enhancement Fields
SPARC3D_ENHANCEMENT_FIELDS = {
    'reconstruction_mode': {
        'label': '🎯 Reconstruction Mode',
        'options': {
            'High Resolution': 'ultra high resolution 1024³ voxel grid',
            'Balanced': 'balanced quality and performance',
            'Fast Preview': 'fast preview mode for quick results',
            'Fine Detail': 'maximum detail preservation mode'
        }
    },
    'surface_type': {
        'label': '🏗️ Surface Type',
        'options': {
            'Watertight': 'watertight closed surface mesh',
            'Open Surface': 'open surface with boundaries',
            'Multi-Component': 'multiple disconnected components',
            'Manifold': 'manifold surface topology'
        }
    },
    'sparse_density': {
        'label': '📊 Sparse Density',
        'options': {
            'Dense': 'dense sparse representation for maximum quality',
            'Balanced': 'balanced sparse density',
            'Sparse': 'sparse representation for efficiency',
            'Adaptive': 'adaptive density based on geometry'
        }
    },
    'geometry_detail': {
        'label': '🔍 Geometry Detail',
        'type': 'multi_checkbox',
        'options': {
            'Sharp Edges': 'preserve sharp edges and corners',
            'Smooth Surfaces': 'smooth surface interpolation',
            'Fine Details': 'capture fine geometric details',
            'Texture Mapping': 'prepare for texture mapping',
            'Normal Preservation': 'preserve surface normals'
        }
    }
}

# Hi3DGen Enhancement Fields
HI3DGEN_ENHANCEMENT_FIELDS = {
    'normal_quality': {
        'label': '🎨 Normal Map Quality',
        'options': {
            'Ultra High': 'ultra high quality normal estimation',
            'High': 'high quality balanced estimation',
            'Standard': 'standard quality for fast processing',
            'Adaptive': 'adaptive quality based on content'
        }
    },
    'geometry_fidelity': {
        'label': '💎 Geometry Fidelity',
        'options': {
            'Maximum': 'maximum fidelity to input image',
            'High': 'high fidelity with optimization',
            'Balanced': 'balanced fidelity and smoothness',
            'Stylized': 'stylized interpretation allowed'
        }
    },
    'surface_complexity': {
        'label': '🏗️ Surface Complexity',
        'options': {
            'Complex': 'complex surface with all details',
            'Moderate': 'moderate complexity',
            'Simplified': 'simplified for real-time use',
            'Adaptive': 'adaptive based on input'
        }
    },
    'material_inference': {
        'label': '🎨 Material Inference',
        'type': 'multi_checkbox',
        'options': {
            'PBR Materials': 'infer PBR material properties',
            'Roughness Map': 'generate roughness map',
            'Metallic Map': 'generate metallic map',
            'Ambient Occlusion': 'compute ambient occlusion',
            'Displacement': 'generate displacement map'
        }
    },
    'output_optimization': {
        'label': '⚡ Output Optimization',
        'type': 'multi_checkbox',
        'options': {
            'Decimation': 'polygon decimation for efficiency',
            'UV Unwrapping': 'automatic UV unwrapping',
            'Tangent Space': 'compute tangent space',
            'LOD Generation': 'generate level of detail',
            'Watertight': 'ensure watertight mesh'
        }
    }
}

# Theme configuration
THEME = {
    "primary": "#7C3AED",  # Electric Purple
    "success": "#10B981",  # Emerald
    "background": "#0A0A0A",  # Deep Black
    "card": "#1F1F1F",
    "text": "#E5E5E5",
    "border": "#333333"
}

# Pricing configuration (for web app)
PRICING_TIERS = {
    "free": {
        "name": "Free",
        "price_usd": 0,
        "price_sar": 0,
        "credits": 50,
        "features": ["Watermarked outputs", "1 project", "Community support"]
    },
    "starter": {
        "name": "Starter",
        "price_usd": 25,
        "price_sar": 94,
        "credits": 500,
        "features": ["No watermarks", "API access", "Email support"]
    },
    "professional": {
        "name": "Professional", 
        "price_usd": 69,
        "price_sar": 259,
        "credits": 2000,
        "features": ["Priority queue", "Advanced models", "Phone support"]
    },
    "studio": {
        "name": "Studio",
        "price_usd": 179,
        "price_sar": 671,
        "credits": 6000,
        "features": ["5 team seats", "Custom models", "Dedicated support"]
    }
}

# Credit costs (1 credit = $0.01 USD)
CREDIT_COSTS = {
    "image_generation": 5,  # 5 credits per image
    "3d_conversion": 45,   # 45 credits per 3D model
    "face_swap": 8,        # 8 credits per face swap
    "video_per_second": 15  # 15 credits per second of video
}

# RunPod configuration
RUNPOD_CONFIG = {
    "gpu_type": "A100",
    "cost_per_second": 0.00025,  # $0.90/hour
    "timeout": 600,  # 10 minutes max
}

# LLM configuration
LLM_CONFIG = {
    "model": "mistral:latest",
    "temperature": 0.7,
    "max_tokens": 500
}

# Video Models Configuration
VIDEO_MODELS = {
    "wan2_1_1.3b": {
        "name": "Wan2.1 1.3B",
        "repo_id": "Wan-AI/Wan2.1-T2V-1.3B",
        "size": "~2.6 GB",
        "vram_required": "8GB+",
        "description": "Consumer GPU friendly with visual text generation",
        "max_frames": 129,
        "fps": 24,
        "features": ["visual_text", "multilingual", "flow_matching"]
    },
    "wan2_1_14b": {
        "name": "Wan2.1 14B",
        "repo_id": "Wan-AI/Wan2.1-T2V-14B",
        "size": "~28 GB",
        "vram_required": "16GB+",
        "description": "Professional quality with 1080p support",
        "max_frames": 129,
        "fps": 24,
        "features": ["visual_text", "multilingual", "1080p", "flow_matching"]
    },
    "hunyuanvideo": {
        "name": "HunyuanVideo",
        "repo_id": "tencent/HunyuanVideo",
        "model_file": "hunyuanvideo_diffusion_pytorch_model.safetensors",
        "size": "~50 GB",
        "vram_required": "24GB+",
        "description": "Cinema-quality with dual-stream architecture",
        "max_frames": 129,
        "fps": 24,
        "features": ["dual_stream", "llama_encoder", "30fps", "cinema_quality"]
    },
    "ltxvideo": {
        "name": "LTX-Video",
        "repo_id": "Lightricks/LTX-Video",
        "size": "~8 GB",
        "vram_required": "12GB+",
        "description": "Real-time generation (4s for 5s video)",
        "max_frames": 121,
        "fps": 30,
        "features": ["real_time", "high_resolution", "artifact_reduction"]
    },
    "mochi_1": {
        "name": "Mochi-1",
        "repo_id": "genmo/mochi-1-preview",
        "size": "~25 GB",
        "vram_required": "24GB+",
        "description": "10B model with smooth 30fps motion",
        "max_frames": 163,
        "fps": 30,
        "features": ["asymmetric_dit", "128x_compression", "smooth_motion"]
    },
    "cogvideox_5b": {
        "name": "CogVideoX-5B",
        "repo_id": "THUDM/CogVideoX-5b",
        "size": "~16 GB",
        "vram_required": "16GB+",
        "description": "Superior image-to-video specialist",
        "max_frames": 49,
        "fps": 8,
        "features": ["i2v_specialist", "lora_support", "interpolation"]
    },
    "cogvideox": {
        "name": "CogVideoX-2B",
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