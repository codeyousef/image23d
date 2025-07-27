"""
Desktop app configuration that imports from main app
"""

import sys
import os
from pathlib import Path

# Add parent directory to path to import from main app
desktop_dir = Path(__file__).parent.parent
project_root = desktop_dir.parent
sys.path.insert(0, str(project_root))

# Import all configurations from main app
from src.hunyuan3d_app.config import (
    # Model definitions
    IMAGE_MODELS,
    GGUF_IMAGE_MODELS,
    GATED_IMAGE_MODELS,
    HUNYUAN3D_MODELS,
    VIDEO_MODELS,
    TEXTURE_PIPELINE_COMPONENTS,
    SPARC3D_MODELS,
    HI3DGEN_MODELS,
    
    # Paths
    MODELS_DIR,
    OUTPUT_DIR,
    CACHE_DIR,
    IMAGE_MODELS_DIR,
    THREED_MODELS_DIR,
    VIDEO_MODELS_DIR,
    
    # Other configs
    QUALITY_PRESETS,
)

# For compatibility with desktop app
FLUX_MODELS = IMAGE_MODELS