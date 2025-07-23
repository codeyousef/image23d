"""Path configurations for the application."""

from pathlib import Path
import os
import tempfile

# --- Base Paths ---
# Use absolute paths based on the project directory
PROJECT_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
MODELS_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "outputs"
CACHE_DIR = PROJECT_DIR / "cache/data"
TEMP_DIR = PROJECT_DIR / "cache/temp"

# --- Model Subdirectories ---
IMAGE_MODELS_DIR = MODELS_DIR / "image"
THREED_MODELS_DIR = MODELS_DIR / "3d"
VIDEO_MODELS_DIR = MODELS_DIR / "video"
LORA_MODELS_DIR = MODELS_DIR / "loras"
GGUF_MODELS_DIR = MODELS_DIR / "gguf"
TEXTURE_MODELS_DIR = MODELS_DIR / "texture_components"

# --- Output Subdirectories ---
IMAGE_OUTPUT_DIR = OUTPUT_DIR / "images"
THREED_OUTPUT_DIR = OUTPUT_DIR / "3d"
VIDEO_OUTPUT_DIR = OUTPUT_DIR / "videos"
PREVIEW_OUTPUT_DIR = OUTPUT_DIR / "previews"

# --- Cache Subdirectories ---
THUMBNAIL_CACHE_DIR = CACHE_DIR / "thumbnails"
ENHANCED_PROMPT_CACHE_DIR = CACHE_DIR / "enhanced_prompts"

# --- Create necessary directories ---
for directory in [
    MODELS_DIR, OUTPUT_DIR, CACHE_DIR, TEMP_DIR,
    IMAGE_MODELS_DIR, THREED_MODELS_DIR, VIDEO_MODELS_DIR,
    LORA_MODELS_DIR, GGUF_MODELS_DIR, TEXTURE_MODELS_DIR,
    IMAGE_OUTPUT_DIR, THREED_OUTPUT_DIR, VIDEO_OUTPUT_DIR,
    PREVIEW_OUTPUT_DIR, THUMBNAIL_CACHE_DIR, ENHANCED_PROMPT_CACHE_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)

# Configure tempfile to use our custom temp directory
tempfile.tempdir = str(TEMP_DIR)