import gc
import logging
import os
import time
import threading
import re
import shutil
import sys
import io
import base64
import tempfile
import psutil
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
from threading import Lock

import gradio as gr
from gradio import Dropdown
import torch
from diffusers import (
    StableDiffusionXLPipeline,
    DiffusionPipeline,
    AutoPipelineForText2Image,
    FluxPipeline,
    FluxTransformer2DModel,
    GGUFQuantizationConfig
)
from huggingface_hub import snapshot_download, HfApi

from .config import (
    IMAGE_MODELS, GATED_IMAGE_MODELS, GGUF_IMAGE_MODELS,
    ALL_IMAGE_MODELS, HUNYUAN3D_MODELS
)
from .memory_manager import get_memory_manager
from .gguf_manager import GGUFModelManager, GGUFModelInfo

logger = logging.getLogger(__name__)

SECRETS_DIR = Path.cwd() / ".secrets"
HF_TOKEN_FILE = SECRETS_DIR / "hf_token.txt"

def save_hf_token(token):
    """Base64 encode and save the Hugging Face token."""
    if not token:
        return
    SECRETS_DIR.mkdir(exist_ok=True)
    encoded_token = base64.b64encode(token.encode('utf-8'))
    HF_TOKEN_FILE.write_bytes(encoded_token)


def load_hf_token():
    """Load and decode the Hugging Face token."""
    if not HF_TOKEN_FILE.exists():
        return None
    try:
        encoded_token = HF_TOKEN_FILE.read_bytes()
        return base64.b64decode(encoded_token).decode('utf-8')
    except Exception as e:
        logger.error(f"Could not load HF token: {e}")
        return None


class ModelManager:
    def __init__(self, models_dir: Path, src_models_dir: Path):
        self.models_dir = models_dir
        self.src_models_dir = src_models_dir
        self.image_model = None
        self.image_model_name = None
        self.hunyuan3d_model = None
        self.hunyuan3d_model_name = None
        self.gguf_manager = GGUFModelManager(cache_dir=str(models_dir / "gguf"))

        self.download_in_progress = False
        self.current_download_model = None
        self.stop_download_flag = False

        self.hf_token = load_hf_token()
        if self.hf_token:
            os.environ["HF_TOKEN"] = self.hf_token

    def set_hf_token(self, token):
        """Set Hugging Face token for gated models"""
        self.hf_token = token
        if token:
            os.environ["HF_TOKEN"] = token
            save_hf_token(token)
            return "✅ Hugging Face token set successfully"
        else:
            # Clear token if empty string is passed
            if HF_TOKEN_FILE.exists():
                HF_TOKEN_FILE.unlink()
            if "HF_TOKEN" in os.environ:
                del os.environ["HF_TOKEN"]
            return "❌ Invalid token or token cleared"

    def get_model_status(self):
        """Get current status of loaded models"""
        vram_used = 0
        if torch.cuda.is_available():
            vram_used = torch.cuda.memory_allocated() / 1024 ** 3

        # Check system requirements
        from .system_checker import check_system_requirements
        sys_req = check_system_requirements()

        status_parts = [
            f"<div class='model-status'>"
        ]

        # Add system requirements warning if needed
        if sys_req["overall_status"] == "error":
            status_parts.append(
                "<div class='system-warning error' style='background-color: #f8d7da; color: #721c24; padding: 10px; margin-bottom: 10px; border-radius: 5px; border: 1px solid #f5c6cb;'>"
                "⚠️ <strong>Warning:</strong> Your system does not meet minimum requirements. "
                "See the System Requirements tab for details."
                "</div>"
            )
        elif sys_req["overall_status"] == "warning":
            status_parts.append(
                "<div class='system-warning warning' style='background-color: #fff3cd; color: #856404; padding: 10px; margin-bottom: 10px; border-radius: 5px; border: 1px solid #ffeeba;'>"
                "⚠️ <strong>Note:</strong> Your system meets minimum but not recommended requirements. "
                "See the System Requirements tab for details."
                "</div>"
            )

        status_parts.extend([
            f"<h3>📊 Model Status</h3>",
            f"<div class='status-grid'>"
        ])

        # Image model status
        if self.image_model:
            status_parts.append(
                f"<div class='status-item success'>🎨 Image: {self.image_model_name}</div>"
            )
        else:
            status_parts.append(
                f"<div class='status-item'>🎨 Image: Not loaded</div>"
            )

        # 3D model status
        if self.hunyuan3d_model:
            status_parts.append(
                f"<div class='status-item success'>🎭 3D: {self.hunyuan3d_model_name}</div>"
            )
        else:
            status_parts.append(
                f"<div class='status-item'>🎭 3D: Not loaded</div>"
            )

        # VRAM status
        status_parts.append(
            f"<div class='status-item'>💾 VRAM: {vram_used:.1f} GB used</div>"
        )

        status_parts.extend(["</div>", "</div>"])

        return "".join(status_parts)

    def check_model_complete(self, model_path, model_type, model_name):
        """Check if a model is completely downloaded"""
        # Check if this is a GGUF model
        from .config import GGUF_IMAGE_MODELS
        if model_name in GGUF_IMAGE_MODELS:
            return self.check_gguf_model_complete(model_name)
        
        model_path = model_path.resolve()
        if not model_path.exists():
            return False

        # Check if this is a HuggingFace cache directory structure
        # HF creates: cache_dir/models--{org}--{model}/snapshots/{hash}/
        subdirs = [d for d in model_path.iterdir() if d.is_dir()]
        hf_cache_dirs = [d for d in subdirs if d.name.startswith("models--")]
        
        if hf_cache_dirs:
            # This is a HF cache structure, need to look inside for the actual model
            for hf_cache_dir in hf_cache_dirs:
                snapshots_dir = hf_cache_dir / "snapshots"
                if snapshots_dir.exists():
                    # Get the latest snapshot (usually there's only one)
                    snapshot_dirs = [d for d in snapshots_dir.iterdir() if d.is_dir()]
                    if snapshot_dirs:
                        # Check the first (and usually only) snapshot
                        actual_model_path = snapshot_dirs[0]
                        # Recursively check this path
                        return self._check_model_files(actual_model_path, model_type, model_name)
            return False
        else:
            # Standard directory structure
            return self._check_model_files(model_path, model_type, model_name)

    def _check_model_files(self, model_path, model_type, model_name):
        """Check if model files are complete in the given path"""
        # Check for essential files based on model type
        if model_type == "image":
            # Special case for FLUX models which have a different structure
            if model_name.startswith("FLUX"):
                # FLUX models have a single large safetensors file at the root
                # The file name format is flux1-dev.safetensors for FLUX.1-dev
                if model_name == "FLUX.1-dev":
                    flux_model_file = "flux1-dev.safetensors"
                else:
                    # For other FLUX models, use the original pattern
                    flux_model_file = f"flux1-{model_name.split('.')[-1].lower()}.safetensors"

                # Essential files for FLUX models
                essential_files = [
                    "model_index.json",  # Pipeline config
                    "scheduler/scheduler_config.json",  # Scheduler config
                    "text_encoder/config.json",  # Text encoder config
                    "text_encoder_2/config.json", # Text encoder 2 config
                    "tokenizer/tokenizer_config.json",  # Tokenizer config
                    "tokenizer_2/tokenizer_config.json", # Tokenizer 2 config
                    "transformer/config.json", # Transformer config
                    "vae/config.json" # VAE config
                ]

                # Check if all essential files exist
                for file in essential_files:
                    file_full_path = model_path / file
                    logger.debug(f"check_model_complete - Checking essential file: {file_full_path}, exists: {file_full_path.is_file()}")
                    if not file_full_path.is_file():
                        return False

                # Check for the main model weights file
                logger.debug(f"check_model_complete - Checking main FLUX model file: {model_path / flux_model_file}, exists: {(model_path / flux_model_file).is_file()}")

                # Check for essential model weight files
                has_text_encoder_weights = any(
                    (model_path / "text_encoder").glob("*.safetensors")
                ) or any(
                    (model_path / "text_encoder").glob("*.bin")
                )

                has_text_encoder_2_weights = any(
                    (model_path / "text_encoder_2").glob("*.safetensors")
                ) or any(
                    (model_path / "text_encoder_2").glob("*.bin")
                )

                has_transformer_weights = any(
                    (model_path / "transformer").glob("*.safetensors")
                ) or any(
                    (model_path / "transformer").glob("*.bin")
                )

                has_vae_weights = any(
                    (model_path / "vae").glob("*.safetensors")
                ) or any(
                    (model_path / "vae").glob("*.bin")
                )

                # Log the status of each component
                logger.debug(f"check_model_complete - FLUX model components status:")
                logger.debug(f"check_model_complete - Main model file: {(model_path / flux_model_file).is_file()}")
                logger.debug(f"check_model_complete - text_encoder weights: {has_text_encoder_weights}")
                logger.debug(f"check_model_complete - text_encoder_2 weights: {has_text_encoder_2_weights}")
                logger.debug(f"check_model_complete - transformer weights: {has_transformer_weights}")
                logger.debug(f"check_model_complete - vae weights: {has_vae_weights}")

                # Return True only if all components are present
                return ((model_path / flux_model_file).is_file() and 
                        has_text_encoder_weights and 
                        has_text_encoder_2_weights and 
                        has_transformer_weights and 
                        has_vae_weights)

            # Standard diffusion models should have these key files
            essential_files = [
                "model_index.json",  # Pipeline config
                "scheduler/scheduler_config.json",  # Scheduler config
                "unet/config.json",  # UNet config
                "text_encoder/config.json",  # Text encoder config
                "tokenizer/tokenizer_config.json",  # Tokenizer config
                "vae/config.json"  # VAE config
            ]

            # Check if all essential files exist
            for file in essential_files:
                if not (model_path / file).exists():
                    return False

            # Also check for actual model weights
            has_unet_weights = any(
                (model_path / "unet").glob("*.safetensors")
            ) or any(
                (model_path / "unet").glob("*.bin")
            )

            has_vae_weights = any(
                (model_path / "vae").glob("*.safetensors")
            ) or any(
                (model_path / "vae").glob("*.bin")
            )

            has_text_encoder_weights = any(
                (model_path / "text_encoder").glob("*.safetensors")
            ) or any(
                (model_path / "text_encoder").glob("*.bin")
            )

            return has_unet_weights and has_vae_weights and has_text_encoder_weights

        else:  # 3D models
            # For Hunyuan3D, check for key files
            essential_files = [
                "config.json",
                "preprocessor_config.json"
            ]

            for file in essential_files:
                if not (model_path / file).exists():
                    return False

            # Check for model weights
            has_weights = any(model_path.glob("*.safetensors")) or any(model_path.glob("*.bin"))
            return has_weights

    def check_missing_components(self, model_type, model_name):
        """Check for missing components required for optimized image generation

        Args:
            model_type: Type of model ('image' or '3d')
            model_name: Name of the model

        Returns:
            List of missing component names, empty list if all components are present
        """
        from .config import GGUF_IMAGE_MODELS, FLUX_COMPONENTS
        
        missing_components = []

        # Check if this is a GGUF model
        if model_name in GGUF_IMAGE_MODELS:
            # GGUF models have a different structure
            config = GGUF_IMAGE_MODELS[model_name]
            
            # Use the helper method that checks multiple locations
            gguf_file = self._find_gguf_file(model_name, config.gguf_file)
            
            if not gguf_file:
                return ["complete model"]
            
            # For FLUX GGUF models, check required components
            if "FLUX" in model_name:
                # Check VAE
                vae_path = self.models_dir / "vae" / FLUX_COMPONENTS["vae"]["filename"]
                if not vae_path.exists():
                    missing_components.append("VAE")
                
                # Check text encoders
                te_dir = self.models_dir / "text_encoders"
                clip_path = te_dir / FLUX_COMPONENTS["text_encoder_clip"]["filename"]
                t5_path = te_dir / FLUX_COMPONENTS["text_encoder_t5"]["filename"]
                
                if not clip_path.exists():
                    missing_components.append("CLIP Text Encoder")
                if not t5_path.exists():
                    missing_components.append("T5 Text Encoder")
            
            return missing_components

        # First check if the model is downloaded at all
        model_path_cache = self.models_dir / model_type / model_name
        model_path_src = self.src_models_dir / model_type / model_name if model_name.startswith("FLUX") else None

        # Determine which path to use
        model_path = None
        if model_path_cache.exists():
            model_path = model_path_cache
        elif model_path_src and model_path_src.exists():
            model_path = model_path_src
        else:
            # Model not downloaded at all
            return ["complete model"]
        
        # Check if this is a HuggingFace cache directory structure
        subdirs = [d for d in model_path.iterdir() if d.is_dir()]
        hf_cache_dirs = [d for d in subdirs if d.name.startswith("models--")]
        
        if hf_cache_dirs:
            # This is a HF cache structure, need to look inside for the actual model
            for hf_cache_dir in hf_cache_dirs:
                snapshots_dir = hf_cache_dir / "snapshots"
                if snapshots_dir.exists():
                    # Get the latest snapshot (usually there's only one)
                    snapshot_dirs = [d for d in snapshots_dir.iterdir() if d.is_dir()]
                    if snapshot_dirs:
                        # Check the first (and usually only) snapshot
                        model_path = snapshot_dirs[0]
                        break

        # For image models, check for specific components
        if model_type == "image":
            # Check for VAE
            has_vae_config = (model_path / "vae" / "config.json").exists()
            has_vae_weights = any((model_path / "vae").glob("*.safetensors")) or any((model_path / "vae").glob("*.bin"))

            if not has_vae_config or not has_vae_weights:
                missing_components.append("VAE")

            # For FLUX models, check additional components
            if model_name.startswith("FLUX"):
                # Check for text encoders
                has_text_encoder_config = (model_path / "text_encoder" / "config.json").exists()
                has_text_encoder_weights = any((model_path / "text_encoder").glob("*.safetensors")) or any((model_path / "text_encoder").glob("*.bin"))

                has_text_encoder_2_config = (model_path / "text_encoder_2" / "config.json").exists()
                has_text_encoder_2_weights = any((model_path / "text_encoder_2").glob("*.safetensors")) or any((model_path / "text_encoder_2").glob("*.bin"))

                has_transformer_config = (model_path / "transformer" / "config.json").exists()
                has_transformer_weights = any((model_path / "transformer").glob("*.safetensors")) or any((model_path / "transformer").glob("*.bin"))

                if not has_text_encoder_config or not has_text_encoder_weights:
                    missing_components.append("Text Encoder")

                if not has_text_encoder_2_config or not has_text_encoder_2_weights:
                    missing_components.append("Text Encoder 2")

                if not has_transformer_config or not has_transformer_weights:
                    missing_components.append("Transformer")
            else:
                # For standard diffusion models
                has_unet_config = (model_path / "unet" / "config.json").exists()
                has_unet_weights = any((model_path / "unet").glob("*.safetensors")) or any((model_path / "unet").glob("*.bin"))

                has_text_encoder_config = (model_path / "text_encoder" / "config.json").exists()
                has_text_encoder_weights = any((model_path / "text_encoder").glob("*.safetensors")) or any((model_path / "text_encoder").glob("*.bin"))

                if not has_unet_config or not has_unet_weights:
                    missing_components.append("UNet")

                if not has_text_encoder_config or not has_text_encoder_weights:
                    missing_components.append("Text Encoder")

        return missing_components

    def _find_gguf_file(self, model_name, gguf_filename):
        """Find GGUF file in either direct path or HF cache structure"""
        # Try multiple possible directory names
        possible_dirs = [
            self.models_dir / "gguf" / model_name,  # e.g. "FLUX.1-dev-Q8"
            self.models_dir / "gguf" / f"{model_name} GGUF (Memory Optimized)",  # With full name
        ]
        
        # Also check if config has a different name
        from .config import GGUF_IMAGE_MODELS
        if model_name in GGUF_IMAGE_MODELS:
            config = GGUF_IMAGE_MODELS[model_name]
            possible_dirs.append(self.models_dir / "gguf" / config.name)
        
        for gguf_dir in possible_dirs:
            if not gguf_dir.exists():
                continue
                
            # First check direct path
            direct_path = gguf_dir / gguf_filename
            if direct_path.exists():
                logger.info(f"Found GGUF file at direct path: {direct_path}")
                return direct_path
                
            # Check HuggingFace cache structure
            # Look for models--org--repo directories
            for item in gguf_dir.iterdir():
                if item.is_dir() and item.name.startswith("models--"):
                    # Check snapshots directory
                    snapshots_dir = item / "snapshots"
                    if snapshots_dir.exists():
                        # Check each snapshot
                        for snapshot in snapshots_dir.iterdir():
                            if snapshot.is_dir():
                                file_path = snapshot / gguf_filename
                                if file_path.exists():
                                    logger.info(f"Found GGUF file in HF cache: {file_path}")
                                    return file_path
        
        logger.warning(f"Could not find GGUF file {gguf_filename} for model {model_name}")
        logger.warning(f"Searched in: {[str(d) for d in possible_dirs]}")
        return None
    
    def check_gguf_model_complete(self, model_name):
        """Check if a GGUF model and its components are fully downloaded"""
        from .config import GGUF_IMAGE_MODELS, FLUX_COMPONENTS
        
        if model_name not in GGUF_IMAGE_MODELS:
            return False
            
        config = GGUF_IMAGE_MODELS[model_name]
        
        # Check main GGUF file using helper method
        gguf_file = self._find_gguf_file(model_name, config.gguf_file)
        logger.info(f"Checking GGUF model {model_name}: found file at {gguf_file}")
        if not gguf_file:
            return False
            
        # For FLUX models, also check required components
        if "FLUX" in model_name:
            # Check VAE
            vae_path = self.models_dir / "vae" / FLUX_COMPONENTS["vae"]["filename"]
            if not vae_path.exists():
                return False
                
            # Check text encoders
            te_dir = self.models_dir / "text_encoders"
            clip_path = te_dir / FLUX_COMPONENTS["text_encoder_clip"]["filename"]
            t5_path = te_dir / FLUX_COMPONENTS["text_encoder_t5"]["filename"]
            
            if not clip_path.exists() or not t5_path.exists():
                return False
        
        return True

    def download_missing_components(self, model_type, model_name, progress):
        """Download missing components for a model

        Args:
            model_type: Type of model ('image' or '3d')
            model_name: Name of the model
            progress: Gradio progress object for UI updates

        Returns:
            Generator yielding progress updates and final status
        """
        try:
            # Check if another download is already in progress
            if self.download_in_progress:
                yield f"""
<div class="error-box">
    <h4>⚠️ Download Already in Progress</h4>
    <p><strong>Model:</strong> {self.current_download_model}</p>
    <p>Please wait for it to complete or stop it before starting a new download.</p>
</div>
"""
                return

            # Reset stop flag and set download in progress
            self.stop_download_flag = False
            self.download_in_progress = True
            self.current_download_model = f"{model_name} (components)"

            from huggingface_hub import HfApi, hf_hub_download, snapshot_download
            import threading
            import time
            import shutil

            # Get model config and repo ID
            if model_type == "image":
                if model_name in IMAGE_MODELS:
                    config = IMAGE_MODELS[model_name]
                    is_gated = False
                elif model_name in GATED_IMAGE_MODELS:
                    config = GATED_IMAGE_MODELS[model_name]
                    is_gated = True
                elif model_name in GGUF_IMAGE_MODELS:
                    # For GGUF models, we only need to download FLUX components
                    from .config import FLUX_COMPONENTS
                    
                    # Check which components are missing
                    missing = self.check_missing_components(model_type, model_name)
                    
                    if not missing or "complete model" in missing:
                        yield f"❌ No missing components to download for {model_name}", *self.get_model_selection_data()
                        self.download_in_progress = False
                        self.current_download_model = None
                        return
                    
                    # Download only the missing FLUX components
                    total_components = len(missing)
                    for idx, component_name in enumerate(missing):
                        if self.stop_download_flag:
                            yield f"⚠️ Download stopped by user", *self.get_model_selection_data()
                            break
                        
                        component_key = None
                        if "VAE" in component_name:
                            component_key = "vae"
                        elif "CLIP" in component_name:
                            component_key = "text_encoder_clip"
                        elif "T5" in component_name:
                            component_key = "text_encoder_t5"
                        
                        if component_key and component_key in FLUX_COMPONENTS:
                            comp_config = FLUX_COMPONENTS[component_key]
                            yield f"📥 Downloading {component_name} ({idx+1}/{total_components})...", *self.get_model_selection_data()
                            
                            # Download the component
                            target_dir = self.models_dir / comp_config["target_dir"]
                            target_dir.mkdir(parents=True, exist_ok=True)
                            
                            try:
                                hf_hub_download(
                                    repo_id=comp_config["repo_id"],
                                    filename=comp_config["filename"],
                                    local_dir=target_dir,
                                    token=self.hf_token if is_gated else None
                                )
                                yield f"✅ Downloaded {component_name} ({idx+1}/{total_components})", *self.get_model_selection_data()
                            except Exception as e:
                                yield f"❌ Failed to download {component_name}: {str(e)}", *self.get_model_selection_data()
                                break
                    
                    yield f"✅ All missing components downloaded for {model_name}", *self.get_model_selection_data()
                    self.download_in_progress = False
                    self.current_download_model = None
                    return
                else:
                    yield f"❌ Unknown model: {model_name}", *self.get_model_selection_data()
                    self.download_in_progress = False
                    self.current_download_model = None
                    return

                repo_id = config.repo_id
            else:
                config = HUNYUAN3D_MODELS[model_name]
                repo_id = config["repo_id"]
                is_gated = False

            # Check for missing components
            missing_components = self.check_missing_components(model_type, model_name)

            if not missing_components:
                yield f"""
<div class="success-box">
    <h4>✅ All Components Present</h4>
    <p><strong>Model:</strong> {model_name}</p>
    <p>All required components are already downloaded.</p>
</div>
""", *self.get_model_selection_data()
                self.download_in_progress = False
                self.current_download_model = None
                return

            if "complete model" in missing_components:
                yield f"""
<div class="error-box">
    <h4>❌ Model Not Downloaded</h4>
    <p>The model <strong>{model_name}</strong> is not downloaded at all.</p>
    <p>Please download the complete model first.</p>
</div>
""", *self.get_model_selection_data()
                self.download_in_progress = False
                self.current_download_model = None
                return

            # Determine model path
            model_path = self.models_dir / model_type / model_name
            model_path.mkdir(parents=True, exist_ok=True)

            # Show absolute path for clarity
            abs_path = model_path.absolute()

            # Check if gated model and token is needed
            if is_gated and not self.hf_token:
                yield f"""
<div class="error-box">
    <h4>🔒 Gated Model - Authentication Required</h4>
    <p><strong>Model:</strong> {model_name}</p>
    <p>This model requires Hugging Face authentication.</p>

    <h5>To download this model:</h5>
    <ol>
        <li>Go to <a href="https://huggingface.co/{repo_id}" target="_blank">the model page</a></li>
        <li>Click "Access repository" and accept the license</li>
        <li>Get your token from <a href="https://huggingface.co/settings/tokens" target="_blank">HF Settings</a></li>
        <li>Enter your token in the "HF Token" field above and try again</li>
    </ol>
</div>
""", *self.get_model_selection_data()
                self.download_in_progress = False
                self.current_download_model = None
                return

            # Prepare for download
            progress(0.0, desc=f"Preparing to download missing components for {model_name}...")

            # Map component names to their folder paths
            component_folders = {
                "VAE": "vae",
                "UNet": "unet",
                "Text Encoder": "text_encoder",
                "Text Encoder 2": "text_encoder_2",
                "Transformer": "transformer"
            }

            # Track download progress
            download_complete = False
            download_error = None
            start_time = time.time()
            total_size = 0
            downloaded_size = 0

            # Function to download a specific component
            def download_component(component_name, folder_name):
                nonlocal total_size, downloaded_size, download_error

                try:
                    # Create component directory if it doesn't exist
                    component_dir = model_path / folder_name
                    component_dir.mkdir(parents=True, exist_ok=True)

                    # Download component files
                    # First, get the list of files in the component folder
                    api = HfApi()
                    try:
                        files = api.list_repo_files(repo_id, folder_name)
                    except Exception as e:
                        logger.error(f"Error listing files for {component_name}: {str(e)}")
                        download_error = e
                        return

                    # Filter files to only include those in the component folder
                    component_files = [f for f in files if f.startswith(f"{folder_name}/")]

                    if not component_files:
                        logger.warning(f"No files found for {component_name} in {folder_name}")
                        download_error = Exception(f"No files found for {component_name}")
                        return

                    # Download each file
                    for file_path in component_files:
                        if self.stop_download_flag:
                            download_error = Exception("Download stopped by user")
                            return

                        try:
                            # Download the file
                            local_file = hf_hub_download(
                                repo_id=repo_id,
                                filename=file_path,
                                local_dir=str(model_path),
                                local_dir_use_symlinks=False,
                                token=self.hf_token if is_gated else None,
                                resume_download=True
                            )

                            # Update progress
                            file_size = os.path.getsize(local_file)
                            total_size += file_size
                            downloaded_size += file_size

                        except Exception as e:
                            logger.error(f"Error downloading {file_path}: {str(e)}")
                            download_error = e
                            return

                except Exception as e:
                    logger.error(f"Error downloading {component_name}: {str(e)}")
                    download_error = e

            # Start download threads for each missing component
            threads = []
            for component in missing_components:
                if component in component_folders:
                    folder = component_folders[component]
                    thread = threading.Thread(
                        target=download_component,
                        args=(component, folder)
                    )
                    thread.daemon = True
                    threads.append((component, thread))
                    thread.start()

            # Update progress while downloading
            total_components = len(threads)
            completed_components = 0

            yield f"""
<div class="info-box">
    <h4>⏳ Downloading Missing Components</h4>
    <p><strong>Model:</strong> {model_name}</p>
    <p><strong>Components:</strong> {", ".join(missing_components)}</p>
    <p>Download in progress...</p>
</div>
"""

            # Wait for all threads to complete
            for component, thread in threads:
                progress_value = 0.1 + (completed_components / total_components * 0.8)
                progress(progress_value, desc=f"Downloading {component}...")

                # Wait for thread to complete with timeout
                max_wait_time = 600  # 10 minutes
                wait_time = 0
                check_interval = 1  # 1 second

                while thread.is_alive() and wait_time < max_wait_time:
                    if self.stop_download_flag:
                        break

                    time.sleep(check_interval)
                    wait_time += check_interval

                    # Update progress
                    elapsed = time.time() - start_time
                    progress_value = 0.1 + (completed_components / total_components * 0.8)
                    progress(progress_value, desc=f"Downloading {component} ({elapsed:.1f}s elapsed)...")

                    # Yield progress update every 5 seconds
                    if wait_time % 5 == 0:
                        yield f"""
<div class="info-box">
    <h4>⏳ Downloading Missing Components</h4>
    <p><strong>Model:</strong> {model_name}</p>
    <p><strong>Components:</strong> {", ".join(missing_components)}</p>
    <p><strong>Current:</strong> Downloading {component} ({elapsed:.1f}s elapsed)</p>
    <p><strong>Progress:</strong> {completed_components}/{total_components} components completed</p>
</div>
"""

                if thread.is_alive():
                    # Thread timed out
                    download_error = Exception(f"Download of {component} timed out after {max_wait_time} seconds")
                    break

                completed_components += 1

            # Check if download was stopped by user
            if self.stop_download_flag:
                yield f"""
<div class="warning-box">
    <h4>⚠️ Download Stopped</h4>
    <p><strong>Model:</strong> {model_name}</p>
    <p>Download of missing components was stopped by user.</p>
</div>
""", *self.get_model_selection_data()
                self.download_in_progress = False
                self.current_download_model = None
                return

            # Check for errors
            if download_error:
                yield f"""
<div class="error-box">
    <h4>❌ Download Failed</h4>
    <p><strong>Model:</strong> {model_name}</p>
    <p><strong>Error:</strong> {str(download_error)}</p>
    <p>Please try again or download the complete model.</p>
</div>
""", *self.get_model_selection_data()
                self.download_in_progress = False
                self.current_download_model = None
                return

            # Download completed successfully
            total_time = time.time() - start_time

            # Format time
            hours, remainder = divmod(total_time, 3600)
            minutes, seconds = divmod(remainder, 60)

            time_str = ""
            if hours > 0:
                time_str += f"{int(hours)}h "
            if minutes > 0 or hours > 0:
                time_str += f"{int(minutes)}m "
            time_str += f"{seconds:.1f}s"

            # Calculate average download speed
            avg_speed = total_size / total_time if total_time > 0 else 0

            yield f"""
<div class="success-box">
    <h4>✅ Components Downloaded Successfully</h4>
    <ul>
        <li><strong>Model:</strong> {model_name}</li>
        <li><strong>Components:</strong> {", ".join(missing_components)}</li>
        <li><strong>Total Size:</strong> {self._format_bytes(total_size)}</li>
        <li><strong>Download Time:</strong> {time_str}</li>
        <li><strong>Average Speed:</strong> {self._format_bytes(avg_speed)}/s</li>
        <li><strong>Location:</strong> {abs_path}</li>
    </ul>
    <p>Model is ready to use!</p>
</div>
""", *self.get_model_selection_data()

            self.download_in_progress = False
            self.current_download_model = None

        except Exception as e:
            logger.error(f"Error downloading missing components: {str(e)}")
            yield f"""
<div class="error-box">
    <h4>❌ Download Failed</h4>
    <p><strong>Model:</strong> {model_name}</p>
    <p><strong>Error:</strong> {str(e)}</p>
    <p>Please try again or download the complete model.</p>
</div>
""", *self.get_model_selection_data()

            self.download_in_progress = False
            self.current_download_model = None

    def stop_download(self):
        """Stop the current download"""
        if self.download_in_progress:
            self.stop_download_flag = True
            return """
<div class="warning-box">
    <h4>⚠️ Stopping Download</h4>
    <p>Stopping download of {self.current_download_model}...</p>
    <p>This may take a moment to complete.</p>
</div>
"""
        else:
            return """
<div class="info-box">
    <h4>ℹ️ No Download in Progress</h4>
    <p>There is no download currently in progress.</p>
</div>
"""

    def download_model(self, model_type, model_name, use_hf_token, force_redownload, progress):
        """Download model with detailed progress tracking

        This method includes several reliability features for large downloads:
        1. Automatic retry mechanism with exponential backoff for network interruptions
        2. Resume capability for interrupted downloads
        3. Extended timeouts for slow connections
        4. Detailed progress tracking and user feedback
        5. Network error detection and recovery

        Args:
            model_type: Type of model to download ("image" or "3d")
            model_name: Name of the model to download
            use_hf_token: Whether to use the Hugging Face token for gated models
            force_redownload: Whether to force a fresh download even if the model exists
            progress: Gradio progress object for UI updates
        """
        # Initialize progress_info at the beginning to avoid UnboundLocalError
        progress_info = None

        try:
            # Check if another download is already in progress
            # Reset the flag if it's incorrectly set (e.g., after app restart)
            download_active = False

            # Check if download thread exists and is alive
            if hasattr(self, '_download_thread'):
                if hasattr(self._download_thread, 'is_alive') and self._download_thread.is_alive():
                    download_active = True
                else:
                    # Thread exists but is not alive or doesn't have is_alive method
                    self.download_in_progress = False
                    self.current_download_model = None
            else:
                # No download thread, reset flags
                self.download_in_progress = False
                self.current_download_model = None

            # Only consider a download in progress if the flag is set AND the thread is active
            if self.download_in_progress and download_active:
                yield f"""
<div class="error-box">
    <h4>⚠️ Download Already in Progress</h4>
    <p><strong>Model:</strong> {self.current_download_model}</p>
    <p>Please wait for it to complete or stop it before starting a new download.</p>
</div>
"""
                return # Exit if download is already active


            # Reset stop flag and set download in progress
            self.stop_download_flag = False
            self.download_in_progress = True
            self.current_download_model = model_name

            from huggingface_hub import HfApi, hf_hub_download, snapshot_download
            from huggingface_hub.constants import DEFAULT_REVISION
            import shutil
            import re
            import io
            import sys
            from threading import Lock

            # Get model config
            if model_type == "image":
                if model_name in IMAGE_MODELS:
                    config = IMAGE_MODELS[model_name]
                    is_gated = False
                elif model_name in GATED_IMAGE_MODELS:
                    config = GATED_IMAGE_MODELS[model_name]
                    is_gated = True
                else:
                    yield f"❌ Unknown model: {model_name}", *self.get_model_selection_data()

                repo_id = config.repo_id
                size = config.size
            else:
                config = HUNYUAN3D_MODELS[model_name]
                repo_id = config["repo_id"]
                size = config["size"]
                is_gated = False

            save_path = self.models_dir / model_type / model_name
            save_path.mkdir(parents=True, exist_ok=True)

            # Show absolute path for clarity
            abs_path = save_path.absolute()

            # Check if already completely downloaded
            if not force_redownload and self.check_model_complete(save_path, model_type, model_name):
                yield f"""
<div class="success-box">
    <h4>✅ Model Already Downloaded</h4>
    <p><strong>Model:</strong> {model_name}</p>
    <p><strong>Location:</strong> {abs_path}</p>
    <p>Ready to use!</p>
    <p style="font-size: 0.9em; color: #666;">If you want to re-download, use the force re-download option.</p>
</div>
""", *self.get_model_selection_data()
                return

            # Check for partial download
            if save_path.exists() and any(save_path.iterdir()):
                if not force_redownload:
                    yield f"""
<div class="warning-box" style="background-color: #fff3cd; color: #664d03; border: 1px solid #ffeaa7; padding: 1em; border-radius: 5px;">
    <h4>⚠️ Partial Download Detected</h4>
    <p><strong>Model:</strong> {model_name}</p>
    <p><strong>Location:</strong> {abs_path}</p>
    <p>This model appears to be partially downloaded. Would you like to:</p>
    <ol>
        <li>Resume the download (automatic)</li>
        <li>Force a fresh download (check the "Force re-download" option)</li>
    </ol>
</div>
""", *self.get_model_selection_data()
                    return
                else:
                    # Clean up partial download
                    progress(0.0, desc=f"Cleaning up partial download of {model_name}...")
                    shutil.rmtree(save_path)
                    save_path.mkdir(parents=True, exist_ok=True)

            # Check if gated model and token is needed
            if is_gated and not self.hf_token:
                yield f"""
<div class="error-box">
    <h4>🔒 Gated Model - Authentication Required</h4>
    <p><strong>Model:</strong> {model_name}</p>
    <p>This model requires Hugging Face authentication.</p>

    <h5>To download this model:</h5>
    <ol>
        <li>Go to <a href="https://huggingface.co/{repo_id}" target="_blank">the model page</a></li>
        <li>Click "Access repository" and accept the license</li>
        <li>Get your token from <a href="https://huggingface.co/settings/tokens" target="_blank">HF Settings</a></li>
        <li>Enter your token in the "HF Token" field above and try again</li>
    </ol>
</div>
""", *self.get_model_selection_data()
                return

            progress(0.0, desc=f"Preparing to download {model_name} to {abs_path}...")

            # Configure download parameters with minimal settings to avoid compatibility issues
            download_kwargs = {
                "repo_id": repo_id,
                "cache_dir": str(save_path),  # Use cache_dir to download directly to our models directory
                "force_download": force_redownload,  # Force fresh download if requested
                "max_workers": 8, # Increase concurrent connections for faster downloads
            }

            # Only add resume_download if not forcing redownload
            if not force_redownload:
                download_kwargs["resume_download"] = True

            if self.hf_token:
                download_kwargs["token"] = self.hf_token

            # Download with progress updates
            download_complete = False
            download_error = None
            start_time = time.time()

            # Create a shared data structure for progress information
            class ProgressInfo:
                def __init__(self):
                    self.total_files = 0
                    self.downloaded_files = 0
                    self.percentage = 0
                    self.elapsed_time = ""
                    self.remaining_time = ""
                    self.speed = ""
                    self.active_downloads = {}  # filename -> {progress, size, speed}
                    self.lock = Lock()
                    self.current_file = "" # To store the name of the file currently being downloaded

                def update(self, total_files=None, downloaded_files=None, percentage=None,
                           elapsed_time=None, remaining_time=None, speed=None):
                    with self.lock:
                        if total_files is not None:
                            self.total_files = total_files
                        if downloaded_files is not None:
                            self.downloaded_files = downloaded_files
                        if percentage is not None:
                            self.percentage = percentage
                        if elapsed_time is not None:
                            self.elapsed_time = elapsed_time
                        if remaining_time is not None:
                            self.remaining_time = remaining_time
                        if speed is not None:
                            self.speed = speed

                def update_file(self, filename, progress, size, speed):
                    with self.lock:
                        filename = filename.strip().replace("Downloading", "").replace(":", "").strip()
                        self.current_file = filename # Update current file
                        if len(filename) > 50:
                            filename = "..." + filename[-47:]
                        if progress == 100 and filename in self.active_downloads:
                            # Delay removal to prevent flickering
                            self.active_downloads[filename] = {"progress": 100, "size": size, "speed": "Done"}
                        elif progress < 100:
                            self.active_downloads[filename] = {"progress": progress, "size": size, "speed": speed}

                def set_current_task(self, task):
                    with self.lock:
                        # Use a special key for generic tasks
                        self.active_downloads["_task_"] = {"task": task}

            progress_info = ProgressInfo() # Initialize at the very beginning

            # Capture stdout and stderr to monitor download progress
            class StdCapture:
                def __init__(self, progress_info):
                    self.progress_info = progress_info
                    self.old_stdout = sys.stdout
                    self.old_stderr = sys.stderr
                    self.buffer = ""

                def write(self, text):
                    # self.old_stream.write(text) # Suppress original output
                    self.buffer += text
                    # Process buffer more aggressively - even without newline/carriage return
                    # This ensures more frequent updates for progress bars
                    if len(self.buffer) > 0:
                        # If we have newlines or carriage returns, process normally
                        if '\r' in self.buffer or '\n' in self.buffer:
                            lines = re.split(r'\r|\n', self.buffer)
                            self.buffer = lines[-1]
                            for line in lines[:-1]:
                                if line.strip():
                                    self.parse_progress(line)
                        # If buffer is getting large without newlines, try to parse it anyway
                        elif len(self.buffer) > 10:
                            if self.buffer.strip():
                                self.parse_progress(self.buffer)
                                self.buffer = ""

                def flush(self,):
                    # Process any remaining buffer content when flushing
                    if self.buffer.strip():
                        self.parse_progress(self.buffer)
                        self.buffer = ""

                    self.old_stdout.flush()
                    self.old_stderr.flush()

                def parse_progress(self, text):
                    # Overall progress: "Fetching 29 files: 14%|█▍ | 4/29 [06:20<54:46, 131.47s/it]"
                    fetch_match = re.search(r"Fetching.*?(\d+)%.*?(\d+)/(\d+)\s*\[([^^]+)<([^,]+),\s*([^^]+)\]", text)
                    if fetch_match:
                        percentage, downloaded, total, elapsed, remaining, speed = fetch_match.groups()
                        self.progress_info.update(
                            total_files=int(total),
                            percentage=int(percentage),
                            downloaded_files=int(downloaded),
                            elapsed_time=elapsed,
                            remaining_time=remaining,
                            speed=speed
                        )
                        return

                    # Alternative overall progress format: "Overall: 14% | Files: 4/29 | Time: 05:23 < 36:25"
                    overall_match = re.search(r"Overall:\s*(\d+)%\s*\|\s*Files:\s*(\d+)/(\d+)\s*\|\s*Time:\s*([^\s<]+)\s*<\s*([^\s]+)", text)
                    if overall_match:
                        percentage, downloaded, total, elapsed, remaining = overall_match.groups()
                        self.progress_info.update(
                            total_files=int(total),
                            percentage=int(percentage),
                            downloaded_files=int(downloaded),
                            elapsed_time=elapsed,
                            remaining_time=remaining
                        )
                        return

                    # Individual file progress: "pytorch_model-00001-of-00003.safetensors:  52%|#| 5.19G/9.95G [23:26<18:23, 4.31MB/s]"
                    file_match = re.search(r"(.+?):\s*(\d+)%\s*\|.*?\|\s*([\d.]+[GMKB]/[\d.]+[GMKB])\s*\[.*?\,\s*([^^]+)\]", text)
                    if file_match:
                        filename, progress, size, speed = file_match.groups()
                        self.progress_info.update_file(filename, int(progress), size, speed)
                        return

                    # Alternative individual file progress format: "• model-00002-of-00002.safetensors [██░░░░░░░░] 23% | 1.03G/4.53G | 1.63MB/s"
                    alt_file_match = re.search(r"•\s+(.+?)\s*\[.+?\]\s*(\d+)%\s*\|\s*([\d.]+[GMKB]/[\d.]+[GMKB])\s*\|\s*([\d.]+[GMkB]+/s)", text)
                    if alt_file_match:
                        filename, progress, size, speed = alt_file_match.groups()
                        self.progress_info.update_file(filename, int(progress), size, speed)
                        return

                    # Completed file format: "• model.safetensors [██████████] 100% | 246M/246M | Done"
                    completed_file_match = re.search(r"•\s+(.+?)\s*\[.+?\]\s*(\d+)%\s*\|\s*([\d.]+[GMKB]/[\d.]+[GMKB])\s*\|\s*(Done)", text)
                    if completed_file_match:
                        filename, progress, size, status = completed_file_match.groups()
                        self.progress_info.update_file(filename, int(progress), size, status)
                        return

                    if "Downloading" in text or "HTTP Request" in text:
                        self.progress_info.set_current_task(text.strip())

            def download_thread():
                nonlocal download_complete, download_error
                try:
                    # Enable hf_transfer for faster downloads
                    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
                    
                    # Try to import hf_transfer
                    try:
                        import hf_transfer
                        logger.info("Using hf_transfer for accelerated downloads")
                    except ImportError:
                        logger.warning("hf_transfer not installed - downloads will be slower")
                        logger.warning("Install with: pip install hf_transfer")
                    
                    std_capture = StdCapture(progress_info)
                    old_stdout = sys.stdout
                    old_stderr = sys.stderr
                    sys.stdout = std_capture
                    sys.stderr = std_capture

                    # Suppress huggingface_hub logs during download
                    hf_logger = logging.getLogger("huggingface_hub")
                    original_hf_level = hf_logger.level
                    hf_logger.setLevel(logging.ERROR)

                    original_sleep = time.sleep
                    def patched_sleep(seconds):
                        # Check stop flag during sleep intervals
                        if self.stop_download_flag: raise Exception("Download stopped by user")
                        original_sleep(seconds)

                    time.sleep = patched_sleep

                    try:
                        # Add retry mechanism with exponential backoff
                        max_retries = 5
                        retry_count = 0
                        retry_delay = 5  # Start with 5 seconds delay

                        while retry_count < max_retries:
                            try:
                                # If this is a retry, log it and update UI
                                if retry_count > 0:
                                    retry_msg = f"Retry attempt {retry_count}/{max_retries} after waiting {retry_delay}s..."
                                    logger.info(retry_msg)

                                    # Update progress info with retry information
                                    progress_info.set_current_task(f"⚠️ {retry_msg}")

                                    # Add a special entry for the retry status
                                    with progress_info.lock:
                                        progress_info.active_downloads["_retry_"] = {
                                            "task": f"Network interruption detected. Automatically retrying ({retry_count}/{max_retries})...",
                                            "is_retry": True,
                                            "attempt": retry_count,
                                            "max_attempts": max_retries
                                        }

                                # Attempt the download
                                snapshot_download(**download_kwargs)
                                download_complete = True

                                # Clear retry status since download resumed successfully
                                with progress_info.lock:
                                    if "_retry_" in progress_info.active_downloads:
                                        del progress_info.active_downloads["_retry_"]

                                break  # Success, exit the retry loop

                            except Exception as e:
                                # Check if we should retry based on the error type
                                error_str = str(e).lower()

                                # Don't retry if user stopped the download
                                if self.stop_download_flag or "download stopped by user" in error_str:
                                    raise Exception("Download stopped by user")

                                # Don't retry for authentication/permission errors
                                if "401" in error_str or "403" in error_str or "authentication" in error_str:
                                    raise

                                # Identify network-related errors that are good candidates for retry
                                is_network_error = any(err in error_str for err in [
                                    "connection", "timeout", "timed out", "reset", 
                                    "network", "socket", "eof", "broken pipe",
                                    "ssl", "handshake", "unreachable", "refused",
                                    "http error", "server error", "502", "503", "504"
                                ])

                                # Log more specific error type for debugging
                                if is_network_error:
                                    logger.info(f"Network-related error detected: {error_str}")
                                    # Continue to retry for network errors
                                else:
                                    logger.warning(f"Non-network error, may not benefit from retries: {error_str}")

                                # Don't retry if we've reached the max retries
                                if retry_count >= max_retries - 1:
                                    raise

                                # Log the error and retry
                                retry_count += 1
                                logger.warning(f"Download error (attempt {retry_count}/{max_retries}): {str(e)}")
                                logger.info(f"Waiting {retry_delay}s before retrying...")

                                # Wait with exponential backoff (5s, 10s, 20s, 40s, 80s)
                                original_sleep(retry_delay)
                                retry_delay *= 2  # Exponential backoff
                    finally:
                        time.sleep = original_sleep
                        sys.stdout = old_stdout
                        sys.stderr = old_stderr
                except Exception as e:
                    download_error = e

            # Start download in background
            progress(0.05, desc=f"Starting download of {model_name} ({size})...")
            self._download_thread = threading.Thread(target=download_thread)
            self._download_thread.start()

            # Update progress periodically while downloading
            while self._download_thread.is_alive():
                elapsed = time.time() - start_time

                if self.stop_download_flag:
                    download_error = Exception("Download stopped by user")
                    break

                with progress_info.lock:
                    total_files = progress_info.total_files
                    percentage = progress_info.percentage
                    downloaded_files = progress_info.downloaded_files
                    elapsed_time = progress_info.elapsed_time
                    remaining_time = progress_info.remaining_time
                    speed = progress_info.speed
                    active_downloads = dict(progress_info.active_downloads)

                progress_value = percentage / 100.0 if total_files > 0 else min(0.95, elapsed / 300.0)

                # --- Build Description String ---
                # Create a self-contained HTML progress bar
                progress_bar_html = f"""
                <div style="margin-bottom: 8px; font-family: sans-serif;">
                    <div style="font-weight: bold; margin-bottom: 4px;">Overall Progress</div>
                    <div style="background-color: #e0e0e0; border-radius: 5px; padding: 2px;">
                        <div style="width: {percentage}%; height: 20px; background: linear-gradient(to right, #64b5f6, #2196f3); border-radius: 4px; text-align: center; color: white; line-height: 20px; font-weight: bold;">
                            {percentage}%
                        </div>
                    </div>
                </div>
                """

                desc_html = f"<div><b>Downloading {model_name}...</b></div>"
                details = []

                if total_files > 0:
                    details.append(f"<b>Overall:</b> {percentage}% | <b>Files:</b> {downloaded_files}/{total_files} | <b>Time:</b> {elapsed_time} &lt; {remaining_time}")
                else:
                    details.append(f"<b>Time Elapsed:</b> {elapsed:.1f}s")

                if active_downloads:
                    details.append("<div style='margin-top: 5px; border-top: 1px solid #eee; padding-top: 5px;'><b>Active Downloads:</b></div>")
                    sorted_downloads = sorted(active_downloads.items(), key=lambda item: item[1].get('progress', 0) == 100)

                    for filename, data in sorted_downloads:
                        if "task" in data:
                            # Special styling for retry information
                            if filename == "_retry_" and data.get("is_retry", False):
                                # Create a more prominent retry status message
                                attempt = data.get("attempt", 0)
                                max_attempts = data.get("max_attempts", 5)
                                retry_progress = (attempt / max_attempts) * 100
                                retry_bar = "█" * (int(retry_progress) // 10) + "░" * (10 - int(retry_progress) // 10)

                                details.append(
                                    f"<div style='margin: 8px 0; padding: 8px; background-color: #fff3cd; border-left: 4px solid #ffc107; border-radius: 4px; color: #664d03;'>"
                                    f"<div style='font-weight: bold; color: #856404;'>⚠️ Automatic Recovery in Progress</div>"
                                    f"<div style='color: #664d03;'>{data['task']}</div>"
                                    f"<div style='margin-top: 5px; font-family: monospace; color: #664d03;'>"
                                    f"[{retry_bar}] Attempt {attempt}/{max_attempts}"
                                    f"</div>"
                                    f"<div style='font-size: 0.9em; margin-top: 5px; color: #664d03;'>The download will continue automatically. Please wait...</div>"
                                    f"</div>"
                                )
                            else:
                                details.append(f"<div style='margin-left: 10px;'>&bull; {data['task']}</div>")
                        else:
                            prog = data.get('progress', 0)
                            progress_bar = "█" * (prog // 10) + "░" * (10 - prog // 10)
                            color = "#66bb6a" if prog == 100 else "#2196f3"
                            details.append(
                                f"<div style='margin-left: 10px; font-family: monospace; font-size: 0.85em; color: {color};'>"
                                f"&bull; {filename}<br>"
                                f"&nbsp;&nbsp;[{progress_bar}] {prog}% | {data.get('size', '')} | {data.get('speed', '')}"
                                f"</div>"
                            )

                full_desc = progress_bar_html + desc_html + "".join(details)
                yield full_desc, *self.get_model_selection_data()
                time.sleep(1)

            # Clean up thread
            self._download_thread.join(timeout=5)  # Wait up to 5 seconds for thread to finish

            # Reset download flags
            self.download_in_progress = False
            self.current_download_model = None

            if download_error:
                if "Download stopped by user" in str(download_error):
                    # Create a more detailed stopped download message
                    download_info = []

                    # Add information about how far the download got
                    if hasattr(progress_info, 'downloaded_files') and progress_info.downloaded_files > 0:
                        download_info.append(f"<strong>Files Downloaded:</strong> {progress_info.downloaded_files}")

                    if hasattr(progress_info, 'total_files') and progress_info.total_files > 0:
                        download_info.append(f"<strong>Total Files Expected:</strong> {progress_info.total_files}")

                        # Calculate percentage if we have both values
                        if hasattr(progress_info, 'downloaded_files') and progress_info.downloaded_files > 0:
                            percentage = (progress_info.downloaded_files / progress_info.total_files) * 100
                            download_info.append(f"<strong>Download Progress:</strong> {percentage:.1f}%")
                    elif hasattr(progress_info, 'percentage') and progress_info.percentage > 0:
                        download_info.append(f"<strong>Download Progress:</strong> {progress_info.percentage}%")

                    # Calculate elapsed time
                    elapsed = time.time() - start_time
                    minutes, seconds = divmod(elapsed, 60)
                    hours, minutes = divmod(minutes, 60)

                    time_str = ""
                    if hours > 0:
                        time_str += f"{int(hours)}h "
                    if minutes > 0 or hours > 0:
                        time_str += f"{int(minutes)}m "
                    time_str += f"{seconds:.1f}s"

                    download_info.append(f"<strong>Time Spent:</strong> {time_str}")

                    # Add information about the last file being downloaded
                    if hasattr(progress_info, 'current_file') and progress_info.current_file:
                        download_info.append(f"<strong>Last File:</strong> {progress_info.current_file}")

                    yield f"""
<div class="warning-box">
    <h4>⚠️ Download Stopped</h4>
    <p><strong>Model:</strong> {model_name}</p>
    <p><strong>Repository:</strong> {repo_id}</p>
    <p><strong>Location:</strong> {abs_path}</p>

    <h5>Download Status When Stopped:</h5>
    <ul>
        {" ".join(f"<li>{info}</li>" for info in download_info)}
    </ul>

    <h5>What Happens Now:</h5>
    <ul>
        <li>Partially downloaded files remain in the destination folder</li>
        <li>You can resume the download by clicking the download button again</li>
        <li>If you want to start fresh, use the "Force re-download" option</li>
    </ul>
</div>
""", *self.get_model_selection_data()
                elif "401" in str(download_error) or "403" in str(download_error):
                    yield f"""
<div class="error-box">
    <h4>🔒 Authentication Failed</h4>
    <p><strong>Model:</strong> {model_name}</p>
    <p><strong>Error:</strong> Access denied</p>
</div>
""", *self.get_model_selection_data()
                else:
                    raise download_error

            # Verify download completeness
            progress(0.95, desc="Verifying download...")

            if not self.check_model_complete(save_path, model_type, model_name):
                # Create a more detailed incomplete download message
                # Check what files are missing
                missing_files = []

                if model_type == "image":
                    # Check for essential files for image models
                    essential_files = [
                        "model_index.json",
                        "scheduler/scheduler_config.json",
                        "unet/config.json",
                        "text_encoder/config.json",
                        "tokenizer/tokenizer_config.json",
                        "vae/config.json"
                    ]

                    for file in essential_files:
                        if not (save_path / file).exists():
                            missing_files.append(file)

                    # Check for model weights
                    if not any((save_path / "unet").glob("*.safetensors")) and not any((save_path / "unet").glob("*.bin")):
                        missing_files.append("unet weights (*.safetensors or *.bin)")

                    if not any((save_path / "vae").glob("*.safetensors")) and not any((save_path / "vae").glob("*.bin")):
                        missing_files.append("vae weights (*.safetensors or *.bin)")

                    if not any((save_path / "text_encoder").glob("*.safetensors")) and not any((save_path / "text_encoder").glob("*.bin")):
                        missing_files.append("text_encoder weights (*.safetensors or *.bin)")

                else:  # 3D models
                    # Check for essential files for 3D models
                    essential_files = [
                        "config.json",
                        "preprocessor_config.json"
                    ]

                    for file in essential_files:
                        if not (save_path / file).exists():
                            missing_files.append(file)

                    # Check for model weights
                    if not any(save_path.glob("*.safetensors")) and not any(save_path.glob("*.bin")):
                        missing_files.append("model weights (*.safetensors or *.bin)")

                # Get information about the download progress
                download_info = []
                if hasattr(progress_info, 'downloaded_files') and progress_info.downloaded_files > 0:
                    download_info.append(f"<strong>Files Downloaded:</strong> {progress_info.downloaded_files}")

                if hasattr(progress_info, 'total_files') and progress_info.total_files > 0:
                    download_info.append(f"<strong>Total Files Expected:</strong> {progress_info.total_files}")

                if hasattr(progress_info, 'percentage') and progress_info.percentage > 0:
                    download_info.append(f"<strong>Download Progress:</strong> {progress_info.percentage}%")

                # Calculate elapsed time
                elapsed = time.time() - start_time
                minutes, seconds = divmod(elapsed, 60)
                hours, minutes = divmod(minutes, 60)

                time_str = ""
                if hours > 0:
                    time_str += f"{int(hours)}h "
                if minutes > 0 or hours > 0:
                    time_str += f"{int(minutes)}m "
                time_str += f"{seconds:.1f}s"

                download_info.append(f"<strong>Time Spent:</strong> {time_str}")

                yield f"""
<div class="error-box">
    <h4>❌ Download Incomplete</h4>
    <p><strong>Model:</strong> {model_name}</p>
    <p><strong>Repository:</strong> {repo_id}</p>
    <p><strong>Location:</strong> {abs_path}</p>

    <h5>Download Status:</h5>
    <ul>
        {" ".join(f"<li>{info}</li>" for info in download_info)}
    </ul>

    <h5>Missing Files:</h5>
    <ul>
        {" ".join(f"<li>{file}</li>" for file in missing_files) if missing_files else "<li>Unknown missing files</li>"}
    </ul>

    <h5>Troubleshooting:</h5>
    <ul>
        <li>Try using the "Force re-download" option to start fresh</li>
        <li>Check your internet connection stability</li>
        <li>Ensure you have enough disk space (check the model size above)</li>
        <li>If using a VPN, try disabling it or changing servers</li>
        <li>For gated models, verify your Hugging Face token is correct</li>
        <li>Check if your firewall or antivirus is blocking the download</li>
        <li>Try downloading at a different time (server might be busy)</li>
        <li>If on a metered connection, ensure you have enough data allowance</li>
        <li>Check the Hugging Face status page for any ongoing issues: <a href="https://status.huggingface.co/" target="_blank">status.huggingface.co</a></li>
    </ul>

    <h5>Technical Information:</h5>
    <ul>
        <li>Download location: {abs_path}</li>
        <li>Repository ID: {repo_id}</li>
        <li>If problems persist, try downloading directly from <a href="https://huggingface.co/{repo_id}/tree/main" target="_blank">huggingface.co/{repo_id}</a></li>
    </ul>
</div>
""", *self.get_model_selection_data()
                return

            # Calculate downloaded size
            total_size = sum(f.stat().st_size for f in save_path.rglob("*") if f.is_file())

            progress(1.0, desc="Download complete!")

            # Create a more detailed success message
            total_time = time.time() - start_time
            minutes, seconds = divmod(total_time, 60)
            hours, minutes = divmod(minutes, 60)

            time_str = ""
            if hours > 0:
                time_str += f"{int(hours)}h "
            if minutes > 0 or hours > 0:
                time_str += f"{int(minutes)}m "
            time_str += f"{seconds:.1f}s"

            # Calculate average download speed
            avg_speed = total_size / total_time if total_time > 0 else 0

            yield f"""
<div class="success-box">
    <h4>✅ Download Complete!</h4>
    <ul>
        <li><strong>Model:</strong> {model_name}</li>
        <li><strong>Repository:</strong> {repo_id}</li>
        <li><strong>Total Size:</strong> {self._format_bytes(total_size)}</li>
        <li><strong>Download Time:</strong> {time_str}</li>
        <li><strong>Average Speed:</strong> {self._format_bytes(avg_speed)}/s</li>
        <li><strong>Files Downloaded:</strong> {progress_info.downloaded_files if progress_info.downloaded_files > 0 else 'Unknown'}</li>
        <li><strong>Location:</strong> {abs_path}</li>
    </ul>
    <p>Model is ready to use!</p>
</div>
""", *self.get_model_selection_data()

        except Exception as e:
            # If the download thread is still alive, it means the download is ongoing in the background.
            # In this case, we log the error but do not update the UI with a "failed" message prematurely.
            if hasattr(self, '_download_thread') and self._download_thread.is_alive():
                logger.error(f"Caught exception during download, but thread is still alive: {str(e)}")
                # Do not yield an error message to the UI, let the download continue in the background.
                return

            # If the thread is not alive, or doesn't exist, then it's a genuine failure or an error
            # that occurred before the thread started/completed.
            # Reset download flags
            self.download_in_progress = False
            self.current_download_model = None

            logger.error(f"Download error: {str(e)}")
            # Create a more detailed error message
            error_message = str(e)

            # Extract more information from the error if possible
            error_details = []
            if "Connection" in error_message or "Timeout" in error_message:
                error_details.append("Network connection issue detected. Check your internet connection.")
            if "disk space" in error_message.lower():
                error_details.append("You may be running out of disk space.")
            if "permission" in error_message.lower():
                error_details.append("File permission issue detected. Check if you have write access to the destination folder.")
            if "token" in error_message.lower() or "auth" in error_message.lower():
                error_details.append("Authentication issue detected. Check your Hugging Face token.")

            # If no specific details were added, add a generic message
            if not error_details:
                error_details.append("Please check your internet connection and try again.")

            # Add information about what was being downloaded
            download_info = [
                f"<strong>Model:</strong> {model_name}",
                f"<strong>Repository:</strong> {repo_id}",
                f"<strong>Destination:</strong> {abs_path}"
            ]

            # Add information about how far the download got
            if progress_info and hasattr(progress_info, 'downloaded_files') and progress_info.downloaded_files > 0:
                download_info.append(f"<strong>Progress:</strong> Downloaded {progress_info.downloaded_files} files before error")

            if progress_info and hasattr(progress_info, 'percentage') and progress_info.percentage > 0:
                download_info.append(f"<strong>Completion:</strong> {progress_info.percentage}% complete before error")

            elapsed = time.time() - start_time
            minutes, seconds = divmod(elapsed, 60)
            hours, minutes = divmod(minutes, 60)

            time_str = ""
            if hours > 0:
                time_str += f"{int(hours)}h "
            if minutes > 0 or hours > 0:
                time_str += f"{int(minutes)}m "
            time_str += f"{seconds:.1f}s"

            download_info.append(f"<strong>Time Elapsed:</strong> {time_str} before error")

            yield f"""
<div class="error-box">
    <h4>❌ Download Failed</h4>
    <p><strong>Error:</strong> {error_message}</p>

    <h5>Download Information:</h5>
    <ul>
        {" ".join(f"<li>{info}</li>" for info in download_info)}
    </ul>

    <h5>Troubleshooting:</h5>
    <ul>
        {" ".join(f"<li>{detail}</li>" for detail in error_details)}
    </ul>

    <p>If the problem persists, try using the "Force re-download" option or check the terminal for more detailed error messages.</p>
</div>
""", *self.get_model_selection_data()


    def _format_bytes(self, bytes):
        """Format bytes to human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes < 1024.0:
                return f"{bytes:.2f} {unit}"
            bytes /= 1024.0
        return f"{bytes:.2f} TB"

    def _get_memory_usage(self):
        """Get current memory usage information"""
        memory_info = {}

        # System RAM usage
        memory_info['ram_used'] = psutil.virtual_memory().used
        memory_info['ram_total'] = psutil.virtual_memory().total
        memory_info['ram_percent'] = psutil.virtual_memory().percent
        memory_info['ram_available'] = psutil.virtual_memory().available

        # GPU VRAM usage if available
        if torch.cuda.is_available():
            memory_info['vram_used'] = torch.cuda.memory_allocated()
            memory_info['vram_reserved'] = torch.cuda.memory_reserved()
            memory_info['vram_total'] = torch.cuda.get_device_properties(0).total_memory
            memory_info['vram_percent'] = (memory_info['vram_used'] / memory_info['vram_total']) * 100
            memory_info['vram_available'] = memory_info['vram_total'] - memory_info['vram_reserved']

        return memory_info

    def _check_memory_availability(self, required_ram_gb=8, required_vram_gb=0):
        """Check if there's enough memory available for loading models

        Args:
            required_ram_gb: Minimum RAM required in GB
            required_vram_gb: Minimum VRAM required in GB

        Returns:
            tuple: (bool, str) - (is_available, message)
        """
        memory_info = self._get_memory_usage()
        required_ram_bytes = required_ram_gb * 1024 * 1024 * 1024
        required_vram_bytes = required_vram_gb * 1024 * 1024 * 1024

        # Check RAM availability
        if memory_info['ram_available'] < required_ram_bytes:
            return False, f"Not enough RAM available. Need {required_ram_gb}GB, but only {self._format_bytes(memory_info['ram_available'])} available."

        # Check VRAM availability if required and available
        if required_vram_gb > 0 and 'vram_available' in memory_info and memory_info['vram_available'] < required_vram_bytes:
            return False, f"Not enough VRAM available. Need {required_vram_gb}GB, but only {self._format_bytes(memory_info['vram_available'])} available."

        return True, "Sufficient memory available"

    def _free_memory(self):
        """Attempt to free up memory by clearing caches and running garbage collection"""
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Run garbage collection
        gc.collect()

        # Log memory after cleanup
        self._log_memory_usage("after memory cleanup")

    def _handle_low_memory_during_loading(self, progress=None):
        """Handle low memory conditions during model loading

        Args:
            progress: Optional progress bar to update

        Returns:
            bool: True if memory was successfully freed, False otherwise
        """
        memory_info = self._get_memory_usage()

        # Check if we're running low on memory (>90% RAM usage)
        if memory_info['ram_percent'] > 90:
            if progress:
                progress(progress.value, desc="Low memory detected, cleaning up...")

            logger.warning(f"Low memory detected during loading: {memory_info['ram_percent']:.1f}% RAM used")

            # Perform aggressive memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Run garbage collection multiple times
            for _ in range(3):
                gc.collect()

            # Check if cleanup was successful
            new_memory_info = self._get_memory_usage()
            if new_memory_info['ram_percent'] < 85:
                logger.info(f"Memory cleanup successful: {new_memory_info['ram_percent']:.1f}% RAM used")
                return True
            else:
                logger.warning(f"Memory cleanup had limited effect: {new_memory_info['ram_percent']:.1f}% RAM used")
                return False

        return True  # Memory is fine

    def download_gguf_model(self, model_name, force_redownload, progress):
        """Download GGUF model with streaming progress updates - generator version"""
        from .config import GGUF_IMAGE_MODELS
        
        if model_name not in GGUF_IMAGE_MODELS:
            yield f"❌ Unknown GGUF model: {model_name}", *self.get_model_selection_data()
            return
            
        config = GGUF_IMAGE_MODELS[model_name]
        model_path = self.models_dir / "gguf" / model_name
        
        try:
            # Check if download is already in progress
            if self.download_in_progress:
                yield f"""
<div class="error-box">
    <h4>⚠️ Download Already in Progress</h4>
    <p><strong>Model:</strong> {self.current_download_model}</p>
    <p>Please wait for it to complete or stop it before starting a new download.</p>
</div>
""", *self.get_model_selection_data()
                return
            
            # Set download flags
            self.download_in_progress = True
            self.current_download_model = f"{model_name} (GGUF)"
            self.stop_download_flag = False
            
            # Progress tracking
            if progress:
                progress(0.01, desc=f"Starting download of {model_name}...")
            
            yield f"""
<div class="info-box">
    <h4>📥 Downloading GGUF Model</h4>
    <p><strong>Model:</strong> {config.name}</p>
    <p><strong>Size:</strong> {config.size}</p>
    <p><strong>Description:</strong> {config.description}</p>
</div>
"""
            
            # Instead of calling _load_gguf_model, we'll inline the download logic here
            # to provide better progress updates
            from .config import FLUX_COMPONENTS
            from huggingface_hub import hf_hub_download
            
            # Create a simple progress info class for GGUF downloads
            class GGUFProgressInfo:
                def __init__(self):
                    self.current_task = ""
                    self.lock = threading.Lock()
                
                def set_current_task(self, task):
                    with self.lock:
                        self.current_task = task
            
            # Create directories for GGUF components
            gguf_dir = self.models_dir / "gguf" / model_name
            gguf_dir.mkdir(parents=True, exist_ok=True)
            
            components_to_download = []
            total_size = 0
            
            # Check what needs to be downloaded
            gguf_file_path = gguf_dir / config.gguf_file
            logger.info(f"Checking GGUF file: {gguf_file_path}")
            logger.info(f"GGUF file exists: {gguf_file_path.exists()}")
            
            if force_redownload or not gguf_file_path.exists():
                components_to_download.append(("GGUF Model", config.gguf_file, gguf_dir, config.repo_id))
                logger.info(f"Will download GGUF model: {config.gguf_file}")
            
            # For FLUX models, inform user about base component requirements
            if "FLUX" in model_name:
                yield f"""
<div class="info-box">
    <h4>ℹ️ Note about FLUX Components</h4>
    <p>GGUF models require FLUX base components (VAE, text encoders) for generation.</p>
    <p>These will be downloaded automatically when you first generate an image.</p>
    <p style="font-size: 0.9em; color: #666;">This approach avoids large upfront downloads and only gets what's needed.</p>
</div>
"""
            
            # Skip the base model pre-download to avoid hanging issues
            if False:  # Disabled due to download hanging issues
                    yield f"""
<div class="info-box">
    <h4>📥 Downloading FLUX Base Model</h4>
    <p>This includes VAE, text encoders, and other components needed for generation.</p>
    <p>This is a one-time download (~5GB total).</p>
</div>
"""
                    try:
                        # Use snapshot_download to get the full model
                        from huggingface_hub import snapshot_download
                        import os
                        
                        # Set environment variables for faster downloads
                        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
                        os.environ["HF_TRANSFER_CONCURRENCY"] = "8"  # Parallel connections
                        
                        # Progress tracking for snapshot download
                        progress_info = GGUFProgressInfo()
                        total_files = 25  # Approximate for FLUX
                        
                        # Create a progress callback
                        def download_callback(progress_data):
                            if isinstance(progress_data, dict) and "downloaded" in progress_data:
                                downloaded = progress_data.get("downloaded", 0)
                                total = progress_data.get("total", 1)
                                percent = (downloaded / total) * 100 if total > 0 else 0
                                progress_info.set_current_task(f"Downloading: {percent:.1f}%")
                        
                        # Track progress in a thread-safe way
                        progress_data = {"current": 0, "total": 25, "percent": 0, "current_file": "", "file_percent": 0}
                        progress_lock = threading.Lock()
                        
                        # Capture output to show progress
                        old_stdout = sys.stdout
                        old_stderr = sys.stderr
                        
                        class DownloadProgressCapture:
                            def __init__(self, progress_callback, progress_data, progress_lock):
                                self.progress_callback = progress_callback
                                self.progress_data = progress_data
                                self.progress_lock = progress_lock
                                self.last_update = time.time()
                                self.buffer = ""
                                
                            def write(self, text):
                                self.buffer += text
                                current_time = time.time()
                                
                                # Process buffer when we have complete lines or every 0.2 seconds
                                if '\n' in self.buffer or '\r' in self.buffer or current_time - self.last_update > 0.2:
                                    lines = self.buffer.split('\n')
                                    self.buffer = lines[-1]  # Keep incomplete line
                                    
                                    for line in lines[:-1]:
                                        if "Fetching" in line and "files:" in line:
                                            # Parse: "Fetching 25 files: 4%|▍ | 1/25"
                                            import re
                                            match = re.search(r'Fetching (\d+) files:\s*(\d+)%.*?(\d+)/(\d+)', line)
                                            if match:
                                                with self.progress_lock:
                                                    self.progress_data["total"] = int(match.group(1))
                                                    self.progress_data["percent"] = int(match.group(2))
                                                    self.progress_data["current"] = int(match.group(3))
                                                
                                                percent = self.progress_data["percent"]
                                                current = self.progress_data["current"]
                                                total = self.progress_data["total"]
                                                
                                                desc = f"Downloading FLUX base model: {current}/{total} files ({percent}%)"
                                                # Only update progress if callback is available
                                                if hasattr(self.progress_callback, '__call__'):
                                                    self.progress_callback(0.1 + (percent / 100.0) * 0.8, desc=desc)
                                                self.last_update = current_time
                                        elif "Downloading" in line or "download" in line.lower():
                                            # Also capture individual file downloads
                                            # Example: "Downloading pytorch_model.safetensors: 100%|████████| 12.5G/12.5G [05:23<00:00, 38.7MB/s]"
                                            file_match = re.search(r'Downloading ([^:]+):\s*(\d+)%', line)
                                            if file_match:
                                                filename = file_match.group(1).strip()
                                                file_percent = int(file_match.group(2))
                                                
                                                # Update progress data with file-specific info
                                                with self.progress_lock:
                                                    self.progress_data["current_file"] = filename
                                                    self.progress_data["file_percent"] = file_percent
                                                    
                                                # Log for debugging
                                                if file_percent % 20 == 0:  # Log every 20%
                                                    logger.info(f"Downloading {filename}: {file_percent}%")
                            
                            def flush(self):
                                pass
                        
                        try:
                            capture = DownloadProgressCapture(progress, progress_data, progress_lock)
                            sys.stdout = capture
                            sys.stderr = capture
                            
                            # Try to use hf_transfer for faster downloads
                            try:
                                import hf_transfer
                                logger.info("Using hf_transfer for faster downloads")
                            except ImportError:
                                logger.warning("hf_transfer not installed. Install with: pip install hf_transfer")
                                logger.warning("Downloads may be slower without hf_transfer")
                                yield f"""
<div class="warning-box">
    <h4>💡 Tip: Speed up downloads</h4>
    <p>Install hf_transfer for faster downloads: <code>pip install hf_transfer</code></p>
</div>
"""
                            
                            # Start download with progress monitoring
                            download_exception = None
                            download_result = None
                            
                            def download_with_exception_handling():
                                nonlocal download_exception, download_result
                                try:
                                    logger.info(f"Starting download of {base_model_id} to {flux_cache_dir}")
                                    download_result = snapshot_download(
                                        repo_id=base_model_id,
                                        local_dir=str(flux_cache_dir),
                                        token=self.hf_token,
                                        ignore_patterns=["*.md", "*.txt", ".gitattributes"],
                                        resume_download=True,
                                        max_workers=4,  # Balance speed and stability
                                        local_files_only=False,
                                        force_download=False
                                    )
                                    logger.info(f"Download completed: {download_result}")
                                except Exception as e:
                                    logger.error(f"Download error: {str(e)}")
                                    download_exception = e
                            
                            download_thread = threading.Thread(target=download_with_exception_handling)
                            download_thread.daemon = True
                            download_thread.start()
                            
                            # Monitor progress and update UI
                            last_percent = -1
                            last_update_time = time.time()
                            start_time = time.time()
                            stuck_counter = 0
                            last_activity_time = time.time()
                            
                            while download_thread.is_alive():
                                # Check for exceptions
                                if download_exception:
                                    logger.error(f"Download failed with exception: {download_exception}")
                                    break
                                
                                with progress_lock:
                                    current = progress_data["current"]
                                    total = progress_data["total"]
                                    percent = progress_data["percent"]
                                
                                current_time = time.time()
                                elapsed = current_time - start_time
                                
                                # Track if progress is stuck
                                if percent != last_percent:
                                    last_activity_time = current_time
                                    stuck_counter = 0
                                else:
                                    stuck_counter += 1
                                
                                # If stuck for too long, provide different messaging
                                time_since_activity = current_time - last_activity_time
                                is_stuck = time_since_activity > 60  # Consider stuck after 60 seconds of no progress
                                
                                # Update more frequently - every percent change or every 1 second
                                if (percent != last_percent and total > 0) or (current_time - last_update_time > 1.0) or (elapsed < 5):
                                    # Also update the progress bar if available
                                    if progress:
                                        if total > 0 and percent > 0:
                                            progress_value = 0.1 + (percent / 100.0) * 0.8
                                            progress(progress_value, desc=f"Downloading FLUX base model: {current}/{total} files ({percent}%)")
                                        else:
                                            # Initial phase
                                            progress_value = min(0.1 + (elapsed / 60.0) * 0.1, 0.2)  # Slowly increase up to 20%
                                            if is_stuck:
                                                progress(progress_value, desc="Download appears slow - checking connection...")
                                            else:
                                                progress(progress_value, desc="Initializing FLUX base model download...")
                                    
                                    # Format elapsed time
                                    elapsed_str = f"{int(elapsed)}s"
                                    if elapsed > 60:
                                        minutes = int(elapsed / 60)
                                        seconds = int(elapsed % 60)
                                        elapsed_str = f"{minutes}m {seconds}s"
                                    
                                    # Get current file info if available
                                    current_file = progress_data.get("current_file", "")
                                    file_percent = progress_data.get("file_percent", 0)
                                    
                                    # Build status message
                                    if percent > 0:
                                        status_msg = f"<p><strong>Progress:</strong> {current}/{total} files ({percent}%)</p>"
                                    else:
                                        if is_stuck and elapsed > 300:  # 5 minutes
                                            status_msg = """<p style="color: #d32f2f;"><strong>⚠️ Download appears to be stuck</strong></p>
                                            <p style="font-size: 0.9em;">Possible issues:</p>
                                            <ul style="font-size: 0.9em;">
                                                <li>Network connectivity problems</li>
                                                <li>HuggingFace servers may be slow</li>
                                                <li>The model may require authentication</li>
                                            </ul>
                                            <p style="font-size: 0.9em;">You can cancel and try again later.</p>"""
                                        elif elapsed > 120:  # 2 minutes
                                            status_msg = "<p><strong>Status:</strong> Still preparing download... This is taking longer than usual.</p>"
                                        elif elapsed > 30:
                                            status_msg = "<p><strong>Status:</strong> Establishing connection to HuggingFace servers...</p>"
                                        else:
                                            status_msg = "<p><strong>Status:</strong> Connecting to HuggingFace servers...</p>"
                                    
                                    # Add current file info if available
                                    file_info = ""
                                    if current_file and file_percent > 0:
                                        file_info = f'<p style="font-size: 0.9em; color: #666;"><strong>Current file:</strong> {current_file} ({file_percent}%)</p>'
                                    
                                    # Skip FLUX download tip since the issue says hf_transfer is already installed
                                    hf_tip = ""
                                    if elapsed > 180 and percent == 0:  # 3 minutes with no progress
                                        hf_tip = """<div style="margin-top: 10px; padding: 10px; background-color: #fff3cd; border-radius: 5px;">
                                        <p style="color: #856404; margin: 0;"><strong>💡 Troubleshooting Tips:</strong></p>
                                        <ul style="color: #856404; margin: 5px 0 0 20px; font-size: 0.9em;">
                                            <li>Try canceling and downloading just the GGUF model first</li>
                                            <li>Check your internet connection</li>
                                            <li>FLUX base model is large (~5GB) and may take time</li>
                                            <li>Consider downloading during off-peak hours</li>
                                        </ul>
                                        </div>"""
                                    
                                    yield f"""
<div class="info-box">
    <h4>📥 Downloading FLUX Base Model</h4>
    {status_msg}
    <p><strong>Elapsed:</strong> {elapsed_str}</p>
    {file_info}
    <p>Download speed depends on your connection to HuggingFace servers.</p>
    <p>Size: ~5GB total</p>
    <div style="margin-top: 10px; background-color: #e0e0e0; border-radius: 5px; padding: 2px;">
        <div style="width: {percent if percent > 0 else 1}%; height: 20px; background: linear-gradient(to right, #64b5f6, #2196f3); border-radius: 4px; text-align: center; color: white; line-height: 20px; font-weight: bold;">
            {f"{percent}%" if percent > 0 else "..."}
        </div>
    </div>
    {hf_tip}
</div>
"""
                                    last_percent = percent
                                    last_update_time = current_time
                                
                                # Add timeout after 10 minutes of being stuck
                                if is_stuck and time_since_activity > 600:
                                    logger.error("Download timeout - no progress for 10 minutes")
                                    download_exception = TimeoutError("Download stuck for over 10 minutes")
                                    break
                                
                                time.sleep(0.2)  # Check more frequently
                            
                            download_thread.join()
                            
                        finally:
                            sys.stdout = old_stdout
                            sys.stderr = old_stderr
                        
                        yield f"""
<div class="success-box">
    <h4>✅ FLUX Base Components Downloaded</h4>
    <p>All required components are now available locally.</p>
</div>
"""
                    except Exception as e:
                        logger.error(f"Failed to download FLUX base components: {str(e)}")
                        yield f"""
<div class="warning-box">
    <h4>⚠️ FLUX Base Download Incomplete</h4>
    <p>Some components may download during generation.</p>
    <p>Error: {str(e)}</p>
</div>
"""
            
            logger.info(f"Components to download: {len(components_to_download)}")
            for comp in components_to_download:
                logger.info(f"  - {comp[0]}: {comp[1]}")
            
            if not components_to_download:
                yield f"""
<div class="success-box">
    <h4>✅ Model Already Downloaded</h4>
    <p>All components for {config.name} are already present.</p>
</div>
""", *self.get_model_selection_data()
                return
            
            # Set up progress tracking
            progress_info = GGUFProgressInfo()
            download_complete = False
            download_error = None
            current_component_idx = 0
            
            # Download thread for GGUF components
            def gguf_download_thread():
                nonlocal download_complete, download_error, current_component_idx
                
                # Capture stdout/stderr to monitor hf_hub_download progress
                import sys
                import re
                
                class ProgressCapture:
                    def __init__(self, progress_info, comp_name):
                        self.progress_info = progress_info
                        self.comp_name = comp_name
                        self.old_stdout = sys.stdout
                        self.old_stderr = sys.stderr
                        self.buffer = ""
                        
                    def write(self, text):
                        self.old_stdout.write(text)  # Still output to console
                        self.buffer += text
                        
                        # Process buffer for progress updates
                        if '\r' in self.buffer or '\n' in self.buffer:
                            lines = re.split(r'[\r\n]+', self.buffer)
                            self.buffer = lines[-1]
                            
                            for line in lines[:-1]:
                                # Parse download progress
                                # Example: "Downloading flux1-dev-Q8_0.gguf: 100%|████████| 12.5G/12.5G [05:23<00:00, 38.7MB/s]"
                                match = re.search(r'(\d+)%\|.*?\|\s*([\d.]+[GMKB])/([\d.]+[GMKB])\s*\[([^<]+)<([^,]+),\s*([^\]]+)\]', line)
                                if match:
                                    percent = match.group(1)
                                    downloaded = match.group(2)
                                    total = match.group(3)
                                    elapsed = match.group(4)
                                    remaining = match.group(5)
                                    speed = match.group(6)
                                    
                                    self.progress_info.set_current_task(
                                        f"{self.comp_name}: {percent}% - {downloaded}/{total} @ {speed} (ETA: {remaining})"
                                    )
                    
                    def flush(self):
                        self.old_stdout.flush()
                
                try:
                    # Download each component
                    for idx, (comp_name, filename, local_dir, repo_id) in enumerate(components_to_download):
                        current_component_idx = idx
                        
                        if self.stop_download_flag:
                            download_error = "Download stopped by user"
                            return
                        
                        # Update progress info
                        progress_info.set_current_task(f"Preparing to download {comp_name}")
                        
                        logger.info(f"Starting download: {comp_name} from {repo_id}/{filename}")
                        logger.info(f"Using token: {'Yes' if self.hf_token else 'No'}")
                        logger.info(f"Target directory: {local_dir}")
                        
                        try:
                            # Enable hf_transfer for faster downloads
                            import os
                            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
                            
                            # Try to use hf_transfer for faster downloads
                            try:
                                import hf_transfer
                                logger.info("Using hf_transfer for faster GGUF download")
                                progress_info.set_current_task(f"🚀 Using hf_transfer for accelerated download of {comp_name}")
                            except ImportError:
                                logger.warning("hf_transfer not available - downloads may be slower")
                                progress_info.set_current_task(f"⚠️ Downloading {comp_name} (install hf_transfer for faster speeds)")
                            
                            # Use hf_hub_download which automatically uses hf_transfer if available
                            from huggingface_hub import hf_hub_download
                            
                            # Create a progress callback for hf_hub_download
                            last_progress = 0
                            start_time = time.time()
                            
                            def download_progress_callback(progress_data):
                                nonlocal last_progress
                                if isinstance(progress_data, dict):
                                    downloaded = progress_data.get("downloaded", 0)
                                    total = progress_data.get("total", 1)
                                    if total > 0:
                                        progress_pct = int((downloaded / total) * 100)
                                        if progress_pct != last_progress:
                                            elapsed = time.time() - start_time
                                            speed = downloaded / elapsed if elapsed > 0 else 0
                                            speed_str = self._format_bytes(speed) + "/s"
                                            
                                            progress_info.set_current_task(
                                                f"Downloading {comp_name}: {progress_pct}% - {self._format_bytes(downloaded)}/{self._format_bytes(total)} @ {speed_str}"
                                            )
                                            last_progress = progress_pct
                                            
                                            # Log every 10%
                                            if progress_pct % 10 == 0:
                                                logger.info(f"GGUF download progress: {progress_pct}% @ {speed_str}")
                            
                            # Set up progress capture
                            capture = ProgressCapture(progress_info, comp_name)
                            old_stdout = sys.stdout
                            old_stderr = sys.stderr
                            
                            try:
                                sys.stdout = capture
                                sys.stderr = capture
                                
                                # Download using hf_hub_download with progress tracking
                                logger.info(f"Downloading {filename} from {repo_id}")
                                file_path = hf_hub_download(
                                    repo_id=repo_id,
                                    filename=filename,
                                    local_dir=str(local_dir),
                                    token=self.hf_token,
                                    resume_download=True,
                                    force_download=False,
                                    local_dir_use_symlinks=False
                                )
                                
                                logger.info(f"Download complete: {comp_name} saved at {file_path}")
                            finally:
                                sys.stdout = old_stdout
                                sys.stderr = old_stderr
                        except Exception as dl_error:
                            logger.error(f"Download failed for {comp_name}: {str(dl_error)}")
                            download_error = str(dl_error)
                            raise
                    
                    download_complete = True
                    
                except Exception as e:
                    download_error = str(e)
                    logger.error(f"GGUF download error: {download_error}")
            
            # Start download thread
            download_thread = threading.Thread(target=gguf_download_thread)
            download_thread.start()
            
            # Monitor progress and yield updates
            total_components = len(components_to_download)
            last_component_idx = -1
            
            # Initial progress at 0%
            if progress:
                progress(0.0, desc="Starting GGUF download...")
            
            # Show initial status
            yield f"""
<div class="info-box">
    <h4>📥 Preparing Download</h4>
    <p><strong>Total Components:</strong> {total_components}</p>
    <p><strong>Status:</strong> Initializing download...</p>
</div>
"""
            
            thread_check_count = 0
            last_status = ""
            while download_thread.is_alive() or (not download_complete and not download_error):
                thread_check_count += 1
                if thread_check_count == 1:
                    logger.info(f"Download thread alive: {download_thread.is_alive()}, complete: {download_complete}, error: {download_error}")
                
                if self.stop_download_flag:
                    yield f"""
<div class="warning-box">
    <h4>⚠️ Download Stopped</h4>
    <p>Download was cancelled by user.</p>
</div>
"""
                    break
                
                # Get current status from progress info
                current_status = progress_info.current_task
                
                # Update UI whenever status changes
                if current_status != last_status:
                    last_status = current_status
                    
                    # Bound check to prevent index errors
                    if current_component_idx < len(components_to_download):
                        comp_name = components_to_download[current_component_idx][0]
                        filename = components_to_download[current_component_idx][1]
                        
                        # Calculate overall progress based on component index and download progress
                        base_progress = current_component_idx / total_components
                        
                        # Extract percentage and details from status if available
                        component_progress = 0
                        download_details = ""
                        
                        if current_status and "%" in current_status:
                            try:
                                # Extract percentage from status like "GGUF Model: 45% - 5.6G/12.5G @ 38.7MB/s (ETA: 00:03:21)"
                                import re
                                match = re.search(r'(\d+)%', current_status)
                                if match:
                                    component_progress = float(match.group(1)) / 100.0
                                
                                # Extract download details
                                details_match = re.search(r'(\d+)%\s*-\s*([^@]+)@\s*([^(]+)(?:\(ETA:\s*([^)]+)\))?', current_status)
                                if details_match:
                                    size_info = details_match.group(2).strip()
                                    speed_info = details_match.group(3).strip()
                                    eta_info = details_match.group(4) if details_match.group(4) else "calculating..."
                                    download_details = f"""
                                    <p style="font-size: 0.9em; color: #666;">
                                        <strong>Size:</strong> {size_info} | 
                                        <strong>Speed:</strong> {speed_info} | 
                                        <strong>ETA:</strong> {eta_info}
                                    </p>"""
                            except Exception as e:
                                logger.debug(f"Error parsing progress: {e}")
                                component_progress = 0
                        
                        # Calculate total progress
                        progress_pct = base_progress + (component_progress / total_components)
                        
                        if progress:
                            progress(progress_pct, desc=current_status or f"Downloading {comp_name}")
                        
                        yield f"""
<div class="info-box">
    <h4>📥 Downloading GGUF Model - Component {current_component_idx + 1}/{total_components}</h4>
    <p><strong>Component:</strong> {comp_name}</p>
    <p><strong>File:</strong> {filename}</p>
    <p><strong>Status:</strong> {current_status or 'Preparing download...'}</p>
    {download_details}
    
    <div style="margin: 10px 0;">
        <div style="font-size: 0.9em; color: #666; margin-bottom: 4px;">Component Progress:</div>
        <div style="background-color: #e0e0e0; border-radius: 5px; padding: 2px;">
            <div style="width: {int(component_progress * 100)}%; height: 20px; background: linear-gradient(to right, #66bb6a, #4caf50); border-radius: 4px; text-align: center; color: white; line-height: 20px; font-weight: bold;">
                {int(component_progress * 100)}%
            </div>
        </div>
    </div>
    
    <div style="margin: 10px 0;">
        <div style="font-size: 0.9em; color: #666; margin-bottom: 4px;">Overall Progress:</div>
        <div style="background-color: #e0e0e0; border-radius: 5px; padding: 2px;">
            <div style="width: {int(progress_pct * 100)}%; height: 20px; background: linear-gradient(to right, #64b5f6, #2196f3); border-radius: 4px; text-align: center; color: white; line-height: 20px; font-weight: bold;">
                {int(progress_pct * 100)}%
            </div>
        </div>
    </div>
    
    {f'<p style="color: #666; font-size: 0.9em; margin-top: 10px;">💡 Using hf_transfer for accelerated downloads</p>' if 'hf_transfer' in sys.modules else ''}
</div>
"""
                
                # Small delay to prevent busy waiting
                time.sleep(0.1)  # Reduced delay for more responsive updates
            
            # Wait for thread to complete
            download_thread.join(timeout=5)
            
            logger.info(f"Download thread finished. Complete: {download_complete}, Error: {download_error}")
            
            # Check for errors
            if download_error:
                yield f"""
<div class="error-box">
    <h4>❌ Download Failed</h4>
    <p><strong>Error:</strong> {download_error}</p>
</div>
"""
                raise Exception(download_error)
            
            # Check if download actually completed
            if download_complete:
                # Success message
                if progress:
                    progress(1.0, desc="All components downloaded!")
                
                yield f"""
<div class="success-box">
    <h4>✅ GGUF Model Downloaded Successfully!</h4>
    <p><strong>Model:</strong> {config.name}</p>
    <p><strong>Location:</strong> {gguf_dir}</p>
    <p>All components have been downloaded and are ready to use.</p>
</div>
""", *self.get_model_selection_data()
            else:
                logger.warning("Download thread ended but download not marked complete")
                yield f"""
<div class="warning-box">
    <h4>⚠️ Download Status Unclear</h4>
    <p>The download process ended but completion status is uncertain.</p>
    <p>Please check the console logs for more details.</p>
</div>
""", *self.get_model_selection_data()
            
        except Exception as e:
            logger.error(f"Error in GGUF download: {str(e)}")
            yield f"""
<div class="error-box">
    <h4>❌ Download Failed</h4>
    <p><strong>Error:</strong> {str(e)}</p>
</div>
""", *self.get_model_selection_data()
        finally:
            # Reset download flags
            self.download_in_progress = False
            self.current_download_model = None

    def _load_gguf_model(self, model_path, config, device_to_use, dtype_to_use, progress):
        """Download and prepare GGUF model components."""
        
        try:
            from .config import FLUX_COMPONENTS
            
            # Create directories for GGUF components
            gguf_dir = self.models_dir / "gguf" / config.name
            gguf_dir.mkdir(parents=True, exist_ok=True)
            
            components_downloaded = 0
            total_components = 4  # GGUF file + VAE + 2 text encoders
            
            # 1. Download GGUF file
            if progress:
                progress(0.1, desc=f"Downloading GGUF file: {config.gguf_file}")
            else:
                logger.info(f"Downloading GGUF file: {config.gguf_file}")
            gguf_file_path = gguf_dir / config.gguf_file
            if not gguf_file_path.exists():
                from huggingface_hub import hf_hub_download
                hf_hub_download(
                    repo_id=config.repo_id,
                    filename=config.gguf_file,
                    local_dir=gguf_dir,
                    token=self.hf_token
                )
            components_downloaded += 1
            
            # 2. Download VAE
            if progress:
                progress(0.3, desc="Downloading FLUX VAE...")
            else:
                logger.info("Downloading FLUX VAE...")
            vae_dir = self.models_dir / "vae"
            vae_dir.mkdir(parents=True, exist_ok=True)
            vae_path = vae_dir / FLUX_COMPONENTS["vae"]["filename"]
            if not vae_path.exists():
                hf_hub_download(
                    repo_id=FLUX_COMPONENTS["vae"]["repo_id"],
                    filename=FLUX_COMPONENTS["vae"]["filename"],
                    local_dir=vae_dir,
                    token=self.hf_token
                )
            components_downloaded += 1
            
            # 3. Download CLIP text encoder
            if progress:
                progress(0.5, desc="Downloading CLIP text encoder...")
            else:
                logger.info("Downloading CLIP text encoder...")
            te_dir = self.models_dir / "text_encoders"
            te_dir.mkdir(parents=True, exist_ok=True)
            clip_path = te_dir / FLUX_COMPONENTS["text_encoder_clip"]["filename"]
            if not clip_path.exists():
                hf_hub_download(
                    repo_id=FLUX_COMPONENTS["text_encoder_clip"]["repo_id"],
                    filename=FLUX_COMPONENTS["text_encoder_clip"]["filename"],
                    local_dir=te_dir,
                    token=self.hf_token
                )
            components_downloaded += 1
            
            # 4. Download T5 text encoder
            if progress:
                progress(0.7, desc="Downloading T5 text encoder...")
            else:
                logger.info("Downloading T5 text encoder...")
            t5_path = te_dir / FLUX_COMPONENTS["text_encoder_t5"]["filename"]
            if not t5_path.exists():
                hf_hub_download(
                    repo_id=FLUX_COMPONENTS["text_encoder_t5"]["repo_id"],
                    filename=FLUX_COMPONENTS["text_encoder_t5"]["filename"],
                    local_dir=te_dir,
                    token=self.hf_token
                )
            components_downloaded += 1
            
            if progress:
                progress(1.0, desc="GGUF model components downloaded!")
            else:
                logger.info("GGUF model components downloaded!")
            
            # Return success message with file locations
            success_message = f"""
<div class="success-box">
    <h4>✅ GGUF Model Downloaded Successfully!</h4>
    <p><strong>Model:</strong> {config.name}</p>
    <p><strong>VRAM Required:</strong> {config.vram_required}</p>
    
    <h5>Downloaded Components:</h5>
    <ul>
        <li>✅ GGUF Model: <code>{gguf_file_path.name}</code></li>
        <li>✅ VAE: <code>{vae_path.name}</code></li>
        <li>✅ CLIP Encoder: <code>{clip_path.name}</code></li>
        <li>✅ T5 Encoder: <code>{t5_path.name}</code></li>
    </ul>
    
    <p><strong>Status:</strong> Ready to use with ComfyUI or manual pipeline setup</p>
    <p><strong>Expected Performance:</strong> 2-3x faster with 50-60% less VRAM usage</p>
    
    <p><em>Note: To use these files, you'll need ComfyUI with GGUF support or wait for full integration in a future update.</em></p>
</div>
"""
            
            logger.info(f"Successfully downloaded GGUF model {config.name}")
            return success_message, None, None
            
        except Exception as e:
            logger.error(f"Failed to download GGUF model: {str(e)}")
            error_message = f"""
<div class="error-box">
    <h4>❌ GGUF Download Failed</h4>
    <p><strong>Error:</strong> {str(e)}</p>
    <p>You can manually download from: <a href="https://huggingface.co/{config.repo_id}" target="_blank">{config.repo_id}</a></p>
</div>
"""
            return error_message, None, None

    def _log_memory_usage(self, stage=""):
        """Log current memory usage"""
        memory_info = self._get_memory_usage()

        log_message = f"Memory usage {stage}: "
        log_message += f"RAM: {self._format_bytes(memory_info['ram_used'])}/{self._format_bytes(memory_info['ram_total'])} ({memory_info['ram_percent']:.1f}%)"

        if 'vram_used' in memory_info:
            log_message += f", VRAM: {self._format_bytes(memory_info['vram_used'])}/{self._format_bytes(memory_info['vram_total'])} ({memory_info['vram_percent']:.1f}%)"

        logger.info(log_message)

    def load_image_model(self, model_name, current_model, current_model_name, device, progress):
        """Load an image generation model"""
        try:
            # Get memory manager
            memory_mgr = get_memory_manager()
            
            # Log initial memory usage
            self._log_memory_usage("before model loading")
            logger.info(f"Memory: {memory_mgr.get_memory_summary()}")

            # Check if model is already loaded
            if current_model and current_model_name == model_name:
                return f"✅ {model_name} is already loaded!", current_model, current_model_name

            # Check memory availability - require at least 8GB RAM
            memory_available, memory_message = self._check_memory_availability(required_ram_gb=8)
            if not memory_available:
                logger.warning(f"Memory check failed: {memory_message}")
                progress(0.1, desc="Attempting to free memory...")
                # Try to free memory
                self._free_memory()
                # Check again
                memory_available, memory_message = self._check_memory_availability(required_ram_gb=8)
                if not memory_available:
                    return f"❌ Not enough memory to load model: {memory_message}", None, None

            # Check if model is downloaded
            model_path = self.models_dir / "image" / model_name

            # Get config - check both regular and gated models
            if model_name in IMAGE_MODELS:
                config = IMAGE_MODELS[model_name]
            elif model_name in GATED_IMAGE_MODELS:
                config = GATED_IMAGE_MODELS[model_name]
            elif model_name in GGUF_IMAGE_MODELS:
                # GGUF models require special handling
                return self._load_gguf_image_model(model_name, current_model, current_model_name, device, progress)
            else:
                return f"❌ Unknown model: {model_name}", None, None

            # Determine the correct model path based on model type
            model_path = (self.models_dir / "image" / model_name).resolve()

            # For FLUX models, also check in src directory if not found in cache
            if not model_path.exists() and model_name.startswith("FLUX"):
                src_model_path = (self.src_models_dir / "image" / model_name).resolve()
                if src_model_path.exists():
                    model_path = src_model_path
                    logger.info(f"Using FLUX model from src directory: {model_path}")

            if not model_path.exists():
                return f"❌ Model {model_name} not found. Please download it first.", None, None
            
            # Check if this is a HuggingFace cache directory structure
            subdirs = [d for d in model_path.iterdir() if d.is_dir()]
            hf_cache_dirs = [d for d in subdirs if d.name.startswith("models--")]
            
            if hf_cache_dirs:
                # This is a HF cache structure, need to look inside for the actual model
                for hf_cache_dir in hf_cache_dirs:
                    snapshots_dir = hf_cache_dir / "snapshots"
                    if snapshots_dir.exists():
                        # Get the latest snapshot (usually there's only one)
                        snapshot_dirs = [d for d in snapshots_dir.iterdir() if d.is_dir()]
                        if snapshot_dirs:
                            # Use the first (and usually only) snapshot
                            model_path = snapshot_dirs[0]
                            logger.info(f"Using model from HuggingFace cache: {model_path}")
                            break
                else:
                    return f"❌ Model {model_name} found but no valid snapshot in HuggingFace cache.", None, None

            # Unload current model
            if current_model:
                progress(0.1, desc="Unloading current model...")
                del current_model
                current_model = None
                current_model_name = None
                self._free_memory()

            # Perform additional memory cleanup before loading new model
            progress(0.2, desc="Preparing system memory...")
            # Force a more aggressive memory cleanup
            import sys
            if 'numpy' in sys.modules:
                # Clear numpy cache if it exists
                try:
                    import numpy as np
                    np.clear_cache()
                except:
                    pass

            # Run garbage collection multiple times to ensure maximum cleanup
            for _ in range(3):
                gc.collect()

            if torch.cuda.is_available():
                # Empty CUDA cache and trigger defragmentation
                torch.cuda.empty_cache()
                # Try to defragment CUDA memory if possible
                try:
                    torch.cuda.memory_stats()  # This can help trigger defragmentation
                except:
                    pass

            self._log_memory_usage("after aggressive cleanup")
            
            # Additional cleanup with memory manager
            memory_mgr.clear_cache_aggressive()
            logger.info(f"Memory after cleanup: {memory_mgr.get_memory_summary()}")
            
            progress(0.3, desc=f"Loading {model_name}...")

            # Use the passed device or default to cuda if available, otherwise cpu
            device_to_use = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

            # Determine dtype based on device
            if device_to_use == "cpu":
                dtype_to_use = torch.float32
            else:
                dtype_to_use = torch.float16

            # Check if this is a GGUF model and handle differently
            if config.is_gguf:
                progress(0.4, desc=f"Downloading GGUF model {model_name}...")
                return self._load_gguf_model(model_path, config, device_to_use, dtype_to_use, progress)
            
            # Check if this is a FLUX model and ensure config.json exists
            elif config.pipeline_class == "FluxPipeline":
                # Check if config.json exists at the root level
                config_json_path = model_path / "config.json"
                if not config_json_path.exists():
                    # Create config.json based on model_index.json
                    import json
                    try:
                        model_index_path = model_path / "model_index.json"
                        if model_index_path.exists():
                            with open(model_index_path, 'r') as f:
                                model_index = json.load(f)

                            # Create a basic config.json with necessary information
                            config_data = {
                                "model_type": "flux",
                                "_class_name": model_index.get("_class_name", "FluxPipeline"),
                                "_diffusers_version": model_index.get("_diffusers_version", "0.30.0")
                            }

                            with open(config_json_path, 'w') as f:
                                json.dump(config_data, f, indent=2)

                            logger.info(f"Created config.json for FLUX model at {config_json_path}")
                        else:
                            logger.error(f"model_index.json not found at {model_index_path}")
                            return f"❌ Error: model_index.json not found at {model_index_path}", None, None
                    except Exception as e:
                        logger.error(f"Error creating config.json: {str(e)}")
                        return f"❌ Error creating config.json: {str(e)}", None, None

                # Check for essential model components before loading
                missing_components = []

                # Check for text_encoder files
                if not any((model_path / "text_encoder").glob("*.safetensors")) and not any((model_path / "text_encoder").glob("*.bin")):
                    missing_components.append("text_encoder weights")

                # Check for text_encoder_2 files
                if not any((model_path / "text_encoder_2").glob("*.safetensors")) and not any((model_path / "text_encoder_2").glob("*.bin")):
                    missing_components.append("text_encoder_2 weights")

                # Check for transformer files
                if not any((model_path / "transformer").glob("*.safetensors")) and not any((model_path / "transformer").glob("*.bin")):
                    missing_components.append("transformer weights")

                # Check for VAE files
                if not any((model_path / "vae").glob("*.safetensors")) and not any((model_path / "vae").glob("*.bin")):
                    missing_components.append("vae weights")

                # If any components are missing, return an error
                if missing_components:
                    error_message = f"❌ Error: Model {model_name} is incomplete. Missing components: {', '.join(missing_components)}.\n"
                    error_message += "Please re-download the model with the 'Force re-download' option checked."
                    logger.error(f"Incomplete model: {model_name}. Missing: {missing_components}")
                    return error_message, None, None

                # Load FLUX model
                try:
                    # Create temporary directory for offloading
                    with tempfile.TemporaryDirectory() as offload_dir:
                        progress(0.4, desc=f"Loading {model_name} with memory optimization...")
                        self._log_memory_usage("before loading FLUX model")

                        # Initialize loaded_model in the outer scope
                        loaded_model = None

                        # Define a custom loading function with memory monitoring
                        def load_with_memory_monitoring():
                            # Set low_cpu_mem_usage=True to reduce memory usage during loading
                            nonlocal loaded_model
                            # Load model without device_map (doesn't work properly with FluxPipeline)
                            loaded_model = FluxPipeline.from_pretrained(
                                str(model_path),
                                torch_dtype=dtype_to_use,
                                use_safetensors=True,
                                low_cpu_mem_usage=True    # Use less CPU memory during loading
                            )
                            
                            # Explicitly move to GPU if requested
                            if device_to_use == "cuda":
                                logger.info("Moving FluxPipeline to GPU...")
                                loaded_model = loaded_model.to(device_to_use)
                                logger.info("FluxPipeline moved to GPU successfully")

                        # Start a monitoring thread to check memory during loading
                        stop_monitoring = threading.Event()

                        def monitor_memory():
                            while not stop_monitoring.is_set():
                                # Check memory and clean up if necessary
                                self._handle_low_memory_during_loading(progress)
                                time.sleep(1)  # Check every second

                        # Start monitoring thread
                        monitor_thread = threading.Thread(target=monitor_memory)
                        monitor_thread.daemon = True
                        monitor_thread.start()

                        try:
                            # Load the model with memory monitoring
                            load_with_memory_monitoring()
                        finally:
                            # Stop monitoring thread
                            stop_monitoring.set()
                            monitor_thread.join(timeout=2)

                        self._log_memory_usage("after loading FLUX model")
                except FileNotFoundError as e:
                    # Extract the missing file path from the error message
                    error_str = str(e)
                    logger.error(f"FileNotFoundError loading FLUX model: {error_str}")

                    # Create a more user-friendly error message
                    error_message = f"❌ Error: Missing file when loading {model_name}.\n"
                    error_message += f"Details: {error_str}\n"
                    error_message += "Please re-download the model with the 'Force re-download' option checked."
                    return error_message, None, None
            # Load based on pipeline class
            elif config.pipeline_class == "AutoPipelineForText2Image":
                # Create temporary directory for offloading
                with tempfile.TemporaryDirectory() as offload_dir:
                    progress(0.4, desc=f"Loading {model_name} with memory optimization...")
                    self._log_memory_usage("before loading AutoPipeline model")

                    # Initialize loaded_model in the outer scope
                    loaded_model = None

                    # Define a custom loading function with memory monitoring
                    def load_with_memory_monitoring():
                        # Set low_cpu_mem_usage=True to reduce memory usage during loading
                        nonlocal loaded_model
                        loaded_model = AutoPipelineForText2Image.from_pretrained(
                            str(model_path),
                            torch_dtype=dtype_to_use,
                            use_safetensors=True,
                            low_cpu_mem_usage=True    # Use less CPU memory during loading
                        )
                        
                        # Explicitly move to GPU if requested
                        if device_to_use == "cuda":
                            logger.info("Moving AutoPipeline to GPU...")
                            loaded_model = loaded_model.to(device_to_use)
                            logger.info("AutoPipeline moved to GPU successfully")

                    # Start a monitoring thread to check memory during loading
                    stop_monitoring = threading.Event()

                    def monitor_memory():
                        while not stop_monitoring.is_set():
                            # Check memory and clean up if necessary
                            self._handle_low_memory_during_loading(progress)
                            time.sleep(1)  # Check every second

                    # Start monitoring thread
                    monitor_thread = threading.Thread(target=monitor_memory)
                    monitor_thread.daemon = True
                    monitor_thread.start()

                    try:
                        # Load the model with memory monitoring
                        load_with_memory_monitoring()
                    finally:
                        # Stop monitoring thread
                        stop_monitoring.set()
                        monitor_thread.join(timeout=2)

                    self._log_memory_usage("after loading AutoPipeline model")
            else:
                # Create temporary directory for offloading
                with tempfile.TemporaryDirectory() as offload_dir:
                    progress(0.4, desc=f"Loading {model_name} with memory optimization...")
                    self._log_memory_usage("before loading DiffusionPipeline model")

                    # Initialize loaded_model in the outer scope
                    loaded_model = None

                    # Define a custom loading function with memory monitoring
                    def load_with_memory_monitoring():
                        # Set low_cpu_mem_usage=True to reduce memory usage during loading
                        nonlocal loaded_model
                        loaded_model = DiffusionPipeline.from_pretrained(
                            str(model_path),
                            torch_dtype=dtype_to_use,
                            use_safetensors=True,
                            low_cpu_mem_usage=True    # Use less CPU memory during loading
                        )
                        
                        # Explicitly move to GPU if requested
                        if device_to_use == "cuda":
                            logger.info("Moving DiffusionPipeline to GPU...")
                            loaded_model = loaded_model.to(device_to_use)
                            logger.info("DiffusionPipeline moved to GPU successfully")

                    # Start a monitoring thread to check memory during loading
                    stop_monitoring = threading.Event()

                    def monitor_memory():
                        while not stop_monitoring.is_set():
                            # Check memory and clean up if necessary
                            self._handle_low_memory_during_loading(progress)
                            time.sleep(1)  # Check every second

                    # Start monitoring thread
                    monitor_thread = threading.Thread(target=monitor_memory)
                    monitor_thread.daemon = True
                    monitor_thread.start()

                    try:
                        # Load the model with memory monitoring
                        load_with_memory_monitoring()
                    finally:
                        # Stop monitoring thread
                        stop_monitoring.set()
                        monitor_thread.join(timeout=2)

                    self._log_memory_usage("after loading DiffusionPipeline model")

            # Enable memory optimizations
            try:
                # Import GPU optimizer for xformers check
                from .gpu_optimizer import get_gpu_optimizer
                gpu_opt = get_gpu_optimizer()
                
                # Try to enable xformers if available and recommended
                xformers_enabled = False
                try:
                    import xformers
                    if hasattr(loaded_model, "enable_xformers_memory_efficient_attention") and gpu_opt.should_use_xformers():
                        try:
                            loaded_model.enable_xformers_memory_efficient_attention()
                            logger.info("Enabled xformers memory efficient attention")
                            xformers_enabled = True
                        except Exception as e:
                            logger.warning(f"Could not enable xformers: {e}")
                except ImportError:
                    logger.info("xformers not installed - using fallback attention slicing")
                        
                # Fall back to sliced attention if xformers not available
                if not xformers_enabled and hasattr(loaded_model, "enable_attention_slicing"):
                    loaded_model.enable_attention_slicing(slice_size="auto")
                    logger.info("Using attention slicing for memory efficiency")
                    
                # Enable VAE optimizations
                if hasattr(loaded_model, "vae"):
                    if hasattr(loaded_model.vae, "enable_slicing"):
                        loaded_model.vae.enable_slicing()
                        logger.info("Enabled VAE slicing")
                    if hasattr(loaded_model.vae, "enable_tiling"):
                        loaded_model.vae.enable_tiling()
                        logger.info("Enabled VAE tiling")
                        
                # Remove safety checker to save memory
                if hasattr(loaded_model, "safety_checker"):
                    loaded_model.safety_checker = None
                    loaded_model.requires_safety_checker = False
                    logger.info("Disabled safety checker to save memory")
                # Disabled CPU offloading to keep models in VRAM for better GPU utilization
                # if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory < 6 * 1024 * 1024 * 1024:  # Less than 6GB VRAM
                #     if hasattr(loaded_model, "enable_sequential_cpu_offload"):
                #         loaded_model.enable_sequential_cpu_offload()
                #         logger.info("Enabled sequential CPU offloading for low memory system")

                # Log successful GPU placement
                if device_to_use == "cuda":
                    logger.info("Model optimization complete, components on GPU")
                
            except Exception as e:
                logger.warning(f"Could not enable memory optimizations: {str(e)}")

            # Log final memory usage and device placement
            self._log_memory_usage("after model loading and optimization")
            
            # Log device placement for model components
            if hasattr(loaded_model, 'device'):
                logger.info(f"Model device: {loaded_model.device}")
            if hasattr(loaded_model, 'hf_device_map'):
                logger.info(f"Model device map: {loaded_model.hf_device_map}")
            
            # Update internal state if no external model was provided
            if current_model is None and device is None:
                self.image_model = loaded_model
                self.image_model_name = model_name

            progress(1.0, desc="Model loaded!")

            return f"✅ Successfully loaded {model_name}", loaded_model, model_name

        except FileNotFoundError as e:
            # Handle missing files with a more user-friendly message
            error_str = str(e)
            logger.error(f"FileNotFoundError loading model: {error_str}")

            # Create a more user-friendly error message
            error_message = f"❌ Error: Missing file when loading {model_name}.\n"
            error_message += f"Details: {error_str}\n"
            error_message += "Please re-download the model with the 'Force re-download' option checked."
            return error_message, None, None
        except Exception as e:
            error_str = str(e)
            logger.error(f"Error loading image model: {error_str}")

            # Create a more detailed error message
            error_message = f"❌ Error loading model: {error_str}\n"

            # Add suggestions based on common error patterns
            if "CUDA out of memory" in error_str:
                error_message += "\nSuggestion: Your GPU ran out of memory. Try using a smaller model or reducing batch size."
            elif "safetensors" in error_str.lower():
                error_message += "\nSuggestion: There might be an issue with the model files. Try re-downloading the model."
            elif "config" in error_str.lower():
                error_message += "\nSuggestion: There might be an issue with the model configuration. Try re-downloading the model."
            elif "not found" in error_str.lower() or "no such file" in error_str.lower():
                error_message += "\nSuggestion: Some model files are missing. Try re-downloading the model with the 'Force re-download' option checked."

            return error_message, None, None

    def load_hunyuan3d_model(self, model_name, current_model, current_model_name, device, progress):
        """Load a Hunyuan3D model"""
        try:
            # Check if model is already loaded
            if current_model and current_model_name == model_name:
                return f"✅ {model_name} is already loaded!", current_model, current_model_name

            # Determine the correct model path
            model_path = (self.models_dir / "3d" / model_name).resolve()

            if not model_path.exists():
                return f"❌ Model {model_name} not found. Please download it first.", None, None
            
            # Check if this is a HuggingFace cache directory structure
            subdirs = [d for d in model_path.iterdir() if d.is_dir()]
            hf_cache_dirs = [d for d in subdirs if d.name.startswith("models--")]
            
            if hf_cache_dirs:
                # This is a HF cache structure, need to look inside for the actual model
                for hf_cache_dir in hf_cache_dirs:
                    snapshots_dir = hf_cache_dir / "snapshots"
                    if snapshots_dir.exists():
                        # Get the latest snapshot (usually there's only one)
                        snapshot_dirs = [d for d in snapshots_dir.iterdir() if d.is_dir()]
                        if snapshot_dirs:
                            # Use the first (and usually only) snapshot
                            model_path = snapshot_dirs[0]
                            logger.info(f"Using 3D model from HuggingFace cache: {model_path}")
                            break
                else:
                    return f"❌ Model {model_name} found but no valid snapshot in HuggingFace cache.", None, None

            # Unload current model
            if current_model:
                progress(0.1, desc="Unloading current model...")
                del current_model
                current_model = None
                current_model_name = None
                torch.cuda.empty_cache()
                gc.collect()

            progress(0.3, desc=f"Loading {model_name}...")

            # Use the passed device or default to cuda if available, otherwise cpu
            device_to_use = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

            # Determine dtype based on device
            if device_to_use == "cpu":
                dtype_to_use = torch.float32
            else:
                dtype_to_use = torch.float16

            # Placeholder for actual Hunyuan3D loading
            # This would be replaced with actual Hunyuan3D loading code
            loaded_model = {"name": model_name, "path": model_path, "dtype": dtype_to_use}

            # Update internal state if no external model was provided
            if current_model is None and device is None:
                self.hunyuan3d_model = loaded_model
                self.hunyuan3d_model_name = model_name

            progress(1.0, desc="Model loaded!")
            return f"✅ Successfully loaded {model_name}", loaded_model, model_name

        except Exception as e:
            logger.error(f"Error loading 3D model: {str(e)}")
            return f"❌ Error loading model: {str(e)}", None, None

    def _load_gguf_image_model(self, model_name, current_model, current_model_name, device, progress):
        """Load a GGUF image generation model using diffusers"""
        try:
            from .config import GGUF_IMAGE_MODELS, FLUX_COMPONENTS
            
            # Check if model is already loaded
            if current_model and current_model_name == model_name:
                return f"✅ {model_name} is already loaded!", current_model, current_model_name
            
            # Get config
            config = GGUF_IMAGE_MODELS[model_name]
            
            # Check if GGUF model is downloaded
            if not self.check_gguf_model_complete(model_name):
                return f"❌ GGUF model {model_name} is not downloaded. Please download it first.", None, None
            
            # Unload current model if any
            if current_model:
                progress(0.1, desc="Unloading current model...")
                del current_model
                current_model = None
                current_model_name = None
                self._free_memory()
            
            progress(0.2, desc=f"Preparing to load GGUF model {model_name}...")
            
            # For the existing FLUX.1-dev-Q8 model, use the already downloaded file
            if model_name == "FLUX.1-dev-Q8":
                # Find the GGUF file
                gguf_file = self._find_gguf_file(model_name, config.gguf_file)
                if not gguf_file:
                    return f"❌ GGUF file not found for {model_name}", None, None
                
                # Create a model info for the existing Q8 model
                model_info = GGUFModelInfo(
                    name=model_name,
                    quantization="8",
                    file_size_gb=12.5,
                    memory_required_gb=14.0,
                    repo_id="city96/FLUX.1-dev-gguf",
                    filename=config.gguf_file,
                    url="",
                    quality_score=0.98,
                    min_vram_gb=12.0
                )
                
                # Use the existing file path
                model_path = gguf_file
                recommended_model = model_info
            else:
                # Get available VRAM and recommend quantization
                available_vram = self.gguf_manager.get_available_vram()
                logger.info(f"Available VRAM: {available_vram:.1f}GB")
                
                # Determine model type (flux-dev or flux-schnell)
                model_type = "flux-dev" if "dev" in model_name.lower() else "flux-schnell"
                
                # Get recommended quantization based on available VRAM
                recommended_model = self.gguf_manager.recommend_quantization(model_type)
                
                if not recommended_model:
                    return f"❌ Not enough VRAM for any GGUF variant. Available: {available_vram:.1f}GB", None, None
                
                # Check if model is downloaded, if not download it
                if not self.gguf_manager.is_model_cached(recommended_model):
                    progress(0.3, desc=f"Downloading {recommended_model.name} ({recommended_model.file_size_gb:.1f}GB)...")
                    try:
                        model_path = self.gguf_manager.download_model(recommended_model)
                    except Exception as e:
                        return f"❌ Failed to download GGUF model: {str(e)}", None, None
                else:
                    model_path = self.gguf_manager.get_model_path(recommended_model)
            
            logger.info(f"Loading GGUF variant: {recommended_model.name} (Q{recommended_model.quantization})")
            
            progress(0.5, desc=f"Loading GGUF transformer ({model_path.stat().st_size / (1024**3):.1f}GB file)... This may take 1-2 minutes")
            
            # Determine compute dtype based on device
            if device == "cpu":
                compute_dtype = torch.float32
            else:
                compute_dtype = torch.bfloat16
            
            try:
                # Load GGUF transformer
                transformer = self.gguf_manager.load_transformer(
                    recommended_model,
                    compute_dtype=compute_dtype,
                    device_map="auto" if device == "cuda" else "cpu",
                    model_path=model_path  # Pass the actual path
                )
                
                progress(0.7, desc="Loading FLUX pipeline components...")
                
                # Create pipeline with GGUF transformer
                base_model_id = "black-forest-labs/FLUX.1-dev" if "dev" in model_name.lower() else "black-forest-labs/FLUX.1-schnell"
                
                # Capture download progress
                import sys
                from io import StringIO
                
                class ProgressCapture:
                    def __init__(self, progress_callback):
                        self.progress_callback = progress_callback
                        self.buffer = StringIO()
                        self.last_update = time.time()
                        
                    def write(self, text):
                        self.buffer.write(text)
                        current_time = time.time()
                        
                        # Update frequently for better feedback
                        if current_time - self.last_update > 0.2 or any(keyword in text for keyword in ["Fetching", "%", "Downloading", "Loading"]):
                            # Clean up the text
                            cleaned_text = text.strip().replace('\r', '').replace('\n', ' ')
                            
                            # Extract progress info
                            if "Fetching" in cleaned_text and "files:" in cleaned_text:
                                # Parse fetching progress: "Fetching 19 files: 21%|██ | 4/19"
                                import re
                                match = re.search(r'Fetching (\d+) files:.*?(\d+)%.*?(\d+)/(\d+)', cleaned_text)
                                if match:
                                    total_files = int(match.group(1))
                                    percent = int(match.group(2))
                                    current = int(match.group(3))
                                    total = int(match.group(4))
                                    # Calculate overall progress (0.7 to 0.9 range)
                                    overall_progress = 0.7 + (percent / 100.0) * 0.2
                                    self.progress_callback(overall_progress, desc=f"Downloading FLUX components: {current}/{total} files ({percent}%)")
                            elif "Downloading" in cleaned_text and "model" in cleaned_text.lower():
                                self.progress_callback(0.75, desc=cleaned_text[:100])  # Limit length
                            elif "Loading" in cleaned_text:
                                self.progress_callback(0.85, desc=cleaned_text[:100])
                            
                            self.last_update = current_time
                    
                    def flush(self):
                        pass
                
                # Redirect stdout to capture progress
                old_stdout = sys.stdout
                old_stderr = sys.stderr
                progress_capture = ProgressCapture(progress)
                
                try:
                    sys.stdout = progress_capture
                    sys.stderr = progress_capture
                    
                    logger.info(f"Downloading FLUX pipeline components from {base_model_id}")
                    
                    # Check if we have the base model cached locally
                    flux_cache_dir = self.models_dir / "flux_base" / base_model_id.replace("/", "--")
                    
                    # For GGUF models, we'll download components individually as needed
                    # This avoids the large FLUX base model download
                    logger.info("Loading FLUX pipeline with GGUF transformer and individual components")
                    
                    try:
                        # Check if we need HF token for gated model
                        if "dev" in model_name.lower() and not self.hf_token:
                            raise Exception("""
                            FLUX.1-dev requires Hugging Face authentication.
                            Please:
                            1. Go to https://huggingface.co/black-forest-labs/FLUX.1-dev
                            2. Accept the license agreement
                            3. Get your token from https://huggingface.co/settings/tokens
                            4. Set it in the Model Manager tab
                            """)
                        
                        # Try loading from HuggingFace directly, which will download only needed components
                        pipeline = FluxPipeline.from_pretrained(
                            base_model_id,
                            transformer=transformer,  # Use our GGUF transformer
                            torch_dtype=compute_dtype,
                            token=self.hf_token if self.hf_token else None,
                            low_cpu_mem_usage=True,
                            local_files_only=False,  # Allow downloading missing components
                            cache_dir=str(self.models_dir / "flux_base")  # Use our cache directory
                        )
                        logger.info("FLUX pipeline loaded successfully with GGUF transformer")
                    except Exception as e:
                        logger.error(f"Failed to load FLUX pipeline: {str(e)}")
                        
                        # If that fails, try a more manual approach
                        logger.info("Attempting alternative loading method...")
                        
                        # Download individual components
                        from transformers import T5EncoderModel, CLIPTextModel
                        from diffusers import AutoencoderKL
                        
                        progress(0.75, desc="Downloading VAE...")
                        vae = AutoencoderKL.from_pretrained(
                            base_model_id,
                            subfolder="vae",
                            torch_dtype=compute_dtype,
                            token=self.hf_token,
                            cache_dir=str(self.models_dir / "flux_base")
                        )
                        
                        progress(0.8, desc="Downloading CLIP text encoder...")
                        text_encoder = CLIPTextModel.from_pretrained(
                            base_model_id,
                            subfolder="text_encoder",
                            torch_dtype=compute_dtype,
                            token=self.hf_token,
                            cache_dir=str(self.models_dir / "flux_base")
                        )
                        
                        progress(0.85, desc="Downloading T5 text encoder...")
                        text_encoder_2 = T5EncoderModel.from_pretrained(
                            base_model_id,
                            subfolder="text_encoder_2",
                            torch_dtype=compute_dtype,
                            token=self.hf_token,
                            cache_dir=str(self.models_dir / "flux_base")
                        )
                        
                        # Create pipeline manually
                        pipeline = FluxPipeline(
                            transformer=transformer,
                            vae=vae,
                            text_encoder=text_encoder,
                            text_encoder_2=text_encoder_2,
                            scheduler=None,  # Will use default
                            tokenizer=None,  # Will be loaded from text encoder
                            tokenizer_2=None  # Will be loaded from text encoder 2
                        )
                finally:
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr
                
                # Enable memory optimizations
                if device == "cuda":
                    pipeline.enable_model_cpu_offload()
                else:
                    pipeline = pipeline.to(device)
                
                # Mark pipeline as using GGUF model
                pipeline._is_gguf_model = True
                pipeline._gguf_model_info = recommended_model
                
                progress(1.0, desc="GGUF model loaded successfully!")
                
                # Store model
                self.image_model = pipeline
                self.image_model_name = model_name
                
                return f"""✅ GGUF model loaded successfully!
Model: {recommended_model.name}
Quantization: {recommended_model.quantization}
Quality Score: {recommended_model.quality_score:.0%}
VRAM Usage: ~{recommended_model.min_vram_gb}GB""", pipeline, model_name
                
            except ImportError as e:
                if "GGUFQuantizationConfig" in str(e) or "gguf" in str(e).lower():
                    return f"""❌ GGUF support not available in your diffusers version.
Please upgrade diffusers: pip install --upgrade diffusers>=0.31.0 gguf""", None, None
                else:
                    raise
            
        except Exception as e:
            logger.error(f"Error loading GGUF model: {str(e)}")
            return f"❌ Failed to load GGUF model: {str(e)}", None, None

    def unload_image_model(self):
        """Unload the current image model"""
        if self.image_model:
            del self.image_model
            self.image_model = None
            self.image_model_name = None
            torch.cuda.empty_cache()
            gc.collect()

    def unload_3d_model(self):
        """Unload the current 3D model"""
        if self.hunyuan3d_model:
            del self.hunyuan3d_model
            self.hunyuan3d_model = None
            self.hunyuan3d_model_name = None
            torch.cuda.empty_cache()
            gc.collect()

    def unload_models(self):
        """Unload all models"""
        self.unload_image_model()
        self.unload_3d_model()
        torch.cuda.empty_cache()
        gc.collect()

    @property
    def any_image_model_downloaded(self):
        """Check if any image model is downloaded"""
        image_model_dir = self.models_dir / "image"
        return image_model_dir.exists() and any(d.is_dir() for d in image_model_dir.iterdir())

    @property
    def any_hunyuan_model_downloaded(self):
        """Check if any hunyuan model is downloaded"""
        hunyuan_model_dir = self.models_dir / "3d"
        return hunyuan_model_dir.exists() and any(d.is_dir() for d in hunyuan_model_dir.iterdir())

    def get_model_selection_data(self):
        """Returns updated choices for model dropdowns."""
        downloaded_image_models = []
        for model_name in ALL_IMAGE_MODELS.keys():
            model_path_cache = self.models_dir / "image" / model_name

            if self.check_model_complete(model_path_cache, "image", model_name):
                logger.debug(f"{model_name} found in cache: True")
                downloaded_image_models.append(model_name)
            elif model_name.startswith("FLUX"):
                model_path_src = self.src_models_dir / "image" / model_name
                is_downloaded_src = self.check_model_complete(model_path_src, "image", model_name)
                logger.debug(f"{model_name} (FLUX) found in src: {is_downloaded_src}")
                if is_downloaded_src:
                    downloaded_image_models.append(model_name)
            else:
                logger.debug(f"{model_name} not found in cache or src.")

        downloaded_hunyuan_models = []
        for model_name in HUNYUAN3D_MODELS.keys():
            model_path = self.models_dir / "3d" / model_name
            if self.check_model_complete(model_path, "3d", model_name):
                downloaded_hunyuan_models.append(model_name)

        image_dropdown_choices = downloaded_image_models
        hunyuan_dropdown_choices = downloaded_hunyuan_models

        logger.debug(f"Final Downloaded Image Models for Dropdown: {downloaded_image_models}")
        logger.debug(f"Final Downloaded Hunyuan Models for Dropdown: {downloaded_hunyuan_models}")

        logger.debug(f"Final Image Dropdown Choices: {image_dropdown_choices}")
        logger.debug(f"Final Hunyuan Dropdown Choices: {hunyuan_dropdown_choices}")

        logger.debug(f"Downloaded Image Models: {downloaded_image_models}")
        logger.debug(f"Downloaded Hunyuan Models: {downloaded_hunyuan_models}")

        logger.debug(f"Final Downloaded Image Models: {downloaded_image_models}")
        logger.debug(f"Final Downloaded Hunyuan Models: {downloaded_hunyuan_models}")

        logger.debug(f"Downloaded Image Models: {downloaded_image_models}")
        logger.debug(f"Downloaded Hunyuan Models: {downloaded_hunyuan_models}")

        logger.debug(f"Final downloaded_image_models: {downloaded_image_models}")
        logger.debug(f"Final downloaded_hunyuan_models: {downloaded_hunyuan_models}")

        logger.debug(f"Downloaded Image Models: {downloaded_image_models}")
        logger.debug(f"Downloaded Hunyuan Models: {downloaded_hunyuan_models}")

        logger.debug(f"Final Downloaded Image Models: {downloaded_image_models}")
        logger.debug(f"Final Downloaded Hunyuan Models: {downloaded_hunyuan_models}")

        logger.debug(f"Final Downloaded Image Models for Dropdown: {downloaded_image_models}")
        logger.debug(f"Final Downloaded Hunyuan Models for Dropdown: {downloaded_hunyuan_models}")

        logger.debug(f"Final Image Dropdown Choices: {image_dropdown_choices}")
        logger.debug(f"Final Hunyuan Dropdown Choices: {hunyuan_dropdown_choices}")

        logger.debug(f"Final downloaded_image_models: {downloaded_image_models}")
        logger.debug(f"Final downloaded_hunyuan_models: {downloaded_hunyuan_models}")

        logger.debug(f"Downloaded Image Models: {downloaded_image_models}")
        logger.debug(f"Downloaded Hunyuan Models: {downloaded_hunyuan_models}")



        selected_image_model = None
        if len(downloaded_image_models) == 1:
            selected_image_model = downloaded_image_models[0]

        selected_hunyuan_model = None
        if len(downloaded_hunyuan_models) == 1:
            selected_hunyuan_model = downloaded_hunyuan_models[0]

        # Return simple values instead of gr.update objects to avoid schema issues
        return image_dropdown_choices, hunyuan_dropdown_choices, image_dropdown_choices, hunyuan_dropdown_choices

    def _format_bytes(self, bytes):
        """Format bytes to human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes < 1024.0:
                return f"{bytes:.2f} {unit}"
            bytes /= 1024.0
        return f"{bytes:.2f} TB"
