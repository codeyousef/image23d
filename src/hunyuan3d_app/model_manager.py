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
    FluxPipeline
)
from huggingface_hub import snapshot_download, HfApi

from .config import (
    IMAGE_MODELS, GATED_IMAGE_MODELS,
    ALL_IMAGE_MODELS, HUNYUAN3D_MODELS
)
from .memory_manager import get_memory_manager

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
        model_path = model_path.resolve()
        if not model_path.exists():
            return False

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
        missing_components = []

        # First check if the model is downloaded at all
        model_path_cache = self.models_dir / model_type / model_name
        model_path_src = self.src_models_dir / model_type / model_name if model_name.startswith("FLUX") else None

        # Determine which path to use
        if model_path_cache.exists():
            model_path = model_path_cache
        elif model_path_src and model_path_src.exists():
            model_path = model_path_src
        else:
            # Model not downloaded at all
            return ["complete model"]

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
                "local_dir": str(save_path),
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
            progress(0.1, desc=f"Downloading GGUF file: {config.gguf_file}")
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
            progress(0.3, desc="Downloading FLUX VAE...")
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
            progress(0.5, desc="Downloading CLIP text encoder...")
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
            progress(0.7, desc="Downloading T5 text encoder...")
            t5_path = te_dir / FLUX_COMPONENTS["text_encoder_t5"]["filename"]
            if not t5_path.exists():
                hf_hub_download(
                    repo_id=FLUX_COMPONENTS["text_encoder_t5"]["repo_id"],
                    filename=FLUX_COMPONENTS["text_encoder_t5"]["filename"],
                    local_dir=te_dir,
                    token=self.hf_token
                )
            components_downloaded += 1
            
            progress(1.0, desc="GGUF model components downloaded!")
            
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

        image_dropdown_update = gr.update(
            choices=image_dropdown_choices,
            value=selected_image_model,
            interactive=len(image_dropdown_choices) > 0
        )
        hunyuan_dropdown_update = gr.update(
            choices=hunyuan_dropdown_choices,
            value=selected_hunyuan_model,
            interactive=len(hunyuan_dropdown_choices) > 0
        )

        return image_dropdown_update, hunyuan_dropdown_update, image_dropdown_update, hunyuan_dropdown_update

    def _format_bytes(self, bytes):
        """Format bytes to human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes < 1024.0:
                return f"{bytes:.2f} {unit}"
            bytes /= 1024.0
        return f"{bytes:.2f} TB"
