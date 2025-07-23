"""Main model management module."""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import torch

from .base import BaseModelManager, ModelInfo
from .download import ModelDownloader
from .loading import ModelLoader
from ..config import (
    IMAGE_MODELS, GATED_IMAGE_MODELS, GGUF_IMAGE_MODELS,
    ALL_IMAGE_MODELS, HUNYUAN3D_MODELS
)
from .gguf import GGUFModelManager
from ..services.websocket import ProgressStreamManager

logger = logging.getLogger(__name__)


class ModelManager(BaseModelManager):
    """Main model manager that coordinates downloading and loading."""
    
    def __init__(self, models_dir: Path, src_models_dir: Path, websocket_server: Optional[ProgressStreamManager] = None):
        super().__init__(models_dir)
        self.src_models_dir = Path(src_models_dir)
        
        # Initialize components
        self.downloader = ModelDownloader(cache_dir=models_dir, websocket_server=websocket_server)
        self.gguf_manager = GGUFModelManager(models_dir=models_dir)
        self.loader = ModelLoader(gguf_manager=self.gguf_manager)
        
        # Model tracking
        self.image_model = None
        self.image_model_name = None
        self.hunyuan3d_model = None
        self.hunyuan3d_model_name = None
        
        # Load HF token if available
        self._load_hf_token()
    
    def _load_hf_token(self):
        """Load Hugging Face token from environment or file."""
        import os
        from ..utils import get_hf_token_from_all_sources, validate_hf_token
        
        # Get token from all sources
        token = get_hf_token_from_all_sources()
        
        if token and validate_hf_token(token):
            logger.info("Loaded valid HF token")
            self.hf_token = token
            self.downloader.set_token(token)
            # Ensure it's in environment for child processes
            os.environ["HF_TOKEN"] = token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = token
        else:
            logger.warning("No valid HF token found - gated models like FLUX.1-dev will require authentication")
            self.hf_token = None
            
    def validate_token_for_model(self, model_name: str) -> Tuple[bool, str]:
        """Validate if token is available for a model that requires it.
        
        Returns:
            Tuple of (is_valid, message)
        """
        # Check if model requires authentication
        requires_auth = False
        if model_name in GATED_IMAGE_MODELS:
            requires_auth = True
        elif "flux" in model_name.lower() and "dev" in model_name.lower():
            requires_auth = True
            
        if not requires_auth:
            return True, "Model does not require authentication"
            
        # Check token
        from ..utils import validate_hf_token
        if self.hf_token and validate_hf_token(self.hf_token):
            return True, "Valid HF token found"
        else:
            return False, (
                "This model requires a Hugging Face token.\n"
                "Please set your token in the 'Model Management' tab or via:\n"
                "export HF_TOKEN='your_token_here'"
            )
    
    def set_hf_token(self, token: str) -> str:
        """Set Hugging Face token for gated models."""
        import os
        
        self.hf_token = token
        self.downloader.set_token(token)
        
        if token:
            # Set in environment for child processes
            os.environ["HF_TOKEN"] = token
            
            # Save to file for persistence
            try:
                from ..utils import save_hf_token
                save_hf_token(token)
                logger.info("HF token saved successfully")
            except Exception as e:
                logger.warning(f"Could not save HF token to file: {e}")
            
            return "✅ Hugging Face token set successfully"
        else:
            # Clear token
            if "HF_TOKEN" in os.environ:
                del os.environ["HF_TOKEN"]
            self.hf_token = None
            return "❌ Token cleared"
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get current status of loaded models."""
        vram_used = 0
        if torch.cuda.is_available():
            vram_used = torch.cuda.memory_allocated() / (1024**3)
        
        return {
            "image_model": self.image_model_name or "None",
            "hunyuan3d_model": self.hunyuan3d_model_name or "None",
            "vram_used": f"{vram_used:.2f} GB",
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }
    
    def get_available_models(self, model_type: str = "all") -> Dict[str, ModelInfo]:
        """Get all available models by type.
        
        Args:
            model_type: Type of models to get ("image", "3d", "all")
            
        Returns:
            Dictionary of model name to ModelInfo
        """
        available = {}
        
        if model_type in ["image", "all"]:
            # Add standard image models
            for name, config in IMAGE_MODELS.items():
                available[name] = ModelInfo(
                    name=name,
                    repo_id=config.repo_id,
                    model_type="image",
                    size_gb=getattr(config, 'size_gb', 0),
                    requires_auth=False,
                    description=config.description,
                    tags=getattr(config, 'tags', []),
                    recommended_vram_gb=getattr(config, 'recommended_vram_gb', 8)
                )
            
            # Add gated image models
            for name, config in GATED_IMAGE_MODELS.items():
                available[name] = ModelInfo(
                    name=name,
                    repo_id=config.repo_id,
                    model_type="image",
                    size_gb=getattr(config, 'size_gb', 0),
                    requires_auth=True,
                    description=config.description,
                    tags=getattr(config, 'tags', []),
                    recommended_vram_gb=getattr(config, 'recommended_vram_gb', 8)
                )
            
            # Add GGUF models
            for name, config in GGUF_IMAGE_MODELS.items():
                available[name] = ModelInfo(
                    name=name,
                    repo_id=config.repo_id,
                    model_type="image_gguf",
                    size_gb=getattr(config, 'size_gb', 0),
                    requires_auth=False,
                    description=config.description,
                    tags=getattr(config, 'tags', ["gguf", "quantized"]),
                    recommended_vram_gb=getattr(config, 'recommended_vram_gb', 6)
                )
        
        if model_type in ["3d", "all"]:
            # Add 3D models
            for name, config in HUNYUAN3D_MODELS.items():
                available[name] = ModelInfo(
                    name=name,
                    repo_id=config["repo_id"],
                    model_type="3d",
                    size_gb=config.get("size_gb", 0),
                    requires_auth=False,
                    description=config.get("description", ""),
                    tags=config.get("tags", ["3d"]),
                    recommended_vram_gb=config.get("recommended_vram_gb", 12)
                )
        
        return available
    
    def check_model_complete(self, model_path: Path, model_type: str, model_name: str) -> bool:
        """Check if a model is completely downloaded.
        
        Args:
            model_path: Path to model directory
            model_type: Type of model
            model_name: Name of model
            
        Returns:
            True if model is complete
        """
        if not model_path.exists():
            return False
        
        # For GGUF models, check for .gguf files
        if model_name in GGUF_IMAGE_MODELS or "gguf" in model_path.parts:
            # GGUF models have .gguf files
            gguf_files = list(model_path.glob("*.gguf"))
            if len(gguf_files) > 0:
                return True
            # Also check recursively in subdirectories
            for subdir in model_path.iterdir():
                if subdir.is_dir():
                    gguf_files = list(subdir.glob("*.gguf"))
                    if len(gguf_files) > 0:
                        return True
        
        # For Hunyuan3D models, check for specific files
        if model_type == "3d" and "hunyuan3d" in model_name.lower():
            # Look for snapshots directory structure
            snapshots_dirs = list(model_path.glob("**/snapshots/*/"))
            if snapshots_dirs:
                # Check if any snapshot has the required files
                for snapshot_dir in snapshots_dirs:
                    if (snapshot_dir / "hunyuan3d-dit-v2-1").exists() or (snapshot_dir / "hunyuan3d-dit-v2-0").exists():
                        return True
            
            # Direct check for hunyuan3d files with actual weights
            dit_v21_path = model_path / "hunyuan3d-dit-v2-1"
            dit_v20_path = model_path / "hunyuan3d-dit-v2-0"
            
            # Check if v2.1 model has actual weight files
            if dit_v21_path.exists():
                has_weights = (
                    any(dit_v21_path.glob("*.pth")) or 
                    any(dit_v21_path.glob("*.pt")) or 
                    any(dit_v21_path.glob("*.safetensors")) or 
                    any(dit_v21_path.glob("*.ckpt")) or
                    any(dit_v21_path.glob("*.fp16.ckpt"))  # Add support for fp16 checkpoints
                )
                if has_weights:
                    return True
                    
            # Check if v2.0 model has actual weight files  
            if dit_v20_path.exists():
                has_weights = (
                    any(dit_v20_path.glob("*.pth")) or 
                    any(dit_v20_path.glob("*.pt")) or 
                    any(dit_v20_path.glob("*.safetensors")) or 
                    any(dit_v20_path.glob("*.ckpt")) or
                    any(dit_v20_path.glob("*.fp16.ckpt"))  # Add support for fp16 checkpoints
                )
                if has_weights:
                    return True
        
        # For standard diffusers models, check for key files
        required_files = [
            "model_index.json",  # Diffusers models
            "config.json",       # Transformers models
        ]
        
        for file in required_files:
            if (model_path / file).exists():
                return True
            # Also check in subdirectories for HuggingFace cache structure
            for subdir in model_path.iterdir():
                if subdir.is_dir():
                    # Check in snapshots
                    snapshots = subdir / "snapshots"
                    if snapshots.exists():
                        for snapshot in snapshots.iterdir():
                            if snapshot.is_dir() and (snapshot / file).exists():
                                return True
                    # Direct check in subdir
                    if (subdir / file).exists():
                        return True
        
        # Check for safetensors or bin files
        has_weights = (
            len(list(model_path.glob("*.safetensors"))) > 0 or
            len(list(model_path.glob("*.bin"))) > 0 or
            len(list(model_path.glob("**/*.safetensors"))) > 0 or  # Check recursively
            len(list(model_path.glob("**/*.bin"))) > 0  # Check recursively
        )
        
        return has_weights
    
    def download_model(
        self,
        model_name: str,
        model_type: str,
        progress_callback: Optional[Any] = None
    ) -> Tuple[bool, str]:
        """Download a model.
        
        Args:
            model_name: Name of the model
            model_type: Type of model (image, 3d)
            progress_callback: Optional progress callback
            
        Returns:
            Tuple of (success, message)
        """
        # Get model info
        all_models = self.get_available_models(model_type)
        if model_name not in all_models:
            return False, f"Unknown model: {model_name}"
        
        model_info = all_models[model_name]
        
        # Determine target directory
        if model_type == "image":
            target_dir = self.models_dir / "image" / model_name
        elif model_type == "3d":
            target_dir = self.models_dir / "3d" / model_name
        else:
            target_dir = self.models_dir / model_type / model_name
        
        # Check if already downloaded
        logger.info(f"Checking if model is complete at: {target_dir}")
        is_complete = self.check_model_complete(target_dir, model_type, model_name)
        logger.info(f"Model complete check: {is_complete}")
        
        if is_complete:
            return True, f"{model_name} is already downloaded"
        
        # Start download
        logger.info(f"Starting download with repo_id: {model_info.repo_id}")
        
        # Check if this is a GGUF model that needs specific files
        specific_files = None
        if model_name in GGUF_IMAGE_MODELS:
            model_config = GGUF_IMAGE_MODELS[model_name]
            if hasattr(model_config, 'gguf_file') and model_config.gguf_file:
                specific_files = [model_config.gguf_file, "README.md", "LICENSE.md"]
                logger.info(f"GGUF model detected, downloading specific files: {specific_files}")
        
        # Create a wrapper callback that logs progress
        def logging_progress_callback(progress):
            logger.debug(f"Download progress update: {progress.percentage}% - {progress.current_file}")
            if progress_callback:
                progress_callback(progress)
        
        return self.downloader.download_model(
            repo_id=model_info.repo_id,
            model_name=model_name,
            model_type=model_type,
            target_dir=target_dir,
            progress_callback=logging_progress_callback,
            specific_files=specific_files
        )
    
    def download_model_concurrent(
        self,
        model_name: str,
        model_type: str,
        progress_callback: Optional[Any] = None,
        priority: int = 0
    ) -> Tuple[bool, str, str]:
        """Download a model with concurrent support.
        
        Returns:
            Tuple of (success, message, download_id)
        """
        # Get model info
        all_models = self.get_available_models(model_type)
        if model_name not in all_models:
            return False, f"Unknown model: {model_name}", ""
        
        model_info = all_models[model_name]
        
        # Determine target directory
        if model_type == "image":
            target_dir = self.models_dir / "image" / model_name
        elif model_type == "3d":
            target_dir = self.models_dir / "3d" / model_name
        else:
            target_dir = self.models_dir / model_type / model_name
        
        # Check if already downloaded
        is_complete = self.check_model_complete(target_dir, model_type, model_name)
        if is_complete:
            return True, f"{model_name} is already downloaded", ""
        
        # Check if this is a GGUF model that needs specific files
        specific_files = None
        if model_name in GGUF_IMAGE_MODELS:
            model_config = GGUF_IMAGE_MODELS[model_name]
            if hasattr(model_config, 'gguf_file') and model_config.gguf_file:
                specific_files = [model_config.gguf_file, "README.md", "LICENSE.md"]
        
        # Use concurrent download
        return self.downloader.download_model_concurrent(
            repo_id=model_info.repo_id,
            model_name=model_name,
            model_type=model_type,
            target_dir=target_dir,
            progress_callback=progress_callback,
            specific_files=specific_files,
            priority=priority
        )
    
    def get_download_status(self) -> Dict[str, Any]:
        """Get status of all downloads."""
        return {
            'active': self.downloader.get_active_downloads(),
            'queued': self.downloader.get_download_queue(),
            'max_concurrent': self.downloader.max_concurrent_downloads
        }
    
    def load_model(self, model_name: str, **kwargs) -> Tuple[bool, str]:
        """Load a model by name.
        
        Args:
            model_name: Name of the model
            **kwargs: Additional loading arguments
            
        Returns:
            Tuple of (success, message)
        """
        # Determine model type
        all_models = self.get_available_models("all")
        if model_name not in all_models:
            return False, f"Unknown model: {model_name}"
        
        model_info = all_models[model_name]
        model_type = model_info.model_type
        
        # Get model path
        if "image" in model_type:
            # Get repo_id from model info
            repo_id = model_info.repo_id
            
            # Find the actual model path from the available locations
            possible_paths = [
                self.models_dir / "image" / model_name,
                self.models_dir / "flux_base" / model_name,
                self.models_dir / "flux_base" / f"models--{model_name.replace('/', '--')}",
                self.models_dir / "gguf" / model_name,
            ]
            
            # Add HuggingFace hub format using repo_id
            if "/" in repo_id:
                repo_name = repo_id.replace('/', '--')
                possible_paths.append(self.models_dir / "flux_base" / repo_name)
            
            # For GGUF models, also check directories by matching GGUF file content
            if model_name in GGUF_IMAGE_MODELS:
                gguf_dir = self.models_dir / "gguf"
                if gguf_dir.exists():
                    model_config = GGUF_IMAGE_MODELS[model_name]
                    expected_gguf = getattr(model_config, 'gguf_file', '')
                    
                    # Check all subdirectories for the expected GGUF file
                    for subdir in gguf_dir.iterdir():
                        if subdir.is_dir():
                            gguf_files = list(subdir.glob("*.gguf"))
                            for gguf_file in gguf_files:
                                if gguf_file.name == expected_gguf:
                                    possible_paths.insert(0, subdir)  # Add to beginning for priority
                                    break
            
            # Find the correct path
            model_path = None
            for path in possible_paths:
                if self.check_model_complete(path, model_type, model_name):
                    model_path = path
                    break
            
            if not model_path:
                return False, f"Model {model_name} not found in any expected location"
            
            # Load the model
            pipeline, message = self.loader.load_image_model(
                model_name=model_name,
                model_path=model_path,
                **kwargs
            )
            
            if pipeline:
                self.image_model = pipeline
                self.image_model_name = model_name
                return True, message
            else:
                return False, message
                
        elif model_type == "3d":
            # Load Hunyuan3D model
            model_path = self.models_dir / "3d" / model_name
            
            # Check if model is downloaded
            if not self.check_model_complete(model_path, model_type, model_name):
                return False, f"Model {model_name} not found or incomplete"
            
            try:
                # Filter out model_type from kwargs as it's not needed for load_hunyuan3d_model
                filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'model_type'}
                
                # Load Hunyuan3D model
                hunyuan3d_model, message = self.loader.load_hunyuan3d_model(
                    model_name=model_name,
                    model_path=model_path,
                    **filtered_kwargs
                )
                
                if hunyuan3d_model:
                    self.hunyuan3d_model = hunyuan3d_model
                    self.hunyuan3d_model_name = model_name
                    return True, message
                else:
                    return False, message
                    
            except Exception as e:
                logger.error(f"Failed to load 3D model {model_name}: {e}")
                return False, f"Error loading 3D model: {str(e)}"
        
        return False, f"Unsupported model type: {model_type}"
    
    def unload_model(self, model_type: str = "image") -> Tuple[bool, str]:
        """Unload a model.
        
        Args:
            model_type: Type of model to unload
            
        Returns:
            Tuple of (success, message)
        """
        if model_type == "image":
            success, message = self.loader.unload_model("image")
            if success:
                self.image_model = None
                self.image_model_name = None
            return success, message
        elif model_type == "3d":
            # 3D model unloading would be implemented here
            self.hunyuan3d_model = None
            self.hunyuan3d_model_name = None
            return True, "3D model unloaded"
        
        return False, f"Unknown model type: {model_type}"
    
    def load_image_model(self, model_name: str, current_model, current_model_name: str, device, progress=None):
        """Load an image model (compatibility method for studio.py).
        
        Args:
            model_name: Name of the model to load
            current_model: Currently loaded model (to check if reload needed)
            current_model_name: Name of currently loaded model
            device: Device to load model on
            progress: Progress callback
            
        Returns:
            Tuple of (status_message, loaded_model, model_name)
        """
        try:
            # Check if already loaded
            if current_model_name == model_name and current_model is not None:
                return "✅ Model already loaded", current_model, current_model_name
            
            # Check if model exists
            downloaded_models = self.get_downloaded_models("image")
            if model_name not in downloaded_models:
                return f"❌ Model {model_name} not found or not downloaded", None, None
            
            # Use the generic load_model method
            success, message = self.load_model(model_name, device=device, progress=progress)
            
            if success:
                # Return the loaded model
                return f"✅ {message}", self.image_model, model_name
            else:
                return f"❌ {message}", None, None
                
        except Exception as e:
            logger.error(f"Failed to load image model {model_name}: {e}")
            return f"❌ Error loading model: {str(e)}", None, None
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model information or None
        """
        all_models = self.get_available_models("all")
        if model_name in all_models:
            return all_models[model_name].to_dict()
        return None
    
    def get_download_progress(self) -> Dict[str, Any]:
        """Get current download progress."""
        if not self.downloader.is_downloading():
            return {"status": "idle"}
        
        progress = self.downloader.download_progress.to_dict()
        progress["status"] = "downloading"
        progress["model"] = self.downloader.current_download_model
        
        return progress
    
    def stop_download(self) -> str:
        """Stop current download."""
        self.downloader.stop_download()
        return "Download stopping..."
    
    def get_downloaded_models(self, model_type: str) -> List[str]:
        """Get list of downloaded models for a given type.
        
        Args:
            model_type: Type of models to check ("image", "3d", "video", etc.)
            
        Returns:
            List of downloaded model names
        """
        downloaded = []
        
        logger.debug(f"Checking for downloaded {model_type} models in {self.models_dir}")
        
        if model_type == "image":
            # Check both regular image models and GGUF models
            models_to_check = ALL_IMAGE_MODELS
            
            # Check standard image models in various directories
            for model_name in models_to_check.keys():
                model_config = models_to_check[model_name]
                
                # Get repo_id from config
                if hasattr(model_config, 'repo_id'):
                    repo_id = model_config.repo_id
                else:
                    repo_id = model_config.get("repo_id", model_name)
                
                # Try different possible locations for the models
                possible_paths = [
                    self.models_dir / "image" / model_name,
                    self.models_dir / "flux_base" / model_name,
                    self.models_dir / "flux_base" / f"models--{model_name.replace('/', '--')}",
                    self.models_dir / "gguf" / model_name,
                ]
                
                # Add HuggingFace hub format using repo_id
                if "/" in repo_id:
                    repo_name = repo_id.replace('/', '--')
                    possible_paths.append(self.models_dir / "flux_base" / repo_name)
                
                for model_path in possible_paths:
                    if self.check_model_complete(model_path, model_type, model_name):
                        downloaded.append(model_name)
                        break
            
            # Also check for GGUF models specifically
            gguf_dir = self.models_dir / "gguf"
            if gguf_dir.exists():
                for gguf_subdir in gguf_dir.iterdir():
                    if gguf_subdir.is_dir():
                        # Check if this directory contains GGUF files
                        gguf_files = list(gguf_subdir.glob("*.gguf"))
                        if gguf_files:
                            # Try to map this to a known model name
                            dir_name = gguf_subdir.name
                            # Look for matching model in GGUF_IMAGE_MODELS
                            matched = False
                            for model_name, model_config in GGUF_IMAGE_MODELS.items():
                                # More flexible matching - check if key parts match
                                # Extract quantization level from both names
                                import re
                                
                                # Extract Q level from model name (e.g., Q8, Q6_K, Q4_K_S)
                                model_q_match = re.search(r'Q\d+(?:_K)?(?:_[SM])?', model_name)
                                dir_q_match = re.search(r'Q\d+(?:_K)?(?:_[SM])?', dir_name)
                                
                                # Also check for model type (dev/schnell)
                                model_type = "schnell" if "schnell" in model_name.lower() else "dev"
                                dir_has_type = model_type in dir_name.lower()
                                
                                # Match if quantization levels match and model type matches
                                if (model_q_match and dir_q_match and 
                                    model_q_match.group() == dir_q_match.group() and
                                    dir_has_type):
                                    if model_name not in downloaded:
                                        downloaded.append(model_name)
                                    matched = True
                                    break
                                
                                # Also check if the GGUF filename matches what's expected
                                expected_gguf = getattr(model_config, 'gguf_file', '')
                                if expected_gguf:
                                    for gguf_file in gguf_files:
                                        if gguf_file.name == expected_gguf:
                                            if model_name not in downloaded:
                                                downloaded.append(model_name)
                                            matched = True
                                            break
                                if matched:
                                    break
                            
                            if not matched:
                                # If no exact match, add with a generic name
                                generic_name = f"GGUF-{dir_name}"
                                if generic_name not in downloaded:
                                    downloaded.append(generic_name)
            
            # Also check the models/image directory for any models that are downloaded but don't match exact names
            image_dir = self.models_dir / "image"
            if image_dir.exists():
                logger.debug(f"Checking image directory: {image_dir}")
                for subdir in image_dir.iterdir():
                    if subdir.is_dir():
                        logger.debug(f"Checking subdirectory: {subdir.name}")
                        # Check if this is a complete model
                        if self.check_model_complete(subdir, "image", subdir.name):
                            logger.debug(f"Model {subdir.name} is complete")
                            # Try to match with known models
                            matched = False
                            dir_name = subdir.name
                            
                            # Check for exact match first
                            if dir_name in models_to_check and dir_name not in downloaded:
                                downloaded.append(dir_name)
                                matched = True
                            else:
                                # Try flexible matching for FLUX models
                                for model_name in models_to_check.keys():
                                    # Direct mapping for known directory names
                                    dir_to_model_map = {
                                        "FLUX.1-dev": "FLUX.1-dev",
                                        "FLUX.1-dev-Q6": "FLUX.1-dev-Q6", 
                                        "FLUX.1-dev-Q8": "FLUX.1-dev-Q8",
                                        "flux1-dev": "FLUX.1-dev",
                                        "flux-dev": "FLUX.1-dev",
                                    }
                                    
                                    if dir_name in dir_to_model_map and dir_to_model_map[dir_name] == model_name:
                                        if model_name not in downloaded:
                                            downloaded.append(model_name)
                                        matched = True
                                        break
                                    
                                    # Handle FLUX model variations with regex
                                    if "FLUX" in model_name and "FLUX" in dir_name:
                                        # Check if it's the same quantization
                                        import re
                                        model_q = re.search(r'Q\d+(?:_K)?(?:_[SM])?', model_name)
                                        dir_q = re.search(r'Q\d+(?:_K)?(?:_[SM])?', dir_name)
                                        
                                        if model_q and dir_q and model_q.group() == dir_q.group():
                                            if model_name not in downloaded:
                                                downloaded.append(model_name)
                                            matched = True
                                            break
                                        elif not model_q and not dir_q:
                                            # Both are non-quantized FLUX models
                                            if ("dev" in model_name.lower() and "dev" in dir_name.lower()) or \
                                               ("schnell" in model_name.lower() and "schnell" in dir_name.lower()):
                                                if model_name not in downloaded:
                                                    downloaded.append(model_name)
                                                matched = True
                                                break
                                    
        elif model_type == "3d":
            models_to_check = HUNYUAN3D_MODELS
            
            # Check 3D models
            for model_name in models_to_check.keys():
                # Try different possible locations
                possible_paths = [
                    self.models_dir / "3d" / model_name,
                    self.models_dir / "3d" / f"models--tencent--{model_name}",
                ]
                
                logger.debug(f"Checking 3D model '{model_name}'...")
                for model_path in possible_paths:
                    logger.debug(f"  Trying path: {model_path}")
                    if self.check_model_complete(model_path, model_type, model_name):
                        logger.info(f"Found complete 3D model '{model_name}' at {model_path}")
                        downloaded.append(model_name)
                        break
                    else:
                        logger.debug(f"  Not complete or not found at {model_path}")
                        
        elif model_type == "video":
            # Import VIDEO_MODELS if it exists
            try:
                from ..config import VIDEO_MODELS
                models_to_check = VIDEO_MODELS
            except ImportError:
                models_to_check = {}
            
            for model_name in models_to_check.keys():
                model_path = self.models_dir / "video" / model_name
                if self.check_model_complete(model_path, model_type, model_name):
                    downloaded.append(model_name)
        else:
            # For other types, check the generic directory
            models_to_check = {}
            model_dir = self.models_dir / model_type
            
            for model_name in models_to_check.keys():
                model_path = model_dir / model_name
                if self.check_model_complete(model_path, model_type, model_name):
                    downloaded.append(model_name)
        
        return downloaded
    
    def get_storage_status(self) -> str:
        """Get storage status information as HTML.
        
        Returns:
            HTML string with storage information
        """
        import psutil
        
        try:
            # Get disk usage for models directory
            disk_usage = psutil.disk_usage(str(self.models_dir.parent))
            
            total_gb = disk_usage.total / (1024**3)
            used_gb = disk_usage.used / (1024**3)
            free_gb = disk_usage.free / (1024**3)
            
            # Calculate models directory size
            models_size = 0
            if self.models_dir.exists():
                for file_path in self.models_dir.rglob("*"):
                    if file_path.is_file():
                        models_size += file_path.stat().st_size
            models_size_gb = models_size / (1024**3)
            
            # Format the status
            status_html = f"""
            <div style='padding: 15px; background: #f8f9fa; border-radius: 8px; font-family: monospace;'>
                <h4>💾 Storage Status</h4>
                <div style='margin: 10px 0;'>
                    <strong>Disk Space:</strong><br>
                    • Total: {total_gb:.1f} GB<br>
                    • Used: {used_gb:.1f} GB<br>
                    • Free: {free_gb:.1f} GB
                </div>
                <div style='margin: 10px 0;'>
                    <strong>Models Storage:</strong><br>
                    • Models Size: {models_size_gb:.2f} GB<br>
                    • Models Path: {str(self.models_dir)}
                </div>
                <div style='margin: 10px 0;'>
                    <div style='background: #e9ecef; height: 20px; border-radius: 10px; overflow: hidden;'>
                        <div style='background: {"#28a745" if free_gb > 10 else "#ffc107" if free_gb > 5 else "#dc3545"}; height: 100%; width: {(used_gb/total_gb)*100:.1f}%; border-radius: 10px;'></div>
                    </div>
                    <small>Disk Usage: {(used_gb/total_gb)*100:.1f}%</small>
                </div>
            </div>
            """
            
            return status_html
            
        except Exception as e:
            return f"""
            <div style='padding: 15px; background: #f8d7da; border-radius: 8px;'>
                <h4>❌ Storage Status Error</h4>
                <p>Could not retrieve storage information: {str(e)}</p>
            </div>
            """
    
    def download_video_model(self, model_name: str):
        """Download a video model (compatibility method)."""
        # Use the concurrent download system
        success, message, download_id = self.download_model_concurrent(
            model_name=model_name,
            model_type="video",
            priority=1
        )
        
        # Yield single result for compatibility
        if success:
            yield f"<div style='color: #059669;'>{message}</div>"
        else:
            yield f"<div style='color: #dc3545;'>{message}</div>"
    
    def download_ip_adapter_model(self, model_name: str):
        """Download an IP adapter model (compatibility method)."""
        # Use the concurrent download system
        success, message, download_id = self.download_model_concurrent(
            model_name=model_name,
            model_type="ip_adapter",
            priority=1
        )
        
        # Yield single result for compatibility
        if success:
            yield f"<div style='color: #059669;'>{message}</div>"
        else:
            yield f"<div style='color: #dc3545;'>{message}</div>"
    
    def download_face_swap_model(self, model_name: str):
        """Download a face swap model (compatibility method)."""
        # Use the concurrent download system
        success, message, download_id = self.download_model_concurrent(
            model_name=model_name,
            model_type="face_swap",
            priority=1
        )
        
        # Yield single result for compatibility
        if success:
            yield f"<div style='color: #059669;'>{message}</div>"
        else:
            yield f"<div style='color: #dc3545;'>{message}</div>"
    
    def download_face_restore_model(self, model_name: str):
        """Download a face restore model (compatibility method)."""
        # Use the concurrent download system
        success, message, download_id = self.download_model_concurrent(
            model_name=model_name,
            model_type="face_restore",
            priority=1
        )
        
        # Yield single result for compatibility
        if success:
            yield f"<div style='color: #059669;'>{message}</div>"
        else:
            yield f"<div style='color: #dc3545;'>{message}</div>"
    
    def download_texture_component(
        self,
        component_name: str,
        component_info: Dict[str, Any],
        priority: int = 1
    ) -> Tuple[bool, str, str]:
        """Download a texture pipeline component.
        
        Args:
            component_name: Name of the component
            component_info: Component configuration dict
            priority: Download priority
            
        Returns:
            Tuple of (success, message, download_id)
        """
        component_type = component_info.get('type', 'model')
        
        if component_type == 'dependency':
            # Handle pip dependencies
            pip_package = component_info.get('pip_package', '')
            if not pip_package:
                return False, f"No pip package specified for {component_name}", ""
            
            # Install pip package
            import subprocess
            try:
                result = subprocess.run(
                    ["pip", "install", pip_package],
                    capture_output=True,
                    text=True,
                    check=True
                )
                return True, f"Successfully installed {pip_package}", ""
            except subprocess.CalledProcessError as e:
                return False, f"Failed to install {pip_package}: {e.stderr}", ""
        
        elif component_type == 'model':
            # Handle model downloads
            if 'url' in component_info:
                # Direct URL download
                target_dir = self.models_dir / "texture_components" / component_name
                
                # Check if already downloaded
                install_path = component_info.get('install_path', '')
                if install_path:
                    check_path = Path(install_path)
                    if check_path.exists():
                        return True, f"{component_name} already downloaded", ""
                
                # Use concurrent download with direct URL
                return self.downloader.download_model_concurrent(
                    repo_id="direct",
                    model_name=component_name,
                    model_type="texture_component",
                    target_dir=target_dir,
                    priority=priority,
                    direct_url=component_info['url']
                )
                
            elif 'repo_id' in component_info:
                # HuggingFace model download
                target_dir = self.models_dir / "texture_components" / component_name
                
                return self.downloader.download_model_concurrent(
                    repo_id=component_info['repo_id'],
                    model_name=component_name,
                    model_type="texture_component",
                    target_dir=target_dir,
                    priority=priority
                )
            else:
                return False, f"No download URL or repo_id for {component_name}", ""
        
        return False, f"Unknown component type: {component_type}", ""