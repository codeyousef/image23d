"""Model Discovery Mixin

Handles checking for models, model completeness, and discovering available models.
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    from ...utils.fs_cache import CachedPath, cached_exists, cached_iterdir
    USE_FS_CACHE = True
except ImportError:
    USE_FS_CACHE = False

from ...config import (
    IMAGE_MODELS, GATED_IMAGE_MODELS, GGUF_IMAGE_MODELS,
    ALL_IMAGE_MODELS, HUNYUAN3D_MODELS
)
try:
    # Try to use lazy loading if available
    from ...config.lazy_loader import (
        IMAGE_MODELS_LAZY, GATED_IMAGE_MODELS_LAZY, 
        GGUF_IMAGE_MODELS_LAZY, HUNYUAN3D_MODELS_LAZY,
        lazy_config_manager
    )
    USE_LAZY_LOADING = True
except ImportError:
    USE_LAZY_LOADING = False
from ..base import ModelInfo

logger = logging.getLogger(__name__)


class ModelDiscoveryMixin:
    """Mixin for model discovery operations"""
    
    @lru_cache(maxsize=32)
    def get_available_models(self, model_type: str = "all") -> Dict[str, ModelInfo]:
        """Get all available models by type.
        
        Args:
            model_type: Type of models to get ("image", "3d", "threed", "sparc3d", "hi3dgen", "all")
            
        Returns:
            Dictionary of model name to ModelInfo
        """
        available = {}
        
        # Handle model type aliases
        if model_type == "threed":
            model_type = "3d"
        
        # Handle sparc3d and hi3dgen as separate model types (return empty for now)
        if model_type == "sparc3d":
            return {}  # No Sparc3D models configured yet
        if model_type == "hi3dgen":
            return {}  # No Hi3DGen models configured yet
        
        if model_type in ["image", "all"]:
            # Use lazy loading if available
            image_models = IMAGE_MODELS_LAZY if USE_LAZY_LOADING else IMAGE_MODELS
            
            # Add standard image models
            for name, config in image_models.items():
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
            gated_models = GATED_IMAGE_MODELS_LAZY if USE_LAZY_LOADING else GATED_IMAGE_MODELS
            for name, config in gated_models.items():
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
            gguf_models = GGUF_IMAGE_MODELS_LAZY if USE_LAZY_LOADING else GGUF_IMAGE_MODELS
            for name, config in gguf_models.items():
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
            hunyuan3d_models = HUNYUAN3D_MODELS_LAZY if USE_LAZY_LOADING else HUNYUAN3D_MODELS
            for name, config in hunyuan3d_models.items():
                # Handle both dict and ThreeDModelConfig objects
                if hasattr(config, 'repo_id'):
                    # It's a ThreeDModelConfig object
                    # Extract size in GB from the size string (e.g., "15GB" -> 15.0)
                    size_gb = 0.0
                    if hasattr(config, 'size') and config.size:
                        size_str = config.size.replace('GB', '').replace('gb', '').strip()
                        try:
                            size_gb = float(size_str)
                        except ValueError:
                            size_gb = 0.0
                    
                    available[name] = ModelInfo(
                        name=config.name,
                        repo_id=config.repo_id,
                        model_type="3d",
                        size_gb=size_gb,
                        requires_auth=False,
                        description=config.description,
                        tags=getattr(config, 'tags', ["3d"]),
                        recommended_vram_gb=12.0  # Default value since ThreeDModelConfig doesn't have this
                    )
                else:
                    # It's a dictionary
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
        # Use cached exists check if available
        exists = cached_exists(model_path) if USE_FS_CACHE else model_path.exists()
        if not exists:
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
            self._check_gguf_models(downloaded)
            
            # Also check the models/image directory for any models that are downloaded but don't match exact names
            self._check_image_directory(downloaded, models_to_check)
                    
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
                from ...config import VIDEO_MODELS
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
    
    def _check_gguf_models(self, downloaded: List[str]) -> None:
        """Check for GGUF models in the gguf directory."""
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
    
    def _check_image_directory(self, downloaded: List[str], models_to_check: Dict[str, Any]) -> None:
        """Check the image directory for downloaded models."""
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
    
    @lru_cache(maxsize=128)
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