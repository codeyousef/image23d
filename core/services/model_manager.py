"""
Core model management service that wraps the existing model manager
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import asyncio

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Wrapper around the existing Hunyuan3D model manager
    Provides a clean interface for both desktop and web apps
    """
    
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self._model_manager = None
        self._initialized = False
        
    async def initialize(self):
        """Initialize the underlying model manager"""
        if self._initialized:
            return
            
        try:
            # Import the existing model manager
            from src.hunyuan3d_app.models.manager import ModelManager as HunyuanModelManager
            from pathlib import Path
            
            # Create instance of existing manager
            src_models_path = Path(__file__).parent.parent.parent / "src" / "hunyuan3d_app" / "models"
            self._model_manager = HunyuanModelManager(
                self.models_dir,
                src_models_path
            )
            
            self._initialized = True
            logger.info("Model manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize model manager: {e}")
            raise
            
    def _ensure_initialized(self):
        """Ensure the model manager is initialized"""
        if not self._initialized:
            raise RuntimeError("Model manager not initialized. Call initialize() first.")
            
    async def load_image_model(self, model_id: str, **kwargs):
        """Load an image generation model"""
        self._ensure_initialized()
        
        # Use the existing model manager's loading logic
        return await asyncio.to_thread(
            self._model_manager.load_image_model,
            model_id,
            **kwargs
        )
        
    async def load_3d_model(self, model_id: str, **kwargs):
        """Load a 3D generation model"""
        self._ensure_initialized()
        
        # Use the existing model manager's loading logic
        return await asyncio.to_thread(
            self._model_manager.load_3d_model,
            model_id,
            **kwargs
        )
        
    async def load_video_model(self, model_id: str, **kwargs):
        """Load a video generation model"""
        self._ensure_initialized()
        
        # Use the existing model manager's loading logic
        return await asyncio.to_thread(
            self._model_manager.load_video_model,
            model_id,
            **kwargs
        )
        
    def is_model_available(self, model_id: str) -> bool:
        """Check if a model is available (downloaded)"""
        self._ensure_initialized()
        
        # Try the wrapped manager first if method exists
        if hasattr(self._model_manager, 'is_model_available'):
            try:
                return self._model_manager.is_model_available(model_id)
            except Exception as e:
                logger.debug(f"Wrapped manager check failed for {model_id}: {e}")
        
        # Fallback to filesystem check
        return self._check_model_filesystem(model_id)
        
    def is_model_loaded(self, model_id: str) -> bool:
        """Check if a model is currently loaded in memory"""
        self._ensure_initialized()
        
        # Try the wrapped manager first if method exists
        if hasattr(self._model_manager, 'is_loaded'):
            try:
                return self._model_manager.is_loaded(model_id)
            except Exception as e:
                logger.debug(f"Wrapped manager loaded check failed for {model_id}: {e}")
        
        # Fallback - check for loaded models in different ways
        if hasattr(self._model_manager, '_loaded_models'):
            return model_id in self._model_manager._loaded_models
        elif hasattr(self._model_manager, 'image_model_name'):
            return (self._model_manager.image_model_name == model_id or
                    self._model_manager.hunyuan3d_model_name == model_id)
        
        return False
        
    async def download_model(self, model_id: str, progress_callback=None):
        """Download a model if not already available"""
        self._ensure_initialized()
        
        return await asyncio.to_thread(
            self._model_manager.download_model,
            model_id,
            progress_callback
        )
        
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a model"""
        self._ensure_initialized()
        return self._model_manager.get_model_info(model_id)
        
    def list_available_models(self, model_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all available models, optionally filtered by type"""
        self._ensure_initialized()
        
        models = []
        
        # Get models from existing config
        if model_type in [None, "image"]:
            try:
                from src.hunyuan3d_app.config import IMAGE_MODELS, GGUF_IMAGE_MODELS
                for model_id, config in {**IMAGE_MODELS, **GGUF_IMAGE_MODELS}.items():
                    models.append({
                        "id": model_id,
                        "type": "image",
                        "name": config.name,
                        "description": config.description,
                        "size": config.size,
                        "vram_required": config.vram_required,
                        "is_available": self.is_model_available(model_id),
                        "is_loaded": self.is_model_loaded(model_id)
                    })
            except ImportError as e:
                logger.warning(f"Failed to import image models config: {e}")
                
        if model_type in [None, "3d"]:
            try:
                # Try src config first, then core config
                try:
                    from src.hunyuan3d_app.config import ALL_3D_MODELS
                except ImportError:
                    from core.config import ALL_3D_MODELS
                    
                for model_id, config in ALL_3D_MODELS.items():
                    models.append({
                        "id": model_id,
                        "type": "3d",
                        "name": config["name"],
                        "description": config["description"],
                        "size": config["size"],
                        "vram_required": config["vram_required"],
                        "is_available": self.is_model_available(model_id),
                        "is_loaded": self.is_model_loaded(model_id)
                    })
            except ImportError as e:
                logger.warning(f"Failed to import 3D models config: {e}")
                
        if model_type in [None, "video"]:
            try:
                from src.hunyuan3d_app.config import VIDEO_MODELS
                for model_id, config in VIDEO_MODELS.items():
                    models.append({
                        "id": model_id,
                        "type": "video",
                        "name": config["name"],
                        "description": config["description"],
                        "size": config["size"],
                        "vram_required": config["vram_required"],
                        "is_available": self.is_model_available(model_id),
                        "is_loaded": self.is_model_loaded(model_id)
                    })
            except ImportError as e:
                logger.warning(f"Failed to import video models config: {e}")
                
        return models
        
    def get_downloaded_models(self, model_type: str) -> list:
        """Get list of downloaded models by type"""
        self._ensure_initialized()
        
        # Use the wrapped manager's method
        if hasattr(self._model_manager, 'get_downloaded_models'):
            return self._model_manager.get_downloaded_models(model_type)
        
        # Fallback - check filesystem
        downloaded = []
        if model_type == "image":
            # Check for image models in various locations
            for model_dir in [self.models_dir / "image", self.models_dir / "gguf"]:
                if model_dir.exists():
                    for item in model_dir.iterdir():
                        if item.is_dir() and self._check_model_complete(item):
                            downloaded.append(item.name)
        elif model_type == "3d":
            model_dir = self.models_dir / "3d"
            if model_dir.exists():
                for item in model_dir.iterdir():
                    if item.is_dir() and self._check_model_complete(item):
                        downloaded.append(item.name)
                        
        return downloaded
    
    def _check_model_complete(self, model_path: Path) -> bool:
        """Check if a model directory contains a complete model"""
        # Check for common model files
        has_model_files = (
            any(model_path.glob("*.safetensors")) or
            any(model_path.glob("*.bin")) or
            any(model_path.glob("*.gguf")) or
            any(model_path.glob("*.pth")) or
            any(model_path.glob("*.ckpt"))
        )
        
        # Check for config files
        has_config = (
            (model_path / "config.json").exists() or
            (model_path / "model_index.json").exists()
        )
        
        # Check for model subdirectories (for diffusers models)
        has_subdirs = any((model_path / subdir).exists() for subdir in ["transformer", "vae", "text_encoder"])
        
        return has_model_files or has_config or has_subdirs
    
    def get_processor_for_model(self, model_id: str, model_type: str, output_dir: Path, prompt_enhancer=None):
        """Get the appropriate processor for a model"""
        if model_type == "3d":
            if "sparc3d" in model_id.lower():
                from ..processors.sparc3d_processor import Sparc3DProcessor
                return Sparc3DProcessor(self, output_dir, prompt_enhancer)
            elif "hi3dgen" in model_id.lower():
                from ..processors.hi3dgen_processor import Hi3DGenProcessor
                return Hi3DGenProcessor(self, output_dir, prompt_enhancer)
            else:
                # Default to standard 3D processor for HunYuan3D models
                from ..processors.threed_processor import ThreeDProcessor
                return ThreeDProcessor(self, output_dir, prompt_enhancer)
        elif model_type == "image":
            from ..processors.image_processor import ImageProcessor
            return ImageProcessor(self, output_dir, prompt_enhancer)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
    def unload_model(self, model_id: str):
        """Unload a model from memory"""
        self._ensure_initialized()
        return self._model_manager.unload_model(model_id)
        
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics"""
        self._ensure_initialized()
        
        import torch
        import psutil
        
        # GPU memory
        gpu_stats = {}
        if torch.cuda.is_available():
            gpu_stats = {
                "allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
                "reserved": torch.cuda.memory_reserved() / 1024**3,    # GB
                "total": torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            }
            
        # System memory
        memory = psutil.virtual_memory()
        system_stats = {
            "used": memory.used / 1024**3,      # GB
            "available": memory.available / 1024**3,  # GB
            "total": memory.total / 1024**3,     # GB
            "percent": memory.percent
        }
        
        return {
            "gpu": gpu_stats,
            "system": system_stats,
            "loaded_models": list(self._model_manager._loaded_models.keys()) if hasattr(self._model_manager, '_loaded_models') else []
        }
    
    def _check_model_filesystem(self, model_id: str) -> bool:
        """Check if model files exist on filesystem"""
        try:
            # Check different model types and their expected directory structures
            if "flux" in model_id.lower() or "image" in model_id.lower():
                return self._check_image_model_filesystem(model_id)
            elif "3d" in model_id.lower() or "hunyuan3d" in model_id.lower():
                return self._check_3d_model_filesystem(model_id)
            else:
                # Generic check - look for any subdirectory with the model name
                model_dir = self.models_dir / model_id
                return model_dir.exists() and model_dir.is_dir()
                
        except Exception as e:
            logger.debug(f"Filesystem check failed for {model_id}: {e}")
            return False
    
    def _check_image_model_filesystem(self, model_id: str) -> bool:
        """Check if image model files exist"""
        # Common locations for image models
        possible_paths = [
            self.models_dir / "image" / model_id,
            self.models_dir / "image" / model_id.replace("-", "_"),
            self.models_dir / "gguf" / model_id,
            self.models_dir / model_id
        ]
        
        for model_path in possible_paths:
            if not model_path.exists():
                continue
                
            # Check for common model files
            expected_files = [
                "config.json",
                "model.safetensors", 
                "pytorch_model.bin",
                f"{model_id}.gguf",
                "model_index.json",
                # Add specific GGUF model names
                "flux1-dev-Q8_0.gguf",
                "flux1-dev-Q6_K.gguf",
                "flux1-dev-Q4_K_S.gguf",
                "flux1-dev.safetensors"
            ]
            
            # If any expected file exists, consider model available
            for file in expected_files:
                if (model_path / file).exists():
                    logger.debug(f"Found {model_id} at {model_path} with file {file}")
                    return True
            
            # Check for subdirectories with model files
            if any(p.is_dir() for p in model_path.iterdir() if p.name in ['transformer', 'vae', 'text_encoder']):
                logger.debug(f"Found {model_id} at {model_path} with component directories")
                return True
            
            # Also check for any .gguf file in the directory
            gguf_files = list(model_path.glob("*.gguf"))
            if gguf_files:
                logger.debug(f"Found {model_id} at {model_path} with GGUF file: {gguf_files[0].name}")
                return True
        
        return False
    
    def _check_3d_model_filesystem(self, model_id: str) -> bool:
        """Check if 3D model files exist"""
        # Common locations for 3D models  
        possible_paths = [
            self.models_dir / "3d" / model_id,
            self.models_dir / "3d" / model_id.replace("-", "_"),
            self.models_dir / model_id
        ]
        
        for model_path in possible_paths:
            if not model_path.exists():
                continue
                
            # Check for common 3D model files
            expected_files = [
                "config.yaml",
                "model.fp16.ckpt", 
                "model.ckpt",
                "model_index.json"
            ]
            
            # If any expected file exists, consider model available
            for file in expected_files:
                if (model_path / file).exists():
                    logger.debug(f"Found 3D model {model_id} at {model_path} with file {file}")
                    return True
            
            # Check for subdirectories with model components
            expected_dirs = ['hunyuan3d-dit-v2-1', 'hunyuan3d-vae-v2-1', 'hunyuan3d-paintpbr-v2-1']
            if any((model_path / d).exists() for d in expected_dirs):
                logger.debug(f"Found 3D model {model_id} at {model_path} with component directories")
                return True
        
        return False