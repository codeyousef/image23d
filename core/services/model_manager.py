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
            from hunyuan3d_app.models.manager import ModelManager as HunyuanModelManager
            from pathlib import Path
            
            # Create instance of existing manager
            src_models_path = Path(__file__).parent.parent.parent / "src" / "hunyuan3d_app" / "models"
            self._model_manager = HunyuanModelManager(
                self.models_dir,
                src_models_path
            )
            
            self._initialized = True
            logger.info("Model manager initialized successfully")
            
        except ImportError as e:
            logger.warning(f"Hunyuan3D models not available: {e}")
            logger.info("Using mock model manager for desktop interface")
            # Use a mock implementation
            self._model_manager = MockModelManager()
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize model manager: {e}")
            # Use mock but still log the error
            self._model_manager = MockModelManager()
            self._initialized = True
            
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
    
    def get_available_models(self, model_type: str = "all") -> Dict[str, Any]:
        """Get available models as a dictionary (for compatibility with desktop app)"""
        self._ensure_initialized()
        
        logger.debug(f"get_available_models called with model_type: {model_type}")
        
        # Delegate to the wrapped manager if it has the method
        if hasattr(self._model_manager, 'get_available_models'):
            model_infos = self._model_manager.get_available_models(model_type)
            logger.debug(f"Wrapped manager returned {len(model_infos)} models for type {model_type}")
            
            # Convert ModelInfo objects to dicts if needed
            models_dict = {}
            for model_id, model_info in model_infos.items():
                if hasattr(model_info, 'to_dict'):
                    # It's a ModelInfo object
                    info_dict = model_info.to_dict()
                    
                    # Filter by model type - only include models that match the requested type
                    actual_model_type = info_dict.get('model_type', '')
                    
                    # Handle type aliases
                    if model_type == "threed":
                        requested_type = "3d"
                    else:
                        requested_type = model_type
                    
                    # Skip models that don't match the requested type (unless requesting "all")
                    if model_type != "all" and actual_model_type != requested_type:
                        logger.debug(f"Filtering out model {model_id} with type {actual_model_type} (requested {requested_type})")
                        continue
                    
                    # Add extra fields that desktop app expects
                    info_dict['is_available'] = self.is_model_available(model_id)
                    info_dict['is_loaded'] = self.is_model_loaded(model_id)
                    models_dict[model_id] = info_dict
                else:
                    # It's already a dict
                    # Apply same filtering for dict models
                    actual_model_type = model_info.get('model_type', model_info.get('type', ''))
                    requested_type = "3d" if model_type == "threed" else model_type
                    
                    if model_type != "all" and actual_model_type != requested_type:
                        logger.debug(f"Filtering out dict model {model_id} with type {actual_model_type} (requested {requested_type})")
                        continue
                        
                    models_dict[model_id] = model_info
                    
            return models_dict
        
        # Otherwise, convert list_available_models output to dict format
        logger.debug(f"Falling back to list_available_models for type {model_type}")
        # Don't convert "threed" to None - pass it through
        list_model_type = None if model_type == "all" else model_type
        models_list = self.list_available_models(list_model_type)
        models_dict = {}
        
        for model in models_list:
            model_id = model.get('id')
            if model_id:
                # Convert to ModelInfo-like dict
                models_dict[model_id] = {
                    'name': model.get('name', model_id),
                    'description': model.get('description', ''),
                    'size': model.get('size', 'Unknown'),
                    'vram_required': model.get('vram_required', 'Unknown'),
                    'type': model.get('type', 'unknown'),
                    'repo_id': model.get('repo_id', model_id),
                    'is_available': model.get('is_available', False),
                    'is_loaded': model.get('is_loaded', False)
                }
        
        return models_dict
        
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
                
        # Handle various 3D model type aliases  
        if model_type in [None, "3d", "threed"]:
            try:
                # Try src config first, then core config
                try:
                    from src.hunyuan3d_app.config import ALL_3D_MODELS, HUNYUAN3D_MODELS
                    # Use HUNYUAN3D_MODELS if available, otherwise ALL_3D_MODELS
                    models_config = HUNYUAN3D_MODELS if 'HUNYUAN3D_MODELS' in locals() else ALL_3D_MODELS
                except ImportError:
                    try:
                        from core.config import ALL_3D_MODELS
                        models_config = ALL_3D_MODELS
                    except ImportError:
                        # Fallback to hardcoded models
                        models_config = {
                            "hunyuan3d-21": {
                                "name": "HunYuan3D 2.1",
                                "description": "Latest HunYuan3D model with improved quality",
                                "size": "15GB",
                                "vram_required": "16GB"
                            },
                            "hunyuan3d-20": {
                                "name": "HunYuan3D 2.0", 
                                "description": "Stable HunYuan3D model",
                                "size": "12GB",
                                "vram_required": "12GB"
                            },
                            "hunyuan3d-2mini": {
                                "name": "HunYuan3D 2 Mini",
                                "description": "Lightweight HunYuan3D model",
                                "size": "8GB", 
                                "vram_required": "8GB"
                            }
                        }
                        
                for model_id, config in models_config.items():
                    # Handle both dict and object configs
                    if hasattr(config, 'name'):
                        # It's a ThreeDModelConfig object
                        model_info = {
                            "id": model_id,
                            "type": "3d",
                            "name": config.name,
                            "description": config.description,
                            "size": config.size,
                            "vram_required": config.vram_required,
                            "is_available": self.is_model_available(model_id),
                            "is_loaded": self.is_model_loaded(model_id)
                        }
                    else:
                        # It's a dictionary
                        model_info = {
                            "id": model_id,
                            "type": "3d",
                            "name": config.get("name", model_id),
                            "description": config.get("description", ""),
                            "size": config.get("size", "Unknown"),
                            "vram_required": config.get("vram_required", "Unknown"),
                            "is_available": self.is_model_available(model_id),
                            "is_loaded": self.is_model_loaded(model_id)
                        }
                    models.append(model_info)
                    
            except Exception as e:
                logger.warning(f"Failed to import 3D models config: {e}")
                
        # Handle Sparc3D models
        if model_type in ["sparc3d"]:
            # Add Sparc3D specific models here if needed
            pass
            
        # Handle Hi3DGen models  
        if model_type in ["hi3dgen"]:
            # Add Hi3DGen specific models here if needed
            pass
                
        if model_type in [None, "video"]:
            try:
                from src.hunyuan3d_app.config import VIDEO_MODELS
                for model_id, config in VIDEO_MODELS.items():
                    # Handle both dict and object configs
                    if hasattr(config, 'name'):
                        # It's a VideoModelConfig object
                        models.append({
                            "id": model_id,
                            "type": "video",
                            "name": config.name,
                            "description": config.description,
                            "size": config.size,
                            "vram_required": config.vram_required,
                            "is_available": self.is_model_available(model_id),
                            "is_loaded": self.is_model_loaded(model_id)
                        })
                    else:
                        # It's a dictionary
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
        # Determine model type from model_id and use the appropriate method
        if hasattr(self._model_manager, 'unload_model_by_type'):
            # Try to determine model type
            if "flux" in model_id.lower() or "sd" in model_id.lower():
                return self._model_manager.unload_model_by_type("image")
            elif "3d" in model_id.lower() or "hunyuan3d" in model_id.lower():
                return self._model_manager.unload_model_by_type("3d")
            else:
                # Default to image
                return self._model_manager.unload_model_by_type("image")
        else:
            # Fallback to parameterless unload_model
            return self._model_manager.unload_model()
        
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


class MockModelManager:
    """Mock model manager for when HunYuan3D models are not available"""
    
    def __init__(self):
        self._loaded_models = {}
        
    def get_available_models(self, model_type=None):
        """Return mock models"""
        if model_type == "3d" or model_type == "threed":
            return {
                "hunyuan3d-21": {
                    "name": "HunYuan3D 2.1",
                    "description": "Latest HunYuan3D model (Mock)",
                    "size": "15GB",
                    "vram_required": "16GB",
                    "model_type": "3d",
                    "is_available": False,
                    "is_loaded": False
                },
                "hunyuan3d-2mini": {
                    "name": "HunYuan3D 2 Mini", 
                    "description": "Lightweight HunYuan3D model (Mock)",
                    "size": "8GB",
                    "vram_required": "8GB",
                    "model_type": "3d",
                    "is_available": False,
                    "is_loaded": False
                }
            }
        elif model_type == "image":
            return {
                "flux-1-dev": {
                    "name": "FLUX.1 Dev",
                    "description": "FLUX.1 Development model (Mock)",
                    "size": "25GB",
                    "vram_required": "24GB",
                    "model_type": "image",
                    "is_available": False,
                    "is_loaded": False
                }
            }
        return {}
        
    def is_model_available(self, model_id):
        return False
        
    def is_model_loaded(self, model_id):
        return model_id in self._loaded_models
        
    def get_downloaded_models(self, model_type):
        return []
        
    def unload_model(self):
        self._loaded_models.clear()
        
    def download_model(self, model_id, model_type):
        logger.info(f"Mock download of {model_id}")
        return True
