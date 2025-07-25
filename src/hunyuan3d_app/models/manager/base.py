"""Base Model Manager

Core model management functionality combining all components.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import torch

from ..base import BaseModelManager, ModelInfo
from ..download import ModelDownloader
from ..loading import ModelLoader
from ...config import (
    IMAGE_MODELS, GATED_IMAGE_MODELS, GGUF_IMAGE_MODELS,
    ALL_IMAGE_MODELS, HUNYUAN3D_MODELS
)
from ..gguf import GGUFModelManager
from ...services.websocket import ProgressStreamManager

from .model_discovery import ModelDiscoveryMixin
from .model_operations import ModelOperationsMixin
from .storage_management import StorageManagementMixin
from .compatibility import CompatibilityMixin

logger = logging.getLogger(__name__)


class ModelManager(
    ModelDiscoveryMixin,
    ModelOperationsMixin,
    StorageManagementMixin,
    CompatibilityMixin,
    BaseModelManager
):
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
        from ...utils import get_hf_token_from_all_sources, validate_hf_token
        
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
        from ...utils import validate_hf_token
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
                from ...utils import save_hf_token
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