"""Compatibility Mixin

Provides backward compatibility methods for legacy code.
"""

import logging
from typing import Optional, Tuple, Any

logger = logging.getLogger(__name__)


class CompatibilityMixin:
    """Mixin for backward compatibility methods"""
    
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
    
    def load_hunyuan3d_model(self, model_name: str, current_model, current_model_name: str, device, progress=None):
        """Load a Hunyuan3D model (compatibility method for studio.py).
        
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
            downloaded_models = self.get_downloaded_models("3d")
            if model_name not in downloaded_models:
                return f"❌ Model {model_name} not found or not downloaded", None, None
            
            # Use the generic load_model method
            success, message = self.load_model(model_name, device=device, progress=progress)
            
            if success:
                # Return the loaded model
                return f"✅ {message}", self.hunyuan3d_model, model_name
            else:
                return f"❌ {message}", None, None
                
        except Exception as e:
            logger.error(f"Failed to load 3D model {model_name}: {e}")
            return f"❌ Error loading model: {str(e)}", None, None
    
    def load_3d_model(self, model_name: str, current_model, current_model_name: str, device, progress=None):
        """Alias for load_hunyuan3d_model for backward compatibility."""
        return self.load_hunyuan3d_model(model_name, current_model, current_model_name, device, progress)
    
    def unload_image_model(self):
        """Unload the image model (compatibility method)."""
        success, message = self.unload_model("image")
        if not success:
            logger.error(f"Failed to unload image model: {message}")
    
    def unload_3d_model(self):
        """Unload the 3D model (compatibility method)."""
        success, message = self.unload_model("3d")
        if not success:
            logger.error(f"Failed to unload 3D model: {message}")
    
    def unload_models(self):
        """Unload all models (compatibility method)."""
        self.unload_image_model()
        self.unload_3d_model()
    
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