"""Model Operations Mixin

Handles model loading, unloading, downloading, and related operations.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Any, Dict

import torch

from ...config import GGUF_IMAGE_MODELS

logger = logging.getLogger(__name__)


class ModelOperationsMixin:
    """Mixin for model-related operations"""
    
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
    
    def unload_model(self) -> Tuple[bool, str]:
        """Unload the currently loaded model.
        
        Returns:
            Tuple of (success, message)
        """
        # Default to unloading image model for backward compatibility
        return self.unload_model_by_type("image")
    
    def unload_model_by_type(self, model_type: str = "image") -> Tuple[bool, str]:
        """Unload a model by type.
        
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
    
    def get_download_status(self) -> Dict[str, Any]:
        """Get status of all downloads."""
        return {
            'active': self.downloader.get_active_downloads(),
            'queued': self.downloader.get_download_queue(),
            'max_concurrent': self.downloader.max_concurrent_downloads
        }
    
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