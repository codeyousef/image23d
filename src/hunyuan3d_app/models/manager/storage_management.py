"""Storage Management Mixin

Handles storage status, model verification, and cleanup operations.
"""

import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


class StorageManagementMixin:
    """Mixin for storage-related operations"""
    
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
    
    def check_missing_components(self, model_type: str, model_name: str) -> list:
        """Check for missing components required for optimized model usage.
        
        Args:
            model_type: Type of model
            model_name: Name of model
            
        Returns:
            List of missing component names
        """
        missing = []
        
        # Check if the model itself is downloaded
        model_path = self._get_model_path(model_type, model_name)
        if not self.check_model_complete(model_path, model_type, model_name):
            missing.append("complete model")
            return missing  # If the model isn't downloaded, no point checking components
        
        # For FLUX models, check for required components
        if model_name.startswith("FLUX") and model_type == "image":
            # Check for VAE
            vae_path = self.models_dir / "vae" / "ae.safetensors"
            if not vae_path.exists():
                missing.append("FLUX VAE")
            
            # Check for text encoders
            t5_path = self.models_dir / "text_encoders" / "t5xxl_fp16.safetensors"
            clip_path = self.models_dir / "text_encoders" / "clip_l.safetensors"
            
            if not t5_path.exists():
                missing.append("T5XXL Text Encoder")
            if not clip_path.exists():
                missing.append("CLIP-L Text Encoder")
        
        # For GGUF models, check if quantization file exists
        if model_name in self._get_gguf_models():
            model_config = self._get_gguf_models()[model_name]
            expected_gguf = getattr(model_config, 'gguf_file', '')
            if expected_gguf:
                # Find the model directory
                gguf_found = False
                for subdir in model_path.rglob("*.gguf"):
                    if subdir.name == expected_gguf:
                        gguf_found = True
                        break
                if not gguf_found:
                    missing.append(f"GGUF file: {expected_gguf}")
        
        return missing
    
    def _get_model_path(self, model_type: str, model_name: str) -> Path:
        """Get the expected path for a model.
        
        Args:
            model_type: Type of model
            model_name: Name of model
            
        Returns:
            Expected model path
        """
        if model_type == "image":
            # Try multiple possible locations
            possible_paths = [
                self.models_dir / "image" / model_name,
                self.models_dir / "flux_base" / model_name,
                self.models_dir / "gguf" / model_name,
            ]
            
            # Check which path exists
            for path in possible_paths:
                if path.exists():
                    return path
            
            # Return default if none exist
            return self.models_dir / "image" / model_name
            
        elif model_type == "3d":
            return self.models_dir / "3d" / model_name
        else:
            return self.models_dir / model_type / model_name
    
    def _get_gguf_models(self) -> Dict[str, Any]:
        """Get GGUF models configuration.
        
        Returns:
            GGUF models dict
        """
        try:
            from ...config import GGUF_IMAGE_MODELS
            return GGUF_IMAGE_MODELS
        except ImportError:
            return {}
    
    def cleanup_orphaned_files(self) -> Dict[str, Any]:
        """Clean up orphaned model files and directories.
        
        Returns:
            Dictionary with cleanup results
        """
        import shutil
        
        cleaned_files = []
        cleaned_dirs = []
        total_freed = 0
        
        try:
            # Check for incomplete downloads (files ending with .tmp, .part, etc.)
            temp_patterns = ["*.tmp", "*.part", "*.download", "*.partial"]
            
            for pattern in temp_patterns:
                for temp_file in self.models_dir.rglob(pattern):
                    if temp_file.is_file():
                        size = temp_file.stat().st_size
                        temp_file.unlink()
                        cleaned_files.append(str(temp_file))
                        total_freed += size
            
            # Check for empty model directories
            for model_type_dir in self.models_dir.iterdir():
                if model_type_dir.is_dir():
                    for model_dir in model_type_dir.iterdir():
                        if model_dir.is_dir():
                            # Check if directory is effectively empty (only hidden files or small metadata)
                            contents = list(model_dir.rglob("*"))
                            if not contents or all(f.name.startswith(".") for f in contents):
                                shutil.rmtree(model_dir)
                                cleaned_dirs.append(str(model_dir))
            
            # Check for orphaned HuggingFace cache structures
            for cache_dir in self.models_dir.rglob("**/blobs"):
                # This is a HF cache structure, check if it has a valid snapshots dir
                parent = cache_dir.parent
                snapshots_dir = parent / "snapshots"
                if not snapshots_dir.exists() or not list(snapshots_dir.iterdir()):
                    # Orphaned cache
                    size = sum(f.stat().st_size for f in parent.rglob("*") if f.is_file())
                    shutil.rmtree(parent)
                    cleaned_dirs.append(str(parent))
                    total_freed += size
            
            return {
                "success": True,
                "cleaned_files": len(cleaned_files),
                "cleaned_dirs": len(cleaned_dirs),
                "freed_space_gb": total_freed / (1024**3),
                "details": {
                    "files": cleaned_files[:10],  # Limit to first 10
                    "dirs": cleaned_dirs[:10]
                }
            }
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return {
                "success": False,
                "error": str(e),
                "cleaned_files": len(cleaned_files),
                "cleaned_dirs": len(cleaned_dirs),
                "freed_space_gb": total_freed / (1024**3)
            }
    
    def verify_model_integrity(self, model_name: str, model_type: str) -> Dict[str, Any]:
        """Verify the integrity of a downloaded model.
        
        Args:
            model_name: Name of the model
            model_type: Type of model
            
        Returns:
            Dictionary with verification results
        """
        model_path = self._get_model_path(model_type, model_name)
        
        if not model_path.exists():
            return {
                "valid": False,
                "errors": ["Model directory not found"],
                "warnings": [],
                "info": {}
            }
        
        errors = []
        warnings = []
        info = {}
        
        # Count files and calculate size
        file_count = 0
        total_size = 0
        for file in model_path.rglob("*"):
            if file.is_file():
                file_count += 1
                total_size += file.stat().st_size
        
        info["file_count"] = file_count
        info["total_size_gb"] = total_size / (1024**3)
        
        # Check for expected files based on model type
        if model_type == "image":
            # Check for diffusers structure
            expected_files = ["model_index.json", "scheduler/scheduler_config.json"]
            for expected in expected_files:
                if not (model_path / expected).exists():
                    # Check in HF cache structure
                    found = False
                    for snapshot in model_path.rglob("*/snapshots/*"):
                        if (snapshot / expected).exists():
                            found = True
                            break
                    if not found:
                        warnings.append(f"Expected file not found: {expected}")
            
            # Check for model weights
            has_weights = (
                list(model_path.rglob("*.safetensors")) or
                list(model_path.rglob("*.bin")) or
                list(model_path.rglob("*.gguf"))
            )
            if not has_weights:
                errors.append("No model weight files found")
        
        elif model_type == "3d":
            # Check for Hunyuan3D specific files
            required_dirs = ["hunyuan3d-dit-v2-1", "hunyuan3d-dit-v2-0"]
            found_model = False
            
            for req_dir in required_dirs:
                model_dir = model_path / req_dir
                if model_dir.exists():
                    # Check for actual model files
                    weight_files = list(model_dir.glob("*.pth")) + \
                                  list(model_dir.glob("*.pt")) + \
                                  list(model_dir.glob("*.safetensors")) + \
                                  list(model_dir.glob("*.ckpt"))
                    if weight_files:
                        found_model = True
                        info["model_variant"] = req_dir
                        break
            
            if not found_model:
                errors.append("No Hunyuan3D model files found")
        
        # Check for corrupted files (0 byte files)
        zero_byte_files = []
        for file in model_path.rglob("*"):
            if file.is_file() and file.stat().st_size == 0:
                zero_byte_files.append(str(file.relative_to(model_path)))
        
        if zero_byte_files:
            warnings.append(f"Found {len(zero_byte_files)} zero-byte files")
            info["zero_byte_files"] = zero_byte_files[:5]  # Limit to first 5
        
        # Overall validity
        valid = len(errors) == 0
        
        return {
            "valid": valid,
            "errors": errors,
            "warnings": warnings,
            "info": info
        }