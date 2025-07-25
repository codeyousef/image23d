"""Model verification and validation utilities"""

import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError

logger = logging.getLogger(__name__)


class ModelVerifier:
    """Handles model verification and validation"""
    
    def __init__(self, hf_token: Optional[str] = None):
        self.api = HfApi()
        self.hf_token = hf_token
    
    def check_model_exists(self, repo_id: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Check if a model exists on Hugging Face Hub
        
        Args:
            repo_id: Repository ID to check
            
        Returns:
            Tuple of (exists, model_info)
        """
        try:
            # Get model info
            model_info = self.api.model_info(repo_id, token=self.hf_token)
            
            # Extract relevant info
            info = {
                "repo_id": model_info.id,
                "author": model_info.author,
                "model_size": getattr(model_info, 'model_size', None),
                "tags": model_info.tags,
                "downloads": getattr(model_info, 'downloads', 0),
                "likes": getattr(model_info, 'likes', 0),
                "private": model_info.private,
                "gated": getattr(model_info, 'gated', False)
            }
            
            return True, info
            
        except RepositoryNotFoundError:
            logger.warning(f"Model {repo_id} not found on Hugging Face Hub")
            return False, None
        except Exception as e:
            logger.error(f"Error checking model {repo_id}: {e}")
            return False, None
    
    def get_model_size(self, repo_id: str) -> Optional[float]:
        """Get the total size of a model in bytes
        
        Args:
            repo_id: Repository ID
            
        Returns:
            Total size in bytes or None
        """
        try:
            # List all files in the repository
            files = self.api.list_repo_files(
                repo_id=repo_id,
                token=self.hf_token
            )
            
            # Get file information
            total_size = 0
            for file_path in files:
                try:
                    file_info = self.api.get_file_info(
                        repo_id=repo_id,
                        filename=file_path,
                        token=self.hf_token
                    )
                    if hasattr(file_info, 'size'):
                        total_size += file_info.size
                except Exception as e:
                    logger.warning(f"Could not get size for {file_path}: {e}")
            
            return total_size if total_size > 0 else None
            
        except Exception as e:
            logger.error(f"Error getting model size for {repo_id}: {e}")
            return None
    
    def verify_download(self, model_path: Path, expected_files: Optional[list] = None) -> Tuple[bool, str]:
        """Verify that a model was downloaded correctly
        
        Args:
            model_path: Path to the downloaded model
            expected_files: Optional list of expected files
            
        Returns:
            Tuple of (is_valid, message)
        """
        if not model_path.exists():
            return False, "Model directory does not exist"
        
        # Check if directory is not empty
        files = list(model_path.rglob("*"))
        if not files:
            return False, "Model directory is empty"
        
        # Check for expected files if provided
        if expected_files:
            missing_files = []
            for expected_file in expected_files:
                file_path = model_path / expected_file
                if not file_path.exists():
                    # Also check in subdirectories
                    found = False
                    for f in files:
                        if f.name == expected_file:
                            found = True
                            break
                    if not found:
                        missing_files.append(expected_file)
            
            if missing_files:
                return False, f"Missing expected files: {', '.join(missing_files)}"
        
        # Check for common model files
        common_patterns = [
            "*.safetensors",
            "*.bin",
            "*.onnx",
            "*.gguf",
            "config.json",
            "model_index.json"
        ]
        
        has_model_files = False
        for pattern in common_patterns:
            if list(model_path.rglob(pattern)):
                has_model_files = True
                break
        
        if not has_model_files:
            return False, "No model files found (safetensors, bin, onnx, gguf, or config files)"
        
        return True, "Model download verified successfully"
    
    def check_disk_space(self, target_dir: Path, required_size: float) -> Tuple[bool, str]:
        """Check if there's enough disk space for download
        
        Args:
            target_dir: Target directory for download
            required_size: Required size in bytes
            
        Returns:
            Tuple of (has_space, message)
        """
        try:
            import shutil
            
            # Get disk usage
            total, used, free = shutil.disk_usage(target_dir)
            
            # Add 10% buffer for safety
            required_with_buffer = required_size * 1.1
            
            if free < required_with_buffer:
                free_gb = free / (1024**3)
                required_gb = required_with_buffer / (1024**3)
                return False, f"Insufficient disk space. Need {required_gb:.2f} GB, have {free_gb:.2f} GB free"
            
            return True, "Sufficient disk space available"
            
        except Exception as e:
            logger.error(f"Error checking disk space: {e}")
            return True, "Could not verify disk space, proceeding anyway"