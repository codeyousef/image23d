"""
Model download utilities with progress tracking
"""

import asyncio
import os
from pathlib import Path
from typing import Optional, Callable, Dict, Any
import logging
from huggingface_hub import snapshot_download, hf_hub_download
import aiohttp

logger = logging.getLogger(__name__)


async def download_model_with_progress(
    model_type: str,
    model_id: str, 
    repo_id: str,
    progress_callback: Optional[Callable[[float], None]] = None,
    models_dir: Optional[Path] = None
) -> bool:
    """
    Download a model with progress tracking
    
    Args:
        model_type: Type of model (image, 3d, video, etc)
        model_id: Local model identifier
        repo_id: HuggingFace repository ID
        progress_callback: Async callback for progress updates (0.0 to 1.0)
        models_dir: Base models directory
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if models_dir is None:
            from ..config import MODELS_DIR
            models_dir = MODELS_DIR
            
        # Determine download path
        download_path = models_dir / model_type / model_id
        download_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting download of {repo_id} to {download_path}")
        
        # Check if already downloaded
        if _check_model_files_exist(download_path, model_type):
            logger.info(f"Model {model_id} already downloaded")
            if progress_callback:
                await progress_callback(1.0)
            return True
        
        # Run download in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        
        def download_with_progress():
            """Download function that tracks progress"""
            from huggingface_hub.file_download import http_get
            from huggingface_hub.utils import filter_repo_objects
            from huggingface_hub.constants import DEFAULT_REVISION
            
            # Get repo info to calculate total size
            from huggingface_hub import list_repo_files, model_info
            try:
                info = model_info(repo_id)
                repo_files = list(list_repo_files(repo_id))
                total_size = 0
                
                # Calculate total size if available
                if hasattr(info, 'siblings'):
                    for file_info in info.siblings:
                        if hasattr(file_info, 'size') and file_info.size:
                            total_size += file_info.size
                
                downloaded_size = 0
                last_progress = 0
                
                def progress_hook(num_bytes: int):
                    """Progress hook for downloads"""
                    nonlocal downloaded_size, last_progress
                    downloaded_size += num_bytes
                    
                    if total_size > 0:
                        progress = min(downloaded_size / total_size, 1.0)
                        # Only update if progress changed significantly (every 1%)
                        if progress - last_progress >= 0.01:
                            last_progress = progress
                            # Schedule callback in async loop
                            asyncio.run_coroutine_threadsafe(
                                progress_callback(progress), loop
                            ) if progress_callback else None
                
                # Download the model
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=str(download_path),
                    local_dir_use_symlinks=False,
                    resume_download=True,
                    revision=DEFAULT_REVISION,
                    # Use tqdm for console progress if no callback
                    tqdm_class=None if progress_callback else None
                )
                
                return True
                
            except Exception as e:
                logger.error(f"Download error: {e}")
                return False
        
        # Execute download in thread pool
        success = await loop.run_in_executor(None, download_with_progress)
        
        if success:
            logger.info(f"Download complete: {model_id}")
            if progress_callback:
                await progress_callback(1.0)
        else:
            logger.error(f"Download failed: {model_id}")
            
        return success
        
    except Exception as e:
        logger.error(f"Download failed for {model_id}: {e}")
        return False


def _check_model_files_exist(model_path: Path, model_type: str) -> bool:
    """Check if model files exist in the directory"""
    if not model_path.exists():
        return False
        
    # Check for common model file patterns
    patterns = [
        '*.safetensors', '*.bin', '*.ckpt', '*.pt', '*.pth',
        '*.gguf', '*.onnx', 'model_index.json', 'config.json'
    ]
    
    for pattern in patterns:
        if list(model_path.glob(pattern)):
            return True
            
    # Check for subdirectories with model files (e.g., transformer/, vae/)
    for subdir in model_path.iterdir():
        if subdir.is_dir():
            for pattern in patterns:
                if list(subdir.glob(pattern)):
                    return True
                    
    return False


def get_model_info(repo_id: str) -> Dict[str, Any]:
    """Get model information from HuggingFace"""
    try:
        from huggingface_hub import model_info
        info = model_info(repo_id)
        return {
            "size": info.size if hasattr(info, 'size') else None,
            "downloads": info.downloads if hasattr(info, 'downloads') else 0,
            "last_modified": info.last_modified if hasattr(info, 'last_modified') else None,
            "files": [f.rfilename for f in info.siblings] if hasattr(info, 'siblings') else []
        }
    except Exception as e:
        logger.error(f"Failed to get model info for {repo_id}: {e}")
        return {}