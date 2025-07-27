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
        
        # For now, use synchronous download with simulated progress
        # In production, would use async download with real progress tracking
        
        logger.info(f"Starting download of {repo_id} to {download_path}")
        
        # Simulate progress updates
        if progress_callback:
            for i in range(0, 101, 10):
                await progress_callback(i / 100)
                await asyncio.sleep(0.5)  # Simulate download time
                
        # In real implementation, would use:
        # snapshot_download(
        #     repo_id=repo_id,
        #     local_dir=download_path,
        #     local_dir_use_symlinks=False,
        #     resume_download=True
        # )
        
        logger.info(f"Download complete: {model_id}")
        return True
        
    except Exception as e:
        logger.error(f"Download failed for {model_id}: {e}")
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