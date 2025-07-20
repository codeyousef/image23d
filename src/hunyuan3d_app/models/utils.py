"""Utility functions for model management."""

import base64
import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch

logger = logging.getLogger(__name__)

SECRETS_DIR = Path.cwd() / ".secrets"
HF_TOKEN_FILE = SECRETS_DIR / "hf_token.txt"


def save_hf_token(token: str) -> bool:
    """Base64 encode and save the Hugging Face token.
    
    Args:
        token: Hugging Face token
        
    Returns:
        True if saved successfully
    """
    if not token:
        return False
    
    try:
        SECRETS_DIR.mkdir(exist_ok=True)
        encoded_token = base64.b64encode(token.encode('utf-8'))
        HF_TOKEN_FILE.write_bytes(encoded_token)
        return True
    except Exception as e:
        logger.error(f"Error saving HF token: {e}")
        return False


def load_hf_token() -> Optional[str]:
    """Load and decode the Hugging Face token.
    
    Returns:
        Decoded token or None
    """
    if not HF_TOKEN_FILE.exists():
        return None
    
    try:
        encoded_token = HF_TOKEN_FILE.read_bytes()
        return base64.b64decode(encoded_token).decode('utf-8')
    except Exception as e:
        logger.error(f"Could not load HF token: {e}")
        return None


def clear_hf_token() -> bool:
    """Clear the saved Hugging Face token.
    
    Returns:
        True if cleared successfully
    """
    try:
        if HF_TOKEN_FILE.exists():
            HF_TOKEN_FILE.unlink()
        if "HF_TOKEN" in os.environ:
            del os.environ["HF_TOKEN"]
        return True
    except Exception as e:
        logger.error(f"Error clearing HF token: {e}")
        return False


def estimate_model_memory(model_size_gb: float, dtype: torch.dtype = torch.float16) -> float:
    """Estimate memory required for a model.
    
    Args:
        model_size_gb: Model size in GB
        dtype: Data type for loading
        
    Returns:
        Estimated memory requirement in GB
    """
    # Base memory is the model size
    base_memory = model_size_gb
    
    # Adjust for dtype
    if dtype == torch.float32:
        base_memory *= 2  # FP32 uses twice the memory of FP16
    elif dtype == torch.int8:
        base_memory *= 0.5  # INT8 uses half the memory
    
    # Add overhead for inference (gradients, activations, etc.)
    # Typically 20-50% overhead
    overhead = base_memory * 0.3
    
    return base_memory + overhead


def get_optimal_dtype(vram_gb: float, model_size_gb: float) -> torch.dtype:
    """Determine optimal dtype based on available VRAM.
    
    Args:
        vram_gb: Available VRAM in GB
        model_size_gb: Model size in GB
        
    Returns:
        Optimal dtype
    """
    # Estimate memory needed for different dtypes
    fp32_memory = estimate_model_memory(model_size_gb, torch.float32)
    fp16_memory = estimate_model_memory(model_size_gb, torch.float16)
    int8_memory = estimate_model_memory(model_size_gb, torch.int8)
    
    # Choose based on available memory
    if vram_gb >= fp32_memory:
        return torch.float32
    elif vram_gb >= fp16_memory:
        return torch.float16
    elif vram_gb >= int8_memory:
        return torch.int8
    else:
        # Default to fp16 and rely on CPU offloading
        return torch.float16


def format_model_size(size_bytes: int) -> str:
    """Format model size for display.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def validate_model_files(model_path: Path, required_files: List[str]) -> Dict[str, bool]:
    """Validate that required model files exist.
    
    Args:
        model_path: Path to model directory
        required_files: List of required file names/patterns
        
    Returns:
        Dictionary of file -> exists status
    """
    validation = {}
    
    for file_pattern in required_files:
        if '*' in file_pattern:
            # Handle glob patterns
            matches = list(model_path.glob(file_pattern))
            validation[file_pattern] = len(matches) > 0
        else:
            # Handle exact file names
            validation[file_pattern] = (model_path / file_pattern).exists()
    
    return validation


def get_model_metadata(model_path: Path) -> Dict[str, Any]:
    """Extract metadata from model files.
    
    Args:
        model_path: Path to model directory
        
    Returns:
        Dictionary of metadata
    """
    metadata = {
        "path": str(model_path),
        "exists": model_path.exists(),
        "size_bytes": 0,
        "file_count": 0,
        "has_config": False,
        "has_weights": False,
        "weight_format": None
    }
    
    if not model_path.exists():
        return metadata
    
    # Count files and calculate size
    for file in model_path.rglob("*"):
        if file.is_file():
            metadata["file_count"] += 1
            metadata["size_bytes"] += file.stat().st_size
            
            # Check for config files
            if file.name in ["config.json", "model_index.json"]:
                metadata["has_config"] = True
            
            # Check for weight files
            if file.suffix in [".bin", ".safetensors", ".gguf"]:
                metadata["has_weights"] = True
                metadata["weight_format"] = file.suffix[1:]  # Remove the dot
    
    metadata["size_formatted"] = format_model_size(metadata["size_bytes"])
    
    return metadata


def cleanup_incomplete_downloads(models_dir: Path) -> List[str]:
    """Clean up incomplete model downloads.
    
    Args:
        models_dir: Root models directory
        
    Returns:
        List of cleaned paths
    """
    cleaned = []
    
    # Look for common incomplete download indicators
    incomplete_indicators = [
        "*.tmp",
        "*.partial",
        "*.downloading",
        ".incomplete"
    ]
    
    for model_type_dir in models_dir.iterdir():
        if not model_type_dir.is_dir():
            continue
            
        for model_dir in model_type_dir.iterdir():
            if not model_dir.is_dir():
                continue
            
            # Check for incomplete indicators
            is_incomplete = False
            for pattern in incomplete_indicators:
                if list(model_dir.glob(pattern)):
                    is_incomplete = True
                    break
            
            # Also check if directory is empty or only has metadata
            if not is_incomplete:
                files = list(model_dir.rglob("*"))
                non_metadata_files = [f for f in files if f.suffix not in [".json", ".txt", ".md"]]
                if len(non_metadata_files) == 0:
                    is_incomplete = True
            
            if is_incomplete:
                try:
                    import shutil
                    shutil.rmtree(model_dir)
                    cleaned.append(str(model_dir))
                    logger.info(f"Cleaned incomplete download: {model_dir}")
                except Exception as e:
                    logger.error(f"Error cleaning {model_dir}: {e}")
    
    return cleaned