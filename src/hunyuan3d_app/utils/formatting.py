"""Data formatting utilities."""

import time
from datetime import datetime, timedelta
from typing import Optional, Union
from pathlib import Path


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted string (e.g., "1.5 GB", "750 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string (e.g., "2h 15m", "45s")
    """
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s" if secs else f"{minutes}m"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m" if minutes else f"{hours}h"


def format_eta(seconds: Optional[int]) -> str:
    """Format estimated time of arrival.
    
    Args:
        seconds: ETA in seconds
        
    Returns:
        Formatted string (e.g., "2 hours", "5 minutes")
    """
    if seconds is None:
        return "Unknown"
    
    if seconds < 60:
        return "Less than a minute"
    elif seconds < 3600:
        minutes = seconds // 60
        return f"{minutes} minute{'s' if minutes > 1 else ''}"
    else:
        hours = seconds // 3600
        return f"{hours} hour{'s' if hours > 1 else ''}"


def format_timestamp(
    timestamp: Optional[Union[float, datetime]] = None,
    format_str: str = "%Y-%m-%d %H:%M:%S"
) -> str:
    """Format timestamp for display.
    
    Args:
        timestamp: Unix timestamp or datetime object (None for current time)
        format_str: strftime format string
        
    Returns:
        Formatted timestamp string
    """
    if timestamp is None:
        dt = datetime.now()
    elif isinstance(timestamp, float):
        dt = datetime.fromtimestamp(timestamp)
    else:
        dt = timestamp
    
    return dt.strftime(format_str)


def format_model_name(model_id: str) -> str:
    """Format model ID for display.
    
    Args:
        model_id: Model identifier
        
    Returns:
        Formatted model name
    """
    # Remove common prefixes
    name = model_id
    for prefix in ["huggingface/", "hf/", "models/"]:
        if name.startswith(prefix):
            name = name[len(prefix):]
    
    # Replace underscores and hyphens
    name = name.replace("_", " ").replace("-", " ")
    
    # Capitalize words
    words = name.split()
    formatted_words = []
    for word in words:
        if word.upper() in ["XL", "SD", "SDXL", "VAE", "LORA", "3D", "2D", "AI", "ML"]:
            formatted_words.append(word.upper())
        elif word.lower() in ["v1", "v2", "v3", "v4", "v5"]:
            formatted_words.append(word.lower())
        else:
            formatted_words.append(word.capitalize())
    
    return " ".join(formatted_words)


def format_progress(current: int, total: int, width: int = 50) -> str:
    """Format progress bar.
    
    Args:
        current: Current value
        total: Total value
        width: Width of progress bar
        
    Returns:
        Progress bar string
    """
    if total == 0:
        percent = 0
    else:
        percent = (current / total) * 100
    
    filled = int(width * current // total) if total > 0 else 0
    bar = "█" * filled + "░" * (width - filled)
    
    return f"|{bar}| {percent:.1f}%"


def format_path(path: Union[str, Path], max_length: int = 50) -> str:
    """Format file path for display.
    
    Args:
        path: File path
        max_length: Maximum length for display
        
    Returns:
        Formatted path string
    """
    path_str = str(Path(path))
    
    if len(path_str) <= max_length:
        return path_str
    
    # Try to keep filename
    path_obj = Path(path_str)
    filename = path_obj.name
    
    if len(filename) >= max_length - 3:
        # Even filename is too long
        return "..." + path_str[-(max_length-3):]
    
    # Shorten directory part
    remaining = max_length - len(filename) - 3
    parent = str(path_obj.parent)
    
    if remaining > 0:
        return parent[:remaining] + ".../" + filename
    else:
        return ".../" + filename


def format_vram_usage(used_gb: float, total_gb: float) -> str:
    """Format VRAM usage for display.
    
    Args:
        used_gb: Used VRAM in GB
        total_gb: Total VRAM in GB
        
    Returns:
        Formatted VRAM usage string
    """
    percent = (used_gb / total_gb * 100) if total_gb > 0 else 0
    return f"{used_gb:.1f}/{total_gb:.1f}GB ({percent:.0f}%)"