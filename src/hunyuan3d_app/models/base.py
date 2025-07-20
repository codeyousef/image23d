"""Base classes and interfaces for model management."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)


class BaseModelManager(ABC):
    """Abstract base class for model managers."""
    
    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def load_model(self, model_name: str, **kwargs) -> Tuple[bool, str]:
        """Load a model by name.
        
        Args:
            model_name: Name of the model to load
            **kwargs: Additional arguments for model loading
            
        Returns:
            Tuple of (success, message)
        """
        pass
    
    @abstractmethod
    def unload_model(self) -> Tuple[bool, str]:
        """Unload the currently loaded model.
        
        Returns:
            Tuple of (success, message)
        """
        pass
    
    @abstractmethod
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with model information or None if not found
        """
        pass


class ModelInfo:
    """Information about a downloadable model."""
    
    def __init__(
        self,
        name: str,
        repo_id: str,
        model_type: str,
        size_gb: float,
        requires_auth: bool = False,
        description: str = "",
        tags: Optional[list] = None,
        recommended_vram_gb: float = 8.0
    ):
        self.name = name
        self.repo_id = repo_id
        self.model_type = model_type
        self.size_gb = size_gb
        self.requires_auth = requires_auth
        self.description = description
        self.tags = tags or []
        self.recommended_vram_gb = recommended_vram_gb
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "repo_id": self.repo_id,
            "model_type": self.model_type,
            "size_gb": self.size_gb,
            "requires_auth": self.requires_auth,
            "description": self.description,
            "tags": self.tags,
            "recommended_vram_gb": self.recommended_vram_gb
        }


class DownloadProgress:
    """Track download progress for models."""
    
    def __init__(self):
        self.total_size = 0
        self.downloaded_size = 0
        self.current_file = ""
        self.percentage = 0
        self.speed = 0
        self.eta = 0
        self.is_complete = False
        self.error = None
        # Additional fields for file-based progress
        self.total_files = 0
        self.completed_files = 0
        # New fields for better tracking
        self.current_file_size = 0
        self.current_file_downloaded = 0
        self.start_time = None
        self.last_update_time = None
        self.download_rate_history = []  # Track speed over time
    
    def update(
        self,
        downloaded: int,
        total: int,
        filename: str = "",
        speed: float = 0,
        eta: float = 0
    ):
        """Update download progress."""
        self.downloaded_size = downloaded
        self.total_size = total
        self.current_file = filename
        self.speed = speed
        self.eta = eta
        
        if total > 0:
            self.percentage = int((downloaded / total) * 100)
        
        if downloaded >= total and total > 0:
            self.is_complete = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for display."""
        return {
            "percentage": self.percentage,
            "downloaded_gb": self.downloaded_size / (1024**3),
            "total_gb": self.total_size / (1024**3),
            "current_file": self.current_file,
            "speed_mbps": self.speed / (1024**2) if self.speed > 0 else 0,
            "eta_minutes": self.eta / 60 if self.eta > 0 else 0,
            "is_complete": self.is_complete,
            "error": self.error,
            # File progress
            "total_files": self.total_files,
            "completed_files": self.completed_files,
            "files_percentage": (self.completed_files / self.total_files * 100) if self.total_files > 0 else 0,
            # Current file progress
            "current_file_size_mb": self.current_file_size / (1024**2) if self.current_file_size > 0 else 0,
            "current_file_downloaded_mb": self.current_file_downloaded / (1024**2) if self.current_file_downloaded > 0 else 0,
            "current_file_percentage": (self.current_file_downloaded / self.current_file_size * 100) if self.current_file_size > 0 else 0
        }