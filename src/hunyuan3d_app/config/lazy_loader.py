"""Lazy loading implementation for model configurations."""

import logging
from functools import lru_cache
from typing import Dict, Any, Optional, Callable

logger = logging.getLogger(__name__)


class LazyModelConfig:
    """Lazy loading wrapper for model configurations."""
    
    def __init__(self, loader_func: Callable[[], Dict[str, Any]]):
        """Initialize with a loader function that returns the config dict."""
        self._loader = loader_func
        self._loaded = False
        self._data = None
    
    def _ensure_loaded(self):
        """Load data if not already loaded."""
        if not self._loaded:
            self._data = self._loader()
            self._loaded = True
    
    def __getitem__(self, key):
        """Get item from lazy-loaded dict."""
        self._ensure_loaded()
        return self._data[key]
    
    def __setitem__(self, key, value):
        """Set item in lazy-loaded dict."""
        self._ensure_loaded()
        self._data[key] = value
    
    def __contains__(self, key):
        """Check if key exists in lazy-loaded dict."""
        self._ensure_loaded()
        return key in self._data
    
    def __iter__(self):
        """Iterate over keys in lazy-loaded dict."""
        self._ensure_loaded()
        return iter(self._data)
    
    def __len__(self):
        """Get length of lazy-loaded dict."""
        self._ensure_loaded()
        return len(self._data)
    
    def get(self, key, default=None):
        """Get item with default value."""
        self._ensure_loaded()
        return self._data.get(key, default)
    
    def keys(self):
        """Get keys from lazy-loaded dict."""
        self._ensure_loaded()
        return self._data.keys()
    
    def values(self):
        """Get values from lazy-loaded dict."""
        self._ensure_loaded()
        return self._data.values()
    
    def items(self):
        """Get items from lazy-loaded dict."""
        self._ensure_loaded()
        return self._data.items()
    
    def to_dict(self):
        """Convert to regular dict."""
        self._ensure_loaded()
        return self._data.copy()


class LazyConfigManager:
    """Manager for lazy loading configurations."""
    
    def __init__(self):
        self._loaders = {}
        self._configs = {}
    
    def register_loader(self, name: str, loader: Callable[[], Dict[str, Any]]):
        """Register a lazy loader for a configuration."""
        self._loaders[name] = loader
        # Create lazy wrapper
        self._configs[name] = LazyModelConfig(loader)
    
    def get_config(self, name: str) -> Optional[LazyModelConfig]:
        """Get a lazy configuration by name."""
        return self._configs.get(name)
    
    @lru_cache(maxsize=32)
    def get_all_model_names(self, config_type: str) -> list:
        """Get all model names for a config type (cached)."""
        config = self.get_config(config_type)
        if config:
            return list(config.keys())
        return []
    
    def preload(self, *config_names):
        """Preload specific configurations."""
        for name in config_names:
            if name in self._configs:
                self._configs[name]._ensure_loaded()
                logger.info(f"Preloaded configuration: {name}")


# Global lazy config manager
lazy_config_manager = LazyConfigManager()


# Loader functions for different model types
@lru_cache(maxsize=1)
def _load_image_models():
    """Load image model configurations."""
    from .model_definitions import IMAGE_MODELS
    logger.info("Loading IMAGE_MODELS configuration")
    return IMAGE_MODELS


@lru_cache(maxsize=1)
def _load_gated_image_models():
    """Load gated image model configurations."""
    from .model_definitions import GATED_IMAGE_MODELS
    logger.info("Loading GATED_IMAGE_MODELS configuration")
    return GATED_IMAGE_MODELS


@lru_cache(maxsize=1)
def _load_gguf_image_models():
    """Load GGUF image model configurations."""
    from .model_definitions import GGUF_IMAGE_MODELS
    logger.info("Loading GGUF_IMAGE_MODELS configuration")
    return GGUF_IMAGE_MODELS


@lru_cache(maxsize=1)
def _load_hunyuan3d_models():
    """Load HunYuan3D model configurations."""
    from .model_definitions import HUNYUAN3D_MODELS_DICT
    logger.info("Loading HUNYUAN3D_MODELS configuration")
    return HUNYUAN3D_MODELS_DICT


@lru_cache(maxsize=1)
def _load_video_models():
    """Load video model configurations from main config."""
    # Import from the main config file since these aren't in model_definitions yet
    import sys
    import importlib
    
    # Check if already imported
    if 'hunyuan3d_app.config' in sys.modules:
        config_module = sys.modules['hunyuan3d_app.config']
    else:
        config_module = importlib.import_module('hunyuan3d_app.config')
    
    logger.info("Loading VIDEO_MODELS configuration")
    return getattr(config_module, 'VIDEO_MODELS', {})


# Register all loaders
lazy_config_manager.register_loader('IMAGE_MODELS', _load_image_models)
lazy_config_manager.register_loader('GATED_IMAGE_MODELS', _load_gated_image_models)
lazy_config_manager.register_loader('GGUF_IMAGE_MODELS', _load_gguf_image_models)
lazy_config_manager.register_loader('HUNYUAN3D_MODELS', _load_hunyuan3d_models)
lazy_config_manager.register_loader('VIDEO_MODELS', _load_video_models)


# Create lazy-loaded proxies for backward compatibility
IMAGE_MODELS_LAZY = lazy_config_manager.get_config('IMAGE_MODELS')
GATED_IMAGE_MODELS_LAZY = lazy_config_manager.get_config('GATED_IMAGE_MODELS')
GGUF_IMAGE_MODELS_LAZY = lazy_config_manager.get_config('GGUF_IMAGE_MODELS')
HUNYUAN3D_MODELS_LAZY = lazy_config_manager.get_config('HUNYUAN3D_MODELS')
VIDEO_MODELS_LAZY = lazy_config_manager.get_config('VIDEO_MODELS')