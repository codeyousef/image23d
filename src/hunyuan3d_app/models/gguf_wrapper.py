"""GGUF Wrapper - backward compatibility wrapper

This module maintains backward compatibility by re-exporting
the StandaloneGGUFPipeline from the refactored gguf_modules package.
"""

# Re-export main class for backward compatibility
from .gguf_modules.wrapper import StandaloneGGUFPipeline

# Re-export any additional classes/functions that might be used elsewhere
from .gguf_modules.loader import GGUFLoader
from .gguf_modules.optimization import GGUFOptimizer
from .gguf_modules.pipeline import GGUFPipelineGenerator

__all__ = [
    'StandaloneGGUFPipeline',
    'GGUFLoader',
    'GGUFOptimizer',
    'GGUFPipelineGenerator'
]