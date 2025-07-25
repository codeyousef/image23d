"""GGUF Model Support Module

Provides standalone GGUF model loading without requiring base model components.
"""

from .wrapper import StandaloneGGUFPipeline
# Import GGUFModelManager from the parent gguf.py file
from ..gguf import GGUFModelManager

__all__ = ['StandaloneGGUFPipeline', 'GGUFModelManager']