"""
NeuralForge Studio Core Module

This module contains shared business logic used by both desktop and web applications.
"""

__version__ = "1.0.0"

from .processors import ImageProcessor, ThreeDProcessor, VideoProcessor, PromptEnhancer
from .services import ModelManager, QueueManager, ServerlessGPUManager

__all__ = [
    "ImageProcessor",
    "ThreeDProcessor", 
    "VideoProcessor",
    "PromptEnhancer",
    "ModelManager",
    "QueueManager",
    "ServerlessGPUManager"
]