"""Model configurations."""

from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class ImageModelConfig:
    """Configuration for image generation models."""
    name: str
    repo_id: str
    pipeline_class: str
    size: str
    vram_required: str
    description: str
    optimal_resolution: Tuple[int, int]
    supports_refiner: bool = False
    is_gguf: bool = False
    gguf_file: str = ""


@dataclass
class ThreeDModelConfig:
    """Configuration for 3D generation models."""
    name: str
    repo_id: str
    size: str
    vram_required: str
    description: str
    optimal_views: int
    supports_pbr: bool = False
    supports_texture: bool = True


@dataclass
class VideoModelConfig:
    """Configuration for video generation models."""
    name: str
    repo_id: str
    size: str
    vram_required: str
    description: str
    max_frames: int
    fps: int = 8


@dataclass
class QualityPreset:
    """Quality preset configurations."""
    name: str
    image_steps: int
    image_guidance: float
    use_refiner: bool
    num_3d_views: int
    mesh_resolution: int
    texture_resolution: int