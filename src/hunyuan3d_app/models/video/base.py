"""Base classes for video generation models

Following the architecture from the Video Implementation Guide:
- BaseVideoModel: Abstract base for all video models
- VideoQualityPreset: Quality configurations for video generation
- Memory optimization and quantization support
"""

import logging
import gc
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
import torch
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class VideoModelType(Enum):
    """Types of video models supported"""
    WAN2_1_1_3B = "wan2_1_1.3b"
    WAN2_1_14B = "wan2_1_14b"
    HUNYUANVIDEO = "hunyuanvideo"
    LTX_VIDEO = "ltx_video"
    MOCHI_1 = "mochi_1"
    COGVIDEOX_5B = "cogvideox_5b"


class VideoFormat(Enum):
    """Video output formats"""
    MP4 = "mp4"
    WEBM = "webm"
    GIF = "gif"
    AVI = "avi"
    MOV = "mov"
    FRAMES = "frames"  # Individual frames


class QuantizationType(Enum):
    """Quantization types for memory optimization"""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8 = "fp8"
    INT8 = "int8"
    NF4 = "nf4"
    Q8_0 = "q8_0"  # GGUF formats
    Q6_K = "q6_k"
    Q5_K_M = "q5_k_m"
    Q4_K_M = "q4_k_m"
    Q3_K_M = "q3_k_m"
    Q2_K = "q2_k"


@dataclass
class VideoQualityPreset:
    """Quality preset configuration for video generation"""
    name: str
    resolution: Tuple[int, int]  # (width, height)
    fps: int
    duration_seconds: float
    inference_steps: int
    guidance_scale: float
    vae_slicing: bool
    cpu_offload: bool
    memory_efficient: bool
    quantization: Optional[QuantizationType] = None
    motion_bucket_id: int = 127  # For motion strength
    noise_aug_strength: float = 0.02  # For stability


# Define quality presets based on the guide
VIDEO_QUALITY_PRESETS = {
    "draft": VideoQualityPreset(
        name="Draft",
        resolution=(512, 288),
        fps=8,
        duration_seconds=2.0,
        inference_steps=20,
        guidance_scale=5.0,
        vae_slicing=True,
        cpu_offload=True,
        memory_efficient=True,
        quantization=QuantizationType.INT8,
        motion_bucket_id=100
    ),
    "standard": VideoQualityPreset(
        name="Standard",
        resolution=(768, 512),
        fps=24,
        duration_seconds=5.0,
        inference_steps=30,
        guidance_scale=7.5,
        vae_slicing=True,
        cpu_offload=False,
        memory_efficient=True,
        quantization=QuantizationType.FP16,
        motion_bucket_id=127
    ),
    "high": VideoQualityPreset(
        name="High Quality",
        resolution=(1024, 576),
        fps=30,
        duration_seconds=5.0,
        inference_steps=50,
        guidance_scale=8.0,
        vae_slicing=False,
        cpu_offload=False,
        memory_efficient=False,
        quantization=QuantizationType.BF16,
        motion_bucket_id=150
    ),
    "ultra": VideoQualityPreset(
        name="Ultra Quality",
        resolution=(1280, 720),
        fps=30,
        duration_seconds=5.0,
        inference_steps=100,
        guidance_scale=10.0,
        vae_slicing=False,
        cpu_offload=False,
        memory_efficient=False,
        quantization=None,  # Full precision
        motion_bucket_id=180,
        noise_aug_strength=0.0  # Minimal noise for best quality
    )
}


@dataclass
class VideoGenerationResult:
    """Result container for video generation"""
    frames: List[Image.Image]
    video_path: Optional[Path] = None
    fps: int = 24
    duration: float = 0.0
    resolution: Tuple[int, int] = (768, 512)
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.duration == 0.0 and self.frames:
            self.duration = len(self.frames) / self.fps


@dataclass 
class VideoModelConfig:
    """Configuration for video models"""
    model_type: VideoModelType
    model_path: Optional[Path] = None
    device: str = "cuda"
    dtype: torch.dtype = torch.float16
    quantization: Optional[QuantizationType] = None
    enable_vae_slicing: bool = True
    enable_cpu_offload: bool = False
    enable_attention_slicing: bool = True
    enable_xformers: bool = True
    chunk_size: int = 8  # Process video in chunks
    max_batch_size: int = 1
    cache_dir: Optional[Path] = None


class BaseVideoModel(ABC):
    """Abstract base class for all video models"""
    
    def __init__(
        self, 
        config: VideoModelConfig
    ):
        self.config = config
        self.device = config.device
        self.dtype = self._get_dtype(config.dtype, config.quantization)
        self.loaded = False
        self.pipeline = None
        
        # Memory optimization flags
        self.vae_slicing_enabled = config.enable_vae_slicing
        self.cpu_offload_enabled = config.enable_cpu_offload
        self.attention_slicing_enabled = config.enable_attention_slicing
        self.xformers_enabled = config.enable_xformers
        
        # Cache directory
        self.cache_dir = config.cache_dir or Path.home() / ".cache" / "huggingface"
        
    def _get_dtype(self, base_dtype: torch.dtype, quantization: Optional[QuantizationType]) -> torch.dtype:
        """Get the appropriate dtype based on quantization settings"""
        if quantization:
            dtype_map = {
                QuantizationType.FP32: torch.float32,
                QuantizationType.FP16: torch.float16,
                QuantizationType.BF16: torch.bfloat16,
                QuantizationType.INT8: torch.int8,
            }
            return dtype_map.get(quantization, base_dtype)
        return base_dtype
        
    @abstractmethod
    def load(self, progress_callback: Optional[Callable] = None) -> bool:
        """Load the model weights"""
        pass
        
    def unload(self):
        """Unload model to free memory"""
        if hasattr(self, 'pipeline') and self.pipeline is not None:
            # Move to CPU first
            if hasattr(self.pipeline, 'to'):
                try:
                    self.pipeline.to('cpu')
                except Exception as e:
                    logger.warning(f"Failed to move pipeline to CPU: {e}")
            
            # Clear the pipeline
            del self.pipeline
            self.pipeline = None
            
        # Clear any other model components
        if hasattr(self, 'vae'):
            del self.vae
        if hasattr(self, 'text_encoder'):
            del self.text_encoder
        if hasattr(self, 'unet'):
            del self.unet
            
        self.loaded = False
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
    @abstractmethod
    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_frames: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        fps: Optional[int] = None,
        motion_bucket_id: int = 127,
        noise_aug_strength: float = 0.02,
        decode_chunk_size: Optional[int] = None,
        seed: Optional[int] = None,
        progress_callback: Optional[Callable] = None,
        **kwargs
    ) -> VideoGenerationResult:
        """Generate video from text prompt"""
        pass
        
    @abstractmethod
    def image_to_video(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        fps: Optional[int] = None,
        motion_bucket_id: int = 127,
        noise_aug_strength: float = 0.02,
        decode_chunk_size: Optional[int] = None,
        seed: Optional[int] = None,
        progress_callback: Optional[Callable] = None,
        **kwargs
    ) -> VideoGenerationResult:
        """Generate video from input image"""
        pass
        
    def enable_memory_optimizations(self):
        """Enable all memory optimization techniques"""
        if hasattr(self, 'pipeline') and self.pipeline:
            # VAE slicing
            if hasattr(self.pipeline, 'vae') and hasattr(self.pipeline.vae, 'enable_slicing'):
                self.pipeline.vae.enable_slicing()
                logger.info("Enabled VAE slicing")
                
            # VAE tiling for very large videos
            if hasattr(self.pipeline, 'vae') and hasattr(self.pipeline.vae, 'enable_tiling'):
                self.pipeline.vae.enable_tiling()
                logger.info("Enabled VAE tiling")
                
            # CPU offloading
            if self.cpu_offload_enabled and hasattr(self.pipeline, 'enable_sequential_cpu_offload'):
                self.pipeline.enable_sequential_cpu_offload()
                logger.info("Enabled sequential CPU offload")
            elif hasattr(self.pipeline, 'enable_model_cpu_offload'):
                self.pipeline.enable_model_cpu_offload()
                logger.info("Enabled model CPU offload")
                
            # Attention slicing
            if self.attention_slicing_enabled and hasattr(self.pipeline, 'enable_attention_slicing'):
                self.pipeline.enable_attention_slicing(1)
                logger.info("Enabled attention slicing")
                
            # xFormers memory efficient attention
            if self.xformers_enabled:
                try:
                    if hasattr(self.pipeline, 'enable_xformers_memory_efficient_attention'):
                        self.pipeline.enable_xformers_memory_efficient_attention()
                        logger.info("Enabled xFormers memory efficient attention")
                except Exception as e:
                    logger.warning(f"Failed to enable xFormers: {e}")
                    
    @abstractmethod
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage in GB"""
        pass
        
    def estimate_memory_requirements(
        self,
        resolution: Tuple[int, int],
        num_frames: int,
        batch_size: int = 1
    ) -> Dict[str, float]:
        """Estimate memory requirements for given parameters"""
        width, height = resolution
        
        # Base estimates (these would be calibrated per model)
        model_params = {
            VideoModelType.WAN2_1_1_3B: 1.3e9,
            VideoModelType.WAN2_1_14B: 14e9,
            VideoModelType.HUNYUANVIDEO: 13e9,
            VideoModelType.LTX_VIDEO: 2e9,
            VideoModelType.MOCHI_1: 10e9,
            VideoModelType.COGVIDEOX_5B: 5e9
        }
        
        base_params = model_params.get(self.config.model_type, 5e9)
        
        # Model weights (assuming FP16)
        model_memory = base_params * 2 / 1e9  # GB
        
        # Activation memory (rough estimate)
        pixels_per_frame = width * height
        total_pixels = pixels_per_frame * num_frames * batch_size
        
        # Latent space (typically 8x smaller)
        latent_pixels = total_pixels / 64
        
        # Estimate activation memory (4 bytes per pixel in latent space)
        activation_memory = latent_pixels * 4 * 4 / 1e9  # GB (x4 for intermediate activations)
        
        # VAE memory for decoding
        vae_memory = total_pixels * 3 * 4 / 1e9  # RGB, float32
        
        # Add overhead
        overhead = 2.0  # GB
        
        total = model_memory + activation_memory + vae_memory + overhead
        
        return {
            "model": model_memory,
            "activations": activation_memory,
            "vae": vae_memory,
            "overhead": overhead,
            "total": total,
            "recommended": total * 1.2  # 20% safety margin
        }
        
    def supports_feature(self, feature: str) -> bool:
        """Check if model supports a specific feature"""
        # Override in specific implementations
        base_features = {
            "text_to_video": True,
            "image_to_video": False,
            "video_to_video": False,
            "inpainting": False,
            "outpainting": False,
            "super_resolution": False,
            "frame_interpolation": False,
            "lora": False,
            "controlnet": False,
            "ip_adapter": False
        }
        return base_features.get(feature, False)
        
    def get_optimal_settings(self, target_vram_gb: float) -> VideoQualityPreset:
        """Get optimal quality preset for available VRAM"""
        # Sort presets by quality (ascending)
        sorted_presets = [
            ("draft", VIDEO_QUALITY_PRESETS["draft"]),
            ("standard", VIDEO_QUALITY_PRESETS["standard"]),
            ("high", VIDEO_QUALITY_PRESETS["high"]),
            ("ultra", VIDEO_QUALITY_PRESETS["ultra"])
        ]
        
        # Find the best preset that fits in available VRAM
        for preset_name, preset in reversed(sorted_presets):
            est_memory = self.estimate_memory_requirements(
                preset.resolution,
                int(preset.fps * preset.duration_seconds)
            )
            
            if est_memory["recommended"] <= target_vram_gb:
                logger.info(f"Selected {preset_name} preset for {target_vram_gb}GB VRAM")
                return preset
                
        # If nothing fits, return draft with additional optimizations
        draft = VIDEO_QUALITY_PRESETS["draft"]
        draft.cpu_offload = True
        draft.vae_slicing = True
        draft.quantization = QuantizationType.INT8
        return draft
        
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.loaded