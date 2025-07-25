"""ComfyUI Integration for Video Generation Models

Provides custom nodes for ComfyUI to use all video models:
- Wan2.1 (1.3B/14B)
- HunyuanVideo (13B)
- LTX-Video (2B)
- Mochi-1 (10B)
- CogVideoX-5B (5B)

Installation:
1. Copy this file to ComfyUI/custom_nodes/
2. Install dependencies: pip install -r requirements.txt
3. Restart ComfyUI
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import torch
import numpy as np
from PIL import Image
import folder_paths
import comfy.utils

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from .base import VideoModelType, VideoQualityPreset, VIDEO_QUALITY_PRESETS
from .wan21 import Wan21VideoModel
from .hunyuanvideo import HunyuanVideoModel
from .ltxvideo import LTXVideoModel
from .mochi import Mochi1Model
from .cogvideox import CogVideoX5BModel
from .memory_optimizer import VideoMemoryOptimizer, auto_optimize_for_hardware

# ComfyUI folder registration
folder_paths.folder_names_and_paths["video_models"] = (
    [os.path.join(folder_paths.models_dir, "video")],
    folder_paths.supported_pt_extensions
)


class VideoModelLoader:
    """Load video generation models"""
    
    MODELS = {
        "Wan2.1 1.3B (8GB VRAM)": VideoModelType.WAN2_1_1_3B,
        "Wan2.1 14B (16GB VRAM)": VideoModelType.WAN2_1_14B,
        "HunyuanVideo (24GB VRAM)": VideoModelType.HUNYUANVIDEO,
        "LTX-Video Real-time (12GB)": VideoModelType.LTX_VIDEO,
        "Mochi-1 30fps (24GB)": VideoModelType.MOCHI_1,
        "CogVideoX-5B I2V (16GB)": VideoModelType.COGVIDEOX_5B
    }
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (list(cls.MODELS.keys()),),
                "device": (["cuda", "cpu"],),
                "precision": (["fp16", "fp32", "bf16"],),
                "auto_optimize": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("VIDEO_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "video/models"
    
    def load_model(self, model: str, device: str, precision: str, auto_optimize: bool):
        """Load selected video model"""
        
        model_type = self.MODELS[model]
        
        # Convert precision
        dtype_map = {
            "fp16": torch.float16,
            "fp32": torch.float32,
            "bf16": torch.bfloat16
        }
        dtype = dtype_map[precision]
        
        # Create model config
        from . import VideoModelConfig, create_video_model
        
        # Create model
        video_model = create_video_model(
            model_type=model_type,
            device=device,
            dtype=precision,
            cache_dir=Path(folder_paths.models_dir) / "video" / "cache"
        )
        
        # Apply auto-optimization
        if auto_optimize:
            auto_optimize_for_hardware(video_model)
            
        # Load model
        success = video_model.load()
        if not success:
            raise RuntimeError(f"Failed to load {model}")
            
        return (video_model,)


class VideoGenerator:
    """Generate videos from text prompts"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("VIDEO_MODEL",),
                "prompt": ("STRING", {"multiline": True}),
                "quality_preset": (["draft", "standard", "high", "ultra", "custom"],),
                "duration_seconds": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 10.0, "step": 0.5}),
                "fps": ("INT", {"default": 24, "min": 8, "max": 60, "step": 1}),
                "width": ("INT", {"default": 768, "min": 256, "max": 1920, "step": 64}),
                "height": ("INT", {"default": 512, "min": 256, "max": 1080, "step": 64}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "guidance_scale": ("FLOAT", {"default": 7.5, "min": 1.0, "max": 20.0}),
                "num_inference_steps": ("INT", {"default": 30, "min": 10, "max": 100}),
                "motion_bucket_id": ("INT", {"default": 127, "min": 1, "max": 255}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "VIDEO_INFO")
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "generate"
    CATEGORY = "video/generation"
    
    def generate(
        self,
        model,
        prompt: str,
        quality_preset: str,
        duration_seconds: float,
        fps: int,
        width: int,
        height: int,
        seed: int,
        negative_prompt: str = "",
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
        motion_bucket_id: int = 127
    ):
        """Generate video frames"""
        
        # Get quality preset
        preset = None
        if quality_preset != "custom":
            preset = VIDEO_QUALITY_PRESETS.get(quality_preset)
            if preset:
                # Override with preset values
                width, height = preset.resolution
                fps = preset.fps
                num_inference_steps = preset.inference_steps
                guidance_scale = preset.guidance_scale
                
        # Calculate frames
        num_frames = int(duration_seconds * fps)
        
        # Handle seed
        if seed == -1:
            seed = None
            
        # Progress callback for ComfyUI
        pbar = comfy.utils.ProgressBar(num_inference_steps)
        def progress_callback(progress, message):
            pbar.update_absolute(int(progress * num_inference_steps), message)
            
        # Generate video
        result = model.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            fps=fps,
            motion_bucket_id=motion_bucket_id,
            seed=seed,
            progress_callback=progress_callback
        )
        
        # Convert frames to ComfyUI format
        frames_np = []
        for frame in result.frames:
            # Convert PIL to numpy
            frame_np = np.array(frame).astype(np.float32) / 255.0
            frames_np.append(frame_np)
            
        # Create info dict
        info = {
            "duration": result.duration,
            "fps": result.fps,
            "resolution": result.resolution,
            "frame_count": len(result.frames),
            "model": str(model.config.model_type),
            "seed": result.metadata.get("seed", seed)
        }
        
        return (frames_np, info)


class ImageToVideo:
    """Generate video from input image"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("VIDEO_MODEL",),
                "image": ("IMAGE",),
                "duration_seconds": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 10.0}),
                "fps": ("INT", {"default": 24, "min": 8, "max": 60}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
            },
            "optional": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "guidance_scale": ("FLOAT", {"default": 6.0, "min": 1.0, "max": 20.0}),
                "num_inference_steps": ("INT", {"default": 50, "min": 10, "max": 100}),
                "motion_bucket_id": ("INT", {"default": 127, "min": 1, "max": 255}),
                "noise_aug_strength": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 1.0}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "VIDEO_INFO")
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "animate"
    CATEGORY = "video/i2v"
    
    def animate(
        self,
        model,
        image,
        duration_seconds: float,
        fps: int,
        seed: int,
        prompt: str = "",
        negative_prompt: str = "",
        guidance_scale: float = 6.0,
        num_inference_steps: int = 50,
        motion_bucket_id: int = 127,
        noise_aug_strength: float = 0.02
    ):
        """Animate input image"""
        
        # Check if model supports I2V
        if not model.supports_feature("image_to_video"):
            raise ValueError(f"Model {model.config.model_type} does not support image-to-video")
            
        # Convert ComfyUI image to PIL
        if isinstance(image, torch.Tensor):
            # Assume shape is [B, H, W, C]
            if len(image.shape) == 4:
                image = image[0]
            image_np = (image.cpu().numpy() * 255).astype(np.uint8)
            image_pil = Image.fromarray(image_np)
        else:
            image_pil = Image.fromarray((image * 255).astype(np.uint8))
            
        # Calculate frames
        num_frames = int(duration_seconds * fps)
        
        # Handle seed
        if seed == -1:
            seed = None
            
        # Progress callback
        pbar = comfy.utils.ProgressBar(num_inference_steps)
        def progress_callback(progress, message):
            pbar.update_absolute(int(progress * num_inference_steps), message)
            
        # Generate video
        result = model.image_to_video(
            image=image_pil,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            fps=fps,
            motion_bucket_id=motion_bucket_id,
            noise_aug_strength=noise_aug_strength,
            seed=seed,
            progress_callback=progress_callback
        )
        
        # Convert frames
        frames_np = []
        for frame in result.frames:
            frame_np = np.array(frame).astype(np.float32) / 255.0
            frames_np.append(frame_np)
            
        # Info
        info = {
            "duration": result.duration,
            "fps": result.fps,
            "resolution": result.resolution,
            "frame_count": len(result.frames),
            "model": str(model.config.model_type),
            "mode": "image_to_video"
        }
        
        return (frames_np, info)


class VideoMemoryOptimizer:
    """Optimize video model memory usage"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("VIDEO_MODEL",),
                "optimization_level": (["minimal", "moderate", "aggressive", "extreme"],),
                "target_vram_gb": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 48.0}),
            }
        }
    
    RETURN_TYPES = ("VIDEO_MODEL", "OPTIMIZATION_INFO")
    FUNCTION = "optimize"
    CATEGORY = "video/optimization"
    
    def optimize(self, model, optimization_level: str, target_vram_gb: float):
        """Apply memory optimizations"""
        
        from . import OptimizationLevel
        
        # Map string to enum
        level_map = {
            "minimal": OptimizationLevel.MINIMAL,
            "moderate": OptimizationLevel.MODERATE,
            "aggressive": OptimizationLevel.AGGRESSIVE,
            "extreme": OptimizationLevel.EXTREME
        }
        level = level_map[optimization_level]
        
        # Create optimizer
        optimizer = VideoMemoryOptimizer()
        
        # Apply optimizations
        if target_vram_gb > 0:
            optimizations = optimizer.optimize_model(model, level, target_vram_gb)
        else:
            optimizations = optimizer.optimize_model(model, level)
            
        # Get memory profile
        profile = optimizer.profile_system()
        
        info = {
            "optimization_level": optimization_level,
            "optimizations_applied": list(optimizations.keys()),
            "gpu_free_gb": profile.gpu_free,
            "gpu_total_gb": profile.gpu_total,
            "can_run": profile.can_run,
            "warnings": profile.warnings
        }
        
        return (model, info)


class VideoLoRALoader:
    """Load LoRA adapters for video models"""
    
    @classmethod
    def INPUT_TYPES(cls):
        # Get LoRA files
        lora_files = folder_paths.get_filename_list("loras")
        
        return {
            "required": {
                "model": ("VIDEO_MODEL",),
                "lora_name": (lora_files,),
                "strength": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.1}),
            }
        }
    
    RETURN_TYPES = ("VIDEO_MODEL",)
    FUNCTION = "load_lora"
    CATEGORY = "video/lora"
    
    def load_lora(self, model, lora_name: str, strength: float):
        """Load LoRA adapter"""
        
        # Check if model supports LoRA
        if not model.supports_feature("lora"):
            raise ValueError(f"Model {model.config.model_type} does not support LoRA")
            
        # Get LoRA path
        lora_path = folder_paths.get_full_path("loras", lora_name)
        
        # Add LoRA to model
        if hasattr(model, 'add_lora'):
            success = model.add_lora(lora_name, lora_path, strength)
            if not success:
                raise RuntimeError(f"Failed to load LoRA: {lora_name}")
                
        return (model,)


class VideoFrameInterpolator:
    """Interpolate frames to increase FPS"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "target_fps": ("INT", {"default": 30, "min": 8, "max": 60}),
                "original_fps": ("INT", {"default": 8, "min": 1, "max": 60}),
                "method": (["linear", "optical_flow"],),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "interpolate"
    CATEGORY = "video/postprocess"
    
    def interpolate(self, frames: List, target_fps: int, original_fps: int, method: str):
        """Interpolate frames to higher FPS"""
        
        if target_fps <= original_fps:
            return (frames,)
            
        interpolated = []
        interpolation_factor = target_fps // original_fps
        
        for i in range(len(frames) - 1):
            interpolated.append(frames[i])
            
            # Add interpolated frames
            for j in range(1, interpolation_factor):
                alpha = j / interpolation_factor
                
                if method == "linear":
                    # Simple linear interpolation
                    interp_frame = (1 - alpha) * frames[i] + alpha * frames[i + 1]
                else:
                    # Placeholder for optical flow
                    interp_frame = (1 - alpha) * frames[i] + alpha * frames[i + 1]
                    
                interpolated.append(interp_frame)
                
        interpolated.append(frames[-1])
        
        return (interpolated,)


class VideoSaver:
    """Save video frames to file"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "fps": ("INT", {"default": 24, "min": 1, "max": 60}),
                "filename_prefix": ("STRING", {"default": "video"}),
                "format": (["mp4", "webm", "gif"],),
                "quality": (["low", "medium", "high"],),
            },
            "optional": {
                "video_info": ("VIDEO_INFO",),
            }
        }
    
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save_video"
    CATEGORY = "video/output"
    
    def save_video(
        self,
        frames: List,
        fps: int,
        filename_prefix: str,
        format: str,
        quality: str,
        video_info: Optional[Dict] = None
    ):
        """Save frames as video file"""
        
        import cv2
        from datetime import datetime
        
        # Create output directory
        output_dir = Path(folder_paths.output_directory) / "videos"
        output_dir.mkdir(exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.{format}"
        output_path = output_dir / filename
        
        # Convert frames to numpy if needed
        frames_np = []
        for frame in frames:
            if isinstance(frame, torch.Tensor):
                frame_np = (frame.cpu().numpy() * 255).astype(np.uint8)
            else:
                frame_np = (frame * 255).astype(np.uint8)
            frames_np.append(frame_np)
            
        if format == "gif":
            # Save as GIF
            images = [Image.fromarray(frame) for frame in frames_np]
            images[0].save(
                output_path,
                save_all=True,
                append_images=images[1:],
                duration=int(1000 / fps),
                loop=0
            )
        else:
            # Save as video
            height, width = frames_np[0].shape[:2]
            
            # Codec selection
            if format == "mp4":
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            else:  # webm
                fourcc = cv2.VideoWriter_fourcc(*'VP90')
                
            # Quality settings
            quality_map = {"low": 50, "medium": 75, "high": 95}
            compression = quality_map[quality]
            
            # Create video writer
            out = cv2.VideoWriter(
                str(output_path),
                fourcc,
                fps,
                (width, height)
            )
            
            # Write frames
            for frame in frames_np:
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
                
            out.release()
            
        # Add metadata if available
        if video_info:
            metadata_path = output_path.with_suffix('.json')
            import json
            with open(metadata_path, 'w') as f:
                json.dump(video_info, f, indent=2)
                
        return {"ui": {"videos": [str(output_path)]}}


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "VideoModelLoader": VideoModelLoader,
    "VideoGenerator": VideoGenerator,
    "ImageToVideo": ImageToVideo,
    "VideoMemoryOptimizer": VideoMemoryOptimizer,
    "VideoLoRALoader": VideoLoRALoader,
    "VideoFrameInterpolator": VideoFrameInterpolator,
    "VideoSaver": VideoSaver
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoModelLoader": "Load Video Model",
    "VideoGenerator": "Generate Video",
    "ImageToVideo": "Image to Video",
    "VideoMemoryOptimizer": "Optimize Video Memory",
    "VideoLoRALoader": "Load Video LoRA",
    "VideoFrameInterpolator": "Interpolate Frames",
    "VideoSaver": "Save Video"
}

# Export for ComfyUI
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']