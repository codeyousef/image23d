"""Video generation module supporting multiple open-source models"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Callable, List
from dataclasses import dataclass
from enum import Enum
import time

import torch
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class VideoModel(Enum):
    """Available video generation models"""
    LTXVIDEO = "LTXVideo"
    WAN_2_1 = "Wan2.1"
    SKYREELS = "SkyReels"


@dataclass
class VideoGenerationParams:
    """Parameters for video generation"""
    prompt: str
    negative_prompt: str = ""
    duration_seconds: float = 5.0
    fps: int = 24
    width: int = 768
    height: int = 512
    motion_strength: float = 1.0
    guidance_scale: float = 7.5
    num_inference_steps: int = 30
    seed: int = -1
    # Character consistency
    character_embeddings: Optional[torch.Tensor] = None
    consistency_strength: float = 0.8
    # Model-specific params
    model_specific: Dict[str, Any] = None


@dataclass
class VideoModelInfo:
    """Information about a video model"""
    name: str
    model_id: str
    capabilities: List[str]
    recommended_settings: Dict[str, Any]
    vram_required: float  # GB
    generation_speed: str  # Description
    max_duration: float  # seconds
    supported_resolutions: List[Tuple[int, int]]
    requires_login: bool = False


class VideoGenerator:
    """Manages video generation with multiple models"""
    
    # Model configurations
    VIDEO_MODELS = {
        VideoModel.LTXVIDEO: VideoModelInfo(
            name="LTX-Video",
            model_id="Lightricks/LTX-Video",
            capabilities=[
                "real-time generation",
                "multiscale rendering",
                "high quality motion",
                "comfyui compatible"
            ],
            recommended_settings={
                "width": 768,
                "height": 512,
                "fps": 24,
                "duration": 5.0,
                "steps": 30
            },
            vram_required=12.0,
            generation_speed="4s for 5s video",
            max_duration=10.0,
            supported_resolutions=[(768, 512), (1024, 576), (1280, 720)]
        ),
        VideoModel.WAN_2_1: VideoModelInfo(
            name="Wan 2.1",
            model_id="alibaba/Wan2.1",
            capabilities=[
                "multilingual prompts",
                "3D causal VAE",
                "text rendering",
                "Chinese support"
            ],
            recommended_settings={
                "width": 640,
                "height": 480,
                "fps": 16,
                "duration": 5.0,
                "steps": 50
            },
            vram_required=16.0,
            generation_speed="4min for 5s video",
            max_duration=8.0,
            supported_resolutions=[(640, 480), (854, 480), (1280, 720)]
        ),
        VideoModel.SKYREELS: VideoModelInfo(
            name="SkyReels V1",
            model_id="skyreels/skyreels-v1",
            capabilities=[
                "cinematic human animation",
                "33 facial expressions",
                "400+ movement types",
                "film-quality output"
            ],
            recommended_settings={
                "width": 1024,
                "height": 576,
                "fps": 30,
                "duration": 4.0,
                "steps": 40
            },
            vram_required=20.0,
            generation_speed="2min for 4s video",
            max_duration=6.0,
            supported_resolutions=[(1024, 576), (1280, 720), (1920, 1080)]
        )
    }
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path("./cache/video")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = {}
        self.current_model = None
        self.current_model_type = None
        
    def get_available_models(self) -> Dict[str, VideoModelInfo]:
        """Get all available video models"""
        return {model.value: info for model, info in self.VIDEO_MODELS.items()}
        
    def get_model_info(self, model_type: VideoModel) -> VideoModelInfo:
        """Get information about a specific model"""
        return self.VIDEO_MODELS.get(model_type)
        
    def load_model(
        self,
        model_type: VideoModel,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        progress_callback: Optional[Callable] = None
    ) -> Tuple[bool, str]:
        """Load a video generation model
        
        Args:
            model_type: Type of model to load
            device: Device to load on
            dtype: Model precision
            progress_callback: Progress callback function
            
        Returns:
            Tuple of (success, message)
        """
        try:
            if self.current_model_type == model_type:
                return True, f"{model_type.value} already loaded"
                
            # Clear previous model
            if self.current_model is not None:
                del self.current_model
                torch.cuda.empty_cache()
                
            model_info = self.VIDEO_MODELS[model_type]
            
            if progress_callback:
                progress_callback(0.1, f"Loading {model_info.name}...")
                
            # Model-specific loading
            if model_type == VideoModel.LTXVIDEO:
                model = self._load_ltxvideo(model_info, device, dtype, progress_callback)
            elif model_type == VideoModel.WAN_2_1:
                model = self._load_wan(model_info, device, dtype, progress_callback)
            elif model_type == VideoModel.SKYREELS:
                model = self._load_skyreels(model_info, device, dtype, progress_callback)
            else:
                return False, f"Unknown model type: {model_type}"
                
            self.current_model = model
            self.current_model_type = model_type
            
            if progress_callback:
                progress_callback(1.0, f"{model_info.name} loaded successfully")
                
            return True, f"{model_info.name} loaded successfully"
            
        except Exception as e:
            logger.error(f"Failed to load video model: {e}")
            return False, f"Failed to load model: {str(e)}"
            
    def _load_ltxvideo(
        self,
        model_info: VideoModelInfo,
        device: str,
        dtype: torch.dtype,
        progress_callback: Optional[Callable]
    ) -> Any:
        """Load LTX-Video model"""
        # This is a placeholder - actual implementation would load the real model
        # For now, we'll create a mock structure
        
        if progress_callback:
            progress_callback(0.3, "Downloading LTX-Video components...")
            
        # In real implementation:
        # from ltxvideo import LTXVideoPipeline
        # pipeline = LTXVideoPipeline.from_pretrained(
        #     model_info.model_id,
        #     torch_dtype=dtype
        # ).to(device)
        
        # Mock implementation
        class MockLTXVideo:
            def __init__(self):
                self.device = device
                self.dtype = dtype
                
            def generate(self, prompt, **kwargs):
                # Mock video generation
                frames = []
                num_frames = int(kwargs.get("num_frames", 120))
                for i in range(num_frames):
                    frame = Image.new("RGB", (768, 512), color=(i, i, i))
                    frames.append(frame)
                return frames
                
        if progress_callback:
            progress_callback(0.8, "Initializing LTX-Video pipeline...")
            
        return MockLTXVideo()
        
    def _load_wan(
        self,
        model_info: VideoModelInfo,
        device: str,
        dtype: torch.dtype,
        progress_callback: Optional[Callable]
    ) -> Any:
        """Load Wan 2.1 model"""
        # Placeholder implementation
        
        if progress_callback:
            progress_callback(0.3, "Downloading Wan 2.1 components...")
            
        class MockWan:
            def __init__(self):
                self.device = device
                self.dtype = dtype
                
            def generate(self, prompt, **kwargs):
                frames = []
                num_frames = int(kwargs.get("num_frames", 80))
                for i in range(num_frames):
                    frame = Image.new("RGB", (640, 480), color=(255-i, i, 128))
                    frames.append(frame)
                return frames
                
        if progress_callback:
            progress_callback(0.8, "Initializing Wan 2.1 pipeline...")
            
        return MockWan()
        
    def _load_skyreels(
        self,
        model_info: VideoModelInfo,
        device: str,
        dtype: torch.dtype,
        progress_callback: Optional[Callable]
    ) -> Any:
        """Load SkyReels model"""
        # Placeholder implementation
        
        if progress_callback:
            progress_callback(0.3, "Downloading SkyReels components...")
            
        class MockSkyReels:
            def __init__(self):
                self.device = device
                self.dtype = dtype
                
            def generate(self, prompt, **kwargs):
                frames = []
                num_frames = int(kwargs.get("num_frames", 120))
                for i in range(num_frames):
                    frame = Image.new("RGB", (1024, 576), color=(i//2, 128, 255-i//2))
                    frames.append(frame)
                return frames
                
        if progress_callback:
            progress_callback(0.8, "Initializing SkyReels pipeline...")
            
        return MockSkyReels()
        
    def generate_video(
        self,
        params: VideoGenerationParams,
        progress_callback: Optional[Callable] = None
    ) -> Tuple[Optional[List[Image.Image]], Dict[str, Any]]:
        """Generate a video
        
        Args:
            params: Video generation parameters
            progress_callback: Progress callback
            
        Returns:
            Tuple of (frames list, info dict)
        """
        if self.current_model is None:
            return None, {"error": "No model loaded"}
            
        try:
            start_time = time.time()
            
            if progress_callback:
                progress_callback(0.0, "Preparing generation...")
                
            # Calculate number of frames
            num_frames = int(params.duration_seconds * params.fps)
            
            # Get model info
            model_info = self.VIDEO_MODELS[self.current_model_type]
            
            # Validate parameters
            if params.duration_seconds > model_info.max_duration:
                logger.warning(
                    f"Requested duration {params.duration_seconds}s exceeds "
                    f"max {model_info.max_duration}s for {model_info.name}"
                )
                params.duration_seconds = model_info.max_duration
                num_frames = int(params.duration_seconds * params.fps)
                
            # Check resolution
            if (params.width, params.height) not in model_info.supported_resolutions:
                # Find closest resolution
                closest = min(
                    model_info.supported_resolutions,
                    key=lambda r: abs(r[0] - params.width) + abs(r[1] - params.height)
                )
                logger.warning(
                    f"Resolution {params.width}x{params.height} not supported, "
                    f"using {closest[0]}x{closest[1]}"
                )
                params.width, params.height = closest
                
            if progress_callback:
                progress_callback(0.1, f"Generating {num_frames} frames...")
                
            # Model-specific generation
            generation_kwargs = {
                "prompt": params.prompt,
                "negative_prompt": params.negative_prompt,
                "num_frames": num_frames,
                "width": params.width,
                "height": params.height,
                "guidance_scale": params.guidance_scale,
                "num_inference_steps": params.num_inference_steps,
                "generator": torch.Generator().manual_seed(
                    params.seed if params.seed >= 0 else torch.randint(0, 2**32, (1,)).item()
                )
            }
            
            # Add character embeddings if provided
            if params.character_embeddings is not None:
                generation_kwargs["ip_adapter_image_embeds"] = params.character_embeddings
                generation_kwargs["ip_adapter_scale"] = params.consistency_strength
                
            # Add model-specific parameters
            if params.model_specific:
                generation_kwargs.update(params.model_specific)
                
            # Generate frames
            if progress_callback:
                # Wrap generation with progress updates
                def generation_progress(step, total):
                    progress = 0.1 + (step / total) * 0.8
                    progress_callback(progress, f"Step {step}/{total}")
                    
                generation_kwargs["callback"] = generation_progress
                generation_kwargs["callback_steps"] = 1
                
            frames = self.current_model.generate(**generation_kwargs)
            
            # Post-process frames if needed
            if progress_callback:
                progress_callback(0.9, "Post-processing frames...")
                
            # Ensure frames are PIL Images
            if frames and not isinstance(frames[0], Image.Image):
                frames = [Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame
                         for frame in frames]
                         
            generation_time = time.time() - start_time
            
            info = {
                "model": self.current_model_type.value,
                "duration": params.duration_seconds,
                "fps": params.fps,
                "resolution": f"{params.width}x{params.height}",
                "frames": len(frames),
                "generation_time": f"{generation_time:.1f}s",
                "seed": generation_kwargs["generator"].initial_seed()
            }
            
            if progress_callback:
                progress_callback(1.0, f"Generated {len(frames)} frames in {generation_time:.1f}s")
                
            return frames, info
            
        except Exception as e:
            logger.error(f"Video generation failed: {e}")
            return None, {"error": str(e)}
            
    def save_video(
        self,
        frames: List[Image.Image],
        output_path: Path,
        fps: int = 24,
        codec: str = "libx264",
        quality: str = "high"
    ) -> bool:
        """Save frames as video file
        
        Args:
            frames: List of PIL Images
            output_path: Output video path
            fps: Frames per second
            codec: Video codec
            quality: Quality preset (low, medium, high)
            
        Returns:
            Success boolean
        """
        try:
            import cv2
            
            if not frames:
                return False
                
            # Get dimensions from first frame
            width, height = frames[0].size
            
            # Set up video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                str(output_path),
                fourcc,
                fps,
                (width, height)
            )
            
            # Write frames
            for frame in frames:
                # Convert PIL to OpenCV format
                cv_frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
                out.write(cv_frame)
                
            out.release()
            return True
            
        except Exception as e:
            logger.error(f"Failed to save video: {e}")
            return False
            
    def create_preview_gif(
        self,
        frames: List[Image.Image],
        output_path: Path,
        fps: int = 10,
        max_frames: int = 50
    ) -> bool:
        """Create a preview GIF from frames
        
        Args:
            frames: List of PIL Images
            output_path: Output GIF path
            fps: Frames per second for GIF
            max_frames: Maximum frames to include
            
        Returns:
            Success boolean
        """
        try:
            if not frames:
                return False
                
            # Limit frames for GIF
            preview_frames = frames[:max_frames]
            
            # Calculate duration per frame in milliseconds
            duration = int(1000 / fps)
            
            # Save as GIF
            preview_frames[0].save(
                output_path,
                save_all=True,
                append_images=preview_frames[1:],
                duration=duration,
                loop=0
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create preview GIF: {e}")
            return False
            
    def estimate_generation_time(
        self,
        model_type: VideoModel,
        params: VideoGenerationParams
    ) -> float:
        """Estimate generation time in seconds
        
        Args:
            model_type: Model to use
            params: Generation parameters
            
        Returns:
            Estimated time in seconds
        """
        model_info = self.VIDEO_MODELS.get(model_type)
        if not model_info:
            return 0.0
            
        # Base estimates (these would be calibrated from real benchmarks)
        base_times = {
            VideoModel.LTXVIDEO: 0.8,  # seconds per second of video
            VideoModel.WAN_2_1: 48.0,   # seconds per second of video
            VideoModel.SKYREELS: 30.0   # seconds per second of video
        }
        
        base_time = base_times.get(model_type, 60.0)
        
        # Adjust for resolution
        base_res = model_info.recommended_settings["width"] * model_info.recommended_settings["height"]
        target_res = params.width * params.height
        resolution_factor = target_res / base_res
        
        # Adjust for steps
        base_steps = model_info.recommended_settings["steps"]
        step_factor = params.num_inference_steps / base_steps
        
        # Calculate estimate
        estimated_time = base_time * params.duration_seconds * resolution_factor * step_factor
        
        return estimated_time
        
    def get_optimal_settings(
        self,
        model_type: VideoModel,
        target_quality: str = "balanced"
    ) -> VideoGenerationParams:
        """Get optimal settings for a model and quality target
        
        Args:
            model_type: Model type
            target_quality: Quality target (fast, balanced, quality)
            
        Returns:
            Optimized parameters
        """
        model_info = self.VIDEO_MODELS.get(model_type)
        if not model_info:
            return VideoGenerationParams(prompt="")
            
        base_settings = model_info.recommended_settings.copy()
        
        # Adjust based on quality target
        if target_quality == "fast":
            base_settings["steps"] = max(10, base_settings["steps"] // 2)
            base_settings["width"] = min(base_settings["width"], 512)
            base_settings["height"] = min(base_settings["height"], 384)
            base_settings["duration"] = min(base_settings["duration"], 3.0)
        elif target_quality == "quality":
            base_settings["steps"] = int(base_settings["steps"] * 1.5)
            # Use highest supported resolution
            if model_info.supported_resolutions:
                max_res = max(model_info.supported_resolutions, key=lambda r: r[0] * r[1])
                base_settings["width"], base_settings["height"] = max_res
                
        return VideoGenerationParams(
            prompt="",
            duration_seconds=base_settings.get("duration", 5.0),
            fps=base_settings.get("fps", 24),
            width=base_settings.get("width", 768),
            height=base_settings.get("height", 512),
            num_inference_steps=base_settings.get("steps", 30)
        )