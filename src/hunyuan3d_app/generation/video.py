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
    LTXVIDEO = "ltxvideo"
    MOCHI_PREVIEW = "mochi-preview"
    COGVIDEOX_5B = "cogvideox-5b"


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
                "high resolution",
                "diffusers compatible"
            ],
            recommended_settings={
                "width": 1216,
                "height": 704,
                "fps": 30,
                "duration": 5.0,
                "steps": 25
            },
            vram_required=24.0,
            generation_speed="real-time generation",
            max_duration=10.0,
            supported_resolutions=[(768, 512), (1024, 576), (1216, 704)]
        ),
        VideoModel.MOCHI_PREVIEW: VideoModelInfo(
            name="Mochi 1 Preview",
            model_id="genmo/mochi-1-preview",
            capabilities=[
                "480p video",
                "efficient generation",
                "low VRAM usage"
            ],
            recommended_settings={
                "width": 848,
                "height": 480,
                "fps": 30,
                "duration": 5.0,
                "steps": 64
            },
            vram_required=12.0,
            generation_speed="~90s for 5s video",
            max_duration=5.0,
            supported_resolutions=[(640, 360), (848, 480), (1024, 576)]
        ),
        VideoModel.COGVIDEOX_5B: VideoModelInfo(
            name="CogVideoX-5B",
            model_id="THUDM/CogVideoX-5b",
            capabilities=[
                "high quality",
                "stable generation",
                "720x480 video"
            ],
            recommended_settings={
                "width": 720,
                "height": 480,
                "fps": 8,
                "duration": 6.0,
                "steps": 50
            },
            vram_required=16.0,
            generation_speed="~3min for 6s video",
            max_duration=6.0,
            supported_resolutions=[(512, 320), (720, 480), (1024, 576)]
        )
    }
    
    def __init__(self, cache_dir: Optional[Path] = None, models_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path("./cache/video")
        self.models_dir = models_dir or Path("./models")
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
            elif model_type == VideoModel.MOCHI_PREVIEW:
                model = self._load_mochi(model_info, device, dtype, progress_callback)
            elif model_type == VideoModel.COGVIDEOX_5B:
                model = self._load_cogvideox(model_info, device, dtype, progress_callback)
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
        try:
            if progress_callback:
                progress_callback(0.3, "Loading LTX-Video components...")
                
            from diffusers import DiffusionPipeline, LTXPipeline, LTXImageToVideoPipeline
            from transformers import T5EncoderModel
            
            # Load the pipeline
            if progress_callback:
                progress_callback(0.5, "Loading transformer model...")
                
            # Check if model exists locally first
            model_path = self.models_dir / "video" / "ltxvideo"
            if model_path.exists():
                pipeline = LTXPipeline.from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                    local_files_only=True
                ).to(device)
            else:
                # Download from HuggingFace
                pipeline = LTXPipeline.from_pretrained(
                    model_info.model_id,
                    torch_dtype=dtype,
                    cache_dir=self.cache_dir
                ).to(device)
                
            # Enable memory optimizations
            if hasattr(pipeline, "enable_model_cpu_offload"):
                pipeline.enable_model_cpu_offload()
            if hasattr(pipeline, "enable_vae_slicing"):
                pipeline.enable_vae_slicing()
                
            if progress_callback:
                progress_callback(0.8, "LTX-Video pipeline ready!")
                
            return pipeline
            
        except ImportError:
            logger.warning("LTX-Video not available, using alternative implementation")
            # Fallback to CogVideoX or other available model
            return self._load_cogvideox_fallback(model_info, device, dtype, progress_callback)
        
    def _load_cogvideox_fallback(
        self,
        model_info: VideoModelInfo,
        device: str,
        dtype: torch.dtype,
        progress_callback: Optional[Callable]
    ) -> Any:
        """Load CogVideoX as fallback"""
        try:
            from diffusers import CogVideoXPipeline
            
            if progress_callback:
                progress_callback(0.3, "Loading CogVideoX as alternative...")
                
            # Use CogVideoX-5b as it's well-supported
            pipeline = CogVideoXPipeline.from_pretrained(
                "THUDM/CogVideoX-5b",
                torch_dtype=dtype,
                cache_dir=self.cache_dir
            )
            
            # Enable optimizations
            pipeline.enable_model_cpu_offload()
            pipeline.enable_vae_slicing()
            
            if progress_callback:
                progress_callback(0.8, "CogVideoX ready!")
                
            return pipeline
            
        except Exception as e:
            logger.error(f"Failed to load CogVideoX: {e}")
            raise
            
    def _load_mochi(
        self,
        model_info: VideoModelInfo,
        device: str,
        dtype: torch.dtype,
        progress_callback: Optional[Callable]
    ) -> Any:
        """Load Mochi Preview model"""
        try:
            if progress_callback:
                progress_callback(0.3, "Loading Mochi Preview...")
                
            from diffusers import MochiPipeline
            
            # Load from local or HuggingFace
            model_path = self.models_dir / "video" / "mochi-preview"
            if model_path.exists():
                pipeline = MochiPipeline.from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                    local_files_only=True
                ).to(device)
            else:
                pipeline = MochiPipeline.from_pretrained(
                    "genmo/mochi-1-preview",
                    torch_dtype=dtype,
                    cache_dir=self.cache_dir
                ).to(device)
                
            # Enable optimizations
            if hasattr(pipeline, "enable_model_cpu_offload"):
                pipeline.enable_model_cpu_offload()
            if hasattr(pipeline, "enable_vae_slicing"):
                pipeline.enable_vae_slicing()
                
            if progress_callback:
                progress_callback(0.8, "Mochi Preview ready!")
                
            return pipeline
            
        except Exception as e:
            logger.warning(f"Failed to load Mochi: {e}")
            # Fallback to CogVideoX
            return self._load_cogvideox_fallback(model_info, device, dtype, progress_callback)
            
    def _load_cogvideox(
        self,
        model_info: VideoModelInfo,
        device: str,
        dtype: torch.dtype,
        progress_callback: Optional[Callable]
    ) -> Any:
        """Load CogVideoX model"""
        try:
            if progress_callback:
                progress_callback(0.3, "Loading CogVideoX-5b...")
                
            from diffusers import CogVideoXPipeline, CogVideoXDPMScheduler
            
            # Load the pipeline
            pipeline = CogVideoXPipeline.from_pretrained(
                model_info.model_id,
                torch_dtype=dtype,
                cache_dir=self.cache_dir
            )
            
            # Use DPM scheduler for better quality
            pipeline.scheduler = CogVideoXDPMScheduler.from_config(
                pipeline.scheduler.config, 
                timestep_spacing="trailing"
            )
            
            # Enable optimizations
            pipeline.enable_model_cpu_offload()
            pipeline.enable_vae_slicing()
            pipeline.vae.enable_tiling()
            
            if progress_callback:
                progress_callback(0.8, "CogVideoX-5b ready!")
                
            return pipeline
            
        except Exception as e:
            logger.error(f"Failed to load CogVideoX: {e}")
            raise
        
        
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
                
            # Generate video based on model type
            if hasattr(self.current_model, 'generate'):
                # For models with generate method
                result = self.current_model.generate(**generation_kwargs)
            else:
                # For standard diffusers pipelines
                result = self.current_model(**generation_kwargs)
            
            # Post-process frames if needed
            if progress_callback:
                progress_callback(0.9, "Post-processing frames...")
                
            # Extract frames from result
            if hasattr(result, 'frames'):
                frames = result.frames
            elif isinstance(result, list):
                frames = result
            elif hasattr(result, 'images'):
                frames = result.images
            else:
                # Handle tensor outputs from video models
                if hasattr(result, '__getitem__') and 'frames' in result:
                    frames_tensor = result['frames'][0]  # Remove batch dimension
                    # Convert tensor to numpy then to PIL
                    frames_np = frames_tensor.cpu().numpy()
                    # Rearrange from (frames, channels, height, width) to list of PIL images
                    frames = []
                    for i in range(frames_np.shape[0]):
                        frame = frames_np[i]
                        # Convert from CHW to HWC if needed
                        if frame.shape[0] in [3, 4]:  # channels first
                            frame = np.transpose(frame, (1, 2, 0))
                        # Normalize to 0-255 range
                        if frame.max() <= 1.0:
                            frame = (frame * 255).astype(np.uint8)
                        frames.append(Image.fromarray(frame))
                else:
                    logger.error(f"Unknown result type from pipeline: {type(result)}")
                    return None, {"error": "Unknown output format from model"}
                
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
            VideoModel.MOCHI_PREVIEW: 18.0,   # seconds per second of video
            VideoModel.COGVIDEOX_5B: 30.0   # seconds per second of video
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