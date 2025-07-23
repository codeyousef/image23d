"""
Core video processing logic shared between platforms
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import uuid

from PIL import Image
import numpy as np

from ..models.generation import VideoGenerationRequest, VideoGenerationResponse, GenerationStatus
from ..models.enhancement import ModelType
from .prompt_enhancer import PromptEnhancer

logger = logging.getLogger(__name__)

class VideoProcessor:
    """Handles video generation with prompt enhancement"""
    
    def __init__(self, model_manager, output_dir: Path, prompt_enhancer: Optional[PromptEnhancer] = None):
        self.model_manager = model_manager
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prompt_enhancer = prompt_enhancer or PromptEnhancer()
        
    async def generate(self, request: VideoGenerationRequest, progress_callback=None) -> VideoGenerationResponse:
        """
        Generate a video based on the request
        
        Args:
            request: Video generation request
            progress_callback: Optional callback for progress updates
            
        Returns:
            Video generation response
        """
        request_id = str(uuid.uuid4())
        response = VideoGenerationResponse(
            request_id=request_id,
            status=GenerationStatus.IN_PROGRESS,
            created_at=datetime.utcnow().isoformat()
        )
        
        try:
            if progress_callback:
                progress_callback(0, "Starting video generation...")
                
            # Handle input image if provided
            input_image = None
            if request.input_image:
                input_image = Image.open(request.input_image)
                
            # Enhance prompt if text-to-video
            enhanced_prompt = request.prompt
            if request.use_enhancement and not request.input_image:
                # Video models typically use similar enhancement to image models
                enhanced_prompt = await self.prompt_enhancer.enhance(
                    request.prompt,
                    ModelType.FLUX_1_DEV,  # Use image enhancement for now
                    request.enhancement_fields
                )
                
            if progress_callback:
                progress_callback(10, "Loading video model...")
                
            # Load the video pipeline
            pipeline = await self._load_pipeline(request.model)
            
            if progress_callback:
                progress_callback(30, "Generating video frames...")
                
            # Prepare generation parameters
            gen_params = self._prepare_generation_params(request, enhanced_prompt, input_image)
            
            # Generate the video
            result = await asyncio.to_thread(
                pipeline.generate,
                **gen_params
            )
            
            if progress_callback:
                progress_callback(80, "Saving video...")
                
            # Save the video
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_filename = f"video_{timestamp}_{request_id}.mp4"
            video_path = self.output_dir / video_filename
            
            # Save video frames or encoded video
            if hasattr(result, 'save'):
                result.save(str(video_path))
            else:
                # Handle frame-based output
                self._save_frames_as_video(result, video_path, request.fps)
                
            if progress_callback:
                progress_callback(90, "Creating preview GIF...")
                
            # Create preview GIF
            gif_path = await self._create_preview_gif(video_path)
            
            # Update response
            response.status = GenerationStatus.COMPLETED
            response.completed_at = datetime.utcnow().isoformat()
            response.video_path = video_path
            response.preview_gif = gif_path
            response.metadata = {
                "model": request.model,
                "prompt": request.prompt,
                "enhanced_prompt": enhanced_prompt if not input_image else None,
                "input_type": "image" if input_image else "text",
                "duration": request.duration,
                "fps": request.fps,
                "motion_scale": request.motion_scale
            }
            
            if progress_callback:
                progress_callback(100, "Video generation complete!")
                
        except Exception as e:
            logger.error(f"Video generation failed: {str(e)}")
            response.status = GenerationStatus.FAILED
            response.error = str(e)
            response.completed_at = datetime.utcnow().isoformat()
            
        return response
        
    async def _load_pipeline(self, model_id: str):
        """Load the video model pipeline"""
        return await asyncio.to_thread(
            self.model_manager.load_video_model,
            model_id
        )
        
    def _prepare_generation_params(self, request: VideoGenerationRequest, prompt: str, input_image: Optional[Image.Image]) -> Dict[str, Any]:
        """Prepare parameters for the video generation pipeline"""
        params = {
            "prompt": prompt,
            "num_frames": int(request.duration * request.fps),
            "fps": request.fps,
            "motion_scale": request.motion_scale,
        }
        
        if input_image:
            params["image"] = input_image
            
        if request.negative_prompt:
            params["negative_prompt"] = request.negative_prompt
            
        if request.seed is not None:
            params["seed"] = request.seed
            
        return params
        
    def _save_frames_as_video(self, frames: list, output_path: Path, fps: int):
        """Save a list of frames as a video file"""
        import cv2
        
        if not frames:
            raise ValueError("No frames to save")
            
        # Get dimensions from first frame
        height, width = np.array(frames[0]).shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Write frames
        for frame in frames:
            # Convert PIL to numpy if needed
            if isinstance(frame, Image.Image):
                frame = np.array(frame)
            # Convert RGB to BGR for OpenCV
            if frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame)
            
        out.release()
        
    async def _create_preview_gif(self, video_path: Path) -> Path:
        """Create a preview GIF from the video"""
        import cv2
        from PIL import Image
        
        gif_path = video_path.with_suffix('.gif')
        
        # Read video
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        # Sample frames for GIF (every nth frame)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_rate = max(1, frame_count // 30)  # Max 30 frames in GIF
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % sample_rate == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize for smaller GIF
                pil_frame = Image.fromarray(frame_rgb)
                pil_frame = pil_frame.resize((320, 240), Image.Resampling.LANCZOS)
                frames.append(pil_frame)
                
            frame_idx += 1
            
        cap.release()
        
        # Save as GIF
        if frames:
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=100,  # 100ms per frame
                loop=0
            )
            
        return gif_path
        
    def validate_request(self, request: VideoGenerationRequest) -> Tuple[bool, Optional[str]]:
        """Validate a video generation request"""
        # Check model availability
        if not self.model_manager.is_model_available(request.model):
            return False, f"Model {request.model} is not available"
            
        # Check duration limits
        if request.duration > 10.0:
            return False, "Video duration too long (max 10 seconds)"
            
        # Check FPS limits
        if request.fps > 60:
            return False, "FPS too high (max 60)"
            
        # Check input
        if request.input_image:
            if not Path(request.input_image).exists():
                return False, "Input image file not found"
        elif not request.prompt:
            return False, "Either prompt or input image is required"
            
        return True, None