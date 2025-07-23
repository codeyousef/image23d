"""
Fixed image processor that uses the actual image generation system
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import uuid

from PIL import Image
import torch
import numpy as np

from ..models.generation import ImageGenerationRequest, ImageGenerationResponse, GenerationStatus
from ..models.enhancement import ModelType
from .prompt_enhancer import PromptEnhancer

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Handles image generation using the actual system"""
    
    def __init__(self, output_dir: Path, prompt_enhancer: Optional[PromptEnhancer] = None):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prompt_enhancer = prompt_enhancer or PromptEnhancer()
        self._studio = None
        
    def _get_studio(self):
        """Lazy load the studio"""
        if self._studio is None:
            # Apply patch for missing method
            import src.hunyuan3d_app.models.check_missing_components_patch
            
            from src.hunyuan3d_app.core.studio import Hunyuan3DStudio
            self._studio = Hunyuan3DStudio()
        return self._studio
        
    async def generate(self, request: ImageGenerationRequest, progress_callback=None) -> ImageGenerationResponse:
        """
        Generate an image based on the request using the actual system
        
        Args:
            request: Image generation request
            progress_callback: Optional callback for progress updates
            
        Returns:
            Image generation response
        """
        request_id = str(uuid.uuid4())
        response = ImageGenerationResponse(
            request_id=request_id,
            status=GenerationStatus.IN_PROGRESS,
            created_at=datetime.utcnow().isoformat()
        )
        
        try:
            # Update progress
            if progress_callback:
                progress_callback(0, "Starting image generation...")
                
            # Enhance prompt if enabled
            enhanced_prompt = request.prompt
            if request.use_enhancement:
                if progress_callback:
                    progress_callback(5, "Enhancing prompt...")
                    
                model_type = self._get_model_type(request.model)
                enhanced_prompt = await self.prompt_enhancer.enhance(
                    request.prompt,
                    model_type,
                    request.enhancement_fields
                )
                response.enhanced_prompt = enhanced_prompt
                
            if progress_callback:
                progress_callback(10, "Loading model and generating...")
                
            # Get the studio instance
            studio = self._get_studio()
            
            # Prepare generation parameters
            gen_params = {
                "prompt": enhanced_prompt,
                "model_name": request.model,
                "negative_prompt": request.negative_prompt or "",
                "width": request.width,
                "height": request.height,
                "num_inference_steps": request.steps,
                "guidance_scale": request.guidance_scale,
                "scheduler": request.scheduler,
                "seed": request.seed
            }
            
            # Create a progress wrapper
            def wrapped_progress(p, msg):
                if progress_callback:
                    # Map internal progress (0-1) to our range (10-90)
                    mapped_progress = 10 + int(p * 80)
                    progress_callback(mapped_progress, msg)
            
            # Generate the image using the actual system
            # Note: generate_image expects individual parameters, not a dict
            result = await asyncio.to_thread(
                studio.generate_image,
                prompt=enhanced_prompt,
                negative_prompt=request.negative_prompt or "",
                model_name=request.model,
                width=request.width,
                height=request.height,
                steps=request.steps,
                guidance_scale=request.guidance_scale,
                seed=request.seed if request.seed is not None else -1,
                progress=wrapped_progress
            )
            
            if progress_callback:
                progress_callback(90, "Saving image...")
                
            # Handle the result - generate_image returns a tuple (image, info)
            if result and result[0] is not None:
                image = result[0]
                info = result[1] if len(result) > 1 else ""
                
                # Save the image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"image_{timestamp}_{request_id}.png"
                image_path = self.output_dir / filename
                
                # Ensure image is PIL Image
                if hasattr(image, 'save'):
                    image.save(image_path)
                else:
                    # Convert if needed
                    pil_image = Image.fromarray(image) if isinstance(image, np.ndarray) else image
                    pil_image.save(image_path)
                
                # Update response
                response.status = GenerationStatus.COMPLETED
                response.completed_at = datetime.utcnow().isoformat()
                response.image_path = image_path
                response.metadata = {
                    "model": request.model,
                    "prompt": request.prompt,
                    "enhanced_prompt": enhanced_prompt,
                    "width": request.width,
                    "height": request.height,
                    "steps": request.steps,
                    "guidance_scale": request.guidance_scale,
                    "seed": gen_params.get("seed"),
                    "info": info
                }
                
                if progress_callback:
                    progress_callback(100, "Image generation complete!")
                    
            else:
                raise ValueError("Image generation returned no output")
                
        except Exception as e:
            logger.error(f"Image generation failed: {str(e)}")
            response.status = GenerationStatus.FAILED
            response.error = str(e)
            response.completed_at = datetime.utcnow().isoformat()
            
        return response
        
    def _get_model_type(self, model_id: str) -> ModelType:
        """Map model ID to ModelType enum"""
        model_id_lower = model_id.lower()
        
        if "flux" in model_id_lower:
            if "schnell" in model_id_lower:
                return ModelType.FLUX_1_SCHNELL
            return ModelType.FLUX_1_DEV
        elif "hunyuan3d" in model_id_lower:
            if "mini" in model_id_lower:
                return ModelType.HUNYUAN_3D_MINI
            elif "2.0" in model_id_lower:
                return ModelType.HUNYUAN_3D_20
            return ModelType.HUNYUAN_3D_21
        elif "sdxl" in model_id_lower:
            return ModelType.SDXL
        else:
            return ModelType.SD15
            
    def validate_request(self, request: ImageGenerationRequest) -> Tuple[bool, Optional[str]]:
        """
        Validate an image generation request
        
        Returns:
            Tuple of (is_valid, error_message)
        """            
        # Check resolution limits
        total_pixels = request.width * request.height
        if total_pixels > 2048 * 2048:
            return False, "Resolution too high (max 2048x2048)"
            
        # Check step limits
        if request.steps > 150:
            return False, "Too many inference steps (max 150)"
            
        # Model validation will be done by the actual system
        return True, None