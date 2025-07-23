"""
Fixed 3D processor that uses the actual HunYuan3D generation system
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import uuid

from PIL import Image
import numpy as np

from ..models.generation import ThreeDGenerationRequest, ThreeDGenerationResponse, GenerationStatus
from ..models.enhancement import ModelType
from .prompt_enhancer import PromptEnhancer

logger = logging.getLogger(__name__)

class ThreeDProcessor:
    """Handles 3D generation using the actual HunYuan3D system"""
    
    def __init__(self, output_dir: Path, prompt_enhancer: Optional[PromptEnhancer] = None):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prompt_enhancer = prompt_enhancer or PromptEnhancer()
        self._generator = None
        
    def _get_generator(self):
        """Lazy load the 3D generator"""
        if self._generator is None:
            from src.hunyuan3d_app.generation.threed import ThreeDGenerator
            self._generator = ThreeDGenerator()
        return self._generator
        
    async def generate(self, request: ThreeDGenerationRequest, progress_callback=None) -> ThreeDGenerationResponse:
        """
        Generate a 3D model based on the request using the actual HunYuan3D system
        
        Args:
            request: 3D generation request
            progress_callback: Optional callback for progress updates
            
        Returns:
            3D generation response
        """
        request_id = str(uuid.uuid4())
        response = ThreeDGenerationResponse(
            request_id=request_id,
            status=GenerationStatus.IN_PROGRESS,
            created_at=datetime.utcnow().isoformat()
        )
        
        try:
            # Create output directory for this generation
            output_subdir = self.output_dir / f"3d_{request_id}"
            output_subdir.mkdir(parents=True, exist_ok=True)
            
            if progress_callback:
                progress_callback(0, "Starting 3D generation...")
                
            # Get the generator
            generator = self._get_generator()
            
            # Handle input image if provided
            input_image = None
            if request.input_image:
                if progress_callback:
                    progress_callback(5, "Loading input image...")
                    
                input_image = await self._prepare_input_image(request.input_image, request.remove_background)
                
            # For text-to-3D, we need to generate an image first
            if not input_image and request.prompt:
                if progress_callback:
                    progress_callback(10, "Text-to-3D: Generating image from prompt...")
                    
                # Use the existing image generation system
                from src.hunyuan3d_app.core.studio_enhanced import Hunyuan3DStudioEnhanced
                studio = Hunyuan3DStudioEnhanced()
                
                # Generate image from prompt
                image_result = await asyncio.to_thread(
                    studio.generate_image_from_text,
                    prompt=request.prompt,
                    model_name="Dreamshaper-XL-Turbo",  # Fast model for preview
                    negative_prompt=request.negative_prompt,
                    width=512,
                    height=512,
                    num_inference_steps=20,
                    guidance_scale=7.5
                )
                
                if image_result and image_result.get("image"):
                    input_image = image_result["image"]
                else:
                    raise ValueError("Failed to generate image from prompt")
                    
            elif not input_image:
                raise ValueError("Either prompt or input image is required")
                
            if progress_callback:
                progress_callback(30, "Converting to 3D model...")
                
            # Map quality preset to actual parameters
            quality_params = self._get_quality_params(request.quality_preset)
            
            # Use the actual 3D generator
            def wrapped_callback(progress, message):
                if progress_callback:
                    # Map generator progress (0-1) to our progress range (30-90)
                    mapped_progress = 30 + int(progress * 60)
                    progress_callback(mapped_progress, message)
                    
            result = await asyncio.to_thread(
                generator.generate_3d,
                image=input_image,
                model_type=request.model,
                quality_preset=request.quality_preset,
                output_format=request.export_formats[0] if request.export_formats else "glb",
                enable_pbr=quality_params.get("enable_pbr", False),
                enable_depth_refinement=quality_params.get("enable_depth_refinement", True),
                progress_callback=wrapped_callback
            )
            
            if progress_callback:
                progress_callback(90, "Finalizing output...")
                
            # Handle the result
            if result and result.get("output_path"):
                model_path = Path(result["output_path"])
                
                # Move to our output directory
                import shutil
                final_path = output_subdir / model_path.name
                shutil.move(str(model_path), str(final_path))
                
                response.model_path = final_path
                
                # Save preview image if available
                if result.get("preview_image"):
                    preview_path = output_subdir / "preview.png"
                    result["preview_image"].save(preview_path)
                    response.preview_images = [preview_path]
                    
                # Export to additional formats if requested
                if len(request.export_formats) > 1:
                    export_paths = await self._export_formats(
                        final_path,
                        request.export_formats[1:],  # Skip first format (already done)
                        output_subdir
                    )
                    response.export_paths = {request.export_formats[0]: final_path, **export_paths}
                else:
                    response.export_paths = {request.export_formats[0]: final_path}
                    
                # Update response metadata
                response.status = GenerationStatus.COMPLETED
                response.completed_at = datetime.utcnow().isoformat()
                response.metadata = {
                    "model": request.model,
                    "prompt": request.prompt,
                    "input_type": "image" if request.input_image else "text",
                    "quality_preset": request.quality_preset,
                    "generation_time": result.get("generation_time"),
                    "model_used": result.get("model_used"),
                    **result.get("metadata", {})
                }
                
                if progress_callback:
                    progress_callback(100, "3D generation complete!")
                    
            else:
                raise ValueError("3D generation returned no output")
                
        except Exception as e:
            logger.error(f"3D generation failed: {str(e)}")
            response.status = GenerationStatus.FAILED
            response.error = str(e)
            response.completed_at = datetime.utcnow().isoformat()
            
        return response
        
    async def _prepare_input_image(self, image_path: str, remove_bg: bool) -> Image.Image:
        """Prepare input image for 3D generation"""
        image = Image.open(image_path)
        
        if remove_bg:
            # Use the existing background removal
            try:
                from src.hunyuan3d_app.utils.image import remove_background
                image = await asyncio.to_thread(remove_background, image)
            except Exception as e:
                logger.warning(f"Background removal failed: {e}, using original image")
                
        # Ensure image is in RGB mode
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        return image
        
    def _get_quality_params(self, preset: str) -> Dict[str, Any]:
        """Get quality parameters for a preset"""
        presets = {
            "draft": {
                "enable_pbr": False,
                "enable_depth_refinement": False,
                "mesh_resolution": 256,
                "texture_resolution": 512
            },
            "standard": {
                "enable_pbr": False,
                "enable_depth_refinement": True,
                "mesh_resolution": 512,
                "texture_resolution": 1024
            },
            "high": {
                "enable_pbr": True,
                "enable_depth_refinement": True,
                "mesh_resolution": 1024,
                "texture_resolution": 2048
            },
            "ultra": {
                "enable_pbr": True,
                "enable_depth_refinement": True,
                "mesh_resolution": 2048,
                "texture_resolution": 4096
            }
        }
        return presets.get(preset, presets["standard"])
        
    async def _export_formats(
        self,
        model_path: Path,
        formats: List[str],
        output_dir: Path
    ) -> Dict[str, Path]:
        """Export 3D model to additional formats"""
        export_paths = {}
        
        try:
            import trimesh
            
            # Load the model
            mesh = trimesh.load(str(model_path))
            
            for fmt in formats:
                try:
                    export_path = output_dir / f"model.{fmt}"
                    
                    if fmt in ["obj", "ply", "stl"]:
                        mesh.export(str(export_path))
                        export_paths[fmt] = export_path
                    else:
                        # For formats not supported by trimesh, copy the original
                        import shutil
                        shutil.copy(model_path, export_path)
                        export_paths[fmt] = export_path
                        
                except Exception as e:
                    logger.warning(f"Failed to export to {fmt}: {e}")
                    
        except Exception as e:
            logger.error(f"Export failed: {e}")
            
        return export_paths
        
    def validate_request(self, request: ThreeDGenerationRequest) -> Tuple[bool, Optional[str]]:
        """Validate a 3D generation request"""
        # Check input
        if request.input_image:
            if not Path(request.input_image).exists():
                return False, "Input image file not found"
        elif not request.prompt:
            return False, "Either prompt or input image is required"
            
        # Check export formats
        valid_formats = ["glb", "obj", "ply", "stl", "fbx", "usdz"]
        for fmt in request.export_formats:
            if fmt not in valid_formats:
                return False, f"Invalid export format: {fmt}"
                
        # Check quality preset
        valid_presets = ["draft", "standard", "high", "ultra"]
        if request.quality_preset not in valid_presets:
            return False, f"Invalid quality preset: {request.quality_preset}"
            
        return True, None