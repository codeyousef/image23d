"""Main 3D processor that coordinates all processing steps"""

import logging
from pathlib import Path
from typing import Optional
from datetime import datetime
import uuid

from ...models.generation import ThreeDGenerationRequest, ThreeDGenerationResponse, GenerationStatus
from ..prompt_enhancer import PromptEnhancer
from .image_processing import ImageProcessor
from .depth_processing import DepthProcessor
from .normal_processing import NormalProcessor
from .mesh_reconstruction import MeshReconstructor
from .texture_generation import TextureGenerator
from .export import ModelExporter
from .validation import RequestValidator

logger = logging.getLogger(__name__)

# Import real HunYuan3D integration
try:
    import sys
    from pathlib import Path
    
    # Add src to path for imports - handle both Unix and Windows paths
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent.parent
    src_path = project_root / "src"
    
    if src_path.exists():
        # Add to sys.path if not already there
        src_path_str = str(src_path)
        if src_path_str not in sys.path:
            sys.path.insert(0, src_path_str)
    
    from hunyuan3d_app.generation.threed import get_3d_generator, generate_3d_model
    from hunyuan3d_app.models.threed.orchestrator import ThreeDOrchestrator
    REAL_HUNYUAN3D_AVAILABLE = True
    logger.info("Real HunYuan3D models available")
except ImportError as e:
    logger.warning(f"Real HunYuan3D models not available: {e}")
    REAL_HUNYUAN3D_AVAILABLE = False


class ThreeDProcessor:
    """Handles 3D generation with prompt enhancement"""
    
    def __init__(self, model_manager, output_dir: Path, prompt_enhancer: Optional[PromptEnhancer] = None):
        self.model_manager = model_manager
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prompt_enhancer = prompt_enhancer or PromptEnhancer()
        
        # Initialize processors
        self.image_processor = ImageProcessor()
        self.depth_processor = DepthProcessor()
        self.normal_processor = NormalProcessor()
        self.mesh_reconstructor = MeshReconstructor()
        self.texture_generator = TextureGenerator()
        self.model_exporter = ModelExporter()
        self.validator = RequestValidator()
        
    async def generate(self, request: ThreeDGenerationRequest, progress_callback=None) -> ThreeDGenerationResponse:
        """
        Generate a 3D model based on the request
        
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
            # Validate request
            is_valid, error_msg = self.validator.validate_request(request)
            if not is_valid:
                raise ValueError(error_msg)
            
            # Create output directory for this generation
            output_subdir = self.output_dir / f"3d_{request_id}"
            output_subdir.mkdir(parents=True, exist_ok=True)
            
            if progress_callback:
                progress_callback("enhance_prompt", 0.0, "Starting 3D generation...")
            
            # Use real HunYuan3D models if available
            if REAL_HUNYUAN3D_AVAILABLE:
                result = await self._generate_with_real_hunyuan3d(
                    request, output_subdir, progress_callback
                )
                
                # Update response with real generation results
                response.model_path = result.get("model_path")
                response.preview_images = result.get("preview_images", [])
                response.export_paths = result.get("export_paths", {})
                response.status = GenerationStatus.COMPLETED
                response.completed_at = datetime.utcnow().isoformat()
                response.metadata = result.get("metadata", {})
                
                if progress_callback:
                    progress_callback("save", 1.0, "3D generation complete!")
                    
            else:
                # Fallback to mock implementation
                logger.warning("Using mock 3D generation - real HunYuan3D models not available")
                result = await self._generate_with_mock_implementation(
                    request, output_subdir, progress_callback
                )
                
                # Update response with mock results
                response.model_path = result.get("model_path")
                response.preview_images = result.get("preview_images", [])
                response.export_paths = result.get("export_paths", {})
                response.status = GenerationStatus.COMPLETED
                response.completed_at = datetime.utcnow().isoformat()
                response.metadata = result.get("metadata", {})
                
        except Exception as e:
            logger.error(f"3D generation failed: {str(e)}")
            response.status = GenerationStatus.FAILED
            response.error = str(e)
            response.completed_at = datetime.utcnow().isoformat()
            
        return response
    
    async def _generate_with_real_hunyuan3d(
        self, 
        request: ThreeDGenerationRequest, 
        output_subdir: Path, 
        progress_callback=None
    ):
        """Generate 3D model using real HunYuan3D implementation"""
        import asyncio
        from PIL import Image
        
        # Handle input image if provided
        input_image = None
        if request.input_image:
            if progress_callback:
                progress_callback("enhance_prompt", 0.02, "Preparing input image...")
            input_image = await self.image_processor.prepare_input_image(
                request.input_image, 
                request.remove_background
            )
            
            # CRITICAL: Save a copy of the actual processed input image for debugging
            if input_image and hasattr(input_image, 'save'):
                import hashlib
                image_hash = hashlib.md5(input_image.tobytes()).hexdigest()[:8]
                logger.info(f"📸 [PROCESSOR] Input image processed, hash: {image_hash}")
                
                # Save the actual input image that will be sent to HunYuan3D
                debug_input_path = output_subdir / f"DEBUG_actual_input_{image_hash}.png"
                input_image.save(debug_input_path)
                logger.info(f"📁 [PROCESSOR] Saved actual input image to: {debug_input_path}")
            
        # Enhance prompt if text-to-3D
        enhanced_prompt = request.prompt
        if request.use_enhancement and not request.input_image:
            if progress_callback:
                progress_callback("enhance_prompt", 0.05, "Enhancing prompt...")
            model_type = self.validator.get_model_type(request.model)
            enhanced_prompt = await self.prompt_enhancer.enhance(
                request.prompt,
                model_type,
                request.enhancement_fields
            )
        
        if progress_callback:
            progress_callback("load_model", 0.1, "Loading HunYuan3D model...")
        
        # Get the 3D generator
        generator = get_3d_generator()
        
        # Create progress wrapper to adapt progress callback format
        def wrapped_progress_callback(*args):
            if progress_callback:
                # Handle both 2-arg and 3-arg callback signatures
                if len(args) == 2:
                    # Old format: (progress_percent, message)
                    progress_percent, message = args
                    # Map progress to appropriate step
                    if progress_percent < 30:
                        progress_callback("load_model", 0.1 + progress_percent * 0.002, message)
                    elif progress_percent < 60:
                        progress_callback("generate", 0.2 + (progress_percent - 30) * 0.015, message)
                    elif progress_percent < 90:
                        progress_callback("postprocess", 0.65 + (progress_percent - 60) * 0.01, message)
                    else:
                        progress_callback("save", 0.95 + (progress_percent - 90) * 0.005, message)
                elif len(args) == 3:
                    # New format: (step, progress, message)
                    step, progress, message = args
                    # Direct pass-through for step-based callbacks
                    progress_callback(step, progress, message)
                else:
                    # Fallback for unexpected formats
                    logger.warning(f"Unexpected progress callback args: {args}")
        
        # Determine input for generation
        if input_image:
            # Image-to-3D: Use the input image directly
            generation_input = input_image
            
            # CRITICAL: Track the image hash before sending to 3D generation
            if hasattr(generation_input, 'tobytes'):
                import hashlib
                image_hash = hashlib.md5(generation_input.tobytes()).hexdigest()[:8]
                logger.info(f"📸 [PROCESSOR] Generation input hash (before 3D gen): {image_hash}")
        else:
            # Text-to-3D: Need to generate an image from text first
            if progress_callback:
                progress_callback("generate", 0.15, "Generating image from text...")
            
            # For now, create a placeholder image with the text
            # In a real implementation, you would use an image generation model
            import numpy as np
            placeholder_image = Image.new('RGB', (512, 512), (128, 128, 128))
            # Add text to the image for better mock behavior
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(placeholder_image)
            try:
                # Try to use a default font
                font = ImageFont.load_default()
                # Word wrap the text
                words = enhanced_prompt.split()
                lines = []
                current_line = ""
                for word in words:
                    if len(current_line + word) < 20:  # Rough character limit
                        current_line += word + " "
                    else:
                        lines.append(current_line.strip())
                        current_line = word + " "
                if current_line:
                    lines.append(current_line.strip())
                
                # Draw the text
                y_offset = 200
                for line in lines[:4]:  # Max 4 lines
                    draw.text((20, y_offset), line, fill=(255, 255, 255), font=font)
                    y_offset += 30
                    
            except Exception as e:
                logger.warning(f"Failed to add text to placeholder image: {e}")
            
            generation_input = placeholder_image
            
            # CRITICAL: Track the generated placeholder image hash
            if hasattr(generation_input, 'tobytes'):
                import hashlib
                image_hash = hashlib.md5(generation_input.tobytes()).hexdigest()[:8]
                logger.info(f"📸 [PROCESSOR] Placeholder generation input hash: {image_hash}")
                # Save the placeholder for comparison
                placeholder_path = output_subdir / f"DEBUG_placeholder_{image_hash}.png"
                generation_input.save(placeholder_path)
                logger.info(f"📁 [PROCESSOR] Saved placeholder image to: {placeholder_path}")
            
            if progress_callback:
                progress_callback("generate", 0.18, "Text-to-image completed, starting 3D generation...")
        
        # CRITICAL: Final tracking before 3D generation
        if hasattr(generation_input, 'tobytes'):
            import hashlib
            final_hash = hashlib.md5(generation_input.tobytes()).hexdigest()[:8]
            logger.info(f"🚀 [PROCESSOR] FINAL IMAGE HASH being sent to 3D generation: {final_hash}")
            logger.info(f"🔍 [PROCESSOR] Final image details: size={generation_input.size}, mode={generation_input.mode}")
        
        # Run real HunYuan3D generation
        result = await asyncio.to_thread(
            generator.generate_3d,
            image=generation_input,
            model_type=request.model,
            quality_preset=request.quality_preset,
            output_format=request.export_formats[0] if request.export_formats else "glb",
            enable_pbr=request.pbr_materials,
            enable_depth_refinement=request.depth_enhancement,
            progress_callback=wrapped_progress_callback
        )
        
        # CRITICAL: Track what comes back from 3D generation
        logger.info(f"📄 [PROCESSOR] 3D generation result keys: {list(result.keys())}")
        if "preview_image" in result:
            logger.info(f"🖼️ [PROCESSOR] Preview image returned from 3D gen: {type(result['preview_image'])}")
        
        # CRITICAL: Track the 3D generation result details
        logger.info(f"📄 [PROCESSOR] 3D generation completed, result keys: {list(result.keys())}")
        
        # Process the result
        model_path = Path(result["output_path"])
        
        # Copy to our output directory
        final_model_path = output_subdir / model_path.name
        import shutil
        shutil.copy2(model_path, final_model_path)
        
        # Handle preview images
        preview_images = []
        if "preview_image" in result and result["preview_image"]:
            preview_path = output_subdir / "preview.png"
            # Handle both PIL Image and path cases
            preview_img = result["preview_image"]
            if hasattr(preview_img, 'save'):  # PIL Image
                preview_img.save(preview_path)
                preview_images.append(preview_path)
                logger.info(f"Saved preview image to {preview_path}")
            elif isinstance(preview_img, (str, Path)) and Path(preview_img).exists():  # Path
                import shutil
                shutil.copy2(preview_img, preview_path)
                preview_images.append(preview_path)
                logger.info(f"Copied preview image to {preview_path}")
            else:
                logger.warning(f"Invalid preview image type: {type(preview_img)}")
        
        # Also check for generated_images which might contain multiple preview views
        if "generated_images" in result and result["generated_images"]:
            for i, img in enumerate(result["generated_images"][:4]):  # Max 4 preview images
                if hasattr(img, 'save'):  # PIL Image
                    preview_path = output_subdir / f"preview_{i}.png"
                    img.save(preview_path)
                    preview_images.append(preview_path)
                    # CRITICAL: Track generated image hashes
                    if hasattr(img, 'tobytes'):
                        import hashlib
                        gen_hash = hashlib.md5(img.tobytes()).hexdigest()[:8]
                        logger.info(f"🖼️ [PROCESSOR] Generated image {i} hash: {gen_hash}")
                    logger.info(f"📁 [PROCESSOR] Saved generated image {i} to {preview_path}")
        
        logger.info(f"📁 [PROCESSOR] Total preview images saved: {len(preview_images)}")
        
        # CRITICAL: Create a summary of all saved images for debugging
        logger.info(f"📈 [PROCESSOR] IMAGE FLOW SUMMARY:")
        debug_files = list(output_subdir.glob("DEBUG_*"))
        for debug_file in debug_files:
            logger.info(f"   - {debug_file.name}")
        if preview_images:
            for preview_path in preview_images:
                logger.info(f"   - FINAL: {preview_path.name}")
        
        # Handle export formats
        export_paths = {}
        for fmt in request.export_formats:
            if fmt.lower() == request.export_formats[0].lower():
                export_paths[fmt] = final_model_path
            else:
                # Try to export to other formats
                try:
                    import trimesh
                    mesh = trimesh.load(final_model_path)
                    export_path = output_subdir / f"model.{fmt.lower()}"
                    mesh.export(export_path)
                    export_paths[fmt] = export_path
                except Exception as e:
                    logger.warning(f"Failed to export to {fmt}: {e}")
        
        return {
            "model_path": final_model_path,
            "preview_images": preview_images,
            "export_paths": export_paths,
            "metadata": {
                "model": request.model,
                "prompt": request.prompt,
                "enhanced_prompt": enhanced_prompt if not input_image else None,
                "input_type": "image" if input_image else "text",
                "quality_preset": request.quality_preset,
                "num_views": request.num_views,
                "mesh_resolution": request.mesh_resolution,
                "texture_resolution": request.texture_resolution,
                "export_formats": request.export_formats,
                "generation_time": result.get("generation_time", 0),
                "model_used": result.get("model_used", request.model),
                "using_real_hunyuan3d": True
            }
        }
    
    async def _generate_with_mock_implementation(
        self, 
        request: ThreeDGenerationRequest, 
        output_subdir: Path, 
        progress_callback=None
    ):
        """Generate 3D model using mock implementation (fallback)"""
        
        # Handle input image if provided
        input_image = None
        if request.input_image:
            input_image = await self.image_processor.prepare_input_image(
                request.input_image, 
                request.remove_background
            )
            
        # Enhance prompt if text-to-3D
        enhanced_prompt = request.prompt
        if request.use_enhancement and not request.input_image:
            if progress_callback:
                progress_callback("enhance_prompt", 0.05, "Enhancing prompt...")
            model_type = self.validator.get_model_type(request.model)
            enhanced_prompt = await self.prompt_enhancer.enhance(
                request.prompt,
                model_type,
                request.enhancement_fields
            )
            
        if progress_callback:
            progress_callback("load_model", 0.1, "Loading 3D model pipeline...")
            
        # Load the 3D pipeline (for now just returns model_id)
        pipeline = await self._load_pipeline(request.model)
        
        if progress_callback:
            progress_callback("generate", 0.2, "Generating multi-view images...")
            
        # Generate multi-view images
        mv_images = await self.image_processor.generate_multiview(
            pipeline,
            enhanced_prompt if not input_image else None,
            input_image,
            request.num_views,
            progress_callback,
            image_model=request.image_model if hasattr(request, 'image_model') else None
        )
        
        # Save preview images
        preview_paths = []
        for i, img in enumerate(mv_images):
            preview_path = output_subdir / f"view_{i:02d}.png"
            img.save(preview_path)
            preview_paths.append(preview_path)
        
        # Generate depth maps with multi-view consistency
        depth_maps = await self.depth_processor.generate_depth_maps(
            mv_images,
            progress_callback
        )
        
        # Estimate normal maps for surface detail
        normal_maps = await self.normal_processor.estimate_normal_maps(
            mv_images,
            depth_maps,
            progress_callback
        )
        
        # Reconstruct 3D model
        mesh_path = await self.mesh_reconstructor.reconstruct_3d(
            pipeline,
            mv_images,
            depth_maps,
            normal_maps,
            output_subdir,
            request.mesh_resolution,
            progress_callback
        )
        
        # Generate textures
        textured_mesh_path = await self.texture_generator.generate_textures(
            pipeline,
            mesh_path,
            request.texture_resolution,
            progress_callback
        )
        
        if progress_callback:
            progress_callback("save", 0.95, "Exporting to requested formats...")
            
        # Export to requested formats
        export_paths = await self.model_exporter.export_formats(
            textured_mesh_path,
            request.export_formats,
            output_subdir
        )
        
        return {
            "model_path": textured_mesh_path,
            "preview_images": preview_paths,
            "export_paths": export_paths,
            "metadata": {
                "model": request.model,
                "prompt": request.prompt,
                "enhanced_prompt": enhanced_prompt if not input_image else None,
                "input_type": "image" if input_image else "text",
                "quality_preset": request.quality_preset,
                "num_views": request.num_views,
                "mesh_resolution": request.mesh_resolution,
                "texture_resolution": request.texture_resolution,
                "export_formats": request.export_formats,
                "using_real_hunyuan3d": False
            }
        }
        
    async def _load_pipeline(self, model_id: str):
        """Load the 3D model pipeline"""
        # For now, we just return the model_id and let the working 3D generator handle it
        # The working implementation will handle the actual model loading
        return model_id
    
    # Re-export validation method for backward compatibility
    def validate_request(self, request: ThreeDGenerationRequest):
        """Validate 3D generation request"""
        return self.validator.validate_request(request)
    
    def _get_model_type(self, model_id: str):
        """Get model type from model ID"""
        return self.validator.get_model_type(model_id)