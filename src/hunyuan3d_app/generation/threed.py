"""3D generation module using the new architecture

This module provides the interface between the UI and the new 3D model system.
It replaces the old placeholder implementation with proper HunYuan3D support.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, Callable
from PIL import Image
import time

from ..models.threed.orchestrator import (
    ThreeDOrchestrator,
    TaskRequirements,
    ModelCapability
)
from ..models.threed.base import QUALITY_PRESETS_3D

logger = logging.getLogger(__name__)


class ThreeDGenerator:
    """Main interface for 3D generation"""
    
    def __init__(self):
        self.orchestrator = ThreeDOrchestrator()
        self.generation_count = 0
        
    def generate_3d(
        self,
        image: Union[Image.Image, str],
        model_type: str = "auto",
        quality_preset: str = "standard",
        output_format: str = "glb",
        enable_pbr: bool = False,
        enable_depth_refinement: bool = True,
        max_generation_time: Optional[float] = None,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Generate 3D model from image
        
        Args:
            image: Input image or path
            model_type: Model to use ("auto", "hunyuan3d-21", "hunyuan3d-2mini", etc.)
            quality_preset: Quality preset ("draft", "standard", "high", "ultra")
            output_format: Output format ("glb", "obj", "ply", etc.)
            enable_pbr: Enable PBR material generation
            enable_depth_refinement: Enable depth map refinement
            max_generation_time: Maximum time allowed for generation
            progress_callback: Progress callback function
            
        Returns:
            Dict containing:
            - output_path: Path to generated 3D file
            - preview_image: Preview render of 3D model
            - generation_time: Time taken for generation
            - model_used: Model that was used
            - metadata: Additional generation metadata
        """
        
        start_time = time.time()
        self.generation_count += 1
        
        try:
            # Validate inputs
            if quality_preset not in QUALITY_PRESETS_3D:
                raise ValueError(f"Invalid quality preset: {quality_preset}")
                
            # Load image if path provided
            if isinstance(image, str):
                image = Image.open(image)
                
            # Prepare task requirements
            required_capabilities = [ModelCapability.IMAGE_TO_3D]
            
            if enable_pbr:
                required_capabilities.append(ModelCapability.PBR_MATERIALS)
                
            if quality_preset in ["high", "ultra"]:
                required_capabilities.append(ModelCapability.HIGH_RESOLUTION)
                
            requirements = TaskRequirements(
                input_type="image",
                quality_preset=quality_preset,
                output_format=output_format,
                required_capabilities=required_capabilities,
                max_generation_time=max_generation_time
            )
            
            # Override model selection if specific model requested
            if model_type != "auto":
                # Map user-friendly names to internal model types
                model_mapping = {
                    "hunyuan3d-21": "hunyuan3d-21",
                    "hunyuan3d-2.1": "hunyuan3d-21",
                    "hunyuan3d-mini": "hunyuan3d-2mini",
                    "hunyuan3d-2mini": "hunyuan3d-2mini",
                    "hi3dgen": "hi3dgen",
                    "sparc3d": "sparc3d"
                }
                
                internal_model = model_mapping.get(model_type.lower())
                if internal_model:
                    # Filter available models
                    from ..models.threed.base import ModelType3D
                    available_models = [
                        m for m in self.orchestrator.available_models
                        if m.value == internal_model
                    ]
                    
                    if available_models:
                        self.orchestrator.available_models = available_models
                    else:
                        logger.warning(f"Model {model_type} not available, using auto selection")
                        
            # Generate 3D model
            result = self.orchestrator.generate(
                image,
                requirements,
                progress_callback
            )
            
            # Get generation time
            generation_time = time.time() - start_time
            
            # Create preview render
            preview_image = self._create_preview(result)
            
            # Prepare response
            return {
                "output_path": result["output_path"],
                "preview_image": preview_image,
                "generation_time": generation_time,
                "model_used": result["model_used"],
                "metadata": {
                    "quality_preset": quality_preset,
                    "output_format": output_format,
                    "enable_pbr": enable_pbr,
                    "enable_depth_refinement": enable_depth_refinement,
                    "selection_reason": result.get("selection_reason", ""),
                    "memory_used_gb": result.get("memory_used_gb", 0),
                    "quantization": result.get("quantization"),
                    "generation_count": self.generation_count,
                    "views_generated": len(result.get("views", [])),
                }
            }
            
        except Exception as e:
            logger.error(f"3D generation failed: {e}")
            raise
            
    def _create_preview(self, result: Dict[str, Any]) -> Optional[Image.Image]:
        """Create preview image of 3D model"""
        
        try:
            # If we have multiple views, create a grid
            views = result.get("views", [])
            if views:
                return self._create_view_grid(views)
                
            # Otherwise, try to render the mesh
            mesh = result.get("mesh")
            if mesh:
                return self._render_mesh_preview(mesh)
                
            return None
            
        except Exception as e:
            logger.error(f"Failed to create preview: {e}")
            return None
            
    def _create_view_grid(self, views: list) -> Image.Image:
        """Create grid of multiple views"""
        
        if not views:
            return None
            
        # Determine grid size
        n = len(views)
        if n == 1:
            return views[0]
        elif n <= 4:
            cols, rows = 2, 2
        elif n <= 6:
            cols, rows = 3, 2
        elif n <= 9:
            cols, rows = 3, 3
        else:
            cols, rows = 4, 3
            
        # Get image size
        img_width, img_height = views[0].size
        
        # Create grid
        grid_width = img_width * cols
        grid_height = img_height * rows
        grid = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))
        
        # Paste images
        for i, view in enumerate(views[:cols*rows]):
            col = i % cols
            row = i // cols
            x = col * img_width
            y = row * img_height
            grid.paste(view, (x, y))
            
        # Resize if too large
        max_size = 1024
        if grid_width > max_size or grid_height > max_size:
            scale = min(max_size / grid_width, max_size / grid_height)
            new_width = int(grid_width * scale)
            new_height = int(grid_height * scale)
            grid = grid.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
        return grid
        
    def _render_mesh_preview(self, mesh) -> Optional[Image.Image]:
        """Render mesh to image"""
        
        try:
            # Simple preview using trimesh
            import trimesh
            import numpy as np
            
            if isinstance(mesh, trimesh.Trimesh):
                # Create scene
                scene = trimesh.Scene(mesh)
                
                # Render to image
                # This is a simple orthographic projection
                # In practice, you might want to use a proper renderer
                png = scene.save_image(resolution=[512, 512])
                
                # Convert to PIL Image
                import io
                image = Image.open(io.BytesIO(png))
                
                return image
                
        except Exception as e:
            logger.error(f"Mesh preview failed: {e}")
            
        return None
        
    def list_available_models(self) -> list:
        """List available 3D models"""
        return self.orchestrator.list_available_models()
        
    def get_quality_presets(self) -> Dict[str, Any]:
        """Get available quality presets"""
        presets = {}
        
        for name, preset in QUALITY_PRESETS_3D.items():
            presets[name] = {
                "name": preset.name,
                "multiview_steps": preset.multiview_steps,
                "multiview_count": preset.multiview_count,
                "reconstruction_resolution": preset.reconstruction_resolution,
                "texture_resolution": preset.texture_resolution,
                "features": {
                    "pbr": preset.use_pbr,
                    "normal_maps": preset.use_normal_maps,
                    "depth_refinement": preset.use_depth_refinement
                },
                "memory_efficient": preset.memory_efficient,
                "supports_quantization": preset.supports_quantization
            }
            
        return presets
        
    def estimate_generation_time(
        self,
        model_type: str,
        quality_preset: str
    ) -> float:
        """Estimate generation time in seconds"""
        
        # Get model profile
        from ..models.threed.orchestrator import MODEL_PROFILES, ModelType3D
        
        model_type_enum = None
        for mt in ModelType3D:
            if mt.value == model_type:
                model_type_enum = mt
                break
                
        if model_type_enum and model_type_enum in MODEL_PROFILES:
            profile = MODEL_PROFILES[model_type_enum]
            return profile.performance_metrics.get(quality_preset, 60.0)
            
        return 60.0  # Default estimate
        
    def cleanup(self):
        """Cleanup resources"""
        self.orchestrator.unload_all()


# Singleton instance
_generator_instance = None


def get_3d_generator() -> ThreeDGenerator:
    """Get or create 3D generator instance"""
    global _generator_instance
    
    if _generator_instance is None:
        _generator_instance = ThreeDGenerator()
        
    return _generator_instance


def generate_3d_model(
    image: Union[Image.Image, str],
    **kwargs
) -> Dict[str, Any]:
    """Convenience function for 3D generation"""
    
    generator = get_3d_generator()
    return generator.generate_3d(image, **kwargs)