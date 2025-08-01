"""3D generation module using the new architecture

This module provides the interface between the UI and the new 3D model system.
It replaces the old placeholder implementation with proper HunYuan3D support.
"""

import logging
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Union, Callable
from PIL import Image
import time
from enum import Enum

from ..models.threed.orchestrator import (
    ThreeDOrchestrator,
    TaskRequirements,
    ModelCapability
)
from ..models.threed.base import QUALITY_PRESETS_3D
from ..services.websocket import create_websocket_progress_callback

logger = logging.getLogger(__name__)


class MemoryProfile(Enum):
    """Memory profiles for different VRAM configurations based on Hunyuan3D-2GP"""
    PROFILE_1 = "profile_1"  # 32GB+ VRAM - Ultra settings
    PROFILE_2 = "profile_2"  # 24GB VRAM - High settings
    PROFILE_3 = "profile_3"  # 16GB VRAM - Standard settings (default)
    PROFILE_4 = "profile_4"  # 9GB VRAM - Memory efficient
    PROFILE_5 = "profile_5"  # 6GB VRAM - Minimal requirements


MEMORY_PROFILE_CONFIGS = {
    MemoryProfile.PROFILE_1: {
        "name": "Ultra (32GB+)",
        "min_vram_gb": 32.0,
        "max_resolution": 2048,
        "batch_size": 8,
        "enable_quantization": False,
        "preferred_quantization": None,
        "use_cpu_offload": False,
        "use_sequential": False,
        "texture_resolution": 4096,
        "multiview_count": 12,
        "enable_pbr": True,
        "description": "Maximum quality for high-end workstations"
    },
    MemoryProfile.PROFILE_2: {
        "name": "High (24GB)",
        "min_vram_gb": 24.0,
        "max_resolution": 1536,
        "batch_size": 6,
        "enable_quantization": False,
        "preferred_quantization": None,
        "use_cpu_offload": False,
        "use_sequential": False,
        "texture_resolution": 2048,
        "multiview_count": 10,
        "enable_pbr": True,
        "description": "High quality for professional GPUs"
    },
    MemoryProfile.PROFILE_3: {
        "name": "Standard (16GB)",
        "min_vram_gb": 16.0,
        "max_resolution": 1024,
        "batch_size": 4,
        "enable_quantization": False,
        "preferred_quantization": None,
        "use_cpu_offload": False,
        "use_sequential": False,
        "texture_resolution": 1024,
        "multiview_count": 8,
        "enable_pbr": True,
        "description": "Default settings for consumer GPUs"
    },
    MemoryProfile.PROFILE_4: {
        "name": "Efficient (9GB)",
        "min_vram_gb": 9.0,
        "max_resolution": 768,
        "batch_size": 2,
        "enable_quantization": True,
        "preferred_quantization": "Q8_0",
        "use_cpu_offload": True,
        "use_sequential": True,
        "texture_resolution": 512,
        "multiview_count": 6,
        "enable_pbr": False,
        "description": "Memory efficient for mid-range GPUs"
    },
    MemoryProfile.PROFILE_5: {
        "name": "Minimal (6GB)",
        "min_vram_gb": 6.0,
        "max_resolution": 512,
        "batch_size": 1,
        "enable_quantization": True,
        "preferred_quantization": "Q4_K_M",
        "use_cpu_offload": True,
        "use_sequential": True,
        "texture_resolution": 512,
        "multiview_count": 4,
        "enable_pbr": False,
        "description": "Minimal requirements for entry-level GPUs"
    }
}


class ThreeDGenerator:
    """Main interface for 3D generation"""
    
    def __init__(self, memory_profile: Optional[MemoryProfile] = None):
        self.orchestrator = ThreeDOrchestrator()
        self.generation_count = 0
        
        # Memory profile system
        self.memory_profile = memory_profile or self._detect_memory_profile()
        self.memory_config = MEMORY_PROFILE_CONFIGS[self.memory_profile]
        
        logger.info(f"Initialized ThreeDGenerator with memory profile: {self.memory_config['name']} ({self.memory_config['min_vram_gb']}GB+)")
        
    def generate_3d(
        self,
        image: Union[Image.Image, str],
        prompt: Optional[str] = None,
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
            prompt: Text description of the desired 3D model (optional)
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
        
        # Create websocket progress callback if none provided
        if progress_callback is None:
            task_id = f"3d_gen_{int(time.time())}_{self.generation_count}"
            progress_callback = create_websocket_progress_callback(task_id, "3d_generation")
            logger.info(f"Created websocket progress callback for task: {task_id}")
        
        try:
            logger.info("\n" + "="*80)
            logger.info(f"[3D_GENERATION] Starting 3D generation")
            logger.info(f"[3D_GENERATION] Model type requested: '{model_type}'")
            logger.info(f"[3D_GENERATION] Quality preset: {quality_preset}")
            
            # Validate inputs
            if quality_preset not in QUALITY_PRESETS_3D:
                raise ValueError(f"Invalid quality preset: {quality_preset}")
                
            # Load image if path provided
            if isinstance(image, str):
                image = Image.open(image)
                
            # Get memory-optimized parameters
            optimized_params = self.get_optimized_generation_params(quality_preset)
            logger.info(f"[3D_GENERATION] Using memory profile {self.memory_config['name']} with optimizations: {optimized_params}")
            
            # Prepare task requirements with memory profile considerations
            required_capabilities = [ModelCapability.IMAGE_TO_3D]
            
            if enable_pbr and optimized_params["enable_pbr"]:
                required_capabilities.append(ModelCapability.PBR_MATERIALS)
                
            if quality_preset in {"high", "ultra"} and not optimized_params["memory_efficient"]:
                required_capabilities.append(ModelCapability.HIGH_RESOLUTION)
                
            requirements = TaskRequirements(
                input_type="image",
                quality_preset=quality_preset,
                output_format=output_format,
                required_capabilities=required_capabilities,
                max_generation_time=max_generation_time,
                # Add memory profile constraints
                memory_profile=optimized_params
            )
            
            # Override model selection if specific model requested
            if model_type != "auto":
                # Map user-friendly names to internal model types
                model_mapping = {
                    "hunyuan3d-21": "hunyuan3d-21",
                    "hunyuan3d-2.1": "hunyuan3d-21",
                    "hunyuan3d-21-turbo": "hunyuan3d-21-turbo",
                    "hunyuan3d-2.1-turbo": "hunyuan3d-21-turbo",
                    "hunyuan3d-mini": "hunyuan3d-2mini",
                    "hunyuan3d-2mini": "hunyuan3d-2mini",
                    "hunyuan3d-mini-turbo": "hunyuan3d-2mini-turbo",
                    "hunyuan3d-2mini-turbo": "hunyuan3d-2mini-turbo",
                    "hi3dgen": "hi3dgen",
                    "hi3dgen-turbo": "hi3dgen-turbo",
                    "sparc3d": "sparc3d",
                    "sparc3d-turbo": "sparc3d-turbo"
                }
                
                logger.info(f"[3D_GENERATION] Looking up model mapping for: '{model_type.lower()}'")
                internal_model = model_mapping.get(model_type.lower())
                logger.info(f"[3D_GENERATION] Mapped to internal model: '{internal_model}'")
                
                if internal_model:
                    # Filter available models
                    from ..models.threed.base import ModelType3D
                    available_models = [
                        m for m in self.orchestrator.available_models
                        if m.value == internal_model
                    ]
                    logger.info(f"[3D_GENERATION] Available models: {[m.value for m in self.orchestrator.available_models]}")
                    logger.info(f"[3D_GENERATION] Filtered models matching '{internal_model}': {[m.value for m in available_models]}")
                    
                    if available_models:
                        # Create a new requirements object with preferred model
                        # Don't modify orchestrator's available models list!
                        requirements.preferred_model = internal_model
                        logger.info(f"[3D_GENERATION] Set preferred model to: '{internal_model}'")
                    else:
                        logger.warning(f"Model {model_type} not available, using auto selection")
                        
            # Generate 3D model
            # Pass prompt through generation parameters
            generation_params = {}
            if prompt:
                generation_params["prompt"] = prompt
                
            result = self.orchestrator.generate(
                image,
                requirements,
                progress_callback,
                **generation_params
            )
            
            # Get generation time
            generation_time = time.time() - start_time
            
            # CRITICAL FIX: Use actual input image as preview instead of mesh render
            preview_image = None
            try:
                # Use the actual input image that was processed, not a mesh render
                # This fixes the user's complaint: "preview.png and generated image in the cache have nothing to do with each other"
                if isinstance(image, Image.Image):
                    preview_image = image.copy()  # Use the actual input image
                    logger.info(f"✅ Using actual input image as preview: {image.size}, {image.mode}")
                    
                    # Track the preview image hash for debugging
                    import hashlib
                    if hasattr(preview_image, 'tobytes'):
                        preview_hash = hashlib.md5(preview_image.tobytes()).hexdigest()[:8]
                        logger.info(f"🌼 [3D_GENERATION] Preview image hash: {preview_hash}")
                else:
                    # Fallback to mesh render only if no input image available
                    logger.info("No input image available, falling back to mesh render for preview")
                    preview_image = self._create_preview(result)
            except Exception as e:
                logger.warning(f"Failed to create preview image: {e}")
                preview_image = None
            
            # Log intermediate outputs for debugging
            if "intermediate_outputs" in result:
                logger.info(f"[3D_GENERATION] Intermediate outputs available: {list(result['intermediate_outputs'].keys())}")
            
            # Track generated images/views
            generated_images = []
            if "intermediate_outputs" in result and "views" in result["intermediate_outputs"]:
                generated_images = result["intermediate_outputs"]["views"]
                logger.info(f"[3D_GENERATION] Found {len(generated_images)} generated view images")
            
            logger.info(f"[3D_GENERATION] Preparing final response...")
                
            # Report completion with generated images info
            if progress_callback:
                if generated_images:
                    progress_callback(1.0, f"Complete! Generated {len(generated_images)} views and 3D model")
                else:
                    progress_callback(1.0, "Complete! Generated 3D model")
            
            # Prepare response
            response = {
                "output_path": result["output_path"],
                "preview_image": preview_image,
                "generation_time": generation_time,
                "model_used": result["model_used"],
                "generated_images": generated_images,  # Add generated images to response
                "metadata": {
                    "quality_preset": quality_preset,
                    "output_format": output_format,
                    "enable_pbr": enable_pbr,
                    "enable_depth_refinement": enable_depth_refinement,
                    "selection_reason": result.get("selection_reason", ""),
                    "memory_used_gb": result.get("memory_used_gb", 0),
                    "quantization": result.get("quantization"),
                    "generation_count": self.generation_count,
                    "views_generated": len(generated_images),
                }
            }
            
            logger.info(f"[3D_GENERATION] Generation completed successfully! Output: {response['output_path']}")
            return response
            
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
    
    def _detect_memory_profile(self) -> MemoryProfile:
        """Auto-detect appropriate memory profile based on available VRAM"""
        try:
            import torch
            if torch.cuda.is_available():
                # Get VRAM of primary GPU
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                logger.info(f"Detected {vram_gb:.1f}GB VRAM on primary GPU")
                
                # Select profile based on available VRAM
                if vram_gb >= 32:
                    return MemoryProfile.PROFILE_1
                elif vram_gb >= 24:
                    return MemoryProfile.PROFILE_2
                elif vram_gb >= 16:
                    return MemoryProfile.PROFILE_3
                elif vram_gb >= 9:
                    return MemoryProfile.PROFILE_4
                else:
                    return MemoryProfile.PROFILE_5
            else:
                logger.warning("CUDA not available, using minimal profile")
                return MemoryProfile.PROFILE_5
        except Exception as e:
            logger.warning(f"Failed to detect VRAM, using default profile: {e}")
            return MemoryProfile.PROFILE_3  # Default to 16GB profile
    
    def set_memory_profile(self, profile: Union[MemoryProfile, str]) -> bool:
        """Set memory profile manually
        
        Args:
            profile: Memory profile enum or string ("profile_1" to "profile_5")
            
        Returns:
            True if profile was set successfully
        """
        try:
            if isinstance(profile, str):
                profile = MemoryProfile(profile)
            
            if profile not in MEMORY_PROFILE_CONFIGS:
                logger.error(f"Invalid memory profile: {profile}")
                return False
            
            old_profile = self.memory_profile
            self.memory_profile = profile
            self.memory_config = MEMORY_PROFILE_CONFIGS[profile]
            
            logger.info(f"Changed memory profile from {MEMORY_PROFILE_CONFIGS[old_profile]['name']} to {self.memory_config['name']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set memory profile: {e}")
            return False
    
    def get_memory_profile_info(self) -> Dict[str, Any]:
        """Get current memory profile information"""
        return {
            "profile": self.memory_profile.value,
            "config": self.memory_config.copy(),
            "detected_vram_gb": self._get_detected_vram()
        }
    
    def _get_detected_vram(self) -> float:
        """Get detected VRAM in GB"""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except:
            pass
        return 0.0
    
    def get_optimized_generation_params(self, quality_preset: str) -> Dict[str, Any]:
        """Get generation parameters optimized for current memory profile
        
        Args:
            quality_preset: Base quality preset
            
        Returns:
            Optimized parameters for the current memory profile
        """
        base_preset = QUALITY_PRESETS_3D.get(quality_preset, QUALITY_PRESETS_3D["standard"])
        
        # Apply memory profile optimizations
        optimized_params = {
            "max_resolution": min(base_preset.reconstruction_resolution, self.memory_config["max_resolution"]),
            "batch_size": self.memory_config["batch_size"],
            "texture_resolution": min(base_preset.texture_resolution, self.memory_config["texture_resolution"]),
            "multiview_count": min(base_preset.multiview_count, self.memory_config["multiview_count"]),
            "enable_quantization": self.memory_config["enable_quantization"],
            "use_cpu_offload": self.memory_config["use_cpu_offload"],
            "use_sequential": self.memory_config["use_sequential"],
            "enable_pbr": base_preset.use_pbr and self.memory_config["enable_pbr"],
            "memory_efficient": base_preset.memory_efficient or self.memory_config["enable_quantization"]
        }
        
        logger.debug(f"Optimized params for {quality_preset} with {self.memory_config['name']}: {optimized_params}")
        return optimized_params
        
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