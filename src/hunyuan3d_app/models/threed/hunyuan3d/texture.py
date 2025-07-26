"""Texture generation component for HunYuan3D."""

import os
import torch
import numpy as np
import trimesh
import logging
from typing import Optional, Dict, Any, Tuple, Union
from pathlib import Path
from PIL import Image

from .config import HunYuan3DConfig, MODEL_VARIANTS
from .utils import raise_not_implemented, validate_device, get_optimal_dtype
from ..base import Base3DModel
from ..memory import optimize_memory_usage

logger = logging.getLogger(__name__)


class HunYuan3DTexture(Base3DModel):
    """Texture generation for 3D meshes."""
    
    def __init__(self, config: HunYuan3DConfig):
        """Initialize texture generator.
        
        Args:
            config: HunYuan3D configuration
        """
        # Get model path from config
        cache_dir = config.cache_dir or Path.home() / ".cache" / "huggingface"
        model_path = Path(cache_dir) / "hunyuan3d" / "texture" / config.model_variant
        device = validate_device(config.device)
        dtype = get_optimal_dtype(device, config.dtype == "float16")
        
        # Initialize base class with required parameters
        super().__init__(model_path=model_path, device=str(device), dtype=dtype)
        
        self.config = config
        
        self.pipeline = None
        self._memory_usage = 0
        
        # Model variant info
        self.variant_info = MODEL_VARIANTS.get(config.model_variant, {})
        self.model_id = self.variant_info.get("texture_model")
        self.supports_pbr = self.variant_info.get("supports_pbr", False)
        
        logger.info(
            f"Initialized HunYuan3D Texture - Model: {self.model_id}, "
            f"PBR: {self.supports_pbr}, Device: {self.device}"
        )
    
    def load(self, progress_callback=None) -> bool:
        """Load the model weights - implements abstract method from Base3DModel."""
        if self.pipeline is not None:
            logger.info("Texture model already loaded")
            self.loaded = True
            return True
        
        # Skip if no texture model for this variant
        if not self.model_id:
            logger.info(f"No texture model for variant {self.config.model_variant}")
            self.loaded = True  # Consider it "loaded" even if no model
            return True
        
        try:
            # Set up paths
            from .setup import get_hunyuan3d_path, fix_import_compatibility
            hunyuan3d_path = get_hunyuan3d_path()
            fix_import_compatibility()
            
            # Load texture pipeline
            self._load_texture_pipeline()
            
            # Apply memory optimizations
            if self.config.enable_model_offloading:
                self._enable_model_offloading()
            
            # Track memory usage
            self._update_memory_usage()
            
            logger.info(
                f"Loaded texture model. Memory usage: {self._memory_usage:.1f}GB"
            )
            self.loaded = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to load texture model: {e}")
            self.loaded = False
            raise
    
    def load_model(self) -> None:
        """Legacy method for backward compatibility."""
        self.load()
    
    def _load_texture_pipeline(self):
        """Load the texture generation pipeline - using official HunYuan3D approach."""
        try:
            # Import required modules
            import huggingface_hub
            from diffusers import DiffusionPipeline
            from diffusers import UniPCMultistepScheduler
            
            # Get custom pipeline path (relative to this file)
            custom_pipeline = os.path.join(
                os.path.dirname(__file__), 
                "..", "..", "..", "..", "..", 
                "Hunyuan3D", "hy3dpaint", "hunyuanpaintpbr"
            )
            custom_pipeline = os.path.abspath(custom_pipeline)
            
            # Check if model already exists locally
            model_path = self._get_model_path()
            
            if model_path.exists():
                logger.info(f"Loading texture model from local path: {model_path}")
                model_load_path = str(model_path)
            else:
                logger.info(f"Downloading texture model: {self.model_id}")
                # Download using snapshot_download like official implementation
                cache_dir = self.config.cache_dir or str(Path.home() / ".cache" / "huggingface")
                model_snapshot_path = huggingface_hub.snapshot_download(
                    repo_id=self.variant_info['repo_id'],
                    allow_patterns=["hunyuan3d-paintpbr-v2-1/*"],
                    cache_dir=cache_dir
                )
                model_load_path = os.path.join(model_snapshot_path, "hunyuan3d-paintpbr-v2-1")
            
            logger.info(f"Loading pipeline from: {model_load_path}")
            
            # Load pipeline exactly like the official implementation
            self.pipeline = DiffusionPipeline.from_pretrained(
                model_load_path,
                custom_pipeline=custom_pipeline,
                torch_dtype=torch.float16  # Use float16 like official implementation
            )
            
            # Configure scheduler like official implementation
            self.pipeline.scheduler = UniPCMultistepScheduler.from_config(
                self.pipeline.scheduler.config, 
                timestep_spacing="trailing"
            )
            self.pipeline.set_progress_bar_config(disable=True)
            self.pipeline.eval()
            
            # Set view size if needed
            if hasattr(self.pipeline, 'unet'):
                setattr(self.pipeline, "view_size", 320)  # Default from official config
            
            # Move to device
            self.pipeline = self.pipeline.to(self.device)
            
        except ImportError as e:
            logger.error(f"Failed to import texture modules: {e}")
            raise RuntimeError(
                f"Failed to import HunYuan3D texture modules: {e}\n"
                "Please ensure hy3dpaint is installed and available."
            )
        except Exception as e:
            logger.error(f"Failed to load texture pipeline: {e}")
            raise
    
    def _get_model_path(self) -> Path:
        """Get local model path."""
        from ....config import MODELS_DIR
        return MODELS_DIR / "3d" / self.config.model_variant / "texture" / self.model_id
    
    def _enable_model_offloading(self):
        """Enable model CPU offloading."""
        if self.pipeline is None:
            return
        
        try:
            if hasattr(self.pipeline, 'enable_model_cpu_offload'):
                self.pipeline.enable_model_cpu_offload()
                logger.info("Enabled texture model CPU offloading")
        except Exception as e:
            logger.warning(f"Failed to enable offloading: {e}")
    
    def _update_memory_usage(self):
        """Update memory usage tracking."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self._memory_usage = torch.cuda.memory_allocated() / 1024**3
    
    def generate_texture(
        self,
        mesh: trimesh.Trimesh,
        prompt: str,
        resolution: Optional[int] = None,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        seed: Optional[int] = None,
        generate_pbr: bool = True,
        **kwargs
    ) -> Dict[str, Union[np.ndarray, Image.Image]]:
        """Generate texture for mesh.
        
        Args:
            mesh: Input mesh
            prompt: Text description for texture
            resolution: Texture resolution
            guidance_scale: Guidance scale
            num_inference_steps: Number of denoising steps
            seed: Random seed
            generate_pbr: Whether to generate PBR materials
            **kwargs: Additional pipeline arguments
            
        Returns:
            Dictionary containing texture maps
        """
        # Use default resolution if not specified
        resolution = resolution or self.config.texture_resolution
        
        # Check if we have a texture model
        if self.pipeline is None:
            if self.model_id:
                self.load_model()
            else:
                raise RuntimeError(
                    f"No texture model available for variant {self.config.model_variant}. "
                    "Texture generation requires a HunYuan3D variant with texture support."
                )
        
        # Ensure pipeline is valid
        if self.pipeline is None or not callable(self.pipeline):
            raise RuntimeError(
                "Texture pipeline is not properly initialized. "
                "Please check that the texture model was loaded successfully."
            )
        
        try:
            logger.info(
                f"Generating texture at {resolution}x{resolution} "
                f"for mesh with {len(mesh.vertices)} vertices"
            )
            
            # Ensure mesh has UV coordinates
            if not hasattr(mesh, 'visual') or not hasattr(mesh.visual, 'uv'):
                logger.info("Generating UV coordinates")
                mesh = self._generate_uvs(mesh)
            
            # Set random seed
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)
            else:
                generator = None
            
            with optimize_memory_usage():
                # Generate texture
                with torch.no_grad():
                    texture_outputs = self.pipeline(
                        mesh=mesh,
                        prompt=prompt,
                        height=resolution,
                        width=resolution,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        generator=generator,
                        output_type="pil",
                        **kwargs
                    )
                
                # Extract texture maps
                texture_maps = self._extract_texture_maps(
                    texture_outputs,
                    generate_pbr and self.supports_pbr
                )
            
            logger.info(f"Generated {len(texture_maps)} texture maps")
            return texture_maps
            
        except Exception as e:
            logger.error(f"Texture generation failed: {e}")
            raise RuntimeError(f"Texture generation failed: {str(e)}")
    
    def _generate_uvs(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Generate UV coordinates for mesh."""
        try:
            # Try to use xatlas for UV unwrapping
            import xatlas
            
            # Pack UV atlas
            vmapping, indices, uvs = xatlas.parametrize(
                mesh.vertices,
                mesh.faces
            )
            
            # Create new mesh with UVs
            mesh_with_uvs = trimesh.Trimesh(
                vertices=mesh.vertices[vmapping],
                faces=indices,
                visual=trimesh.visual.TextureVisuals(uv=uvs)
            )
            
            return mesh_with_uvs
            
        except ImportError:
            logger.warning("xatlas not available, using simple UV projection")
            # Fallback to spherical UV mapping
            return self._spherical_uv_projection(mesh)
    
    def _spherical_uv_projection(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Apply spherical UV projection."""
        vertices = mesh.vertices
        
        # Center vertices
        center = vertices.mean(axis=0)
        centered = vertices - center
        
        # Convert to spherical coordinates
        r = np.linalg.norm(centered, axis=1)
        theta = np.arctan2(centered[:, 1], centered[:, 0])
        phi = np.arccos(np.clip(centered[:, 2] / (r + 1e-8), -1, 1))
        
        # Map to UV coordinates
        u = (theta + np.pi) / (2 * np.pi)
        v = phi / np.pi
        
        # Create UV coordinates
        uv = np.stack([u, v], axis=1)
        
        # Create visual with UVs
        mesh.visual = trimesh.visual.TextureVisuals(uv=uv)
        
        return mesh
    
    def _extract_texture_maps(
        self,
        outputs: Any,
        generate_pbr: bool
    ) -> Dict[str, Image.Image]:
        """Extract texture maps from pipeline outputs."""
        texture_maps = {}
        
        # Extract base color / albedo
        if hasattr(outputs, 'images'):
            texture_maps['albedo'] = outputs.images[0]
        elif isinstance(outputs, list):
            texture_maps['albedo'] = outputs[0]
        else:
            texture_maps['albedo'] = outputs
        
        # Extract PBR maps if available
        if generate_pbr and hasattr(outputs, 'pbr_maps'):
            pbr_maps = outputs.pbr_maps
            
            if 'normal' in pbr_maps:
                texture_maps['normal'] = pbr_maps['normal']
            if 'metallic' in pbr_maps:
                texture_maps['metallic'] = pbr_maps['metallic']
            if 'roughness' in pbr_maps:
                texture_maps['roughness'] = pbr_maps['roughness']
            if 'ao' in pbr_maps:
                texture_maps['ao'] = pbr_maps['ao']
        
        return texture_maps
    
    def _generate_fallback_texture(
        self,
        mesh: trimesh.Trimesh,
        resolution: int
    ) -> Dict[str, np.ndarray]:
        """This method should not be used - proper texture generation required."""
        raise RuntimeError(
            "Texture generation failed. No fallback texture available. "
            "Please ensure the HunYuan3D texture pipeline is properly configured."
        )
    
    def apply_texture_to_mesh(
        self,
        mesh: trimesh.Trimesh,
        texture_maps: Dict[str, Union[np.ndarray, Image.Image]]
    ) -> trimesh.Trimesh:
        """Apply texture maps to mesh.
        
        Args:
            mesh: Input mesh
            texture_maps: Dictionary of texture maps
            
        Returns:
            Textured mesh
        """
        if 'albedo' not in texture_maps:
            logger.warning("No albedo map found")
            return mesh
        
        # Get albedo texture
        albedo = texture_maps['albedo']
        if isinstance(albedo, np.ndarray):
            albedo = Image.fromarray(albedo)
        
        # Create texture visual
        mesh.visual = trimesh.visual.texture.TextureVisuals(
            image=albedo,
            uv=mesh.visual.uv if hasattr(mesh.visual, 'uv') else None
        )
        
        # Store additional maps as metadata
        if 'normal' in texture_maps:
            mesh.metadata['normal_map'] = texture_maps['normal']
        if 'metallic' in texture_maps:
            mesh.metadata['metallic_map'] = texture_maps['metallic']
        if 'roughness' in texture_maps:
            mesh.metadata['roughness_map'] = texture_maps['roughness']
        
        return mesh
    
    def unload_model(self) -> None:
        """Unload model to free memory."""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self._memory_usage = 0
            logger.info("Unloaded texture model")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage in GB - implements abstract method from Base3DModel."""
        return {
            "total": self._memory_usage,
            "model": self._memory_usage,
            "cache": 0.0
        }
    
    def _update_memory_usage(self) -> None:
        """Update tracked memory usage."""
        if torch.cuda.is_available():
            # Get GPU memory usage
            self._memory_usage = torch.cuda.memory_allocated(self.device) / 1024**3
        else:
            # Estimate based on model type
            self._memory_usage = 3.0  # Texture models are typically smaller