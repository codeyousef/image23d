"""HunYuan3D 2.1 implementation with GGUF quantization support

Based on the 3D Implementation Guide:
- Two-stage pipeline: multi-view diffusion + sparse-view reconstruction
- GGUF quantization for memory efficiency
- Support for 2.1, 2.0, and 2mini variants
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
import sys
import os
import torch
import numpy as np
from PIL import Image
import trimesh
from huggingface_hub import hf_hub_download, snapshot_download

# Setup logger early
logger = logging.getLogger(__name__)

# Add HunYuan3D paths to sys.path
hunyuan_base = Path(__file__).parent.parent.parent.parent.parent / "Hunyuan3D"
if hunyuan_base.exists():
    sys.path.insert(0, str(hunyuan_base / "hy3dshape"))
    sys.path.insert(0, str(hunyuan_base / "hy3dpaint"))

# Set environment variable for HunYuan3D models
# Use forward slashes for cross-platform compatibility
models_path = Path(__file__).parent.parent.parent.parent.parent / "models" / "3d"
# Ensure the path is absolute and uses forward slashes
models_path_str = str(models_path.resolve()).replace('\\', '/')
os.environ['HY3DGEN_MODELS'] = models_path_str
logger.info(f"Set HY3DGEN_MODELS to: {models_path_str}")

from .base import (
    Base3DPipeline,
    MultiViewModel,
    ReconstructionModel,
    TextureModel,
    QualityPreset3D,
    QUALITY_PRESETS_3D,
    IntermediateFormat
)
from .memory import ThreeDMemoryManager
from .intermediate import (
    DepthEstimator,
    NormalEstimator,
    UVUnwrapper,
    TextureSynthesizer,
    PBRMaterialGenerator
)
from ..gguf_wrapper import StandaloneGGUFPipeline


@dataclass
class HunYuan3DConfig:
    """Configuration for HunYuan3D models"""
    model_variant: str = "hunyuan3d-21"  # or "hunyuan3d-2mv", "hunyuan3d-2mini"
    use_gguf: bool = True
    gguf_quantization: Optional[str] = "Q8_0"  # Q8_0, Q6_K, Q5_K_S, Q4_K_M
    device: str = "cuda"
    dtype: torch.dtype = torch.float16
    
    # Model paths
    model_base_path: Path = Path("models/3d")
    
    # Pipeline settings
    enable_intermediate_processing: bool = True
    enable_pbr_materials: bool = False
    enable_depth_refinement: bool = True
    
    # Memory settings
    sequential_offload: bool = False
    cpu_offload: bool = False
    attention_slicing: bool = True
    
    def __post_init__(self):
        """Fix paths after initialization"""
        # Convert to absolute path
        if not self.model_base_path.is_absolute():
            # Get the project root (5 levels up from this file)
            project_root = Path(__file__).parent.parent.parent.parent.parent
            self.model_base_path = (project_root / self.model_base_path).resolve()
    
    def get_model_id(self) -> str:
        """Get HuggingFace model ID"""
        model_ids = {
            "hunyuan3d-21": "tencent/HunYuan3D-2.1",
            "hunyuan3d-2mv": "tencent/HunYuan3D-2.0-DiT-MV",
            "hunyuan3d-2mini": "tencent/HunYuan3D-2.0-mini"
        }
        return model_ids.get(self.model_variant, model_ids["hunyuan3d-21"])


class HunYuan3DMultiView(MultiViewModel):
    """Multi-view generation component for HunYuan3D"""
    
    def __init__(
        self,
        config: HunYuan3DConfig,
        model_path: Optional[Path] = None
    ):
        self.config = config
        self.model_path = model_path or config.model_base_path / "multiview"
        super().__init__(self.model_path, config.device, config.dtype)
        
        self.pipeline = None
        self.is_gguf_wrapped = False
        
    def load(self, progress_callback=None) -> bool:
        """Load multi-view diffusion model"""
        try:
            if progress_callback:
                progress_callback(0.1, "Loading HunYuan3D multi-view model...")
                
            model_id = self.config.get_model_id()
            
            # Check if GGUF model exists
            if self.config.use_gguf and self.config.gguf_quantization:
                gguf_path = self.model_path / f"multiview_{self.config.gguf_quantization}.gguf"
                
                if gguf_path.exists():
                    logger.info(f"Loading GGUF model from {gguf_path}")
                    # Load with GGUF wrapper
                    from ..gguf_wrapper import StandaloneGGUFPipeline
                    self.pipeline = StandaloneGGUFPipeline(
                        model_path=str(gguf_path),
                        model_type="hunyuan3d_mv",
                        device=self.config.device
                    )
                    self.is_gguf_wrapped = True
                else:
                    logger.warning(f"GGUF model not found at {gguf_path}, loading standard model")
                    
            if self.pipeline is None:
                logger.info(f"Loading HunYuan3D multi-view model: {model_id}")
                
                # Try to import HunYuan3D
                try:
                    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
                    from hy3dshape.rembg import BackgroundRemover
                    
                    logger.info("Successfully imported HunYuan3D modules")
                    
                    # Check if model weights exist
                    model_variant_path = self.config.model_base_path / self.config.model_variant
                    
                    # Log paths for debugging
                    logger.info(f"Looking for model at: {model_variant_path}")
                    logger.info(f"Absolute path: {model_variant_path.resolve()}")
                    logger.info(f"Path exists: {model_variant_path.exists()}")
                    logger.info(f"HY3DGEN_MODELS env: {os.environ.get('HY3DGEN_MODELS', 'not set')}")
                    logger.info(f"Current working directory: {os.getcwd()}")
                    logger.info(f"model_base_path type: {type(self.config.model_base_path)}")
                    
                    # Try to list parent directory
                    if self.config.model_base_path.exists():
                        logger.info(f"Contents of {self.config.model_base_path}:")
                        for item in self.config.model_base_path.iterdir():
                            logger.info(f"  - {item.name}")
                    
                    if model_variant_path.exists():
                        logger.info(f"Found existing model at {model_variant_path}")
                        
                        # Check for specific model components
                        dit_path = model_variant_path / "hunyuan3d-dit-v2-1"
                        vae_path = model_variant_path / "hunyuan3d-vae-v2-1"
                        
                        if dit_path.exists() and vae_path.exists():
                            logger.info("Found HunYuan3D components - loading pipeline")
                            logger.info(f"DIT path: {dit_path}")
                            logger.info(f"VAE path: {vae_path}")
                            
                            if progress_callback:
                                progress_callback(0.5, "Loading HunYuan3D pipeline...")
                            
                            try:
                                # Check if we need to use the dit/vae paths directly
                                logger.info(f"Attempting to load pipeline from: {model_variant_path}")
                                
                                # Try loading with specific component paths
                                import torch
                                
                                # The HunYuan3D pipeline expects just the model name
                                # Since we set HY3DGEN_MODELS environment variable
                                logger.info("Calling Hunyuan3DDiTFlowMatchingPipeline.from_pretrained...")
                                try:
                                    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                                        self.config.model_variant,  # Just "hunyuan3d-21"
                                        subfolder="hunyuan3d-dit-v2-1",  # DIT subfolder
                                        device=self.config.device,
                                        dtype=self.config.dtype,
                                        use_safetensors=False,  # We have .ckpt files
                                        variant="fp16"  # Our models are fp16
                                    )
                                    logger.info(f"Pipeline returned: {pipeline}")
                                    logger.info(f"Pipeline type after creation: {type(pipeline)}")
                                    
                                    # Check if pipeline was created successfully
                                    if pipeline is None:
                                        raise RuntimeError("Pipeline.from_pretrained returned None")
                                    
                                    self.pipeline = pipeline
                                    logger.info(f"self.pipeline assigned: {self.pipeline}")
                                except Exception as e:
                                    logger.error(f"Exception during pipeline creation: {e}")
                                    logger.error(f"Exception type: {type(e).__name__}")
                                    import traceback
                                    logger.error(f"Full traceback: {traceback.format_exc()}")
                                    raise
                                
                                # Move to device if needed
                                if hasattr(self.pipeline, 'to') and self.pipeline is not None:
                                    self.pipeline = self.pipeline.to(self.config.device)
                                
                                # Also initialize background remover
                                try:
                                    self.bg_remover = BackgroundRemover()
                                except Exception as e:
                                    logger.warning(f"Failed to initialize background remover: {e}")
                                    self.bg_remover = None
                                
                                logger.info("HunYuan3D pipeline loaded successfully")
                                logger.info(f"Pipeline type: {type(self.pipeline)}")
                                logger.info(f"Pipeline attributes: {dir(self.pipeline)[:10]}...")  # First 10 attrs
                                
                            except Exception as e:
                                logger.error(f"Failed to load HunYuan3D pipeline: {e}")
                                logger.error(f"Error type: {type(e).__name__}")
                                import traceback
                                logger.error(f"Traceback: {traceback.format_exc()}")
                                
                                # Try alternative loading method
                                logger.info("Trying alternative loading method...")
                                try:
                                    # Maybe it expects a config file
                                    config_path = dit_path / "config.yaml"
                                    if config_path.exists():
                                        logger.info(f"Found config at: {config_path}")
                                        # Try loading with config
                                        from omegaconf import OmegaConf
                                        config = OmegaConf.load(config_path)
                                        logger.info(f"Config loaded: {config}")
                                    
                                    # Try manual initialization
                                    from hy3dshape.models import DiT_hy_v2_1
                                    from hy3dshape.scheduler import PNDMScheduler
                                    
                                    # Load model manually
                                    model = DiT_hy_v2_1()
                                    ckpt_path = dit_path / "model.fp16.ckpt"
                                    if ckpt_path.exists():
                                        logger.info(f"Loading checkpoint from: {ckpt_path}")
                                        state_dict = torch.load(ckpt_path, map_location=self.config.device)
                                        model.load_state_dict(state_dict)
                                        model = model.to(self.config.device, dtype=self.config.dtype)
                                        
                                        # Create pipeline wrapper
                                        self.pipeline = type('Pipeline', (), {
                                            'model': model,
                                            '__call__': lambda self, image, **kwargs: self.model(image, **kwargs)
                                        })()
                                        
                                        logger.info("Loaded model manually")
                                    else:
                                        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
                                        
                                except Exception as e2:
                                    logger.error(f"Alternative loading also failed: {e2}")
                                    self._raise_not_implemented(f"Failed to load pipeline: {str(e)}")
                        else:
                            self._raise_not_implemented("Model components (dit/vae) not found")
                    else:
                        # Use forward slashes in error message for clarity
                        path_str = str(model_variant_path).replace('\\', '/')
                        self._raise_not_implemented(f"Model not found at {path_str}")
                        
                except ImportError as e:
                    logger.error(f"Failed to import HunYuan3D modules: {e}")
                    self._raise_not_implemented(
                        "HunYuan3D modules not found. Please ensure:\\n"
                        "1. Hunyuan3D repository is cloned\\n"
                        "2. Dependencies are installed: pip install -e ./Hunyuan3D\\n"
                        "3. Check if hy3dshape module exists"
                    )
                
            # Mark as loaded if pipeline was created
            self.loaded = (self.pipeline is not None)
            
            if not self.loaded:
                logger.error("Pipeline was not created - model not loaded")
                return False
            
            if progress_callback:
                progress_callback(1.0, "Multi-view model loaded")
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to load multi-view model: {e}")
            return False
            
    def _raise_not_implemented(self, reason: str):
        """Raise error with clear message about what's not implemented"""
        raise NotImplementedError(
            f"HunYuan3D 2.1 generation failed: {reason}\n"
            f"To use HunYuan3D 2.1, ensure:\n"
            f"1. The Hunyuan3D repository is cloned\n"
            f"2. All dependencies are installed\n"
            f"3. Model weights are downloaded to {self.config.model_base_path}"
        )
    
    def unload(self):
        """Unload model to free memory"""
        if self.pipeline is not None:
            if hasattr(self.pipeline, 'unload'):
                self.pipeline.unload()
            del self.pipeline
            self.pipeline = None
            
        self.loaded = False
        torch.cuda.empty_cache()
        
    def generate_views(
        self,
        image: Union[Image.Image, torch.Tensor],
        num_views: int = 6,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
        seed: Optional[int] = None,
        progress_callback=None
    ) -> List[Image.Image]:
        """Generate multiple views from input image"""
        
        if not self.loaded:
            raise RuntimeError("Model not loaded")
            
        # Convert image to tensor if needed
        if isinstance(image, Image.Image):
            # Preprocess image
            image = image.convert("RGB")
            image = image.resize((512, 512))  # Standard size
            
        try:
            # HunYuan3D doesn't generate multiple views directly
            # It generates a 3D mesh from a single image
            # For multi-view consistency, we'll return the input image
            # The actual 3D generation happens in the reconstruction phase
            
            logger.info("HunYuan3D uses single image input for 3D generation")
            
            # Remove background if needed
            if hasattr(self, 'bg_remover') and image.mode != 'RGBA':
                if progress_callback:
                    progress_callback(0.5, "Removing background...")
                try:
                    image = self.bg_remover(image)
                except Exception as e:
                    logger.warning(f"Background removal failed: {e}")
            
            # Return single processed image
            # HunYuan3D will handle the multi-view generation internally
            return [image]
            
        except Exception as e:
            logger.error(f"View generation failed: {e}")
            raise
            
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage"""
        if not self.loaded:
            return {"total": 0.0}
            
        # Estimate based on model variant
        base_usage = {
            "hunyuan3d-21": 8.0,
            "hunyuan3d-2mv": 6.0,
            "hunyuan3d-2mini": 4.0
        }
        
        usage = base_usage.get(self.config.model_variant, 8.0)
        
        # Adjust for quantization
        if self.config.use_gguf and self.config.gguf_quantization:
            quant_factors = {
                "Q8_0": 0.5,
                "Q6_K": 0.4,
                "Q5_K_S": 0.35,
                "Q4_K_M": 0.3
            }
            usage *= quant_factors.get(self.config.gguf_quantization, 0.5)
            
        return {"total": usage}


class HunYuan3DReconstruction(ReconstructionModel):
    """HunYuan3D mesh generation component"""
    
    def __init__(
        self,
        config: HunYuan3DConfig,
        model_path: Optional[Path] = None,
        multiview_model: Optional['HunYuan3DMultiView'] = None
    ):
        self.config = config
        self.model_path = model_path or config.model_base_path / "reconstruction"
        super().__init__(self.model_path, config.device, config.dtype)
        
        self.multiview_model = multiview_model  # Reference to multiview model with pipeline
        
    def load(self, progress_callback=None) -> bool:
        """Load reconstruction model"""
        try:
            if progress_callback:
                progress_callback(0.1, "Loading reconstruction model...")
                
            # TODO: Load actual reconstruction model
            # For now, just mark as loaded
            logger.info("Loading HunYuan3D reconstruction model")
            
            self.loaded = True
            
            if progress_callback:
                progress_callback(1.0, "Reconstruction model loaded")
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to load reconstruction model: {e}")
            return False
            
    def unload(self):
        """Unload model"""
        if self.model is not None:
            del self.model
            self.model = None
        self.loaded = False
        torch.cuda.empty_cache()
        
    def reconstruct(
        self,
        views: List[Image.Image],
        resolution: int = 256,
        use_depth: bool = False,
        use_normals: bool = False,
        progress_callback=None
    ) -> trimesh.Trimesh:
        """Generate 3D mesh using HunYuan3D"""
        
        if not self.loaded:
            raise RuntimeError("Model not loaded")
            
        # Check if we have access to the HunYuan3D pipeline
        if not self.multiview_model or not hasattr(self.multiview_model, 'pipeline'):
            raise RuntimeError("HunYuan3D pipeline not available in reconstruction model")
            
        try:
            # HunYuan3D expects a single RGBA image
            if views and len(views) > 0:
                image = views[0]  # Use first image (should be preprocessed with bg removed)
                
                if progress_callback:
                    progress_callback(0.1, "Starting HunYuan3D mesh generation...")
                
                # Ensure image is RGBA
                if image.mode != 'RGBA':
                    image = image.convert('RGBA')
                
                # Generate mesh using HunYuan3D pipeline
                if progress_callback:
                    progress_callback(0.3, "Running HunYuan3D inference...")
                
                # Call the pipeline - it returns List[List[trimesh.Trimesh]]
                result = self.multiview_model.pipeline(
                    image=image,
                    num_inference_steps=50,  # Default steps
                    guidance_scale=7.5,      # Default guidance
                    output_type="trimesh"
                )
                
                if isinstance(result, list) and len(result) > 0:
                    # Extract the first mesh from the nested list structure
                    if isinstance(result[0], list) and len(result[0]) > 0:
                        mesh = result[0][0]  # First batch, first mesh
                    else:
                        mesh = result[0]  # Direct mesh
                    
                    if progress_callback:
                        progress_callback(0.9, "Processing mesh...")
                    
                    # Ensure it's a trimesh object
                    if not isinstance(mesh, trimesh.Trimesh):
                        # Convert if needed
                        logger.warning(f"Got unexpected mesh type: {type(mesh)}")
                        if hasattr(mesh, 'vertices') and hasattr(mesh, 'faces'):
                            vertices = mesh.vertices
                            faces = mesh.faces
                            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                        else:
                            raise ValueError(f"Generated mesh has invalid format: {type(mesh)}")
                    
                    # Log mesh info
                    logger.info(f"Generated mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
                    
                    if progress_callback:
                        progress_callback(1.0, "Mesh generation complete")
                    
                    return mesh
                else:
                    raise ValueError("HunYuan3D pipeline returned no mesh")
            else:
                raise ValueError("No input image provided")
            
        except Exception as e:
            logger.error(f"HunYuan3D reconstruction failed: {e}")
            raise RuntimeError(f"3D generation failed: {str(e)}")
            
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage"""
        if not self.loaded:
            return {"total": 0.0}
            
        # Estimate usage
        return {"total": 4.0}  # 4GB for reconstruction model


class HunYuan3DTexture(TextureModel):
    """Texture generation component"""
    
    def __init__(
        self,
        config: HunYuan3DConfig,
        model_path: Optional[Path] = None
    ):
        self.config = config
        self.model_path = model_path or config.model_base_path / "texture"
        super().__init__(self.model_path, config.device, config.dtype)
        
        self.synthesizer = TextureSynthesizer(config.device)
        self.pbr_generator = PBRMaterialGenerator(config.device)
        
    def load(self, progress_callback=None) -> bool:
        """Load texture model"""
        self.loaded = True
        return True
        
    def unload(self):
        """Unload model"""
        self.loaded = False
        
    def generate_texture(
        self,
        mesh: trimesh.Trimesh,
        views: List[Image.Image],
        resolution: int = 1024,
        use_pbr: bool = False,
        progress_callback=None
    ) -> Dict[str, Image.Image]:
        """Generate textures for mesh"""
        
        if not self.loaded:
            raise RuntimeError("Model not loaded")
            
        try:
            # UV unwrapping
            if progress_callback:
                progress_callback(0.2, "UV unwrapping...")
                
            unwrapper = UVUnwrapper()
            uv_data = unwrapper.process(mesh, resolution)
            
            # Texture synthesis
            if progress_callback:
                progress_callback(0.5, "Synthesizing texture...")
                
            base_texture = self.synthesizer.process(mesh, views, uv_data, resolution)
            
            result = {"diffuse": base_texture}
            
            # Generate PBR maps if requested
            if use_pbr:
                if progress_callback:
                    progress_callback(0.8, "Generating PBR materials...")
                    
                pbr_maps = self.pbr_generator.process(base_texture, mesh)
                result.update(pbr_maps)
                
            if progress_callback:
                progress_callback(1.0, "Texture generation complete")
                
            return result
            
        except Exception as e:
            logger.error(f"Texture generation failed: {e}")
            raise
            
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage"""
        return {"total": 1.0}  # Minimal for texture processing


class HunYuan3DPipeline(Base3DPipeline):
    """Complete HunYuan3D pipeline"""
    
    def __init__(
        self,
        config: HunYuan3DConfig,
        memory_manager: Optional[ThreeDMemoryManager] = None
    ):
        self.config = config
        
        # Initialize components
        multiview = HunYuan3DMultiView(config)
        reconstruction = HunYuan3DReconstruction(config, multiview_model=multiview)
        texture = HunYuan3DTexture(config)
        
        # Initialize intermediate processors
        processors = {}
        if config.enable_intermediate_processing:
            processors["depth"] = DepthEstimator(config.device)
            processors["normal"] = NormalEstimator(config.device)
            
        super().__init__(
            multiview_model=multiview,
            reconstruction_model=reconstruction,
            texture_model=texture,
            intermediate_processors=processors,
            device=config.device
        )
        
        self.memory_manager = memory_manager
        
    def load_models(self, components: List[str] = None, progress_callback=None) -> bool:
        """Load specified components or all"""
        
        if components is None:
            components = ["multiview", "reconstruction", "texture"]
            
        total_steps = len(components)
        
        for i, component in enumerate(components):
            if progress_callback:
                progress = i / total_steps
                progress_callback(progress, f"Loading {component}...")
                
            try:
                if component == "multiview" and self.multiview_model:
                    if not self.multiview_model.load(progress_callback):
                        logger.error(f"Failed to load {component}")
                        return False
                        
                elif component == "reconstruction" and self.reconstruction_model:
                    if not self.reconstruction_model.load(progress_callback):
                        logger.error(f"Failed to load {component}")
                        return False
                        
                elif component == "texture" and self.texture_model:
                    if not self.texture_model.load(progress_callback):
                        logger.error(f"Failed to load {component}")
                        return False
            except Exception as e:
                logger.error(f"Exception loading {component}: {e}")
                return False
                    
        if progress_callback:
            progress_callback(1.0, "All models loaded")
            
        return True
        
    def generate(
        self,
        image: Union[Image.Image, torch.Tensor],
        quality_preset: Union[str, QualityPreset3D] = "standard",
        output_format: str = "glb",
        progress_callback=None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate 3D model from image"""
        
        # Get quality preset
        if isinstance(quality_preset, str):
            preset = QUALITY_PRESETS_3D.get(quality_preset, QUALITY_PRESETS_3D["standard"])
        else:
            preset = quality_preset
            
        # Memory check
        if self.memory_manager:
            required_memory = self._estimate_memory_requirement(preset)
            can_proceed, msg = self.memory_manager.check_memory_available(required_memory)
            if not can_proceed:
                raise RuntimeError(f"Insufficient memory: {msg}")
                
        try:
            # Stage 1: Multi-view generation
            if progress_callback:
                progress_callback(0.0, "Generating multiple views...")
                
            views = self.multiview_model.generate_views(
                image,
                num_views=preset.multiview_count,
                num_inference_steps=preset.multiview_steps,
                progress_callback=lambda p, m: progress_callback(p * 0.3, m) if progress_callback else None,
                **kwargs
            )
            
            # Store intermediate data
            self.intermediate_data.add(IntermediateFormat.MULTIVIEW_IMAGES, views)
            
            # Optional: Generate depth maps
            if preset.use_depth_refinement and "depth" in self.intermediate_processors:
                if progress_callback:
                    progress_callback(0.3, "Estimating depth maps...")
                    
                depth_estimator = self.intermediate_processors["depth"]
                depths = depth_estimator.estimate_multiview_depth(views)
                self.intermediate_data.add(IntermediateFormat.DEPTH_MAP, depths)
                
            # Stage 2: 3D reconstruction
            if progress_callback:
                progress_callback(0.4, "Reconstructing 3D mesh...")
                
            mesh = self.reconstruction_model.reconstruct(
                views,
                resolution=preset.reconstruction_resolution,
                use_depth=preset.use_depth_refinement,
                use_normals=preset.use_normal_maps,
                progress_callback=lambda p, m: progress_callback(0.4 + p * 0.3, m) if progress_callback else None
            )
            
            # Stage 3: Texture generation
            if progress_callback:
                progress_callback(0.7, "Generating textures...")
                
            textures = self.texture_model.generate_texture(
                mesh,
                views,
                resolution=preset.texture_resolution,
                use_pbr=preset.use_pbr,
                progress_callback=lambda p, m: progress_callback(0.7 + p * 0.3, m) if progress_callback else None
            )
            
            # Apply textures to mesh
            if "diffuse" in textures:
                material = trimesh.visual.material.SimpleMaterial(
                    image=textures["diffuse"]
                )
                mesh.visual = trimesh.visual.TextureVisuals(
                    material=material
                )
                
            # Export to requested format
            if progress_callback:
                progress_callback(0.95, f"Exporting to {output_format}...")
                
            output_path = self._export_mesh(mesh, output_format, textures)
            
            # Clear intermediate data if memory efficient mode
            if preset.memory_efficient:
                self.clear_intermediate_data()
                
            if progress_callback:
                progress_callback(1.0, "3D generation complete!")
                
            return {
                "mesh": mesh,
                "textures": textures,
                "output_path": output_path,
                "views": views,
                "preset": preset.name,
                "format": output_format
            }
            
        except Exception as e:
            logger.error(f"3D generation failed: {e}")
            raise
            
    def _estimate_memory_requirement(self, preset: QualityPreset3D) -> float:
        """Estimate memory requirement for generation"""
        
        # Base requirements
        base = 4.0  # 4GB base
        
        # Add for resolution
        if preset.reconstruction_resolution >= 512:
            base += 2.0
        if preset.reconstruction_resolution >= 1024:
            base += 4.0
            
        # Add for texture resolution
        if preset.texture_resolution >= 2048:
            base += 1.0
        if preset.texture_resolution >= 4096:
            base += 2.0
            
        # Add for features
        if preset.use_pbr:
            base += 1.0
        if preset.use_depth_refinement:
            base += 0.5
            
        return base
        
    def _export_mesh(
        self,
        mesh: trimesh.Trimesh,
        format: str,
        textures: Dict[str, Image.Image]
    ) -> Path:
        """Export mesh to file"""
        
        output_dir = Path("outputs/3d")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        import time
        timestamp = int(time.time())
        filename = f"hunyuan3d_{timestamp}.{format}"
        output_path = output_dir / filename
        
        # Export based on format
        if format == "glb":
            mesh.export(str(output_path))
        elif format == "obj":
            mesh.export(str(output_path))
            # Also save textures
            for tex_type, tex_img in textures.items():
                tex_path = output_path.with_suffix(f".{tex_type}.png")
                tex_img.save(str(tex_path))
        else:
            # Other formats
            mesh.export(str(output_path))
            
        return output_path
        
    def supports_quantization(self) -> bool:
        """Check if pipeline supports GGUF quantization"""
        return self.config.use_gguf


class HunYuan3DModel:
    """High-level interface for HunYuan3D"""
    
    def __init__(
        self,
        variant: str = "hunyuan3d-21",
        use_gguf: bool = True,
        quantization: Optional[str] = "Q8_0",
        device: str = "cuda"
    ):
        self.config = HunYuan3DConfig(
            model_variant=variant,
            use_gguf=use_gguf,
            gguf_quantization=quantization,
            device=device
        )
        
        # Create memory manager
        cache_dir = Path("cache/hunyuan3d")
        self.memory_manager = ThreeDMemoryManager(cache_dir)
        
        # Create pipeline
        self.pipeline = HunYuan3DPipeline(self.config, self.memory_manager)
        
    def load(self, components: List[str] = None, progress_callback=None) -> bool:
        """Load model components"""
        return self.pipeline.load_models(components, progress_callback)
        
    def generate(
        self,
        image: Union[Image.Image, str],
        quality: str = "standard",
        output_format: str = "glb",
        **kwargs
    ) -> Dict[str, Any]:
        """Generate 3D model from image"""
        
        # Load image if path provided
        if isinstance(image, str):
            image = Image.open(image)
            
        return self.pipeline.generate(
            image,
            quality_preset=quality,
            output_format=output_format,
            **kwargs
        )
        
    def unload(self):
        """Unload all models"""
        if self.pipeline.multiview_model:
            self.pipeline.multiview_model.unload()
        if self.pipeline.reconstruction_model:
            self.pipeline.reconstruction_model.unload()
        if self.pipeline.texture_model:
            self.pipeline.texture_model.unload()