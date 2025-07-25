"""Texture generation component for HunYuan3D."""

import os
import torch
import numpy as np
import trimesh
import logging
from typing import Optional, Dict, Any, Tuple, Union, List
from pathlib import Path
from PIL import Image
from torchvision import transforms

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
        
        # DINO v2 model for texture generation
        self.dino_v2 = None
        self.dino_ckpt_path = "facebook/dinov2-giant"
        
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
            
            # Note: torch.load security check is disabled globally in __init__.py 
            # This allows HunYuan3D models to load properly without CVE-2025-32434 errors
            
            # Load pipeline exactly like the official implementation
            self.pipeline = DiffusionPipeline.from_pretrained(
                model_load_path,
                custom_pipeline=custom_pipeline,
                torch_dtype=torch.float16,  # Use float16 like official implementation
                device_map=None  # Don't use automatic device mapping
            )
            
            logger.info("Pipeline loading completed, configuring components...")
            
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
            
            # Move to device - ensure all components are moved
            logger.info(f"Moving pipeline to device: {self.device}")
            self.pipeline = self.pipeline.to(self.device)
            
            # Double-check all components are on the correct device
            self._ensure_pipeline_on_device()
            
            # Validate that all pipeline components are properly loaded
            logger.info("Validating pipeline components are fully loaded...")
            
            # Check that essential components are loaded
            essential_components = ['unet', 'scheduler']
            for component_name in essential_components:
                if hasattr(self.pipeline, component_name):
                    component = getattr(self.pipeline, component_name)
                    if component is None:
                        logger.warning(f"Component {component_name} is None after loading")
                    else:
                        logger.debug(f"Component {component_name} loaded successfully: {type(component)}")
                else:
                    logger.warning(f"Component {component_name} not found in pipeline")
            
            # Additional validation for text/image encoders if they exist
            optional_components = ['text_encoder', 'image_encoder', 'vae']
            for component_name in optional_components:
                if hasattr(self.pipeline, component_name):
                    component = getattr(self.pipeline, component_name)
                    if component is not None:
                        logger.debug(f"Optional component {component_name} loaded: {type(component)}")
            
            logger.info("Pipeline setup and validation completed successfully")
            
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
    
    def _initialize_dino_v2(self):
        """Initialize DINO v2 model for feature extraction."""
        if self.dino_v2 is not None:
            return  # Already initialized
        
        try:
            # Import HunYuan3D's DINO v2 module
            import sys
            from pathlib import Path
            
            # Add HunYuan3D path to sys.path temporarily
            hunyuan3d_path = Path(__file__).parent.parent.parent.parent.parent / "Hunyuan3D"
            hy3dpaint_path = hunyuan3d_path / "hy3dpaint"
            
            if hy3dpaint_path.exists() and str(hy3dpaint_path) not in sys.path:
                sys.path.insert(0, str(hy3dpaint_path))
            
            # Import Dino_v2 from HunYuan3D
            from hunyuanpaintpbr.unet.modules import Dino_v2
            
            logger.info(f"Initializing DINO v2 model from: {self.dino_ckpt_path}")
            self.dino_v2 = Dino_v2(self.dino_ckpt_path).to(torch.float16)
            self.dino_v2 = self.dino_v2.to(self.device)
            self.dino_v2.eval()
            
            logger.info("DINO v2 model initialized successfully")
            
        except ImportError as e:
            logger.error(f"Failed to import DINO v2 module: {e}")
            logger.warning("Texture generation may fail without DINO features")
        except Exception as e:
            logger.error(f"Failed to initialize DINO v2: {e}")
            logger.warning("Continuing without DINO v2 features")
    
    def _update_memory_usage(self):
        """Update memory usage tracking."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self._memory_usage = torch.cuda.memory_allocated() / 1024**3
    
    def generate_texture(
        self,
        mesh: trimesh.Trimesh,
        prompt: str,
        input_image: Optional[Image.Image] = None,
        resolution: Optional[int] = None,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        seed: Optional[int] = None,
        generate_pbr: bool = True,
        **kwargs
    ) -> Dict[str, Union[np.ndarray, Image.Image]]:
        """Generate texture for mesh.
        
        Args:
            mesh: Input mesh for UV reference
            prompt: Text description for texture
            input_image: Original input image used for 3D generation (required)
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
                try:
                    # Validate mesh before UV generation
                    if not hasattr(mesh, 'vertices') or not hasattr(mesh, 'faces'):
                        raise ValueError("Mesh is missing vertices or faces")
                    
                    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
                        raise ValueError(f"Mesh has no geometry: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
                    
                    logger.info(f"Mesh validation passed: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
                    mesh = self._generate_uvs(mesh)
                    logger.info("UV coordinate generation completed successfully")
                    
                except Exception as e:
                    logger.error(f"UV generation failed: {e}")
                    logger.warning("Continuing without UV coordinates - texture generation may fail")
                    # Create a simple UV visual if none exists
                    if not hasattr(mesh, 'visual'):
                        mesh.visual = trimesh.visual.ColorVisuals()
            
            # Set random seed
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)
            else:
                generator = None
            
            with optimize_memory_usage():
                # Generate texture
                with torch.no_grad():
                    # HunYuan3D texture pipeline expects an image as primary input
                    if input_image is None:
                        raise ValueError("HunYuan3D texture generation requires an input image")
                    
                    logger.info(f"Generating texture from input image: {input_image.size}")
                    logger.info(f"Device: {self.device}, Pipeline device: {next(self.pipeline.unet.parameters()).device}")
                    
                    # Ensure input image is in the correct format
                    # The HunYuan3D pipeline expects PIL images, not tensors
                    if not isinstance(input_image, Image.Image):
                        raise ValueError("Input image must be a PIL Image")
                    
                    # Validate and prepare input image
                    input_image = self._validate_input_image(input_image, resolution)
                    
                    # Prepare mesh conditions for HunYuan3D texture pipeline
                    cached_condition = self._prepare_mesh_conditions(mesh, resolution, input_image)
                    
                    # Ensure all pipeline components are on the same device
                    self._ensure_pipeline_on_device()
                    
                    texture_outputs = self.pipeline(
                        images=[input_image],  # HunYuan3D expects 'images' as list, not 'image'
                        prompt=prompt,
                        height=resolution,
                        width=resolution,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        generator=generator,
                        output_type="pil",
                        **cached_condition,  # Pass mesh-rendered conditions
                        **kwargs
                    )
                
                # Extract texture maps
                texture_maps = self._extract_texture_maps(
                    texture_outputs,
                    generate_pbr and self.supports_pbr
                )
            
            logger.info(f"Generated {len(texture_maps)} texture maps")
            return texture_maps
            
        except RuntimeError as e:
            if "Expected all tensors to be on the same device" in str(e):
                logger.error(f"Device mismatch error: {e}")
                logger.error("This usually indicates a problem with the pipeline configuration")
                logger.error(f"Current device: {self.device}")
                if hasattr(self, 'pipeline') and self.pipeline:
                    if hasattr(self.pipeline, 'vae'):
                        logger.error(f"VAE device: {next(self.pipeline.vae.parameters()).device}")
                    if hasattr(self.pipeline, 'unet'):
                        logger.error(f"UNet device: {next(self.pipeline.unet.parameters()).device}")
                raise RuntimeError(f"Device mismatch in texture generation pipeline: {str(e)}")
            else:
                logger.error(f"Texture generation failed: {e}")
                raise RuntimeError(f"Texture generation failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in texture generation: {e}")
            raise RuntimeError(f"Texture generation failed: {str(e)}")
    
    def _generate_uvs(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Generate UV coordinates for mesh."""
        try:
            # Try to use xatlas for UV unwrapping
            import xatlas
            logger.info("Using xatlas for UV unwrapping")
            
            # Validate mesh data for xatlas
            if not isinstance(mesh.vertices, np.ndarray) or not isinstance(mesh.faces, np.ndarray):
                raise ValueError("Mesh vertices and faces must be numpy arrays")
            
            # Check mesh complexity - if too complex, skip xatlas to avoid hanging
            num_vertices = len(mesh.vertices)
            num_faces = len(mesh.faces)
            
            if num_vertices > 50000 or num_faces > 100000:
                logger.warning(f"Mesh too complex for xatlas ({num_vertices} vertices, {num_faces} faces), using spherical projection")
                return self._spherical_uv_projection(mesh)
            
            # Pack UV atlas with timeout protection
            logger.debug(f"Calling xatlas.parametrize with vertices shape: {mesh.vertices.shape}, faces shape: {mesh.faces.shape}")
            
            # Use threading to implement a timeout for xatlas
            import threading
            import queue
            import time
            
            result_queue = queue.Queue()
            exception_queue = queue.Queue()
            
            def xatlas_worker():
                try:
                    logger.info("Starting xatlas UV parametrization...")
                    start_time = time.time()
                    
                    vmapping, indices, uvs = xatlas.parametrize(
                        mesh.vertices.astype(np.float32),  # Ensure float32 for xatlas
                        mesh.faces.astype(np.uint32)       # Ensure uint32 for xatlas
                    )
                    
                    elapsed_time = time.time() - start_time
                    logger.info(f"xatlas completed in {elapsed_time:.2f} seconds")
                    
                    result_queue.put((vmapping, indices, uvs))
                except Exception as e:
                    exception_queue.put(e)
            
            # Start worker thread
            worker_thread = threading.Thread(target=xatlas_worker)
            worker_thread.daemon = True
            worker_thread.start()
            
            # Wait for result with timeout (30 seconds)
            timeout_seconds = 30
            worker_thread.join(timeout=timeout_seconds)
            
            if worker_thread.is_alive():
                logger.error(f"xatlas timed out after {timeout_seconds} seconds, falling back to spherical projection")
                return self._spherical_uv_projection(mesh)
            
            # Check for exceptions
            if not exception_queue.empty():
                raise exception_queue.get()
            
            # Get result
            if result_queue.empty():
                raise RuntimeError("xatlas worker finished but no result available")
            
            vmapping, indices, uvs = result_queue.get()
            
            logger.info(f"xatlas UV generation successful: {len(vmapping)} vertices mapped, {len(uvs)} UV coordinates")
            
            # Create new mesh with UVs
            mesh_with_uvs = trimesh.Trimesh(
                vertices=mesh.vertices[vmapping],
                faces=indices,
                visual=trimesh.visual.TextureVisuals(uv=uvs)
            )
            
            return mesh_with_uvs
            
        except ImportError:
            logger.warning("xatlas library not available, using spherical UV projection fallback")
            return self._spherical_uv_projection(mesh)
        except Exception as e:
            logger.error(f"xatlas UV generation failed: {e}")
            logger.warning("Falling back to spherical UV projection")
            return self._spherical_uv_projection(mesh)
    
    def _spherical_uv_projection(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Apply spherical UV projection."""
        try:
            logger.info("Applying spherical UV projection fallback")
            vertices = mesh.vertices
            
            if vertices.shape[0] == 0:
                raise ValueError("No vertices to project")
            
            # Center vertices
            center = vertices.mean(axis=0)
            centered = vertices - center
            
            # Convert to spherical coordinates
            r = np.linalg.norm(centered, axis=1)
            
            # Handle case where r is zero (all points at center)
            r = np.maximum(r, 1e-8)
            
            theta = np.arctan2(centered[:, 1], centered[:, 0])
            phi = np.arccos(np.clip(centered[:, 2] / r, -1, 1))
            
            # Map to UV coordinates
            u = (theta + np.pi) / (2 * np.pi)
            v = phi / np.pi
            
            # Ensure UV coordinates are valid
            u = np.clip(u, 0.0, 1.0)
            v = np.clip(v, 0.0, 1.0)
            
            # Create UV coordinates
            uv = np.stack([u, v], axis=1)
            
            logger.info(f"Spherical UV projection successful: {len(uv)} UV coordinates generated")
            
            # Create visual with UVs
            mesh.visual = trimesh.visual.TextureVisuals(uv=uv)
            
            return mesh
            
        except Exception as e:
            logger.error(f"Spherical UV projection failed: {e}")
            logger.warning("Creating mesh without UV coordinates")
            # Return mesh with basic visual
            if not hasattr(mesh, 'visual'):
                mesh.visual = trimesh.visual.ColorVisuals()
            return mesh
    
    def _ensure_pipeline_on_device(self):
        """Ensure all pipeline components are on the correct device."""
        try:
            if self.pipeline is None:
                return
            
            # Check if pipeline needs to be moved to device
            if hasattr(self.pipeline, 'vae') and self.pipeline.vae is not None:
                # Check VAE device
                vae_device = next(self.pipeline.vae.parameters()).device
                if vae_device != self.device:
                    logger.info(f"Moving VAE from {vae_device} to {self.device}")
                    self.pipeline.vae = self.pipeline.vae.to(self.device)
            
            if hasattr(self.pipeline, 'unet') and self.pipeline.unet is not None:
                # Check UNet device
                unet_device = next(self.pipeline.unet.parameters()).device
                if unet_device != self.device:
                    logger.info(f"Moving UNet from {unet_device} to {self.device}")
                    self.pipeline.unet = self.pipeline.unet.to(self.device)
            
            if hasattr(self.pipeline, 'text_encoder') and self.pipeline.text_encoder is not None:
                # Check text encoder device
                text_encoder_device = next(self.pipeline.text_encoder.parameters()).device
                if text_encoder_device != self.device:
                    logger.info(f"Moving text encoder from {text_encoder_device} to {self.device}")
                    self.pipeline.text_encoder = self.pipeline.text_encoder.to(self.device)
                    
            # The pipeline itself may not have a settable device attribute
            # This is normal - the components are what matter
                
            logger.debug("All pipeline components verified to be on the correct device")
            
        except Exception as e:
            logger.error(f"Failed to ensure pipeline on device: {e}")
            # Continue anyway - the pipeline might still work
    
    def _validate_input_image(self, image: Image.Image, resolution: int) -> Image.Image:
        """Validate and prepare input image for texture generation.
        
        Args:
            image: Input PIL image
            resolution: Target resolution
            
        Returns:
            Validated and prepared PIL image
        """
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                logger.info(f"Converting image from {image.mode} to RGB")
                image = image.convert('RGB')
            
            # Resize if needed
            if image.size != (resolution, resolution):
                logger.info(f"Resizing image from {image.size} to ({resolution}, {resolution})")
                image = image.resize((resolution, resolution), Image.Resampling.LANCZOS)
            
            # Validate image data
            img_array = np.array(image)
            if img_array.shape != (resolution, resolution, 3):
                raise ValueError(f"Invalid image shape after processing: {img_array.shape}")
            
            # Check for valid pixel values
            if img_array.min() < 0 or img_array.max() > 255:
                logger.warning(f"Image has invalid pixel values: min={img_array.min()}, max={img_array.max()}")
                img_array = np.clip(img_array, 0, 255)
                image = Image.fromarray(img_array.astype(np.uint8))
            
            logger.debug(f"Input image validated: {image.size}, {image.mode}")
            return image
            
        except Exception as e:
            logger.error(f"Failed to validate input image: {e}")
            raise ValueError(f"Invalid input image: {str(e)}")
    
    def _prepare_mesh_conditions(self, mesh: trimesh.Trimesh, resolution: int, input_image: Image.Image) -> Dict[str, Any]:
        """Prepare mesh conditions (normal/position maps) for HunYuan3D texture pipeline.
        
        Args:
            mesh: Input mesh
            resolution: Resolution for rendered maps
            input_image: Original input image for DINO feature extraction
            
        Returns:
            Dictionary containing cached_condition data for the pipeline
        """
        try:
            logger.info("Preparing mesh conditions for texture generation...")
            
            # For now, create simple placeholder conditions
            # TODO: Implement proper mesh rendering for normal/position maps
            cached_condition = {}
            
            # Create simple normal and position maps as placeholders
            # In a full implementation, these would be rendered from the mesh
            
            # Create a simple normal map (pointing up)
            normal_map = Image.new('RGB', (resolution, resolution), (128, 128, 255))  # Normal pointing up
            cached_condition["images_normal"] = [[normal_map]]  # List of lists format expected
            
            # Create a simple position map (gradient)
            position_map = Image.new('RGB', (resolution, resolution), (0, 0, 0))
            # Add some gradient to simulate depth
            import numpy as np
            pos_array = np.zeros((resolution, resolution, 3), dtype=np.uint8)
            for i in range(resolution):
                for j in range(resolution):
                    # Simple depth gradient
                    depth = int((i + j) / (2 * resolution) * 255)
                    pos_array[i, j] = [depth, depth, depth]
            
            position_map = Image.fromarray(pos_array)
            cached_condition["images_position"] = [[position_map]]  # List of lists format expected
            
            # Check if pipeline uses DINO and add hidden states
            if hasattr(self.pipeline, 'unet') and hasattr(self.pipeline.unet, 'use_dino') and self.pipeline.unet.use_dino:
                logger.info("Pipeline uses DINO, extracting features from input image...")
                
                # Initialize DINO v2 if not already done
                self._initialize_dino_v2()
                
                if self.dino_v2 is not None:
                    try:
                        # Process input image through DINO v2
                        with torch.no_grad():
                            dino_hidden_states = self.dino_v2(input_image)
                            cached_condition["dino_hidden_states"] = dino_hidden_states
                            logger.info(f"Added DINO hidden states with shape: {dino_hidden_states.shape}")
                    except Exception as e:
                        logger.error(f"Failed to extract DINO features: {e}")
                        logger.warning("Continuing without DINO features")
                else:
                    logger.warning("DINO v2 not initialized, continuing without DINO features")
            
            logger.info("Mesh conditions prepared")
            return cached_condition
            
        except Exception as e:
            logger.error(f"Failed to prepare mesh conditions: {e}")
            logger.warning("Using empty conditions - texture generation may fail")
            return {}
    
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
        
        if self.dino_v2 is not None:
            del self.dino_v2
            self.dino_v2 = None
            
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
            base_memory = 3.0  # Texture models are typically smaller
            # Add DINO v2 memory if loaded (approximately 2.5GB for DINOv2-giant)
            if self.dino_v2 is not None:
                base_memory += 2.5
            self._memory_usage = base_memory