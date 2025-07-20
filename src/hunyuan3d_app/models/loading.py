"""Model loading functionality."""

import gc
import logging
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Union

import torch
from diffusers import (
    StableDiffusionXLPipeline,
    DiffusionPipeline,
    AutoPipelineForText2Image,
    FluxPipeline,
    FluxTransformer2DModel,
    GGUFQuantizationConfig
)

from ..utils.memory import get_memory_manager
from ..config import ALL_IMAGE_MODELS, GGUF_IMAGE_MODELS

logger = logging.getLogger(__name__)


class ModelLoader:
    """Handles loading and unloading of AI models."""
    
    def __init__(self, device: str = "cuda", gguf_manager=None):
        self.device = device
        self.memory_manager = get_memory_manager()
        
        # Currently loaded models
        self.loaded_models = {}
        self.model_configs = {}
        
        # GGUF manager for standalone GGUF loading
        self.gguf_manager = gguf_manager
    
    def load_image_model(
        self,
        model_name: str,
        model_path: Path,
        progress_callback: Optional[Any] = None,
        **kwargs
    ) -> Tuple[Any, str]:
        """Load an image generation model.
        
        Args:
            model_name: Name of the model
            model_path: Path to the model files
            progress_callback: Optional progress callback
            **kwargs: Additional loading arguments
            
        Returns:
            Tuple of (model_pipeline, message)
        """
        try:
            # Unload any existing image model
            if "image" in self.loaded_models:
                self.unload_model("image")
            
            # Update progress
            if progress_callback:
                progress_callback(0.1, f"Preparing to load {model_name}...")
            
            # Get model configuration
            model_config = ALL_IMAGE_MODELS.get(model_name, {})
            is_gguf = model_name in GGUF_IMAGE_MODELS
            
            # Clean memory before loading
            self.memory_manager.cleanup_memory()
            
            # Load based on model type
            if is_gguf:
                pipeline = self._load_gguf_model(model_name, model_path, progress_callback)
            else:
                pipeline = self._load_standard_model(model_name, model_path, model_config, progress_callback)
            
            # Validate pipeline before storing
            if not self._validate_pipeline(pipeline, model_name):
                error_msg = f"Pipeline validation failed for {model_name} - not a real AI model"
                logger.error(error_msg)
                return None, error_msg
            
            # Store loaded model
            self.loaded_models["image"] = pipeline
            self.model_configs["image"] = {
                "name": model_name,
                "path": str(model_path),
                "is_gguf": is_gguf,
                "config": model_config
            }
            
            # Final progress
            if progress_callback:
                progress_callback(1.0, f"Successfully loaded {model_name}")
            
            return pipeline, f"Successfully loaded {model_name}"
            
        except Exception as e:
            error_msg = f"Error loading {model_name}: {str(e)}"
            logger.error(error_msg)
            return None, error_msg
    
    def _load_standard_model(
        self,
        model_name: str,
        model_path: Path,
        model_config: Dict[str, Any],
        progress_callback: Optional[Any]
    ) -> Any:
        """Load a standard diffusion model."""
        logger.info(f"Loading standard model {model_name} from {model_path}")
        
        # Determine pipeline class
        pipeline_class = DiffusionPipeline
        if "flux" in model_name.lower():
            pipeline_class = FluxPipeline
        elif "sdxl" in model_name.lower():
            pipeline_class = StableDiffusionXLPipeline
        
        # Update progress
        if progress_callback:
            progress_callback(0.3, "Loading model components...")
        
        # Load with appropriate settings
        load_kwargs = {
            "pretrained_model_name_or_path": str(model_path),
            "torch_dtype": torch.float16,
            "use_safetensors": True,
            "local_files_only": True
        }
        
        # Get HF token for gated models
        from ..utils import get_hf_token_from_all_sources, validate_hf_token
        
        hf_token = get_hf_token_from_all_sources()
        if hf_token and validate_hf_token(hf_token) and "flux" in model_name.lower():
            load_kwargs["token"] = hf_token
            logger.info("Using HF token for FLUX model access")
        elif "flux" in model_name.lower() and "dev" in model_name.lower():
            logger.warning("FLUX.1-dev requires authentication - ensure HF token is set")
        
        # Only add variant if fp16 files exist
        if (model_path / "transformer" / "diffusion_pytorch_model.fp16.safetensors").exists():
            load_kwargs["variant"] = "fp16"
        
        # Add device map for large models
        size = getattr(model_config, 'size', "0 GB")
        if isinstance(size, str):
            # Handle size strings like "~24 GB"
            import re
            size_match = re.search(r'(\d+)', size)
            size_gb = int(size_match.group(1)) if size_match else 0
        else:
            size_gb = size
        
        # For FLUX models, use special handling due to their size
        if "flux" in model_name.lower() and size_gb > 6:
            # Don't use device_map for FLUX - causes device mismatch issues
            # But we'll enable CPU offload after loading
            logger.info(f"Large FLUX model ({size_gb}GB) - will use CPU offload instead of device_map")
            load_kwargs["low_cpu_mem_usage"] = True  # Help with loading large models
        elif size_gb > 6:
            load_kwargs["device_map"] = "balanced"
            logger.info(f"Using device_map='balanced' for large model ({size_gb}GB)")
        
        # Load the pipeline
        pipeline = pipeline_class.from_pretrained(**load_kwargs)
        
        # Update progress
        if progress_callback:
            progress_callback(0.7, "Optimizing model...")
        
        # Optimize based on available VRAM
        vram_gb = 0
        if torch.cuda.is_available():
            vram_gb = torch.cuda.mem_get_info()[0] / (1024**3)
        
        if vram_gb < 8:
            logger.info("Enabling CPU offload due to limited VRAM")
            try:
                # Check if device_map was used
                if hasattr(pipeline, "hf_device_map") and pipeline.hf_device_map:
                    logger.info("Pipeline has device_map, skipping CPU offload")
                    # Device map already handles memory management
                elif hasattr(pipeline, "enable_model_cpu_offload"):
                    pipeline.enable_model_cpu_offload()
                elif hasattr(pipeline, "to"):
                    pipeline = pipeline.to(self.device)
            except Exception as e:
                logger.warning(f"Could not enable CPU offload: {e}")
                # Fallback to moving to device
                if hasattr(pipeline, "to"):
                    pipeline = pipeline.to(self.device)
        elif hasattr(pipeline, "to"):
            pipeline = pipeline.to(self.device)
        
        # Enable memory efficient attention
        if hasattr(pipeline, "enable_attention_slicing"):
            pipeline.enable_attention_slicing()
        
        if hasattr(pipeline, "enable_xformers_memory_efficient_attention"):
            try:
                pipeline.enable_xformers_memory_efficient_attention()
                logger.info("Enabled xformers memory efficient attention")
            except Exception as e:
                logger.warning(f"Could not enable xformers: {e}")
        
        return pipeline
    
    def _load_gguf_model(
        self,
        model_name: str,
        model_path: Path,
        progress_callback: Optional[Any]
    ) -> Any:
        """Load a GGUF quantized model."""
        logger.info(f"Loading GGUF model {model_name} from {model_path}")
        
        if progress_callback:
            progress_callback(0.1, "Preparing GGUF model loading...")
        
        try:
            # Use the GGUF manager to load the model
            from ..models.gguf import GGUFModelManager
            
            # Find the GGUF file
            gguf_files = list(model_path.glob("*.gguf"))
            if not gguf_files:
                raise FileNotFoundError(f"No GGUF files found in {model_path}")
            
            gguf_file = gguf_files[0]  # Use the first GGUF file found
            logger.info(f"Using GGUF file: {gguf_file}")
            
            if progress_callback:
                progress_callback(0.3, "Loading GGUF transformer...")
            
            # Check if we have local components first before trying to use diffusers GGUF support
            # Look for other required components locally
            flux_base_dir = model_path.parent.parent / "flux_base"
            local_components_available = False
            
            # Check for local FLUX model components
            possible_flux_paths = [
                flux_base_dir / "black-forest-labs--FLUX.1-dev",
                flux_base_dir / "models--black-forest-labs--FLUX.1-dev",
            ]
            
            flux_model_path = None
            for path in possible_flux_paths:
                if path.exists():
                    # Check if it has the required components
                    snapshots = path / "snapshots"
                    if snapshots.exists():
                        snapshot_dirs = list(snapshots.iterdir())
                        if snapshot_dirs:
                            snapshot = snapshot_dirs[0]
                            # Check for essential components
                            if ((snapshot / "text_encoder").exists() and 
                                (snapshot / "text_encoder_2").exists() and 
                                (snapshot / "vae").exists()):
                                flux_model_path = snapshot
                                local_components_available = True
                                logger.info(f"Found local FLUX components at {flux_model_path}")
                                break
            
            # Skip trying to load full FLUX pipeline - we want to use GGUF
            # The original code was loading the full model which ignores the GGUF file
            
            # Load GGUF model using the simplified wrapper
            logger.info("Loading GGUF model...")
            
            try:
                from ..models.gguf_wrapper import load_standalone_gguf
                
                if progress_callback:
                    progress_callback(0.6, "Loading GGUF model with official API...")
                
                pipeline, status = load_standalone_gguf(
                    gguf_path=gguf_file,
                    model_name=model_name,
                    device=self.device,
                    progress_callback=lambda p, msg: progress_callback(0.6 + p * 0.3, msg) if progress_callback else None
                )
                
                if pipeline:
                    logger.info(f"GGUF model loaded successfully: {status}")
                    # Mark as GGUF model
                    pipeline._is_gguf_model = True
                    return pipeline
                else:
                    logger.warning(f"GGUF loading failed: {status}")
                    
            except Exception as wrapper_error:
                logger.error(f"GGUF loading error: {wrapper_error}")
                import traceback
                traceback.print_exc()
            
            # If all GGUF loading attempts fail, raise the error to trigger mock fallback
            raise NotImplementedError("Could not load GGUF model with available methods")
            
        except Exception as e:
            logger.error(f"Failed to load GGUF model {model_name}: {e}")
            gguf_error = str(e)  # Store the error for later use
            
            # Try to load regular FLUX model as fallback
            try:
                if progress_callback:
                    progress_callback(0.8, "Attempting to load standard FLUX model...")
                
                # Determine model ID based on model name
                model_id = "black-forest-labs/FLUX.1-dev"  # Default
                if "schnell" in model_name.lower():
                    model_id = "black-forest-labs/FLUX.1-schnell"
                
                # Check if regular model exists locally
                from pathlib import Path
                
                # Try multiple possible paths for the model
                possible_model_paths = [
                    model_path.parent.parent / "flux_base" / model_id.replace('/', '--'),
                    model_path.parent.parent / "flux_base" / f"models--{model_id.replace('/', '--')}",
                    model_path.parent.parent / "image" / model_id.split('/')[-1],
                ]
                
                regular_model_path = None
                for path in possible_model_paths:
                    if path.exists():
                        regular_model_path = path
                        logger.info(f"Found regular FLUX model at {regular_model_path}")
                        break
                
                if not regular_model_path:
                    logger.warning(f"Could not find regular FLUX model in any expected location")
                
                from diffusers import FluxPipeline
                
                load_kwargs = {
                    "torch_dtype": torch.float16,
                    "use_safetensors": True,
                }
                # Don't specify variant - let it use whatever is available
                
                if regular_model_path and regular_model_path.exists():
                    try:
                        # Check if it's a HuggingFace cache structure
                        snapshots_dir = regular_model_path / "snapshots"
                        if snapshots_dir.exists():
                            # Find the latest snapshot
                            snapshots = list(snapshots_dir.iterdir())
                            if snapshots:
                                snapshot_path = snapshots[-1]  # Use latest snapshot
                                logger.info(f"Loading from snapshot: {snapshot_path}")
                                pipeline = FluxPipeline.from_pretrained(
                                    str(snapshot_path),
                                    **load_kwargs
                                )
                                logger.info("Loaded FLUX model from HuggingFace cache snapshot")
                            else:
                                raise ValueError("No snapshots found in cache")
                        else:
                            # Direct model directory
                            pipeline = FluxPipeline.from_pretrained(
                                str(regular_model_path),
                                **load_kwargs
                            )
                            logger.info("Loaded FLUX model from local directory")
                    except Exception as e:
                        logger.warning(f"Failed to load from {regular_model_path}: {e}")
                        # Try with model ID and local files
                        pipeline = FluxPipeline.from_pretrained(
                            model_id,
                            local_files_only=True,
                            cache_dir=str(model_path.parent.parent / "flux_base"),
                            **load_kwargs
                        )
                else:
                    # Download if needed
                    logger.info("Downloading FLUX model components...")
                    # For downloads, always pass token if available
                    if hf_token:
                        load_kwargs["token"] = hf_token
                    pipeline = FluxPipeline.from_pretrained(
                        model_id,
                        **load_kwargs
                    )
                
                # Optimize pipeline
                if hasattr(pipeline, "enable_attention_slicing"):
                    pipeline.enable_attention_slicing()
                
                if torch.cuda.is_available():
                    if hasattr(pipeline, "enable_model_cpu_offload"):
                        pipeline.enable_model_cpu_offload()
                    else:
                        pipeline = pipeline.to("cuda")
                
                if progress_callback:
                    progress_callback(1.0, f"Loaded {model_name} (standard version)")
                
                logger.info(f"Successfully loaded standard FLUX model as fallback for {model_name}")
                return pipeline
                
            except Exception as fallback_error:
                logger.error(f"Failed to load standard FLUX model: {fallback_error}")
                
            # No mock fallback - fail with clear error
            error_msg = f"Failed to load GGUF model {model_name}: {gguf_error}\nAlso failed to load standard FLUX model: {fallback_error}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def load_hunyuan3d_model(
        self,
        model_name: str,
        model_path: Path,
        device: str = "cuda",
        progress_callback: Optional[Any] = None
    ) -> Tuple[Any, str]:
        """Load a Hunyuan3D model.
        
        Args:
            model_name: Name of the model
            model_path: Path to model directory
            device: Device to load on
            progress_callback: Optional progress callback
            
        Returns:
            Tuple of (model, message)
        """
        try:
            logger.info(f"Loading Hunyuan3D model {model_name} from {model_path}")
            
            if progress_callback:
                progress_callback(0.1, "Preparing Hunyuan3D model loading...")
            
            # Look for actual model files
            # Check for snapshot directory structure
            snapshots_dirs = list(model_path.glob("**/snapshots/*/"))
            actual_model_path = None
            
            if snapshots_dirs:
                # Use the latest snapshot
                actual_model_path = snapshots_dirs[-1]
            else:
                # Direct path
                actual_model_path = model_path
            
            logger.info(f"Looking for model files in {actual_model_path}")
            
            # Check what type of Hunyuan3D model this is
            if "21" in model_name or "2.1" in model_name:
                model_version = "2.1"
            elif "20" in model_name or "2.0" in model_name:
                model_version = "2.0"
            else:
                model_version = "2.1"  # Default
            
            if progress_callback:
                progress_callback(0.3, f"Loading Hunyuan3D {model_version} components...")
            
            # Try to load the actual Hunyuan3D model
            try:
                # First, check if we have the model files
                dit_path = actual_model_path / f"hunyuan3d-dit-v{model_version.replace('.', '-')}"
                if not dit_path.exists():
                    # Try alternative naming
                    dit_path = actual_model_path / "dit"
                    if not dit_path.exists():
                        # Check if the files exist at all
                        logger.info(f"Looking for DIT model in: {actual_model_path}")
                        logger.info(f"Contents: {list(actual_model_path.iterdir())}")
                        raise FileNotFoundError(f"Could not find Hunyuan3D DIT model files in {actual_model_path}")
                
                # Use the proper Hunyuan3D wrapper
                from .hunyuan3d_wrapper import load_hunyuan3d_model
                
                if progress_callback:
                    progress_callback(0.4, "Loading Hunyuan3D model with full inference support...")
                
                model, status = load_hunyuan3d_model(
                    actual_model_path, 
                    model_version, 
                    device,
                    lambda p, msg: progress_callback(0.4 + p * 0.5, msg) if progress_callback else None
                )
                
                if model:
                    logger.info(f"Successfully loaded Hunyuan3D model {model_name}: {status}")
                    return model, status
                else:
                    raise RuntimeError(f"Failed to load Hunyuan3D model: {status}")
                
            except Exception as e:
                logger.error(f"Failed to load actual Hunyuan3D model: {e}")
                # Fall back to a mock model for now
                if progress_callback:
                    progress_callback(0.9, "Using Hunyuan3D placeholder...")
                
                # No placeholder - fail properly
                error_msg = f"Failed to load Hunyuan3D model: {e}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
                
        except Exception as e:
            logger.error(f"Failed to load Hunyuan3D model {model_name}: {e}")
            return None, f"Failed to load Hunyuan3D model: {str(e)}"
    
    def unload_model(self, model_type: str) -> Tuple[bool, str]:
        """Unload a model and free memory.
        
        Args:
            model_type: Type of model to unload (e.g., "image", "3d")
            
        Returns:
            Tuple of (success, message)
        """
        try:
            if model_type not in self.loaded_models:
                return True, f"No {model_type} model loaded"
            
            # Get the model
            model = self.loaded_models[model_type]
            
            # Clear from loaded models
            del self.loaded_models[model_type]
            if model_type in self.model_configs:
                del self.model_configs[model_type]
            
            # Delete the model
            del model
            
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return True, f"Successfully unloaded {model_type} model"
            
        except Exception as e:
            error_msg = f"Error unloading {model_type} model: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def get_loaded_models(self) -> Dict[str, Dict[str, Any]]:
        """Get information about loaded models.
        
        Returns:
            Dictionary of loaded model information
        """
        return {
            model_type: {
                "name": config.get("name"),
                "path": config.get("path"),
                "is_gguf": config.get("is_gguf", False),
                "vram_usage_gb": self._get_model_vram_usage(model_type)
            }
            for model_type, config in self.model_configs.items()
        }
    
    def _get_model_vram_usage(self, model_type: str) -> float:
        """Estimate VRAM usage of a loaded model.
        
        Args:
            model_type: Type of model
            
        Returns:
            VRAM usage in GB
        """
        if model_type not in self.loaded_models:
            return 0.0
        
        # This is a rough estimate
        try:
            model = self.loaded_models[model_type]
            if hasattr(model, "device") and model.device.type == "cuda":
                # Get current VRAM usage
                return torch.cuda.memory_allocated(model.device) / (1024**3)
        except:
            pass
        
        return 0.0
    
    def _validate_pipeline(self, pipeline: Any, model_name: str) -> bool:
        """Validate that a pipeline is capable of real AI generation.
        
        Args:
            pipeline: The pipeline to validate
            model_name: Name of the model
            
        Returns:
            True if pipeline is valid for real AI generation
        """
        if pipeline is None:
            logger.error(f"Pipeline is None for {model_name}")
            return False
        
        # Check for mock/dummy indicators
        pipeline_type = type(pipeline).__name__
        if any(word in pipeline_type.lower() for word in ['mock', 'dummy', 'placeholder', 'procedural']):
            logger.error(f"Pipeline {pipeline_type} appears to be a mock/dummy implementation")
            return False
        
        # For GGUF models, check if they have a real pipeline
        if hasattr(pipeline, '_is_gguf_model') and pipeline._is_gguf_model:
            if hasattr(pipeline, '_is_placeholder') and pipeline._is_placeholder:
                logger.error(f"GGUF model {model_name} is in placeholder mode")
                return False
            if hasattr(pipeline, '_real_pipeline') and pipeline._real_pipeline is None:
                logger.error(f"GGUF model {model_name} has no real pipeline")
                return False
        
        # Check for required methods
        if not callable(pipeline):
            logger.error(f"Pipeline for {model_name} is not callable")
            return False
        
        logger.info(f"Pipeline validation passed for {model_name} ({pipeline_type})")
        return True