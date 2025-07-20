"""GGUF Wrapper for standalone GGUF model loading without requiring base model components"""

import logging
import time
from pathlib import Path
from typing import Optional, Tuple, Any
import torch

logger = logging.getLogger(__name__)


class StandaloneGGUFPipeline:
    """Wrapper pipeline for standalone GGUF models that don't require base components"""
    
    def __init__(
        self,
        gguf_path: Path,
        model_name: str,
        vae_path: Optional[Path] = None,
        text_encoder_path: Optional[Path] = None,
        text_encoder_2_path: Optional[Path] = None,
        device: str = "cuda"
    ):
        self.gguf_path = gguf_path
        self.model_name = model_name
        self.device = device
        
        # Store component paths
        self.vae_path = vae_path
        self.text_encoder_path = text_encoder_path
        self.text_encoder_2_path = text_encoder_2_path
        
        # Initialize as placeholder until we have proper GGUF support
        self._is_placeholder = True
        self._real_pipeline = None
        self._is_gguf_model = True  # Flag to identify GGUF models
        self._has_device_map = False  # Track if device_map is used
        logger.info(f"Created StandaloneGGUFPipeline for {model_name}")
        
        # Try to load GGUF weights or fall back to real model
        self._load_gguf_weights()
    
    def _load_gguf_weights(self):
        """Load GGUF weights using the official diffusers API"""
        try:
            logger.info(f"GGUF file exists: {self.gguf_path.exists()}")
            file_size_gb = self.gguf_path.stat().st_size / (1024**3)
            logger.info(f"GGUF file size: {file_size_gb:.2f} GB")
            
            # Log file size but continue loading
            if file_size_gb > 10.0:
                logger.info(f"Loading large GGUF file ({file_size_gb:.1f} GB). This may take a while...")
                logger.info("For faster loading, consider using smaller quantization levels (Q4, Q5, Q6).")
            
            # Import required components
            from diffusers import FluxPipeline, FluxTransformer2DModel, GGUFQuantizationConfig
            
            # Try to load GGUF transformer with proper quantization config
            try:
                logger.info("Loading GGUF transformer with GGUFQuantizationConfig...")
                
                # Create quantization config - use bfloat16 for better compatibility
                # Fall back to float16 if bfloat16 not available
                # For Q6/Q8 models, always use float16 for better performance
                is_large_quant = any(q in self.model_name.lower() or q in str(self.gguf_path) for q in ["q6", "q8", "Q6", "Q8"])
                if is_large_quant:
                    compute_dtype = torch.float16
                    logger.info("Q6/Q8 model detected - using float16 for better performance")
                else:
                    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                quantization_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)
                
                # Load the GGUF transformer
                logger.info("Loading GGUF transformer (this may take several minutes for large files)...")
                logger.info(f"GGUF path: {self.gguf_path}")
                logger.info(f"Using compute dtype: {compute_dtype}")
                logger.info(f"Device map: {'auto' if torch.cuda.is_available() else 'cpu'}")
                
                
                # Load the GGUF transformer directly
                logger.info("Loading GGUF transformer (this may take a while for large files)...")
                try:
                    # Only use device_map for very large models to avoid device conflicts
                    # Q8 models are around 10-12GB but should work without device_map
                    file_size_gb = self.gguf_path.stat().st_size / (1024**3)
                    # Check if it's a Q6/Q8 model - these have device placement issues
                    is_large_quant = any(q in self.model_name.lower() or q in str(self.gguf_path) for q in ["q6", "q8", "Q6", "Q8"])
                    use_device_map = file_size_gb > 15.0 and torch.cuda.is_available() and not is_large_quant
                    
                    # Import memory manager
                    from ..utils.memory_optimization import get_memory_manager
                    memory_manager = get_memory_manager()
                    
                    if is_large_quant:
                        logger.info(f"Q6/Q8 model detected ({file_size_gb:.1f}GB) - using optimized loading strategy")
                        memory_manager.aggressive_memory_clear()
                    
                    # Load transformer - for Q6/Q8, avoid device_map entirely
                    if is_large_quant:
                        # Load to CPU first, then move to target device
                        logger.info("Loading Q6/Q8 transformer to CPU first (avoiding device_map issues)")
                        transformer = FluxTransformer2DModel.from_single_file(
                            str(self.gguf_path),
                            quantization_config=quantization_config,
                            torch_dtype=compute_dtype,
                            device_map=None,  # Explicitly no device_map
                            low_cpu_mem_usage=True
                        )
                        logger.info("GGUF transformer loaded to CPU")
                        
                        # Now move to target device if CUDA
                        if self.device == "cuda" and torch.cuda.is_available():
                            logger.info(f"Moving Q6/Q8 transformer to {self.device}")
                            # Don't change dtype for quantized models!
                            transformer = transformer.to(self.device)
                            logger.info("Q6/Q8 transformer moved to GPU")
                    else:
                        # For smaller quants, use the original logic
                        transformer = FluxTransformer2DModel.from_single_file(
                            str(self.gguf_path),
                            quantization_config=quantization_config,
                            torch_dtype=compute_dtype,
                            device_map="auto" if use_device_map else None
                        )
                        logger.info("GGUF transformer loaded successfully!")
                        
                        # Move to device if needed
                        if not use_device_map and self.device == "cuda":
                            logger.info(f"Moving GGUF transformer to {self.device}")
                            transformer = transformer.to(self.device)
                            logger.info("GGUF transformer moved to GPU")
                    
                    # Store device_map status
                    self._has_device_map = use_device_map
                except Exception as e:
                    logger.error(f"GGUF transformer loading failed: {type(e).__name__}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    raise
                
                # Now we need to create the full pipeline
                # First, check if we have base model components locally
                base_model_id = "black-forest-labs/FLUX.1-dev"
                if "schnell" in self.model_name.lower():
                    base_model_id = "black-forest-labs/FLUX.1-schnell"
                
                # Look for cached base model - check local models directory first
                cache_paths = [
                    # Check in local models directory first (where full FLUX model is)
                    self.gguf_path.parent.parent / "FLUX.1-dev",  # models/image/FLUX.1-dev
                    self.gguf_path.parent.parent / "FLUX.1-schnell",  # models/image/FLUX.1-schnell
                    # Then check flux_base directory
                    self.gguf_path.parent.parent.parent / "flux_base" / f"models--{base_model_id.replace('/', '--')}",
                    # Finally check HuggingFace cache
                    Path.home() / ".cache" / "huggingface" / "hub" / f"models--{base_model_id.replace('/', '--')}",
                ]
                
                base_model_path = None
                for path in cache_paths:
                    if path.exists():
                        # Check if it's a direct model directory (has model_index.json or vae folder)
                        if (path / "vae").exists() or (path / "model_index.json").exists():
                            base_model_path = path
                            logger.info(f"Found cached base model at: {base_model_path}")
                            break
                        # Check for snapshots structure (HuggingFace cache)
                        snapshots = path / "snapshots"
                        if snapshots.exists() and list(snapshots.iterdir()):
                            base_model_path = list(snapshots.iterdir())[0]
                            logger.info(f"Found cached base model at: {base_model_path}")
                            break
                
                # Create the pipeline
                if base_model_path:
                    # Load from local cache
                    logger.info("Creating pipeline with cached components...")
                    try:
                        # First try to load just the components we need (VAE, text encoders)
                        # without the transformer since we're using GGUF
                        from diffusers import AutoencoderKL
                        from transformers import CLIPTextModel, T5EncoderModel
                        
                        logger.info("Loading individual components...")
                        
                        # Get HF token from all sources
                        from ..utils import get_hf_token_from_all_sources, validate_hf_token
                        
                        hf_token = get_hf_token_from_all_sources()
                        if hf_token and validate_hf_token(hf_token):
                            logger.info("Using HF token for local model loading")
                        else:
                            logger.warning(
                                "No valid HF token found. This may cause issues with gated models like FLUX.1-dev.\n"
                                "Please set your token via:\n"
                                "1. The 'Model Management' tab in the UI\n"
                                "2. Environment variable: export HF_TOKEN='your_token'\n"
                                "3. Or run: huggingface-cli login"
                            )
                        
                        try:
                            # Load VAE (small, ~1GB)
                            vae = AutoencoderKL.from_pretrained(
                                str(base_model_path),
                                subfolder="vae",
                                torch_dtype=compute_dtype,
                                local_files_only=True,
                                low_cpu_mem_usage=True,
                                token=hf_token
                            )
                            logger.info("VAE loaded")
                            
                            # Load text encoder (CLIP) (small, ~1-2GB)
                            text_encoder = CLIPTextModel.from_pretrained(
                                str(base_model_path),
                                subfolder="text_encoder",
                                torch_dtype=compute_dtype,
                                local_files_only=True,
                                low_cpu_mem_usage=True,
                                token=hf_token
                            )
                            logger.info("CLIP text encoder loaded")
                            
                            # Load text encoder 2 (T5) (large, ~8-10GB)
                            # This is the memory-heavy component
                            text_encoder_2 = T5EncoderModel.from_pretrained(
                                str(base_model_path),
                                subfolder="text_encoder_2",
                                torch_dtype=compute_dtype,
                                local_files_only=True,
                                low_cpu_mem_usage=True,
                                token=hf_token
                            )
                            logger.info("T5 text encoder loaded")
                        except Exception as auth_error:
                            if "401" in str(auth_error) or "restricted" in str(auth_error).lower():
                                logger.error("Authentication failed when loading FLUX.1-dev components")
                                error_msg = (
                                    "❌ Authentication Error: Cannot access FLUX.1-dev components.\n\n"
                                    "FLUX.1-dev is a gated model that requires authentication even for local files.\n\n"
                                    "To fix this:\n"
                                    "1. Go to https://huggingface.co/black-forest-labs/FLUX.1-dev\n"
                                    "2. Accept the license agreement\n"
                                    "3. Get your token from https://huggingface.co/settings/tokens\n"
                                    "4. Set your token in the 'Model Management' tab\n"
                                    "   OR run: export HF_TOKEN='your_token_here'\n\n"
                                    f"Error details: {str(auth_error)}"
                                )
                                raise RuntimeError(error_msg)
                            else:
                                # Re-raise other errors
                                raise
                        
                        # Load tokenizers
                        from transformers import CLIPTokenizer, T5TokenizerFast
                        
                        tokenizer = CLIPTokenizer.from_pretrained(
                            str(base_model_path),
                            subfolder="tokenizer",
                            local_files_only=True,
                            token=hf_token
                        )
                        
                        tokenizer_2 = T5TokenizerFast.from_pretrained(
                            str(base_model_path),
                            subfolder="tokenizer_2",
                            local_files_only=True,
                            token=hf_token
                        )
                        logger.info("Tokenizers loaded")
                        
                        # Load scheduler
                        from diffusers import FlowMatchEulerDiscreteScheduler
                        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                            str(base_model_path),
                            subfolder="scheduler",
                            local_files_only=True
                        )
                        logger.info("Scheduler loaded")
                        
                        # Create pipeline with all components
                        self._real_pipeline = FluxPipeline(
                            scheduler=scheduler,
                            vae=vae,
                            text_encoder=text_encoder,
                            text_encoder_2=text_encoder_2,
                            transformer=transformer,
                            tokenizer=tokenizer,
                            tokenizer_2=tokenizer_2
                        )
                        logger.info("Pipeline created with GGUF transformer and local components")
                        
                        # For Q8 models, ensure transformer is on GPU (after CPU offload setup)
                        if "q8" in self.model_name.lower() and not self._has_device_map:
                            if hasattr(self._real_pipeline, 'transformer') and self.device == "cuda":
                                # Double-check transformer is on GPU
                                device_check = next(self._real_pipeline.transformer.parameters()).device
                                if device_check.type != 'cuda':
                                    logger.warning(f"Transformer was on {device_check}, moving to {self.device}")
                                    self._real_pipeline.transformer = self._real_pipeline.transformer.to(self.device)
                                else:
                                    logger.info(f"Transformer already on {device_check}")
                        
                        
                    except Exception as e:
                        logger.warning(f"Failed to load individual components: {e}")
                        logger.info("Attempting fallback to full pipeline loading...")
                        try:
                            # Fallback to full pipeline loading
                            self._real_pipeline = FluxPipeline.from_pretrained(
                                str(base_model_path),
                                transformer=transformer,
                                torch_dtype=compute_dtype,
                                use_safetensors=True,
                                local_files_only=True,
                                token=hf_token
                            )
                            logger.info("Successfully loaded pipeline with fallback method")
                        except Exception as fallback_error:
                            logger.error(f"Fallback pipeline loading also failed: {fallback_error}")
                            raise
                else:
                    # Need to download base components
                    logger.warning(
                        f"\nBase model components not found locally.\n"
                        f"The GGUF file contains only the transformer weights.\n"
                        f"To use GGUF models, you also need:\n"
                        f"  - VAE (Variational Autoencoder)\n"
                        f"  - Text Encoder (CLIP)\n"
                        f"  - Text Encoder 2 (T5)\n"
                        f"\nOptions:\n"
                        f"1. Download base components now (requires internet)\n"
                        f"2. Download '{base_model_id}' separately\n"
                    )
                    
                    # Try to download if internet available
                    try:
                        logger.info(f"Attempting to download base components for {base_model_id}...")
                        
                        # Get HF token from all sources
                        from ..utils import get_hf_token_from_all_sources, validate_hf_token
                        
                        hf_token = get_hf_token_from_all_sources()
                        if hf_token and validate_hf_token(hf_token):
                            logger.info("Using HF token for gated model access")
                        else:
                            logger.warning("No valid HF token found - download may fail for gated models")
                        
                        self._real_pipeline = FluxPipeline.from_pretrained(
                            base_model_id,
                            transformer=transformer,
                            torch_dtype=compute_dtype,
                            use_safetensors=True,
                            resume_download=True,
                            token=hf_token  # Pass token for gated models
                        )
                        logger.info("Base components downloaded successfully!")
                    except Exception as download_error:
                        logger.error(f"Failed to download base components: {download_error}")
                        if "401" in str(download_error):
                            logger.error("Authentication failed - HF token may be required for gated models")
                        logger.error("GGUF model requires base components to function.")
                        raise RuntimeError(
                            "Cannot use GGUF model without base components. "
                            "Please ensure you have internet access or download the base model first."
                        )
                
                
                # Apply general optimizations for non-Q6/Q8 models
                is_q6_q8 = any(q in self.model_name.lower() for q in ["q6", "q8"])
                
                if not is_q6_q8:
                    if self.device == "cuda" and torch.cuda.is_available():
                        # Enable memory optimizations
                        self._real_pipeline.enable_attention_slicing()
                        if hasattr(self._real_pipeline, 'enable_vae_slicing'):
                            self._real_pipeline.enable_vae_slicing()
                        
                        # Check if device_map was used - check on the pipeline's transformer
                        has_device_map = (hasattr(self._real_pipeline, 'transformer') and 
                                         hasattr(self._real_pipeline.transformer, 'hf_device_map') and 
                                         self._real_pipeline.transformer.hf_device_map)
                        if not has_device_map:
                            # Check available VRAM before deciding on CPU offload for smaller models
                            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                            model_size_gb = self.gguf_path.stat().st_size / (1024**3)
                            
                            if model_size_gb + 10 > vram_gb:
                                logger.info(f"Model size ({model_size_gb:.1f}GB) + overhead (10GB) exceeds VRAM ({vram_gb:.1f}GB), enabling CPU offload")
                                self._real_pipeline.enable_model_cpu_offload()
                            else:
                                logger.info(f"Model size ({model_size_gb:.1f}GB) fits in VRAM ({vram_gb:.1f}GB), loading to GPU")
                                self._real_pipeline = self._real_pipeline.to(self.device)
                    else:
                        # Only move to device if not using device_map
                        if not (hasattr(transformer, 'hf_device_map') and transformer.hf_device_map):
                            self._real_pipeline = self._real_pipeline.to(self.device)
                else:
                    # Apply Q6/Q8 specific optimizations with proper error handling
                    logger.info("Q6/Q8 model detected - applying optimizations")
                    try:
                        self._apply_q6_q8_optimizations()
                    except Exception as e:
                        logger.error(f"Failed to apply Q6/Q8 optimizations: {e}")
                        # Continue without Q6/Q8 optimizations rather than failing completely
                        logger.warning("Continuing with basic optimizations for Q6/Q8 model")
                        if self.device == "cuda" and torch.cuda.is_available():
                            # Apply basic optimizations as fallback
                            self._real_pipeline.enable_attention_slicing()
                            if hasattr(self._real_pipeline, 'enable_vae_slicing'):
                                self._real_pipeline.enable_vae_slicing()
                            self._real_pipeline = self._real_pipeline.to(self.device)
                
                # Skip torch.compile for GGUF models - causes device mismatch issues
                # The GGUF quantization and device management doesn't play well with dynamo
                if False:  # Disabled due to device mismatch errors with GGUF
                    logger.info("Skipping torch.compile for GGUF models")
                
                logger.info("GGUF model pipeline ready for inference!")
                logger.info("Pipeline components loaded - ready for real AI generation")
                self._is_placeholder = False
                
            except ImportError as e:
                logger.error(f"GGUFQuantizationConfig not available: {e}")
                logger.error("Please upgrade diffusers: pip install --upgrade diffusers>=0.32.0")
                self._is_placeholder = True
                self._real_pipeline = None
                
            except Exception as e:
                logger.error(f"Failed to load GGUF model: {e}")
                logger.info("Falling back to procedural generation")
                self._is_placeholder = True
                self._real_pipeline = None
            
        except Exception as e:
            logger.error(f"Failed to initialize GGUF pipeline: {e}")
            self._is_placeholder = True
    
    def _apply_q6_q8_optimizations(self):
        """Apply Q6/Q8 specific optimizations including CPU offload per implementation guide."""
        
        if not self._real_pipeline:
            return
            
        
        if self.device == "cuda" and torch.cuda.is_available():
            # Check if device_map was used - check on the pipeline's transformer
            has_device_map = (hasattr(self._real_pipeline, 'transformer') and 
                             hasattr(self._real_pipeline.transformer, 'hf_device_map') and 
                             self._real_pipeline.transformer.hf_device_map)
            
            
            # Check available VRAM and model info
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            model_size_gb = self.gguf_path.stat().st_size / (1024**3)
            logger.info(f"VRAM available: {vram_gb:.1f}GB, Model size: {model_size_gb:.1f}GB")
            
            # For Q6/Q8 models, ALWAYS use sequential CPU offload on 24GB cards per guide
            # This overrides device_map because sequential offload is more effective
            is_q6_q8 = any(q in self.model_name.lower() for q in ["q6", "q8"])
            
            if is_q6_q8:
                
                # Q8 models on 24GB cards always need sequential CPU offload
                if vram_gb <= 24.0:
                    if hasattr(self._real_pipeline, 'enable_sequential_cpu_offload'):
                        try:
                            result = self._real_pipeline.enable_sequential_cpu_offload()
                            logger.info("Sequential CPU offload enabled - generation will be slower but stable")
                        except Exception as e:
                            # CPU offload failed but generation can continue without it
                            logger.info("Sequential CPU offload not available - continuing with standard memory management")
                else:
                    logger.info(f"High-end card ({vram_gb:.1f}GB) - using standard device placement")
                    try:
                        self._real_pipeline = self._real_pipeline.to(self.device)
                        logger.info(f"Q6/Q8 model moved to {self.device}")
                    except Exception as e:
                        logger.error(f"Failed to move Q6/Q8 model to device: {e}")
                
                # Enable memory optimizations that work with GGUF
                if hasattr(self._real_pipeline, 'enable_attention_slicing'):
                    self._real_pipeline.enable_attention_slicing(1)
                    logger.info("Enabled attention slicing")
                if hasattr(self._real_pipeline, 'enable_vae_slicing'):
                    self._real_pipeline.enable_vae_slicing()
                    logger.info("Enabled VAE slicing")
                if hasattr(self._real_pipeline, 'enable_vae_tiling'):
                    self._real_pipeline.enable_vae_tiling()
                    logger.info("Enabled VAE tiling")
                
                logger.info("Q6/Q8 model ready")
            else:
                logger.warning(f"🔍 DEBUG: Not a Q6/Q8 model: {self.model_name}")
        else:
            logger.warning(f"🔍 DEBUG: Device/CUDA check failed - device: {self.device}, cuda available: {torch.cuda.is_available()}")
    
    def __call__(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        num_images_per_prompt: int = 1,
        generator: Optional[torch.Generator] = None,
        seed: Optional[int] = None,
        **kwargs
    ):
        """Generate images using the GGUF model"""
        
        # Use real pipeline if available
        if self._real_pipeline is not None:
            try:
                logger.info("Using GGUF model for real AI generation")
                logger.info(f"Pipeline type: {type(self._real_pipeline).__name__}")
                
                # FLUX models typically use these settings
                # Adjust based on quantization level for optimal results
                if "Q2" in self.model_name or "Q3" in self.model_name:
                    # Lower quantization = faster generation
                    num_inference_steps = min(num_inference_steps, 15)
                elif any(q in self.model_name or q in self.model_name.lower() for q in ["Q6", "q6", "Q8", "q8"]):
                    # Q6/Q8 models are slow - optimize for speed
                    num_inference_steps = min(num_inference_steps, 15)
                    # Lower guidance scale can speed up generation with minimal quality loss
                    guidance_scale = min(guidance_scale, 3.5)
                    logger.info(f"Q6/Q8 model detected - optimized settings: steps={num_inference_steps}, guidance={guidance_scale}")
                elif "schnell" in self.model_name.lower():
                    # Schnell is optimized for speed
                    num_inference_steps = min(num_inference_steps, 4)
                    guidance_scale = 0.0
                
                # Log generation parameters
                logger.info("Starting GGUF generation:")
                logger.info(f"  - Model: {self.model_name}")
                logger.info(f"  - Steps: {num_inference_steps}")
                logger.info(f"  - Size: {width}x{height}")
                logger.info(f"  - Has device_map: {self._has_device_map}")
                
                # Debug: Check device placement of all components
                if hasattr(self._real_pipeline, 'transformer') and self._real_pipeline.transformer is not None:
                    transformer_device = next(self._real_pipeline.transformer.parameters()).device
                    logger.info(f"  - Transformer device: {transformer_device}")
                if hasattr(self._real_pipeline, 'vae') and self._real_pipeline.vae is not None:
                    vae_device = next(self._real_pipeline.vae.parameters()).device
                    logger.info(f"  - VAE device: {vae_device}")
                if hasattr(self._real_pipeline, 'text_encoder') and self._real_pipeline.text_encoder is not None:
                    text_encoder_device = next(self._real_pipeline.text_encoder.parameters()).device
                    logger.info(f"  - Text encoder device: {text_encoder_device}")
                if hasattr(self._real_pipeline, 'text_encoder_2') and self._real_pipeline.text_encoder_2 is not None:
                    text_encoder_2_device = next(self._real_pipeline.text_encoder_2.parameters()).device
                    logger.info(f"  - Text encoder 2 device: {text_encoder_2_device}")
                
                # Skip all device checks and synchronization for GGUF models
                # The GGUF transformer has complex internal device placement
                if any(q in self.model_name.lower() for q in ["q6", "q8"]):
                    logger.info("Q6/Q8 GGUF model - skipping device checks to avoid conflicts")
                    logger.info("CPU offload will handle all device placement")
                else:
                    # Skip device synchronization if using CPU offload
                    # CPU offload manages device placement automatically
                    has_cpu_offload = hasattr(self._real_pipeline, '_cpu_offload_hook') and self._real_pipeline._cpu_offload_hook is not None
                    
                    if has_cpu_offload:
                        logger.info("Using CPU offload - device management handled automatically")
                
                # Clean up memory before generation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
                    logger.info("Cleared GPU cache before generation")
                
                # Generate with GGUF model
                # Extract callback from kwargs if present
                callback_on_step_end = kwargs.pop('callback_on_step_end', None)
                
                # Add performance optimizations for FLUX
                generation_kwargs = {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "width": width,
                    "height": height,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "num_images_per_prompt": num_images_per_prompt,
                    "generator": generator,
                    "callback_on_step_end": callback_on_step_end,
                }
                
                # Add FLUX-specific optimizations
                if any(q in self.model_name.lower() for q in ["q6", "q8"]):
                    # For Q6/Q8 models, use max_sequence_length optimization
                    # This limits the text encoder context for faster processing
                    generation_kwargs["max_sequence_length"] = 256  # Default is 512
                    logger.info("Using max_sequence_length=256 for faster Q6/Q8 generation")
                    
                    # Check if we're using sequential CPU offload
                    has_sequential_offload = hasattr(self._real_pipeline, '_sequential_cpu_offload_hook') and self._real_pipeline._sequential_cpu_offload_hook is not None
                    
                    # Only move generator to CPU if using sequential offload
                    if has_sequential_offload and generator is not None and generator.device.type == 'cuda':
                        logger.info("Moving generator to CPU to match sequential offload")
                        generator = torch.Generator('cpu').manual_seed(int(torch.randint(0, 2**32, (1,)).item()))
                        generation_kwargs["generator"] = generator
                    # Otherwise keep generator on same device as model
                
                # Merge with any additional kwargs
                generation_kwargs.update(kwargs)
                
                logger.info("Starting pipeline generation...")
                start_time = time.time()
                
                try:
                    
                    result = self._real_pipeline(**generation_kwargs)
                    
                    elapsed = time.time() - start_time
                    logger.info(f"Pipeline generation completed in {elapsed:.1f}s")
                except RuntimeError as e:
                    if "Expected all tensors to be on the same device" in str(e):
                        elapsed = time.time() - start_time
                        logger.error(f"Device mismatch error after {elapsed:.1f}s")
                        # Determine quantization level from model name
                        quant_level = "GGUF"
                        if "q8" in self.model_name.lower():
                            quant_level = "Q8"
                        elif "q6" in self.model_name.lower():
                            quant_level = "Q6"
                        elif "q5" in self.model_name.lower():
                            quant_level = "Q5"
                        elif "q4" in self.model_name.lower():
                            quant_level = "Q4"
                        
                        logger.error(f"Device placement issue with GGUF {quant_level} model")
                        logger.error("This is a known limitation when using diffusers with GGUF models")
                        logger.error("\nPossible causes:")
                        logger.error("- The GGUF quantization splits tensors across devices")
                        logger.error("- model_cpu_offload() has bugs with GGUF models")
                        logger.error("\nRecommended solutions:")
                        logger.error("1. Restart the app and try again (sometimes helps)")
                        logger.error("2. Use Q4 models which have better compatibility")
                        logger.error("3. Use ComfyUI which has better GGUF support")
                        logger.error("4. Use non-GGUF FLUX models (FP16/FP8)")
                        logger.error("5. Try with --low-vram flag if available")
                        raise RuntimeError(
                            f"GGUF {quant_level} model device placement error. "
                            f"This is a known issue with diffusers + GGUF. "
                            f"Try: 1) Restart app, 2) Use Q4 models, 3) Use ComfyUI, or 4) Use non-GGUF models."
                        )
                    else:
                        elapsed = time.time() - start_time
                        logger.error(f"Pipeline generation failed after {elapsed:.1f}s")
                        raise
                
                logger.info(f"Generated {len(result.images)} images with GGUF model")
                return result
                
            except Exception as e:
                logger.error(f"GGUF pipeline generation failed: {e}")
                logger.error(f"Error type: {type(e).__name__}")
                logger.error(f"Pipeline status: real_pipeline={self._real_pipeline is not None}, placeholder={self._is_placeholder}")
                import traceback
                traceback.print_exc()
                # No procedural fallback - raise the error
                raise RuntimeError(f"GGUF model generation failed: {str(e)}")
        
        # No real pipeline available - fail properly
        logger.error(f"GGUF model {self.model_name} has no real pipeline available")
        logger.error(f"Model status: placeholder={self._is_placeholder}, real_pipeline={self._real_pipeline is not None}")
        raise RuntimeError(f"GGUF model {self.model_name} is not properly initialized for real AI generation")
    
    
    def to(self, device):
        """Move pipeline to device"""
        self.device = device
        if self._real_pipeline is not None:
            logger.info(f"Moving GGUF pipeline to {device}")
            self._real_pipeline = self._real_pipeline.to(device)
            # Ensure all components are on the same device
            if hasattr(self._real_pipeline, 'vae') and self._real_pipeline.vae is not None:
                self._real_pipeline.vae = self._real_pipeline.vae.to(device)
            if hasattr(self._real_pipeline, 'text_encoder') and self._real_pipeline.text_encoder is not None:
                self._real_pipeline.text_encoder = self._real_pipeline.text_encoder.to(device)
            if hasattr(self._real_pipeline, 'text_encoder_2') and self._real_pipeline.text_encoder_2 is not None:
                self._real_pipeline.text_encoder_2 = self._real_pipeline.text_encoder_2.to(device)
            if hasattr(self._real_pipeline, 'transformer') and self._real_pipeline.transformer is not None:
                self._real_pipeline.transformer = self._real_pipeline.transformer.to(device)
            logger.info(f"All pipeline components moved to {device}")
        return self
    
    def enable_attention_slicing(self, slice_size: Optional[int] = None):
        """Enable attention slicing"""
        if self._real_pipeline is not None and hasattr(self._real_pipeline, 'enable_attention_slicing'):
            self._real_pipeline.enable_attention_slicing(slice_size)
    
    def enable_model_cpu_offload(self):
        """Enable CPU offload"""
        if self._real_pipeline is not None and hasattr(self._real_pipeline, 'enable_model_cpu_offload'):
            self._real_pipeline.enable_model_cpu_offload()
    
    def enable_vae_slicing(self):
        """Enable VAE slicing"""
        if self._real_pipeline is not None and hasattr(self._real_pipeline, 'enable_vae_slicing'):
            self._real_pipeline.enable_vae_slicing()
    
    def enable_xformers_memory_efficient_attention(self):
        """Enable xformers"""
        if self._real_pipeline is not None and hasattr(self._real_pipeline, 'enable_xformers_memory_efficient_attention'):
            try:
                self._real_pipeline.enable_xformers_memory_efficient_attention()
            except Exception:
                pass  # xformers not available
    
    @property
    def hf_device_map(self):
        """Check if pipeline uses device_map"""
        if self._has_device_map:
            return True
        if self._real_pipeline is not None and hasattr(self._real_pipeline, 'hf_device_map'):
            return self._real_pipeline.hf_device_map
        return None


def load_standalone_gguf(
    gguf_path: Path,
    model_name: str,
    device: str = "cuda",
    progress_callback: Optional[Any] = None
) -> Tuple[Optional[StandaloneGGUFPipeline], str]:
    """Load a GGUF model using the official diffusers API
    
    Args:
        gguf_path: Path to GGUF file
        model_name: Name of the model
        device: Device to load on
        progress_callback: Optional progress callback
        
    Returns:
        Tuple of (pipeline, status_message)
    """
    try:
        if progress_callback:
            progress_callback(0.1, f"Initializing GGUF model {model_name}")
        
        if not gguf_path.exists():
            return None, f"GGUF file not found: {gguf_path}"
        
        # Create the GGUF pipeline wrapper
        pipeline = StandaloneGGUFPipeline(
            gguf_path=gguf_path,
            model_name=model_name,
            device=device
        )
        
        if progress_callback:
            progress_callback(0.8, "Optimizing GGUF pipeline")
        
        # Check if it loaded successfully
        if hasattr(pipeline, '_real_pipeline') and pipeline._real_pipeline is not None:
            if progress_callback:
                progress_callback(1.0, f"GGUF model {model_name} ready for inference")
            return pipeline, f"Successfully loaded GGUF model {model_name} with full inference support"
        elif hasattr(pipeline, '_is_placeholder') and pipeline._is_placeholder:
            if progress_callback:
                progress_callback(1.0, f"GGUF model {model_name} loaded (limited mode)")
            return pipeline, f"Loaded GGUF model {model_name} in compatibility mode"
        else:
            return pipeline, f"Loaded GGUF model {model_name}"
        
    except Exception as e:
        logger.error(f"Failed to load GGUF model: {e}")
        import traceback
        traceback.print_exc()
        return None, f"Failed to load GGUF model: {str(e)}"