"""Core FLUX model implementations based on Flux.1 Dev Implementation Guide.

This module provides the fundamental FLUX pipeline implementations including
standard and GGUF quantized variants with proper device management and optimizations.
"""

import torch
import gc
import time
import logging
from pathlib import Path
from typing import Optional, Any, Dict, Union
from diffusers import FluxPipeline, FluxTransformer2DModel, GGUFQuantizationConfig
from diffusers.utils import make_image_grid

logger = logging.getLogger(__name__)


class FluxGenerator:
    """Basic FLUX.1 implementation with optimal settings."""
    
    def __init__(self, model_id="black-forest-labs/FLUX.1-dev", device="cuda"):
        self.device = device
        self.model_id = model_id
        self.pipe = None
        self._setup_pipeline()
    
    def _setup_pipeline(self):
        """Initialize the Flux pipeline with optimal settings."""
        logger.info(f"Loading Flux.1 model from {self.model_id}...")
        
        # Get HF token for gated models
        from ..utils import get_hf_token_from_all_sources, validate_hf_token
        
        hf_token = get_hf_token_from_all_sources()
        if hf_token and validate_hf_token(hf_token):
            logger.info("Using HF token for FLUX model access")
        else:
            logger.warning("No valid HF token found - this may cause issues with FLUX.1-dev")
        
        # Load pipeline with optimized settings
        self.pipe = FluxPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,  # Use bfloat16 for better performance
            device_map="auto" if torch.cuda.device_count() > 1 else None,
            token=hf_token  # Pass token for gated models
        )
        
        # Move to device if not using device_map
        if not hasattr(self.pipe, 'hf_device_map') or not self.pipe.hf_device_map:
            self.pipe = self.pipe.to(self.device)
        
        # Memory optimizations
        self._apply_memory_optimizations()
        
        logger.info("Model loaded successfully!")
    
    def _apply_memory_optimizations(self):
        """Apply memory optimization techniques."""
        # Import memory manager
        from ..utils.memory_optimization import get_memory_manager
        memory_manager = get_memory_manager()
        
        # Apply comprehensive VAE optimizations through memory manager
        self.pipe = memory_manager.optimize_model_for_memory(self.pipe)
        
        # Only use CPU offload for non-GGUF models
        if not hasattr(self, '_is_gguf_model'):
            # Check available VRAM
            if torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if vram_gb < 16:  # Use CPU offload for limited VRAM
                    logger.info(f"Limited VRAM ({vram_gb:.1f}GB), enabling CPU offload")
                    self.pipe.enable_model_cpu_offload()
    
    def generate_image(self, 
                      prompt: str,
                      negative_prompt: Optional[str] = None,
                      height: int = 1024,
                      width: int = 1024,
                      num_inference_steps: int = 28,  # Optimal for FLUX.1-dev
                      guidance_scale: float = 3.5,    # Default distilled CFG scale
                      seed: Optional[int] = None,
                      max_sequence_length: int = 512) -> Any:
        """Generate image with optimized parameters.
        
        Note: FLUX.1-dev uses distilled CFG. The guidance_scale parameter
        here represents the distilled CFG scale, not regular CFG.
        For FLUX.1-dev, always use CFG=1.0 internally.
        """
        
        generator = torch.Generator(device=self.device).manual_seed(seed) if seed else None
        
        # Clear GPU cache before generation
        self._clear_cache()
        
        start_time = time.time()
        
        with torch.no_grad():
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                guidance_scale=guidance_scale,  # This is distilled CFG scale
                num_inference_steps=num_inference_steps,
                max_sequence_length=max_sequence_length,  # Optimize for memory
                generator=generator
            )
            image = result.images[0]
        
        generation_time = time.time() - start_time
        logger.info(f"Generation completed in {generation_time:.2f} seconds")
        
        return image
    
    def _clear_cache(self):
        """Clear GPU memory cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        gc.collect()


class FluxGGUFGenerator:
    """GGUF quantized FLUX implementation with proper device management."""
    
    def __init__(self, 
                 gguf_path: Union[str, Path] = "city96/FLUX.1-dev-gguf",
                 quantization_level: str = "Q8_0",
                 device: str = "cuda"):
        self.device = device
        self.gguf_path = gguf_path
        self.quantization_level = quantization_level
        self.pipe = None
        self._is_gguf_model = True  # Flag for identification
        self._setup_gguf_pipeline()
    
    def _setup_gguf_pipeline(self):
        """Setup GGUF quantized pipeline following the guide exactly."""
        logger.info(f"Loading GGUF model: {self.quantization_level}")
        
        # Determine compute dtype based on quantization level
        is_large_quant = self.quantization_level in ["Q6_K", "Q8_0"]
        if is_large_quant:
            compute_dtype = torch.float16
            logger.info("Q6/Q8 model detected - using float16 for better performance")
        else:
            compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        # Construct model path/URL
        if isinstance(self.gguf_path, str) and self.gguf_path.startswith("city96/"):
            # HuggingFace model ID
            model_filename = f"flux1-dev-{self.quantization_level}.gguf"
            gguf_file = f"https://huggingface.co/{self.gguf_path}/blob/main/{model_filename}"
        else:
            # Local file path
            gguf_file = str(self.gguf_path)
        
        logger.info(f"Loading GGUF transformer from: {gguf_file}")
        
        # Create quantization config
        quantization_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)
        
        # Load quantized transformer - NO device_map for Q6/Q8!
        if is_large_quant:
            logger.info("Loading Q6/Q8 transformer to CPU first (avoiding device_map issues)")
            transformer = FluxTransformer2DModel.from_single_file(
                gguf_file,
                quantization_config=quantization_config,
                torch_dtype=compute_dtype,
                device_map=None,  # Explicitly no device_map
                low_cpu_mem_usage=True
            )
            
            # Move to GPU after loading
            if self.device == "cuda" and torch.cuda.is_available():
                logger.info(f"Moving Q6/Q8 transformer to {self.device}")
                # Don't change dtype for quantized models!
                transformer = transformer.to(self.device)
                logger.info("Q6/Q8 transformer moved to GPU")
        else:
            # For smaller quants, can use auto device_map
            transformer = FluxTransformer2DModel.from_single_file(
                gguf_file,
                quantization_config=quantization_config,
                torch_dtype=compute_dtype,
                device_map="auto" if torch.cuda.device_count() > 1 else None
            )
            if self.device == "cuda" and not hasattr(transformer, 'hf_device_map'):
                transformer = transformer.to(self.device)
        
        # Load base pipeline with quantized transformer
        logger.info("Loading base pipeline components...")
        
        # Get HF token for gated models
        from ..utils import get_hf_token_from_all_sources, validate_hf_token
        
        hf_token = get_hf_token_from_all_sources()
        if hf_token and validate_hf_token(hf_token):
            logger.info("Using HF token for FLUX base components access")
        else:
            logger.warning("No valid HF token found - this may cause authentication errors")
        
        # Check for local model first
        from pathlib import Path
        import os
        
        # Get the project root directory
        project_root = Path(__file__).parent.parent.parent.parent
        logger.info(f"Project root: {project_root}")
        
        local_model_paths = [
            # Absolute paths to local models
            project_root / "models" / "image" / "FLUX.1-dev",
            project_root / "models" / "flux_base" / "FLUX.1-dev",
            # HuggingFace cache last (may be incomplete)
            Path.home() / ".cache" / "huggingface" / "hub" / "models--black-forest-labs--FLUX.1-dev",
        ]
        
        logger.info("Checking for local FLUX.1-dev models in:")
        for path in local_model_paths:
            logger.info(f"  - {path} (exists: {path.exists()})")
        
        local_model_path = None
        for path in local_model_paths:
            if not path.exists():
                continue
                
            # For project models, check if complete
            if "hunyuan3d-app" in str(path):
                # Ensure all required components exist
                required_components = ["vae", "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2"]
                all_present = all((path / comp).exists() for comp in required_components)
                
                if all_present:
                    local_model_path = path
                    logger.info(f"Found complete local FLUX.1-dev model at: {local_model_path}")
                    break
                else:
                    logger.warning(f"Incomplete model at {path}, missing components")
            else:
                # For HF cache, check structure
                if (path / "vae").exists() or (path / "model_index.json").exists():
                    local_model_path = path
                    logger.info(f"Found FLUX.1-dev model at: {local_model_path}")
                    break
                    
                # Check for snapshots structure
                snapshots = path / "snapshots"
                if snapshots.exists() and list(snapshots.iterdir()):
                    # Check if snapshot is complete
                    for snapshot in snapshots.iterdir():
                        if snapshot.is_dir():
                            # Check for text_encoder_2 files specifically
                            te2_path = snapshot / "text_encoder_2"
                            if te2_path.exists() and list(te2_path.glob("*.safetensors")):
                                local_model_path = snapshot
                                logger.info(f"Found FLUX.1-dev model in cache at: {local_model_path}")
                                break
                    if local_model_path:
                        break
        
        if local_model_path:
            # Load from local path
            self.pipe = FluxPipeline.from_pretrained(
                str(local_model_path),
                transformer=transformer,
                torch_dtype=compute_dtype,
                token=hf_token,
                local_files_only=True
            )
            logger.info("Loaded FLUX base components from local model")
        else:
            # Download if not found locally
            logger.info("Local FLUX.1-dev not found, downloading...")
            self.pipe = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                transformer=transformer,
                torch_dtype=compute_dtype,
                token=hf_token
            )
        
        # Apply optimizations based on quantization level
        self._apply_gguf_optimizations(is_large_quant)
        
        logger.info("GGUF model loaded successfully!")
    
    def _apply_gguf_optimizations(self, is_large_quant: bool):
        """Apply memory and performance optimizations for GGUF models."""
        # Import memory manager
        from ..utils.memory_optimization import get_memory_manager
        memory_manager = get_memory_manager()
        
        # Apply comprehensive VAE optimizations through memory manager
        self.pipe = memory_manager.optimize_model_for_memory(self.pipe)
        
        # Set memory format for better performance
        if hasattr(self.pipe.transformer, 'to'):
            self.pipe.transformer.to(memory_format=torch.channels_last)
        
        # Import memory manager
        from ..utils.memory_optimization import get_memory_manager
        memory_manager = get_memory_manager()
        
        # Clear memory before loading
        memory_manager.aggressive_memory_clear()
        
        # Device placement strategy with improved memory management
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            # Q8: ~12GB, Q6: ~10GB, others: ~6GB
            model_size_gb = 12 if "q8" in self.quantization_level.lower() else (10 if "q6" in self.quantization_level.lower() else 6)
            
            # Check memory with proper overhead calculation
            memory_check = memory_manager.check_memory_for_model(model_size_gb, overhead_gb=10.0)
            
            logger.info(f"Memory check for {self.quantization_level}:")
            logger.info(f"  Total VRAM: {vram_gb:.1f}GB")
            logger.info(f"  Available: {memory_check['available_gb']:.1f}GB")
            logger.info(f"  Required: {memory_check['required_gb']:.1f}GB (model: {model_size_gb}GB + overhead: 10GB)")
            
            # For Q6/Q8 models, always use sequential offload on 24GB cards
            if is_large_quant:
                # Q8 models need ~22GB total (12GB model + 10GB overhead)
                # On 24GB cards, this leaves very little room, so always offload
                logger.info("Q6/Q8 model detected - using sequential CPU offload for stability")
                self.pipe.enable_sequential_cpu_offload()
                logger.info("Sequential CPU offload enabled - this will be slower but prevents OOM")
            else:
                # For smaller quants, use memory check
                if not memory_check["has_enough_memory"]:
                    logger.warning(f"Insufficient memory: {memory_check['deficit_gb']:.1f}GB deficit")
                    logger.info("Using model CPU offload")
                    self.pipe.enable_model_cpu_offload()
                else:
                    logger.info(f"Sufficient memory available, moving pipeline to {self.device}")
                    self.pipe = self.pipe.to(self.device)
    
    def generate_image(self,
                      prompt: str,
                      negative_prompt: Optional[str] = None,
                      height: int = 1024,
                      width: int = 1024,
                      num_inference_steps: Optional[int] = None,
                      guidance_scale: float = 3.5,
                      seed: Optional[int] = None,
                      max_sequence_length: int = 256) -> Any:
        """Generate image with GGUF model.
        
        Optimized settings based on quantization level.
        """
        
        # Optimize steps based on quantization
        if num_inference_steps is None:
            if "Q2" in self.quantization_level or "Q3" in self.quantization_level:
                num_inference_steps = 15  # Faster for lower quality
            elif "Q6" in self.quantization_level or "Q8" in self.quantization_level:
                num_inference_steps = 15  # Optimized for Q6/Q8 speed
                guidance_scale = min(guidance_scale, 3.5)  # Lower guidance for speed
            else:
                num_inference_steps = 20  # Balanced
        
        # Import memory manager
        from ..utils.memory_optimization import get_memory_manager
        memory_manager = get_memory_manager()
        
        # Pre-generation memory check and clearing
        memory_manager.aggressive_memory_clear()
        memory_manager.monitor_memory_usage("pre-generation")
        
        # Generator setup - handle device placement
        if hasattr(self.pipe, '_sequential_cpu_offload_hook') and self.pipe._sequential_cpu_offload_hook:
            # Use CPU generator for sequential offload
            generator = torch.Generator('cpu').manual_seed(seed) if seed else None
        else:
            generator = torch.Generator(self.device).manual_seed(seed) if seed else None
        
        logger.info(f"Starting GGUF generation:")
        logger.info(f"  - Model: {self.quantization_level}")
        logger.info(f"  - Steps: {num_inference_steps}")
        logger.info(f"  - Size: {width}x{height}")
        logger.info(f"  - Guidance: {guidance_scale}")
        
        start_time = time.time()
        
        with torch.no_grad():
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                max_sequence_length=max_sequence_length,  # Lower for Q6/Q8
                generator=generator
            )
            image = result.images[0]
        
        generation_time = time.time() - start_time
        logger.info(f"GGUF generation completed in {generation_time:.2f}s")
        
        # Post-generation memory cleanup
        memory_manager.monitor_memory_usage("post-generation")
        memory_manager.aggressive_memory_clear()
        
        return image
    
    def benchmark_quantization_levels(self, prompt: str, test_levels: Optional[list] = None):
        """Benchmark different quantization levels."""
        if test_levels is None:
            test_levels = ["Q8_0", "Q5_K_M", "Q4_K_S"]
        
        results = {}
        original_level = self.quantization_level
        
        for quant_level in test_levels:
            try:
                logger.info(f"\nTesting {quant_level}...")
                
                # Reload with different quantization
                self.quantization_level = quant_level
                self._setup_gguf_pipeline()
                
                # Test generation
                torch.cuda.reset_peak_memory_stats()
                start_time = time.time()
                
                image = self.generate_image(prompt, num_inference_steps=10)
                
                generation_time = time.time() - start_time
                peak_memory = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
                
                results[quant_level] = {
                    'generation_time': generation_time,
                    'peak_memory_gb': peak_memory,
                    'image': image,
                    'success': True
                }
                
                logger.info(f"{quant_level}: {generation_time:.2f}s, {peak_memory:.2f}GB")
                
            except Exception as e:
                logger.error(f"Failed to test {quant_level}: {e}")
                results[quant_level] = {'success': False, 'error': str(e)}
        
        # Restore original
        self.quantization_level = original_level
        self._setup_gguf_pipeline()
        
        return results