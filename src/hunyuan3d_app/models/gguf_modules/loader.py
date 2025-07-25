"""GGUF model loading functionality"""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple, Any
import torch

logger = logging.getLogger(__name__)


class GGUFLoader:
    """Handles loading GGUF weights and creating pipelines"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        
    def load_gguf_transformer(self, gguf_path: Path, model_name: str) -> Tuple[Any, bool]:
        """Load GGUF transformer with proper quantization config
        
        Returns:
            Tuple of (transformer, has_device_map)
        """
        logger.info(f"GGUF file exists: {gguf_path.exists()}")
        file_size_gb = gguf_path.stat().st_size / (1024**3)
        logger.info(f"GGUF file size: {file_size_gb:.2f} GB")
        
        # Log file size but continue loading
        if file_size_gb > 10.0:
            logger.info(f"Loading large GGUF file ({file_size_gb:.1f} GB). This may take a while...")
            logger.info("For faster loading, consider using smaller quantization levels (Q4, Q5, Q6).")
        
        # Import required components
        from diffusers import FluxTransformer2DModel, GGUFQuantizationConfig
        
        # Create quantization config
        is_large_quant = any(q in model_name.lower() or q in str(gguf_path) for q in ["q6", "q8", "Q6", "Q8"])
        if is_large_quant:
            compute_dtype = torch.float16
            logger.info("Q6/Q8 model detected - using float16 for better performance")
        else:
            compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        quantization_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)
        
        logger.info("Loading GGUF transformer (this may take several minutes for large files)...")
        logger.info(f"GGUF path: {gguf_path}")
        logger.info(f"Using compute dtype: {compute_dtype}")
        
        # Check if it's a Q6/Q8 model - these have device placement issues
        use_device_map = file_size_gb > 15.0 and torch.cuda.is_available() and not is_large_quant
        
        # Import memory manager
        from ...utils.memory_optimization import get_memory_manager
        memory_manager = get_memory_manager()
        
        if is_large_quant:
            logger.info(f"Q6/Q8 model detected ({file_size_gb:.1f}GB) - using optimized loading strategy")
            memory_manager.aggressive_memory_clear()
        
        # Load transformer
        has_device_map = False
        
        try:
            if is_large_quant:
                # Load to CPU first, then move to target device
                logger.info("Loading Q6/Q8 model to CPU first...")
                transformer = FluxTransformer2DModel.from_single_file(
                    str(gguf_path),
                    quantization_config=quantization_config,
                    torch_dtype=compute_dtype,
                    low_cpu_mem_usage=True
                )
                logger.info("Q6/Q8 transformer loaded to CPU")
                
                # Move to target device after loading
                if self.device == "cuda" and torch.cuda.is_available():
                    memory_manager.aggressive_memory_clear()
                    logger.info(f"Moving Q6/Q8 transformer to {self.device}...")
                    transformer = transformer.to(self.device)
                    logger.info(f"Q6/Q8 transformer moved to {self.device}")
            elif use_device_map:
                logger.info(f"Using device_map='auto' for large model ({file_size_gb:.1f}GB)")
                transformer = FluxTransformer2DModel.from_single_file(
                    str(gguf_path),
                    quantization_config=quantization_config,
                    torch_dtype=compute_dtype,
                    device_map="auto",
                    low_cpu_mem_usage=True
                )
                has_device_map = True
            else:
                # Regular loading for smaller models
                transformer = FluxTransformer2DModel.from_single_file(
                    str(gguf_path),
                    quantization_config=quantization_config,
                    torch_dtype=compute_dtype,
                    low_cpu_mem_usage=True
                )
                logger.info("GGUF transformer loaded to CPU")
                
                # Move to device
                if self.device == "cuda" and torch.cuda.is_available():
                    logger.info(f"Moving transformer to {self.device}...")
                    transformer = transformer.to(self.device)
                    logger.info(f"Transformer moved to {self.device}")
                    
        except Exception as load_error:
            logger.warning(f"Failed to load with device_map, retrying without: {load_error}")
            # Fallback to regular loading
            transformer = FluxTransformer2DModel.from_single_file(
                str(gguf_path),
                quantization_config=quantization_config,
                torch_dtype=compute_dtype,
                low_cpu_mem_usage=True
            )
            logger.info("GGUF transformer loaded (fallback method)")
            
            # Move to device
            if self.device == "cuda" and torch.cuda.is_available():
                logger.info(f"Moving transformer to {self.device}...")
                transformer = transformer.to(self.device)
                logger.info(f"Transformer moved to {self.device}")
                
        logger.info("GGUF transformer loaded successfully")
        return transformer, has_device_map
    
    def load_flux_components(self, base_model_path: Path, compute_dtype: torch.dtype) -> dict:
        """Load FLUX model components
        
        Returns:
            Dictionary with loaded components
        """
        from diffusers import AutoencoderKL
        from transformers import CLIPTextModel, T5EncoderModel, CLIPTokenizer, T5TokenizerFast
        from diffusers import FlowMatchEulerDiscreteScheduler
        
        components = {}
        
        # Get HF token from environment
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not hf_token:
            logger.warning("No HF_TOKEN found in environment. May fail to load gated models.")
        
        try:
            # Load VAE
            components['vae'] = AutoencoderKL.from_pretrained(
                str(base_model_path),
                subfolder="vae",
                torch_dtype=compute_dtype,
                local_files_only=True,
                low_cpu_mem_usage=True,
                token=hf_token
            )
            logger.info("VAE loaded")
            
            # Load text encoder (CLIP)
            components['text_encoder'] = CLIPTextModel.from_pretrained(
                str(base_model_path),
                subfolder="text_encoder",
                torch_dtype=compute_dtype,
                local_files_only=True,
                low_cpu_mem_usage=True,
                token=hf_token
            )
            logger.info("CLIP text encoder loaded")
            
            # Load text encoder 2 (T5)
            components['text_encoder_2'] = T5EncoderModel.from_pretrained(
                str(base_model_path),
                subfolder="text_encoder_2",
                torch_dtype=compute_dtype,
                local_files_only=True,
                low_cpu_mem_usage=True,
                token=hf_token
            )
            logger.info("T5 text encoder loaded")
            
            # Load tokenizers
            components['tokenizer'] = CLIPTokenizer.from_pretrained(
                str(base_model_path),
                subfolder="tokenizer",
                local_files_only=True,
                token=hf_token
            )
            
            components['tokenizer_2'] = T5TokenizerFast.from_pretrained(
                str(base_model_path),
                subfolder="tokenizer_2",
                local_files_only=True,
                token=hf_token
            )
            logger.info("Tokenizers loaded")
            
            # Load scheduler
            components['scheduler'] = FlowMatchEulerDiscreteScheduler.from_pretrained(
                str(base_model_path),
                subfolder="scheduler",
                local_files_only=True
            )
            logger.info("Scheduler loaded")
            
            return components
            
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
                raise