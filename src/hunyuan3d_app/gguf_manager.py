"""GGUF Model Manager for efficient FLUX model loading and inference"""

import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List
from dataclasses import dataclass
import torch
from diffusers import FluxPipeline, FluxTransformer2DModel
from transformers import T5EncoderModel, CLIPTextModel
from huggingface_hub import hf_hub_download
import requests

logger = logging.getLogger(__name__)


@dataclass
class GGUFModelInfo:
    """Information about a GGUF model variant"""
    name: str
    quantization: str
    file_size_gb: float
    memory_required_gb: float
    repo_id: str
    filename: str
    url: str
    quality_score: float = 0.95  # Quality relative to full model
    min_vram_gb: float = 8.0  # Minimum VRAM required


class GGUFModelManager:
    """Manages GGUF model loading and inference for FLUX models"""
    
    # Available GGUF models with their specifications
    FLUX_DEV_MODELS = {
        "Q8_0": GGUFModelInfo(
            name="FLUX.1-dev-Q8_0",
            quantization="8",
            file_size_gb=12.5,
            memory_required_gb=14.0,
            repo_id="city96/FLUX.1-dev-gguf",
            filename="flux1-dev-Q8_0.gguf",
            url="https://huggingface.co/city96/FLUX.1-dev-gguf/resolve/main/flux1-dev-Q8_0.gguf",
            quality_score=0.98,
            min_vram_gb=12.0
        ),
        "Q5_K_S": GGUFModelInfo(
            name="FLUX.1-dev-Q5_K_S",
            quantization="5",
            file_size_gb=8.0,
            memory_required_gb=10.0,
            repo_id="city96/FLUX.1-dev-gguf",
            filename="flux1-dev-Q5_K_S.gguf",
            url="https://huggingface.co/city96/FLUX.1-dev-gguf/resolve/main/flux1-dev-Q5_K_S.gguf",
            quality_score=0.95,
            min_vram_gb=8.0
        ),
        "Q4_K_S": GGUFModelInfo(
            name="FLUX.1-dev-Q4_K_S",
            quantization="4",
            file_size_gb=6.5,
            memory_required_gb=8.0,
            repo_id="city96/FLUX.1-dev-gguf",
            filename="flux1-dev-Q4_K_S.gguf",
            url="https://huggingface.co/city96/FLUX.1-dev-gguf/resolve/main/flux1-dev-Q4_K_S.gguf",
            quality_score=0.90,
            min_vram_gb=6.0
        ),
    }
    
    FLUX_SCHNELL_MODELS = {
        "Q8_0": GGUFModelInfo(
            name="FLUX.1-schnell-Q8_0",
            quantization="8",
            file_size_gb=12.5,
            memory_required_gb=14.0,
            repo_id="city96/FLUX.1-schnell-gguf",
            filename="flux1-schnell-Q8_0.gguf",
            url="https://huggingface.co/city96/FLUX.1-schnell-gguf/resolve/main/flux1-schnell-Q8_0.gguf",
            quality_score=0.98,
            min_vram_gb=12.0
        ),
    }
    
    def __init__(self, cache_dir: str = None, models_dir: Path = None, device: str = "cuda"):
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        elif models_dir:
            self.cache_dir = models_dir / "gguf"
        else:
            self.cache_dir = Path.home() / ".cache" / "huggingface" / "gguf"
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.loaded_components = {}
        
    def load_gguf_flux_model(
        self, 
        model_name: str,
        gguf_file_path: Path,
        vae_path: Optional[Path] = None,
        text_encoder_path: Optional[Path] = None,
        text_encoder_2_path: Optional[Path] = None,
        dtype: torch.dtype = torch.float16,
        progress_callback=None
    ) -> Tuple[Optional[FluxPipeline], str]:
        """
        Load a FLUX GGUF model with its components
        
        Args:
            model_name: Name of the model
            gguf_file_path: Path to the GGUF transformer file
            vae_path: Path to VAE model
            text_encoder_path: Path to CLIP text encoder
            text_encoder_2_path: Path to T5 text encoder
            dtype: Data type for loading (float16 recommended)
            progress_callback: Callback for progress updates
            
        Returns:
            Tuple of (pipeline, status_message)
        """
        try:
            if progress_callback:
                progress_callback(0.1, desc=f"Loading GGUF transformer from {gguf_file_path.name}")
            
            # Load the GGUF transformer
            logger.info(f"Loading GGUF transformer from {gguf_file_path}")
            
            # For FLUX models, we need to load the transformer from the GGUF file
            # The diffusers library supports loading from single files
            transformer = FluxTransformer2DModel.from_single_file(
                str(gguf_file_path),
                torch_dtype=dtype,
                low_cpu_mem_usage=True
            )
            
            if progress_callback:
                progress_callback(0.3, desc="Loading VAE")
            
            # Load VAE if provided
            vae = None
            if vae_path and vae_path.exists():
                logger.info(f"Loading VAE from {vae_path}")
                from diffusers import AutoencoderKL
                vae = AutoencoderKL.from_pretrained(
                    str(vae_path.parent),
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True
                )
            
            if progress_callback:
                progress_callback(0.5, desc="Loading text encoders")
            
            # Load text encoders
            text_encoder = None
            if text_encoder_path and text_encoder_path.exists():
                logger.info(f"Loading CLIP text encoder from {text_encoder_path}")
                text_encoder = CLIPTextModel.from_pretrained(
                    str(text_encoder_path.parent),
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True
                )
            
            text_encoder_2 = None
            if text_encoder_2_path and text_encoder_2_path.exists():
                logger.info(f"Loading T5 text encoder from {text_encoder_2_path}")
                text_encoder_2 = T5EncoderModel.from_pretrained(
                    str(text_encoder_2_path.parent),
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True
                )
            
            if progress_callback:
                progress_callback(0.7, desc="Creating pipeline")
            
            # Create the pipeline with loaded components
            pipeline = FluxPipeline(
                transformer=transformer,
                vae=vae,
                text_encoder=text_encoder,
                text_encoder_2=text_encoder_2,
                tokenizer=None,  # Will be loaded from text encoder
                tokenizer_2=None,  # Will be loaded from text encoder 2
                scheduler=None  # Will use default scheduler
            )
            
            # Move to device
            if self.device == "cuda" and torch.cuda.is_available():
                if progress_callback:
                    progress_callback(0.8, desc="Moving to GPU")
                pipeline = pipeline.to(self.device)
                
                # Enable memory efficient attention if available
                try:
                    pipeline.enable_attention_slicing()
                    pipeline.enable_vae_slicing()
                except:
                    pass
            
            if progress_callback:
                progress_callback(1.0, desc="GGUF model loaded successfully")
            
            # Store loaded components for memory management
            self.loaded_components = {
                "transformer": transformer,
                "vae": vae,
                "text_encoder": text_encoder,
                "text_encoder_2": text_encoder_2,
                "pipeline": pipeline
            }
            
            return pipeline, f"✅ Successfully loaded GGUF model {model_name}"
            
        except Exception as e:
            logger.error(f"Error loading GGUF model: {str(e)}")
            return None, f"❌ Failed to load GGUF model: {str(e)}"
    
    def estimate_memory_usage(self, quantization: str) -> Dict[str, float]:
        """
        Estimate memory usage for different quantization levels
        
        Args:
            quantization: Quantization level (Q4_K_S, Q5_K_S, Q8_0, etc.)
            
        Returns:
            Dictionary with memory estimates in GB
        """
        # Approximate memory usage for FLUX models
        base_memory = {
            "Q4_K_S": 6.5,   # 4-bit quantization
            "Q5_K_S": 8.0,   # 5-bit quantization
            "Q8_0": 12.5,    # 8-bit quantization
            "F16": 24.0,     # Full precision
        }
        
        # Add overhead for other components
        vae_memory = 0.5
        text_encoder_memory = 1.0
        overhead = 1.5
        
        quantization_key = quantization.upper()
        if quantization_key in base_memory:
            total = base_memory[quantization_key] + vae_memory + text_encoder_memory + overhead
            return {
                "transformer": base_memory[quantization_key],
                "vae": vae_memory,
                "text_encoders": text_encoder_memory,
                "overhead": overhead,
                "total": total
            }
        else:
            # Default to Q8 estimate
            return self.estimate_memory_usage("Q8_0")
    
    def unload_model(self):
        """Unload the current GGUF model and free memory"""
        for component_name, component in self.loaded_components.items():
            if component is not None:
                del component
        
        self.loaded_components = {}
        
        # Force garbage collection
        import gc
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("GGUF model unloaded and memory freed")
    
    def get_available_vram(self) -> float:
        """Get available VRAM in GB"""
        if torch.cuda.is_available():
            # Get free memory in bytes
            free_memory = torch.cuda.mem_get_info()[0]
            # Convert to GB
            return free_memory / (1024 ** 3)
        return 0.0
    
    def recommend_quantization(self, model_type: str = "flux-dev") -> Optional[GGUFModelInfo]:
        """Recommend best GGUF model based on available VRAM"""
        available_vram = self.get_available_vram()
        
        if model_type == "flux-dev":
            models = self.FLUX_DEV_MODELS
        else:
            models = self.FLUX_SCHNELL_MODELS
        
        # Sort by memory requirement (ascending)
        sorted_models = sorted(
            models.values(),
            key=lambda x: x.memory_required_gb
        )
        
        # Find best model that fits in available VRAM
        for model in sorted_models:
            if model.memory_required_gb <= available_vram:
                return model
        
        # If no model fits, return the smallest one
        return sorted_models[0] if sorted_models else None
    
    def is_model_cached(self, model_info: GGUFModelInfo) -> bool:
        """Check if model is already downloaded"""
        model_path = self.cache_dir / model_info.filename
        return model_path.exists()
    
    def get_model_path(self, model_info: GGUFModelInfo) -> Path:
        """Get path to model file"""
        return self.cache_dir / model_info.filename
    
    def download_model(self, model_info: GGUFModelInfo, token: Optional[str] = None) -> Path:
        """Download GGUF model"""
        model_path = self.get_model_path(model_info)
        
        if model_path.exists():
            logger.info(f"Model already cached: {model_path}")
            return model_path
        
        logger.info(f"Downloading {model_info.name} from {model_info.repo_id}")
        
        # Download using huggingface_hub
        downloaded_path = hf_hub_download(
            repo_id=model_info.repo_id,
            filename=model_info.filename,
            local_dir=str(self.cache_dir),
            token=token
        )
        
        return Path(downloaded_path)
    
    def load_transformer(
        self, 
        model_info: GGUFModelInfo,
        compute_dtype: torch.dtype = torch.float16,
        device_map: str = "auto",
        model_path: Optional[Path] = None
    ) -> FluxTransformer2DModel:
        """Load GGUF transformer model"""
        # Use provided path or get from model info
        if model_path is None:
            model_path = self.get_model_path(model_info)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        logger.info(f"Loading GGUF transformer from {model_path}")
        
        # Load transformer from GGUF file using GGUFQuantizationConfig
        try:
            from diffusers import FluxTransformer2DModel, GGUFQuantizationConfig
            
            # Create quantization config for GGUF models
            quantization_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)
            
            logger.info(f"Loading GGUF model with quantization config (compute_dtype={compute_dtype})")
            logger.info(f"Model file size: {model_path.stat().st_size / (1024**3):.2f} GB")
            
            # Load the GGUF transformer with proper quantization config
            # This can take a while for large models
            start_time = time.time()
            transformer = FluxTransformer2DModel.from_single_file(
                str(model_path),
                quantization_config=quantization_config,
                torch_dtype=compute_dtype,
                device_map=device_map,
                resume_download=True,
                local_files_only=False
            )
            
            load_time = time.time() - start_time
            logger.info(f"GGUF transformer loaded successfully in {load_time:.1f} seconds")
            
        except ImportError as ie:
            logger.error(f"GGUFQuantizationConfig not available: {str(ie)}")
            logger.warning("Please upgrade diffusers: pip install --upgrade diffusers gguf")
            # Fallback for older diffusers versions
            try:
                transformer = FluxTransformer2DModel.from_single_file(
                    str(model_path),
                    torch_dtype=compute_dtype,
                    device_map=device_map,
                    low_cpu_mem_usage=False,
                    ignore_mismatched_sizes=True
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load GGUF model. Please upgrade diffusers: {str(e)}")
                
        except Exception as e:
            logger.error(f"Failed to load GGUF model: {str(e)}")
            # Try one more fallback with minimal parameters
            logger.warning("Attempting minimal fallback loading")
            try:
                transformer = FluxTransformer2DModel.from_single_file(
                    str(model_path),
                    torch_dtype=compute_dtype
                )
            except Exception as e2:
                raise RuntimeError(f"All loading attempts failed: {str(e2)}")
        
        return transformer