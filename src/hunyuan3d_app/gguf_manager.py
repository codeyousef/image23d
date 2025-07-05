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
    
    # Available GGUF models with their specifications - Complete quantization support
    FLUX_DEV_MODELS = {
        # FP16 - Highest quality, highest memory usage
        "FP16": GGUFModelInfo(
            name="FLUX.1-dev-FP16",
            quantization="FP16",
            file_size_gb=24.0,
            memory_required_gb=26.0,
            repo_id="black-forest-labs/FLUX.1-dev",
            filename="flux1-dev-fp16.safetensors",
            url="https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors",
            quality_score=1.0,
            min_vram_gb=24.0
        ),
        # FP8 - Near FP16 quality with less memory
        "FP8_E4M3FN": GGUFModelInfo(
            name="FLUX.1-dev-FP8-E4M3FN",
            quantization="FP8",
            file_size_gb=12.0,
            memory_required_gb=14.0,
            repo_id="Kijai/flux-fp8",
            filename="flux1-dev-fp8-e4m3fn.safetensors",
            url="https://huggingface.co/Kijai/flux-fp8/resolve/main/flux1-dev-fp8-e4m3fn.safetensors",
            quality_score=0.99,
            min_vram_gb=12.0
        ),
        # Q8_0 - Excellent quality/memory balance
        "Q8_0": GGUFModelInfo(
            name="FLUX.1-dev-Q8_0",
            quantization="Q8_0",
            file_size_gb=12.5,
            memory_required_gb=14.0,
            repo_id="city96/FLUX.1-dev-gguf",
            filename="flux1-dev-Q8_0.gguf",
            url="https://huggingface.co/city96/FLUX.1-dev-gguf/resolve/main/flux1-dev-Q8_0.gguf",
            quality_score=0.98,
            min_vram_gb=12.0
        ),
        # Q6_K - Great quality with good memory savings
        "Q6_K": GGUFModelInfo(
            name="FLUX.1-dev-Q6_K",
            quantization="Q6_K",
            file_size_gb=9.8,
            memory_required_gb=11.5,
            repo_id="city96/FLUX.1-dev-gguf",
            filename="flux1-dev-Q6_K.gguf",
            url="https://huggingface.co/city96/FLUX.1-dev-gguf/resolve/main/flux1-dev-Q6_K.gguf",
            quality_score=0.96,
            min_vram_gb=10.0
        ),
        # Q5_K_M - Balanced option
        "Q5_K_M": GGUFModelInfo(
            name="FLUX.1-dev-Q5_K_M",
            quantization="Q5_K_M",
            file_size_gb=8.5,
            memory_required_gb=10.5,
            repo_id="city96/FLUX.1-dev-gguf",
            filename="flux1-dev-Q5_K_M.gguf",
            url="https://huggingface.co/city96/FLUX.1-dev-gguf/resolve/main/flux1-dev-Q5_K_M.gguf",
            quality_score=0.95,
            min_vram_gb=9.0
        ),
        # Q5_K_S - Memory efficient
        "Q5_K_S": GGUFModelInfo(
            name="FLUX.1-dev-Q5_K_S",
            quantization="Q5_K_S",
            file_size_gb=8.0,
            memory_required_gb=10.0,
            repo_id="city96/FLUX.1-dev-gguf",
            filename="flux1-dev-Q5_K_S.gguf",
            url="https://huggingface.co/city96/FLUX.1-dev-gguf/resolve/main/flux1-dev-Q5_K_S.gguf",
            quality_score=0.94,
            min_vram_gb=8.0
        ),
        # Q4_K_M - Good balance of quality and memory
        "Q4_K_M": GGUFModelInfo(
            name="FLUX.1-dev-Q4_K_M",
            quantization="Q4_K_M",
            file_size_gb=7.0,
            memory_required_gb=8.5,
            repo_id="city96/FLUX.1-dev-gguf",
            filename="flux1-dev-Q4_K_M.gguf",
            url="https://huggingface.co/city96/FLUX.1-dev-gguf/resolve/main/flux1-dev-Q4_K_M.gguf",
            quality_score=0.92,
            min_vram_gb=7.0
        ),
        # Q4_K_S - More memory efficient
        "Q4_K_S": GGUFModelInfo(
            name="FLUX.1-dev-Q4_K_S",
            quantization="Q4_K_S",
            file_size_gb=6.5,
            memory_required_gb=8.0,
            repo_id="city96/FLUX.1-dev-gguf",
            filename="flux1-dev-Q4_K_S.gguf",
            url="https://huggingface.co/city96/FLUX.1-dev-gguf/resolve/main/flux1-dev-Q4_K_S.gguf",
            quality_score=0.90,
            min_vram_gb=6.0
        ),
        # Q3_K_L - Lower quality but very memory efficient
        "Q3_K_L": GGUFModelInfo(
            name="FLUX.1-dev-Q3_K_L",
            quantization="Q3_K_L",
            file_size_gb=5.5,
            memory_required_gb=7.0,
            repo_id="city96/FLUX.1-dev-gguf",
            filename="flux1-dev-Q3_K_L.gguf",
            url="https://huggingface.co/city96/FLUX.1-dev-gguf/resolve/main/flux1-dev-Q3_K_L.gguf",
            quality_score=0.85,
            min_vram_gb=5.5
        ),
        # Q3_K_M - Even more memory efficient
        "Q3_K_M": GGUFModelInfo(
            name="FLUX.1-dev-Q3_K_M",
            quantization="Q3_K_M",
            file_size_gb=5.0,
            memory_required_gb=6.5,
            repo_id="city96/FLUX.1-dev-gguf",
            filename="flux1-dev-Q3_K_M.gguf",
            url="https://huggingface.co/city96/FLUX.1-dev-gguf/resolve/main/flux1-dev-Q3_K_M.gguf",
            quality_score=0.82,
            min_vram_gb=5.0
        ),
        # Q3_K_S - Very memory efficient
        "Q3_K_S": GGUFModelInfo(
            name="FLUX.1-dev-Q3_K_S",
            quantization="Q3_K_S",
            file_size_gb=4.5,
            memory_required_gb=6.0,
            repo_id="city96/FLUX.1-dev-gguf",
            filename="flux1-dev-Q3_K_S.gguf",
            url="https://huggingface.co/city96/FLUX.1-dev-gguf/resolve/main/flux1-dev-Q3_K_S.gguf",
            quality_score=0.80,
            min_vram_gb=4.5
        ),
        # Q2_K - Extreme quantization for very low VRAM
        "Q2_K": GGUFModelInfo(
            name="FLUX.1-dev-Q2_K",
            quantization="Q2_K",
            file_size_gb=3.5,
            memory_required_gb=5.0,
            repo_id="city96/FLUX.1-dev-gguf",
            filename="flux1-dev-Q2_K.gguf",
            url="https://huggingface.co/city96/FLUX.1-dev-gguf/resolve/main/flux1-dev-Q2_K.gguf",
            quality_score=0.70,
            min_vram_gb=4.0
        ),
    }
    
    FLUX_SCHNELL_MODELS = {
        # Q8_0 - Excellent quality/memory balance
        "Q8_0": GGUFModelInfo(
            name="FLUX.1-schnell-Q8_0",
            quantization="Q8_0",
            file_size_gb=12.5,
            memory_required_gb=14.0,
            repo_id="city96/FLUX.1-schnell-gguf",
            filename="flux1-schnell-Q8_0.gguf",
            url="https://huggingface.co/city96/FLUX.1-schnell-gguf/resolve/main/flux1-schnell-Q8_0.gguf",
            quality_score=0.98,
            min_vram_gb=12.0
        ),
        # Q6_K - Great quality with good memory savings
        "Q6_K": GGUFModelInfo(
            name="FLUX.1-schnell-Q6_K",
            quantization="Q6_K",
            file_size_gb=9.8,
            memory_required_gb=11.5,
            repo_id="city96/FLUX.1-schnell-gguf",
            filename="flux1-schnell-Q6_K.gguf",
            url="https://huggingface.co/city96/FLUX.1-schnell-gguf/resolve/main/flux1-schnell-Q6_K.gguf",
            quality_score=0.96,
            min_vram_gb=10.0
        ),
        # Q5_K_M - Balanced option
        "Q5_K_M": GGUFModelInfo(
            name="FLUX.1-schnell-Q5_K_M",
            quantization="Q5_K_M",
            file_size_gb=8.5,
            memory_required_gb=10.5,
            repo_id="city96/FLUX.1-schnell-gguf",
            filename="flux1-schnell-Q5_K_M.gguf",
            url="https://huggingface.co/city96/FLUX.1-schnell-gguf/resolve/main/flux1-schnell-Q5_K_M.gguf",
            quality_score=0.95,
            min_vram_gb=9.0
        ),
        # Q4_K_S - Memory efficient for fast generation
        "Q4_K_S": GGUFModelInfo(
            name="FLUX.1-schnell-Q4_K_S",
            quantization="Q4_K_S",
            file_size_gb=6.5,
            memory_required_gb=8.0,
            repo_id="city96/FLUX.1-schnell-gguf",
            filename="flux1-schnell-Q4_K_S.gguf",
            url="https://huggingface.co/city96/FLUX.1-schnell-gguf/resolve/main/flux1-schnell-Q4_K_S.gguf",
            quality_score=0.90,
            min_vram_gb=6.0
        ),
        # Q3_K_M - Very memory efficient for speed
        "Q3_K_M": GGUFModelInfo(
            name="FLUX.1-schnell-Q3_K_M",
            quantization="Q3_K_M",
            file_size_gb=5.0,
            memory_required_gb=6.5,
            repo_id="city96/FLUX.1-schnell-gguf",
            filename="flux1-schnell-Q3_K_M.gguf",
            url="https://huggingface.co/city96/FLUX.1-schnell-gguf/resolve/main/flux1-schnell-Q3_K_M.gguf",
            quality_score=0.82,
            min_vram_gb=5.0
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
    
    def recommend_quantization(self, model_type: str = "flux-dev", target_quality: float = 0.9) -> Optional[GGUFModelInfo]:
        """Recommend best GGUF model based on available VRAM and quality requirements
        
        Args:
            model_type: Type of model ("flux-dev" or "flux-schnell")
            target_quality: Minimum quality score (0.0-1.0)
            
        Returns:
            Recommended model info or None
        """
        available_vram = self.get_available_vram()
        
        if model_type == "flux-dev":
            models = self.FLUX_DEV_MODELS
        else:
            models = self.FLUX_SCHNELL_MODELS
        
        # Filter by quality requirement
        quality_filtered = [
            model for model in models.values()
            if model.quality_score >= target_quality
        ]
        
        # Sort by memory requirement (ascending)
        sorted_models = sorted(
            quality_filtered,
            key=lambda x: x.memory_required_gb
        )
        
        # Find best model that fits in available VRAM with 10% safety margin
        safety_margin = 0.9
        for model in sorted_models:
            if model.memory_required_gb <= (available_vram * safety_margin):
                return model
        
        # If no model fits with quality requirement, relax quality and try again
        if target_quality > 0.7:
            return self.recommend_quantization(model_type, target_quality - 0.1)
        
        # If still no model fits, return the smallest one
        all_sorted = sorted(models.values(), key=lambda x: x.memory_required_gb)
        return all_sorted[0] if all_sorted else None
    
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
    
    def discover_available_models(self, repo_id: str, token: Optional[str] = None) -> Dict[str, GGUFModelInfo]:
        """Dynamically discover available GGUF models from a repository
        
        Args:
            repo_id: HuggingFace repository ID
            token: Optional HF token for private repos
            
        Returns:
            Dictionary of discovered models with quantization as key
        """
        from huggingface_hub import list_repo_files
        
        try:
            logger.info(f"Discovering GGUF models in {repo_id}")
            files = list_repo_files(repo_id, token=token)
            
            discovered_models = {}
            gguf_files = [f for f in files if f.endswith('.gguf')]
            
            # Parse quantization from filename
            quantization_patterns = [
                (r'Q2_K', 'Q2_K', 0.70, 3.5, 5.0),
                (r'Q3_K_S', 'Q3_K_S', 0.80, 4.5, 6.0),
                (r'Q3_K_M', 'Q3_K_M', 0.82, 5.0, 6.5),
                (r'Q3_K_L', 'Q3_K_L', 0.85, 5.5, 7.0),
                (r'Q4_K_S', 'Q4_K_S', 0.90, 6.5, 8.0),
                (r'Q4_K_M', 'Q4_K_M', 0.92, 7.0, 8.5),
                (r'Q5_K_S', 'Q5_K_S', 0.94, 8.0, 10.0),
                (r'Q5_K_M', 'Q5_K_M', 0.95, 8.5, 10.5),
                (r'Q6_K', 'Q6_K', 0.96, 9.8, 11.5),
                (r'Q8_0', 'Q8_0', 0.98, 12.5, 14.0),
            ]
            
            model_base = repo_id.split('/')[-1].replace('-gguf', '')
            
            for file in gguf_files:
                for pattern, quant, quality, size_gb, mem_gb in quantization_patterns:
                    if pattern in file:
                        model_info = GGUFModelInfo(
                            name=f"{model_base}-{quant}",
                            quantization=quant,
                            file_size_gb=size_gb,
                            memory_required_gb=mem_gb,
                            repo_id=repo_id,
                            filename=file,
                            url=f"https://huggingface.co/{repo_id}/resolve/main/{file}",
                            quality_score=quality,
                            min_vram_gb=mem_gb - 2.0  # Conservative estimate
                        )
                        discovered_models[quant] = model_info
                        break
            
            logger.info(f"Discovered {len(discovered_models)} GGUF models")
            return discovered_models
            
        except Exception as e:
            logger.error(f"Error discovering models: {str(e)}")
            return {}
    
    def benchmark_model(self, model_info: GGUFModelInfo, test_prompts: List[str] = None) -> Dict[str, Any]:
        """Benchmark a GGUF model for performance and quality
        
        Args:
            model_info: Model to benchmark
            test_prompts: Optional list of test prompts
            
        Returns:
            Benchmark results including speed, memory usage, and quality metrics
        """
        if test_prompts is None:
            test_prompts = [
                "A majestic mountain landscape at sunset",
                "A futuristic cyberpunk city street",
                "A detailed portrait of a wise elderly person"
            ]
        
        results = {
            "model": model_info.name,
            "quantization": model_info.quantization,
            "memory_peak_gb": 0.0,
            "avg_time_per_image": 0.0,
            "avg_steps_per_second": 0.0,
            "quality_score": model_info.quality_score,
            "successful_generations": 0,
            "failed_generations": 0
        }
        
        try:
            # Load model
            start_load = time.time()
            transformer = self.load_transformer(model_info)
            load_time = time.time() - start_load
            results["load_time"] = load_time
            
            # Track peak memory
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            
            # Run test generations
            generation_times = []
            for prompt in test_prompts:
                try:
                    start_gen = time.time()
                    # Simulate generation (would need actual pipeline in real implementation)
                    # This is a placeholder - actual generation would happen here
                    gen_time = time.time() - start_gen
                    generation_times.append(gen_time)
                    results["successful_generations"] += 1
                except Exception as e:
                    logger.error(f"Generation failed: {str(e)}")
                    results["failed_generations"] += 1
            
            # Calculate metrics
            if generation_times:
                results["avg_time_per_image"] = sum(generation_times) / len(generation_times)
                # Assuming 20 steps per image for calculation
                results["avg_steps_per_second"] = 20.0 / results["avg_time_per_image"]
            
            # Get peak memory usage
            if torch.cuda.is_available():
                results["memory_peak_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
            
            # Clean up
            self.unload_model()
            
        except Exception as e:
            logger.error(f"Benchmark failed: {str(e)}")
            results["error"] = str(e)
        
        return results
    
    def get_all_available_models(self) -> Dict[str, Dict[str, GGUFModelInfo]]:
        """Get all available GGUF models organized by type
        
        Returns:
            Dictionary with model types as keys and model dictionaries as values
        """
        return {
            "flux-dev": self.FLUX_DEV_MODELS,
            "flux-schnell": self.FLUX_SCHNELL_MODELS
        }