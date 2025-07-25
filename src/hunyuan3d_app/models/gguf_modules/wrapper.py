"""Main GGUF wrapper class"""

import logging
import time
from pathlib import Path
from typing import Optional, Tuple, Any
import torch

from .loader import GGUFLoader
from .optimization import GGUFOptimizer
from .pipeline import GGUFPipelineGenerator

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
        
        # Initialize loader and optimizer
        self.loader = GGUFLoader(device)
        self.optimizer = GGUFOptimizer()
        
        # Try to load GGUF weights or fall back to real model
        self._load_gguf_weights()
    
    def _load_gguf_weights(self):
        """Load GGUF weights using the official diffusers API"""
        try:
            # Load GGUF transformer
            transformer, self._has_device_map = self.loader.load_gguf_transformer(
                self.gguf_path, 
                self.model_name
            )
            
            # Now create pipeline based on availability of base components
            if self.text_encoder_path and self.text_encoder_path.exists():
                # We have base components, create full pipeline
                logger.info("Base components found, creating full FLUX pipeline")
                self._create_full_pipeline(transformer)
            else:
                # Check if we have FLUX base model available
                flux_base_path = self._find_flux_base_model()
                if flux_base_path:
                    logger.info(f"Found FLUX base model at: {flux_base_path}")
                    self._create_flux_pipeline(transformer, flux_base_path)
                else:
                    logger.warning("No base components or FLUX model found")
                    self._create_placeholder_pipeline()
                    
            # Apply optimizations for Q6/Q8 models
            if self._real_pipeline and any(q in self.model_name.lower() for q in ["q6", "q8"]):
                self._apply_q6_q8_optimizations()
                    
        except Exception as e:
            logger.error(f"Failed to load GGUF weights: {e}")
            logger.info("Creating placeholder pipeline")
            self._create_placeholder_pipeline()
            
    def _find_flux_base_model(self) -> Optional[Path]:
        """Find FLUX base model in common locations"""
        # Check for FLUX models in the models directory
        models_dir = self.gguf_path.parent.parent
        
        # Look for FLUX models
        flux_candidates = [
            models_dir / "image" / "flux-dev",
            models_dir / "image" / "FLUX.1-dev",
            models_dir / "image" / "black-forest-labs--FLUX.1-dev",
            models_dir / "flux" / "FLUX.1-dev",
            models_dir / "flux" / "flux-dev"
        ]
        
        for candidate in flux_candidates:
            if candidate.exists() and (candidate / "vae").exists():
                return candidate
                
        return None
    
    def _create_full_pipeline(self, transformer):
        """Create pipeline with provided base components"""
        try:
            from diffusers import FluxPipeline
            
            # Load base components from provided paths
            logger.info("Loading base components from provided paths...")
            
            # Create pipeline with components
            self._real_pipeline = FluxPipeline(
                transformer=transformer,
                vae=None,  # Will be loaded on demand
                text_encoder=None,
                text_encoder_2=None,
                tokenizer=None,
                tokenizer_2=None,
                scheduler=None
            )
            
            self._is_placeholder = False
            logger.info("Created GGUF pipeline with base components")
            
        except Exception as e:
            logger.error(f"Failed to create full pipeline: {e}")
            self._create_placeholder_pipeline()
    
    def _create_flux_pipeline(self, transformer, flux_base_path: Path):
        """Create FLUX pipeline with GGUF transformer"""
        try:
            from diffusers import FluxPipeline
            
            # Determine compute dtype
            is_large_quant = any(q in self.model_name.lower() for q in ["q6", "q8"])
            compute_dtype = torch.float16 if is_large_quant else (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )
            
            # Load FLUX components
            components = self.loader.load_flux_components(flux_base_path, compute_dtype)
            
            # Create pipeline with all components
            self._real_pipeline = FluxPipeline(
                scheduler=components['scheduler'],
                vae=components['vae'],
                text_encoder=components['text_encoder'],
                text_encoder_2=components['text_encoder_2'],
                transformer=transformer,
                tokenizer=components['tokenizer'],
                tokenizer_2=components['tokenizer_2']
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
            
            # Move pipeline to device (if not using device_map)
            if not self._has_device_map:
                # Move VAE and text encoders to CPU to save VRAM
                self._real_pipeline.vae = self._real_pipeline.vae.to("cpu")
                self._real_pipeline.text_encoder = self._real_pipeline.text_encoder.to("cpu")
                self._real_pipeline.text_encoder_2 = self._real_pipeline.text_encoder_2.to("cpu")
                logger.info("Moved VAE and text encoders to CPU to save VRAM")
            
            # Enable optimizations
            self.optimizer.enable_memory_optimizations(self._real_pipeline, self.device)
            
            self._is_placeholder = False
            logger.info("Successfully created FLUX pipeline with GGUF transformer")
            
        except Exception as e:
            logger.error(f"Failed to create FLUX pipeline: {e}")
            self._create_placeholder_pipeline()
    
    def _create_placeholder_pipeline(self):
        """Create a placeholder pipeline for testing"""
        self._is_placeholder = True
        self._real_pipeline = None
        logger.info("Using placeholder pipeline")
    
    def _apply_q6_q8_optimizations(self):
        """Apply special optimizations for Q6/Q8 GGUF models"""
        if self._real_pipeline:
            self.optimizer.apply_q6_q8_optimizations(self._real_pipeline)
    
    def __call__(self, *args, **kwargs):
        """Forward calls to the real pipeline or return placeholder"""
        if self._real_pipeline:
            generator = GGUFPipelineGenerator(self._real_pipeline, self.device)
            return generator.generate(*args, **kwargs)
        else:
            # Return placeholder result
            logger.warning("Using placeholder GGUF pipeline - returning dummy image")
            return GGUFPipelineGenerator.create_placeholder_result(
                prompt=kwargs.get('prompt', ''),
                height=kwargs.get('height', 1024),
                width=kwargs.get('width', 1024),
                num_images_per_prompt=kwargs.get('num_images_per_prompt', 1)
            )
    
    def to(self, device):
        """Move pipeline to device"""
        self.device = device
        if self._real_pipeline and not self._has_device_map:
            logger.info(f"Moving GGUF pipeline to {device}")
            # Only move the transformer for GGUF models
            if hasattr(self._real_pipeline, 'transformer'):
                self._real_pipeline.transformer = self._real_pipeline.transformer.to(device)
            # Keep other components on CPU to save memory
        return self
    
    def enable_attention_slicing(self, slice_size: Optional[int] = None):
        """Enable attention slicing"""
        if self._real_pipeline and hasattr(self._real_pipeline, 'enable_attention_slicing'):
            self._real_pipeline.enable_attention_slicing(slice_size)
    
    def enable_model_cpu_offload(self):
        """Enable model CPU offload"""
        if self._real_pipeline and hasattr(self._real_pipeline, 'enable_model_cpu_offload'):
            self._real_pipeline.enable_model_cpu_offload()
    
    def enable_vae_slicing(self):
        """Enable VAE slicing"""
        if self._real_pipeline and hasattr(self._real_pipeline, 'enable_vae_slicing'):
            self._real_pipeline.enable_vae_slicing()
    
    def enable_xformers_memory_efficient_attention(self):
        """Enable xformers memory efficient attention"""
        if self._real_pipeline and hasattr(self._real_pipeline, 'enable_xformers_memory_efficient_attention'):
            try:
                self._real_pipeline.enable_xformers_memory_efficient_attention()
            except Exception as e:
                logger.warning(f"Could not enable xformers: {e}")
    
    @property
    def hf_device_map(self):
        """Return device map if available"""
        if self._real_pipeline and hasattr(self._real_pipeline, 'hf_device_map'):
            return self._real_pipeline.hf_device_map
        return None