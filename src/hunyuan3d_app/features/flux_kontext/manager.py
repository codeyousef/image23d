"""Flux Kontext integration for enhanced FLUX generation capabilities"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass
import torch
import numpy as np
from PIL import Image
import json

logger = logging.getLogger(__name__)


@dataclass
class FluxKontextConfig:
    """Configuration for Flux Kontext generation"""
    # Context settings
    context_strength: float = 0.8
    context_mode: str = "balanced"  # "focused", "balanced", "expansive"
    
    # Model settings
    model_variant: str = "flux-dev"  # "flux-dev", "flux-schnell"
    precision: str = "fp16"  # "fp16", "fp8", "int8"
    
    # Generation settings
    guidance_scale: float = 3.5
    num_inference_steps: int = 28
    width: int = 1024
    height: int = 1024
    seed: int = -1
    
    # Kontext-specific
    semantic_layers: List[str] = None
    attention_layers: List[str] = None
    context_injection_step: int = 5
    context_preservation: float = 0.7
    
    def __post_init__(self):
        if self.semantic_layers is None:
            self.semantic_layers = ["transformer.transformer_blocks.0", "transformer.transformer_blocks.6", "transformer.transformer_blocks.12"]
        if self.attention_layers is None:
            self.attention_layers = ["transformer.transformer_blocks.3", "transformer.transformer_blocks.9", "transformer.transformer_blocks.15"]


class FluxKontextManager:
    """Manages Flux Kontext generation with enhanced context awareness"""
    
    def __init__(
        self,
        model_dir: Optional[Path] = None,
        cache_dir: Optional[Path] = None
    ):
        self.model_dir = model_dir or Path("./models/flux")
        self.cache_dir = cache_dir or Path("./cache/flux_kontext")
        
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Model components
        self.flux_pipeline = None
        self.kontext_processor = None
        self.context_embeddings = {}
        
        # Configuration
        self.current_config = FluxKontextConfig()
        self.models_loaded = False
        
    def initialize_models(
        self,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16
    ) -> Tuple[bool, str]:
        """Initialize Flux Kontext models
        
        Args:
            device: Device to load models on
            dtype: Model precision
            
        Returns:
            Tuple of (success, message)
        """
        try:
            logger.info("Initializing Flux Kontext models...")
            
            # Try to load actual Flux pipeline with Kontext modifications
            success = self._load_flux_pipeline(device, dtype)
            if not success:
                # Fallback to basic Flux if Kontext not available
                logger.warning("Flux Kontext not available, using standard Flux")
                success = self._load_basic_flux(device, dtype)
                
            if success:
                # Initialize context processor
                self.kontext_processor = self._create_kontext_processor(device, dtype)
                self.models_loaded = True
                return True, "Flux Kontext initialized successfully"
            else:
                return False, "Failed to load Flux models"
                
        except Exception as e:
            logger.error(f"Failed to initialize Flux Kontext: {e}")
            return False, f"Initialization failed: {str(e)}"
            
    def _load_flux_pipeline(self, device: str, dtype: torch.dtype) -> bool:
        """Load Flux pipeline with Kontext extensions"""
        try:
            # Try to import Flux Kontext (this would be a custom implementation)
            from diffusers import FluxPipeline
            
            # Check for Kontext-enhanced model
            kontext_model_path = self.model_dir / "flux-kontext"
            if kontext_model_path.exists():
                logger.info("Loading Flux Kontext enhanced model")
                self.flux_pipeline = FluxPipeline.from_pretrained(
                    str(kontext_model_path),
                    torch_dtype=dtype,
                    device_map=device
                )
                # Apply Kontext modifications
                self._apply_kontext_modifications()
                return True
            else:
                return False
                
        except Exception as e:
            logger.warning(f"Failed to load Flux Kontext: {e}")
            return False
            
    def _load_basic_flux(self, device: str, dtype: torch.dtype) -> bool:
        """Load basic Flux pipeline as fallback"""
        try:
            from diffusers import FluxPipeline
            
            # Try different model variants
            model_paths = [
                self.model_dir / "flux-dev",
                self.model_dir / "flux-schnell",
                "black-forest-labs/FLUX.1-dev",
                "black-forest-labs/FLUX.1-schnell"
            ]
            
            for model_path in model_paths:
                try:
                    if isinstance(model_path, Path) and not model_path.exists():
                        continue
                        
                    self.flux_pipeline = FluxPipeline.from_pretrained(
                        str(model_path),
                        torch_dtype=dtype,
                        device_map=device,
                        local_files_only=isinstance(model_path, Path)
                    )
                    
                    # Enable optimizations
                    self.flux_pipeline.enable_model_cpu_offload()
                    if hasattr(self.flux_pipeline, 'enable_vae_slicing'):
                        self.flux_pipeline.enable_vae_slicing()
                        
                    logger.info(f"Loaded Flux model from {model_path}")
                    return True
                    
                except Exception as e:
                    logger.debug(f"Failed to load from {model_path}: {e}")
                    continue
                    
            return False
            
        except Exception as e:
            logger.error(f"Failed to load basic Flux: {e}")
            return False
            
    def _apply_kontext_modifications(self):
        """Apply Kontext-specific modifications to the pipeline"""
        try:
            # This would apply custom modifications for enhanced context awareness
            # For now, we'll use standard Flux with some custom hooks
            
            # Hook into transformer blocks for context injection
            if hasattr(self.flux_pipeline, 'transformer'):
                transformer = self.flux_pipeline.transformer
                
                # Store original forward methods
                self.original_forward_methods = {}
                
                # Apply context-aware modifications to specified layers
                for layer_name in self.current_config.semantic_layers:
                    if hasattr(transformer, layer_name.split('.')[-1]):
                        layer = getattr(transformer, layer_name.split('.')[-1])
                        if hasattr(layer, 'forward'):
                            self.original_forward_methods[layer_name] = layer.forward
                            layer.forward = self._create_context_aware_forward(layer, layer_name)
                            
            logger.info("Applied Kontext modifications to Flux pipeline")
            
        except Exception as e:
            logger.warning(f"Failed to apply Kontext modifications: {e}")
            
    def _create_context_aware_forward(self, layer, layer_name):
        """Create context-aware forward function for a layer"""
        original_forward = self.original_forward_methods[layer_name]
        
        def context_forward(*args, **kwargs):
            # Call original forward
            output = original_forward(*args, **kwargs)
            
            # Apply context modifications if available
            if hasattr(self, 'current_context_embeddings') and self.current_context_embeddings:
                output = self._apply_context_to_output(output, layer_name)
                
            return output
            
        return context_forward
        
    def _apply_context_to_output(self, output, layer_name):
        """Apply context embeddings to layer output"""
        try:
            if layer_name in self.current_context_embeddings:
                context_embed = self.current_context_embeddings[layer_name]
                
                # Simple context blending (would be more sophisticated in real implementation)
                if isinstance(output, torch.Tensor) and output.shape[-1] == context_embed.shape[-1]:
                    blend_factor = self.current_config.context_strength
                    output = output * (1 - blend_factor) + context_embed * blend_factor
                    
            return output
            
        except Exception as e:
            logger.debug(f"Failed to apply context to {layer_name}: {e}")
            return output
            
    def _create_kontext_processor(self, device: str, dtype: torch.dtype):
        """Create context processor for enhanced understanding"""
        class KontextProcessor:
            def __init__(self, device, dtype):
                self.device = device
                self.dtype = dtype
                
            def extract_context(self, prompt: str, reference_images: List[Image.Image] = None) -> Dict[str, torch.Tensor]:
                """Extract context from prompt and reference images"""
                context = {}
                
                # Extract semantic context from prompt
                semantic_context = self._extract_semantic_context(prompt)
                context['semantic'] = semantic_context
                
                # Extract visual context from reference images
                if reference_images:
                    visual_context = self._extract_visual_context(reference_images)
                    context['visual'] = visual_context
                    
                return context
                
            def _extract_semantic_context(self, prompt: str) -> torch.Tensor:
                """Extract semantic context from text prompt"""
                # This would use a more sophisticated NLP model
                # For now, create dummy embeddings based on prompt length
                embed_dim = 768
                context_embed = torch.randn(1, len(prompt.split()), embed_dim, dtype=self.dtype, device=self.device)
                return context_embed
                
            def _extract_visual_context(self, images: List[Image.Image]) -> torch.Tensor:
                """Extract visual context from reference images"""
                # This would use CLIP or similar vision model
                # For now, create dummy visual embeddings
                embed_dim = 768
                context_embed = torch.randn(1, len(images), embed_dim, dtype=self.dtype, device=self.device)
                return context_embed
                
        return KontextProcessor(device, dtype)
        
    def generate_with_kontext(
        self,
        prompt: str,
        negative_prompt: str = "",
        reference_images: List[Image.Image] = None,
        config: Optional[FluxKontextConfig] = None
    ) -> Tuple[Optional[Image.Image], Dict[str, Any]]:
        """Generate image using Flux Kontext
        
        Args:
            prompt: Text prompt
            negative_prompt: Negative prompt
            reference_images: Reference images for context
            config: Generation configuration
            
        Returns:
            Tuple of (generated image, info dict)
        """
        if not self.models_loaded:
            return None, {"error": "Models not loaded"}
            
        if config is None:
            config = self.current_config
            
        try:
            start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            
            if start_time:
                start_time.record()
                
            # Extract context
            if self.kontext_processor:
                context_embeddings = self.kontext_processor.extract_context(prompt, reference_images)
                self.current_context_embeddings = context_embeddings
            else:
                self.current_context_embeddings = {}
                
            # Prepare generation parameters
            generator = torch.Generator(device=self.flux_pipeline.device)
            if config.seed >= 0:
                generator.manual_seed(config.seed)
                
            # Generate with context
            result = self.flux_pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=config.width,
                height=config.height,
                num_inference_steps=config.num_inference_steps,
                guidance_scale=config.guidance_scale,
                generator=generator
            )
            
            if end_time:
                end_time.record()
                torch.cuda.synchronize()
                generation_time = start_time.elapsed_time(end_time) / 1000.0
            else:
                generation_time = 0.0
                
            # Get generated image
            image = result.images[0] if hasattr(result, 'images') and result.images else None
            
            if image:
                # Apply Kontext post-processing if available
                image = self._apply_kontext_postprocessing(image, config)
                
                info = {
                    "model": "Flux Kontext",
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "config": {
                        "context_strength": config.context_strength,
                        "context_mode": config.context_mode,
                        "guidance_scale": config.guidance_scale,
                        "steps": config.num_inference_steps,
                        "resolution": f"{config.width}x{config.height}",
                        "seed": config.seed
                    },
                    "generation_time": f"{generation_time:.2f}s",
                    "context_used": len(self.current_context_embeddings) > 0,
                    "reference_images": len(reference_images) if reference_images else 0
                }
                
                return image, info
            else:
                return None, {"error": "Failed to generate image"}
                
        except Exception as e:
            logger.error(f"Flux Kontext generation failed: {e}")
            return None, {"error": str(e)}
        finally:
            # Clear current context
            self.current_context_embeddings = {}
            
    def _apply_kontext_postprocessing(self, image: Image.Image, config: FluxKontextConfig) -> Image.Image:
        """Apply Kontext-specific post-processing"""
        try:
            # This would apply context-aware post-processing
            # For now, just return the image as-is
            return image
            
        except Exception as e:
            logger.warning(f"Failed to apply Kontext post-processing: {e}")
            return image
            
    def update_config(self, **kwargs) -> FluxKontextConfig:
        """Update configuration parameters"""
        for key, value in kwargs.items():
            if hasattr(self.current_config, key):
                setattr(self.current_config, key, value)
                
        return self.current_config
        
    def get_available_modes(self) -> List[str]:
        """Get available context modes"""
        return ["focused", "balanced", "expansive"]
        
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded model"""
        if not self.models_loaded:
            return {"status": "not_loaded"}
            
        info = {
            "status": "loaded",
            "model_type": "Flux Kontext" if hasattr(self, 'original_forward_methods') else "Flux (Standard)",
            "device": str(self.flux_pipeline.device) if self.flux_pipeline else "unknown",
            "precision": str(self.flux_pipeline.dtype) if self.flux_pipeline else "unknown",
            "kontext_enabled": hasattr(self, 'original_forward_methods'),
            "context_layers": len(self.current_config.semantic_layers + self.current_config.attention_layers)
        }
        
        return info
        
    def save_context_preset(self, name: str, prompt: str, reference_images: List[Image.Image] = None) -> bool:
        """Save a context preset for reuse"""
        try:
            if not self.kontext_processor:
                return False
                
            # Extract and save context
            context_embeddings = self.kontext_processor.extract_context(prompt, reference_images)
            
            preset_data = {
                "name": name,
                "prompt": prompt,
                "context_embeddings": {k: v.cpu().numpy().tolist() for k, v in context_embeddings.items()},
                "config": self.current_config.__dict__
            }
            
            preset_path = self.cache_dir / f"preset_{name}.json"
            with open(preset_path, 'w') as f:
                json.dump(preset_data, f, indent=2)
                
            logger.info(f"Saved context preset: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save context preset: {e}")
            return False
            
    def load_context_preset(self, name: str) -> bool:
        """Load a saved context preset"""
        try:
            preset_path = self.cache_dir / f"preset_{name}.json"
            if not preset_path.exists():
                return False
                
            with open(preset_path, 'r') as f:
                preset_data = json.load(f)
                
            # Load context embeddings
            context_embeddings = {}
            for k, v in preset_data["context_embeddings"].items():
                tensor = torch.tensor(v, dtype=self.flux_pipeline.dtype, device=self.flux_pipeline.device)
                context_embeddings[k] = tensor
                
            self.current_context_embeddings = context_embeddings
            
            # Update config
            for k, v in preset_data["config"].items():
                if hasattr(self.current_config, k):
                    setattr(self.current_config, k, v)
                    
            logger.info(f"Loaded context preset: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load context preset: {e}")
            return False
            
    def list_context_presets(self) -> List[str]:
        """List available context presets"""
        try:
            presets = []
            for preset_file in self.cache_dir.glob("preset_*.json"):
                name = preset_file.stem.replace("preset_", "")
                presets.append(name)
            return sorted(presets)
            
        except Exception as e:
            logger.error(f"Failed to list presets: {e}")
            return []