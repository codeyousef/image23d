"""Production-ready FLUX pipeline combining all components.

This module provides a unified interface that combines all FLUX modules
into a production-ready pipeline with automatic optimization, error handling,
and comprehensive features.
"""

import torch
import gc
import logging
import time
from typing import Optional, Dict, Any, List, Union, Tuple
from PIL import Image
from pathlib import Path
from dataclasses import dataclass
import json
import hashlib
from datetime import datetime

# Import all FLUX components
from .flux_core import FluxGenerator, FluxGGUFGenerator
from .flux_guidance import AdvancedFluxGuidance, FluxSamplerOptimizer
from .flux_acceleration import AcceleratedFluxGenerator, HybridAccelerationPipeline
from .flux_prompts import FluxPromptOptimizer, PromptTemplate
from .flux_optimization import OptimizedFluxGenerator, MemoryOptimizer
from .flux_controlnet import FluxControlNetGenerator, ControlNetProcessor
from .flux_enhance import PostProcessingPipeline, StyleEnhancer, CompositeEnhancer

logger = logging.getLogger(__name__)


@dataclass
class GenerationRequest:
    """Unified generation request structure."""
    prompt: str
    negative_prompt: Optional[str] = None
    style: str = "photorealistic"
    quality: str = "high"  # low, medium, high, ultra
    width: int = 1024
    height: int = 1024
    num_images: int = 1
    seed: Optional[int] = None
    
    # Advanced options
    use_controlnet: bool = False
    control_image: Optional[Union[str, Image.Image]] = None
    control_type: str = "depth"
    control_strength: float = 1.0
    
    # Optimization options
    use_acceleration: bool = True
    acceleration_method: str = "auto"  # auto, hyperflux, fluxturbo, none
    compile_model: bool = True
    
    # Enhancement options
    enhance_output: bool = True
    upscale_factor: float = 1.0
    face_restoration: bool = False
    
    # Model options
    model_variant: str = "auto"  # auto, base, gguf_q8, gguf_q6, etc
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = {}
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            if isinstance(value, Image.Image):
                # Convert PIL Image to path or base64
                value = "<PIL.Image>"
            data[field] = value
        return data
    
    def get_cache_key(self) -> str:
        """Generate cache key for this request."""
        # Create deterministic key from parameters
        key_data = {
            "prompt": self.prompt,
            "style": self.style,
            "quality": self.quality,
            "size": f"{self.width}x{self.height}",
            "seed": self.seed,
            "model": self.model_variant
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()


@dataclass
class GenerationResult:
    """Result of a generation request."""
    images: List[Image.Image]
    metadata: Dict[str, Any]
    timing: Dict[str, float]
    success: bool = True
    error: Optional[str] = None
    
    def save(self, output_dir: Union[str, Path], prefix: str = "flux"):
        """Save results to directory."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        saved_paths = []
        for i, img in enumerate(self.images):
            filename = f"{prefix}_{timestamp}_{i}.png"
            path = output_dir / filename
            img.save(path, "PNG")
            saved_paths.append(str(path))
        
        # Save metadata
        metadata_path = output_dir / f"{prefix}_{timestamp}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump({
                "metadata": self.metadata,
                "timing": self.timing,
                "images": saved_paths
            }, f, indent=2)
        
        return saved_paths


class ProductionFluxPipeline:
    """Production-ready FLUX pipeline with all features integrated."""
    
    def __init__(self,
                 base_model: str = "black-forest-labs/FLUX.1-dev",
                 device: str = "cuda",
                 enable_caching: bool = True,
                 cache_dir: Optional[Path] = None):
        
        self.base_model = base_model
        self.device = device
        self.enable_caching = enable_caching
        self.cache_dir = cache_dir or Path("cache/flux_cache")
        
        # Component instances
        self.generators = {}
        self.prompt_optimizer = FluxPromptOptimizer()
        self.guidance = AdvancedFluxGuidance()
        self.post_processor = PostProcessingPipeline(device)
        self.memory_optimizer = MemoryOptimizer()
        
        # Performance tracking
        self.performance_stats = {
            "total_generations": 0,
            "total_time": 0,
            "average_time": 0,
            "cache_hits": 0
        }
        
        # Initialize default generator
        self._setup_default_generator()
    
    def _setup_default_generator(self):
        """Setup default generator based on available resources."""
        # Check available VRAM
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"Available VRAM: {vram_gb:.1f}GB")
            
            if vram_gb >= 24:
                # High VRAM - use full model with all optimizations
                self._setup_generator("base", FluxGenerator(self.base_model, self.device))
                self._setup_generator("optimized", OptimizedFluxGenerator(
                    self.generators["base"].pipe,
                    enable_compile=True
                ))
            elif vram_gb >= 12:
                # Medium VRAM - use Q8 GGUF
                self._setup_generator("gguf_q8", FluxGGUFGenerator(
                    quantization_level="Q8_0",
                    device=self.device
                ))
            else:
                # Low VRAM - use Q4 GGUF
                self._setup_generator("gguf_q4", FluxGGUFGenerator(
                    quantization_level="Q4_K_S",
                    device=self.device
                ))
        else:
            # CPU only
            logger.warning("No GPU available, using CPU (will be slow)")
            self._setup_generator("cpu", FluxGenerator(self.base_model, "cpu"))
    
    def _setup_generator(self, name: str, generator: Any):
        """Register a generator."""
        self.generators[name] = generator
        logger.info(f"Registered generator: {name}")
    
    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Main generation entry point with full pipeline."""
        
        start_time = time.time()
        timing = {}
        
        try:
            # Step 1: Check cache
            if self.enable_caching and request.seed is not None:
                cached = self._check_cache(request)
                if cached:
                    self.performance_stats["cache_hits"] += 1
                    return cached
            
            # Step 2: Select appropriate generator
            generator = self._select_generator(request)
            
            # Step 3: Optimize prompt
            prompt_start = time.time()
            optimized = self.prompt_optimizer.optimize_prompt(
                request.prompt,
                style=request.style,
                quality_preset=request.quality
            )
            request.prompt = optimized["prompt"]
            if not request.negative_prompt:
                request.negative_prompt = optimized["negative_prompt"]
            timing["prompt_optimization"] = time.time() - prompt_start
            
            # Step 4: Apply guidance settings
            guidance_config = self.guidance.get_guidance_config(request.style)
            
            # Step 5: Generate images
            gen_start = time.time()
            
            if request.use_controlnet and request.control_image:
                images = self._generate_with_controlnet(request, generator)
            else:
                images = self._generate_standard(request, generator, guidance_config)
            
            timing["generation"] = time.time() - gen_start
            
            # Step 6: Post-process if requested
            if request.enhance_output:
                enhance_start = time.time()
                images = self._enhance_images(images, request)
                timing["enhancement"] = time.time() - enhance_start
            
            # Step 7: Create result
            total_time = time.time() - start_time
            timing["total"] = total_time
            
            # Update stats
            self.performance_stats["total_generations"] += len(images)
            self.performance_stats["total_time"] += total_time
            self.performance_stats["average_time"] = (
                self.performance_stats["total_time"] / 
                self.performance_stats["total_generations"]
            )
            
            result = GenerationResult(
                images=images,
                metadata=self._create_metadata(request, optimized),
                timing=timing,
                success=True
            )
            
            # Cache result if applicable
            if self.enable_caching and request.seed is not None:
                self._cache_result(request, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return GenerationResult(
                images=[],
                metadata={},
                timing={"error": time.time() - start_time},
                success=False,
                error=str(e)
            )
    
    def _select_generator(self, request: GenerationRequest) -> Any:
        """Select appropriate generator based on request."""
        
        if request.model_variant != "auto":
            # User specified variant
            if request.model_variant in self.generators:
                return self.generators[request.model_variant]
            else:
                logger.warning(f"Requested variant {request.model_variant} not available")
        
        # Auto-select based on quality and available resources
        if request.quality == "ultra" and "optimized" in self.generators:
            return self.generators["optimized"]
        elif request.quality == "low" and "gguf_q4" in self.generators:
            return self.generators["gguf_q4"]
        elif "gguf_q8" in self.generators:
            return self.generators["gguf_q8"]
        elif "base" in self.generators:
            return self.generators["base"]
        else:
            # Return first available
            return next(iter(self.generators.values()))
    
    def _generate_standard(self,
                          request: GenerationRequest,
                          generator: Any,
                          guidance_config: Dict[str, float]) -> List[Image.Image]:
        """Standard generation without ControlNet."""
        
        # Determine steps based on quality and acceleration
        steps = self._get_steps_for_quality(request.quality, request.use_acceleration)
        
        # Check if generator supports acceleration
        if request.use_acceleration and hasattr(generator, 'ultra_fast_generation'):
            # Use accelerated generation
            images = []
            for i in range(request.num_images):
                seed = request.seed + i if request.seed else None
                img, _ = generator.ultra_fast_generation(
                    prompt=request.prompt,
                    negative_prompt=request.negative_prompt,
                    height=request.height,
                    width=request.width,
                    num_inference_steps=steps,
                    guidance_scale=guidance_config["distilled_cfg"],
                    seed=seed
                )
                images.append(img)
        else:
            # Standard generation
            if hasattr(generator, 'generate_optimized'):
                # Optimized generator
                images = generator.generate_optimized(
                    prompt=request.prompt,
                    negative_prompt=request.negative_prompt,
                    height=request.height,
                    width=request.width,
                    num_inference_steps=steps,
                    guidance_scale=guidance_config["distilled_cfg"],
                    seed=request.seed,
                    num_images=request.num_images
                )
            else:
                # Basic generator
                images = []
                for i in range(request.num_images):
                    seed = request.seed + i if request.seed else None
                    img = generator.generate_image(
                        prompt=request.prompt,
                        negative_prompt=request.negative_prompt,
                        height=request.height,
                        width=request.width,
                        num_inference_steps=steps,
                        guidance_scale=guidance_config["distilled_cfg"],
                        seed=seed
                    )
                    images.append(img)
        
        return images
    
    def _generate_with_controlnet(self,
                                 request: GenerationRequest,
                                 generator: Any) -> List[Image.Image]:
        """Generation with ControlNet guidance."""
        
        # Create ControlNet generator
        controlnet_gen = FluxControlNetGenerator(
            base_model=self.base_model,
            control_type=request.control_type,
            device=self.device
        )
        
        # Generate
        images = []
        for i in range(request.num_images):
            seed = request.seed + i if request.seed else None
            img = controlnet_gen.generate_controlled(
                prompt=request.prompt,
                control_image=request.control_image,
                negative_prompt=request.negative_prompt,
                height=request.height,
                width=request.width,
                guidance_scale=3.5,  # Adjust for ControlNet
                controlnet_conditioning_scale=request.control_strength,
                seed=seed
            )
            images.append(img)
        
        return images
    
    def _enhance_images(self,
                       images: List[Image.Image],
                       request: GenerationRequest) -> List[Image.Image]:
        """Apply post-processing enhancements."""
        
        # Get style-appropriate enhancement
        enhancer = CompositeEnhancer(self.post_processor)
        
        enhanced = []
        for img in images:
            if request.upscale_factor > 1.0:
                # Progressive upscale for quality
                img = enhancer.progressive_upscale(
                    img,
                    target_scale=request.upscale_factor,
                    stages=2 if request.upscale_factor > 2 else 1
                )
            else:
                # Just enhance without upscaling
                img = enhancer.smart_enhance(
                    img,
                    prompt=request.prompt,
                    auto_detect=True
                )
            
            enhanced.append(img)
        
        return enhanced
    
    def _get_steps_for_quality(self, quality: str, use_acceleration: bool) -> int:
        """Get optimal steps for quality level."""
        
        if use_acceleration:
            # Accelerated steps
            steps_map = {
                "low": 4,
                "medium": 8,
                "high": 12,
                "ultra": 16
            }
        else:
            # Standard steps
            steps_map = {
                "low": 15,
                "medium": 20,
                "high": 28,
                "ultra": 40
            }
        
        return steps_map.get(quality, 20)
    
    def _create_metadata(self, request: GenerationRequest, optimized: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive metadata."""
        return {
            "request": request.to_dict(),
            "optimization": {
                "token_count": optimized.get("token_count", -1),
                "prompt_type": optimized.get("prompt_type"),
                "guidance_adjustment": optimized.get("guidance_adjustment", 0)
            },
            "pipeline": {
                "base_model": self.base_model,
                "device": self.device,
                "generator_used": type(self._select_generator(request)).__name__
            },
            "timestamp": datetime.now().isoformat()
        }
    
    def _check_cache(self, request: GenerationRequest) -> Optional[GenerationResult]:
        """Check if result is cached."""
        cache_key = request.get_cache_key()
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            # Load cached result
            # Implementation depends on caching strategy
            pass
        
        return None
    
    def _cache_result(self, request: GenerationRequest, result: GenerationResult):
        """Cache generation result."""
        # Implementation depends on caching strategy
        pass
    
    def batch_generate(self,
                      requests: List[GenerationRequest],
                      parallel: bool = False) -> List[GenerationResult]:
        """Generate multiple requests."""
        
        results = []
        total_requests = len(requests)
        
        for i, request in enumerate(requests):
            logger.info(f"Processing request {i+1}/{total_requests}")
            result = self.generate(request)
            results.append(result)
            
            # Memory cleanup between batches
            if i % 5 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return results
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get detailed performance report."""
        memory_stats = self.memory_optimizer.get_memory_stats()
        
        return {
            "performance": self.performance_stats,
            "memory": memory_stats,
            "generators": list(self.generators.keys()),
            "capabilities": {
                "max_resolution": self._get_max_resolution(),
                "acceleration_available": any(
                    hasattr(g, 'ultra_fast_generation') 
                    for g in self.generators.values()
                ),
                "controlnet_available": True,  # Always available
                "enhancement_available": True
            }
        }
    
    def _get_max_resolution(self) -> Tuple[int, int]:
        """Get maximum supported resolution based on available memory."""
        if not torch.cuda.is_available():
            return (512, 512)
        
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        if vram_gb >= 24:
            return (2048, 2048)
        elif vram_gb >= 16:
            return (1536, 1536)
        elif vram_gb >= 12:
            return (1024, 1024)
        elif vram_gb >= 8:
            return (768, 768)
        else:
            return (512, 512)


# Convenience functions
def create_production_pipeline(model_variant: Optional[str] = None) -> ProductionFluxPipeline:
    """Create production pipeline with auto-configuration."""
    
    if model_variant:
        # Specific variant requested
        if "gguf" in model_variant.lower():
            base_model = "city96/FLUX.1-dev-gguf"
        else:
            base_model = "black-forest-labs/FLUX.1-dev"
    else:
        # Auto-detect best variant
        base_model = "black-forest-labs/FLUX.1-dev"
    
    pipeline = ProductionFluxPipeline(base_model)
    
    # Setup additional variants if requested
    if model_variant == "all":
        # Setup all available variants
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            if vram_gb >= 16:
                # Can handle multiple variants
                pipeline._setup_generator("gguf_q6", FluxGGUFGenerator(
                    quantization_level="Q6_K",
                    device="cuda"
                ))
                
                # Setup acceleration
                from .flux_acceleration import AcceleratedFluxGenerator
                pipeline._setup_generator("hyperflux", AcceleratedFluxGenerator(
                    acceleration_method="hyperflux"
                ))
    
    return pipeline


def quick_generate(prompt: str,
                  style: str = "photorealistic",
                  quality: str = "high",
                  **kwargs) -> Image.Image:
    """Quick generation with minimal setup."""
    
    pipeline = create_production_pipeline()
    
    request = GenerationRequest(
        prompt=prompt,
        style=style,
        quality=quality,
        **kwargs
    )
    
    result = pipeline.generate(request)
    
    if result.success and result.images:
        return result.images[0]
    else:
        raise RuntimeError(f"Generation failed: {result.error}")


# Example presets
class ProductionPresets:
    """Common production presets."""
    
    PRESETS = {
        "fast_preview": GenerationRequest(
            prompt="",
            quality="low",
            width=512,
            height=512,
            use_acceleration=True,
            enhance_output=False
        ),
        "standard_1k": GenerationRequest(
            prompt="",
            quality="high",
            width=1024,
            height=1024,
            use_acceleration=True,
            enhance_output=True,
            upscale_factor=1.0
        ),
        "high_quality_2k": GenerationRequest(
            prompt="",
            quality="ultra",
            width=1024,
            height=1024,
            use_acceleration=False,
            enhance_output=True,
            upscale_factor=2.0
        ),
        "3d_asset": GenerationRequest(
            prompt="",
            style="3d_render",
            quality="high",
            width=1024,
            height=1024,
            use_controlnet=True,
            control_type="depth",
            enhance_output=True
        )
    }
    
    @classmethod
    def get_preset(cls, name: str, prompt: str, **overrides) -> GenerationRequest:
        """Get preset with prompt and overrides."""
        if name not in cls.PRESETS:
            raise ValueError(f"Unknown preset: {name}")
        
        preset = cls.PRESETS[name]
        # Create copy with prompt
        request = GenerationRequest(
            prompt=prompt,
            negative_prompt=preset.negative_prompt,
            style=preset.style,
            quality=preset.quality,
            width=preset.width,
            height=preset.height,
            num_images=preset.num_images,
            seed=preset.seed,
            use_controlnet=preset.use_controlnet,
            control_image=preset.control_image,
            control_type=preset.control_type,
            control_strength=preset.control_strength,
            use_acceleration=preset.use_acceleration,
            acceleration_method=preset.acceleration_method,
            compile_model=preset.compile_model,
            enhance_output=preset.enhance_output,
            upscale_factor=preset.upscale_factor,
            face_restoration=preset.face_restoration,
            model_variant=preset.model_variant
        )
        
        # Apply overrides
        for key, value in overrides.items():
            if hasattr(request, key):
                setattr(request, key, value)
        
        return request