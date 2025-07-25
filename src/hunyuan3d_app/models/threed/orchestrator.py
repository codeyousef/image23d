"""Multi-model orchestrator for 3D generation

Implements the orchestration framework from Managing Multiple AI Models guide:
- Dynamic model switching based on task requirements
- Memory-aware scheduling
- Model compatibility checks
- Pipeline composition
"""

import logging
import traceback
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable, Tuple
import torch
from PIL import Image

from .base import ModelType3D, QualityPreset3D, QUALITY_PRESETS_3D
from .memory import ThreeDMemoryManager, ModelSwapper
from .hunyuan3d import HunYuan3DModel, HunYuan3DConfig

logger = logging.getLogger(__name__)


class ModelCapability(Enum):
    """Capabilities of different 3D models"""
    TEXT_TO_3D = "text_to_3d"
    IMAGE_TO_3D = "image_to_3d"
    MULTIVIEW_GENERATION = "multiview_generation"
    SPARSE_RECONSTRUCTION = "sparse_reconstruction"
    DENSE_RECONSTRUCTION = "dense_reconstruction"
    TEXTURE_SYNTHESIS = "texture_synthesis"
    PBR_MATERIALS = "pbr_materials"
    HIGH_RESOLUTION = "high_resolution"
    FAST_GENERATION = "fast_generation"
    GGUF_SUPPORT = "gguf_support"


@dataclass
class ModelProfile:
    """Profile for a 3D model"""
    model_type: ModelType3D
    capabilities: List[ModelCapability]
    memory_requirements: Dict[str, float]  # Quality preset -> GB
    performance_metrics: Dict[str, float]  # Quality preset -> seconds
    supported_formats: List[str]
    max_resolution: int
    supports_quantization: bool
    quantization_levels: List[str]


# Model profiles based on the guides
MODEL_PROFILES = {
    ModelType3D.HUNYUAN3D_21: ModelProfile(
        model_type=ModelType3D.HUNYUAN3D_21,
        capabilities=[
            ModelCapability.IMAGE_TO_3D,
            ModelCapability.MULTIVIEW_GENERATION,
            ModelCapability.SPARSE_RECONSTRUCTION,
            ModelCapability.TEXTURE_SYNTHESIS,
            ModelCapability.PBR_MATERIALS,
            ModelCapability.HIGH_RESOLUTION,
            ModelCapability.GGUF_SUPPORT
        ],
        memory_requirements={
            "draft": 6.0,
            "standard": 8.0,
            "high": 12.0,
            "ultra": 16.0
        },
        performance_metrics={
            "draft": 30.0,
            "standard": 60.0,
            "high": 120.0,
            "ultra": 240.0
        },
        supported_formats=["glb", "obj", "ply", "stl", "fbx", "usdz"],
        max_resolution=1024,
        supports_quantization=True,
        quantization_levels=["Q8_0", "Q6_K", "Q5_K_S", "Q4_K_M"]
    ),
    ModelType3D.HUNYUAN3D_2MINI: ModelProfile(
        model_type=ModelType3D.HUNYUAN3D_2MINI,
        capabilities=[
            ModelCapability.IMAGE_TO_3D,
            ModelCapability.MULTIVIEW_GENERATION,
            ModelCapability.SPARSE_RECONSTRUCTION,
            ModelCapability.TEXTURE_SYNTHESIS,
            ModelCapability.FAST_GENERATION,
            ModelCapability.GGUF_SUPPORT,
            # Note: Mini model has basic PBR support at lower quality
            # According to guide, it's for "rapid prototyping"
            ModelCapability.PBR_MATERIALS  # Added based on guide's multi-variant support
        ],
        memory_requirements={
            # According to guide: Mini variant for rapid prototyping (4× faster)
            "draft": 2.0,    # Reduced for true rapid prototyping
            "standard": 3.0,  # Reduced from 4.0
            "high": 4.0,     # Reduced from 6.0
            "ultra": 6.0     # Reduced from 8.0
        },
        performance_metrics={
            # 4× faster according to guide
            "draft": 7.5,    # 30/4 = 7.5 seconds
            "standard": 15.0, # 60/4 = 15 seconds  
            "high": 30.0,    # 120/4 = 30 seconds
            "ultra": 60.0    # 240/4 = 60 seconds
        },
        supported_formats=["glb", "obj", "ply"],
        max_resolution=512,
        supports_quantization=True,
        quantization_levels=["Q8_0", "Q6_K", "Q5_K_S", "Q4_K_M"]
    ),
    ModelType3D.HI3DGEN: ModelProfile(
        model_type=ModelType3D.HI3DGEN,
        capabilities=[
            ModelCapability.IMAGE_TO_3D,
            ModelCapability.DENSE_RECONSTRUCTION,
            ModelCapability.TEXTURE_SYNTHESIS,
            ModelCapability.HIGH_RESOLUTION
        ],
        memory_requirements={
            "draft": 8.0,
            "standard": 10.0,
            "high": 14.0,
            "ultra": 18.0
        },
        performance_metrics={
            "draft": 45.0,
            "standard": 90.0,
            "high": 180.0,
            "ultra": 300.0
        },
        supported_formats=["glb", "obj", "ply", "fbx"],
        max_resolution=2048,
        supports_quantization=False,
        quantization_levels=[]
    ),
    ModelType3D.SPARC3D: ModelProfile(
        model_type=ModelType3D.SPARC3D,
        capabilities=[
            ModelCapability.IMAGE_TO_3D,
            ModelCapability.SPARSE_RECONSTRUCTION,
            ModelCapability.HIGH_RESOLUTION,
            ModelCapability.TEXTURE_SYNTHESIS
        ],
        memory_requirements={
            "draft": 10.0,
            "standard": 12.0,
            "high": 16.0,
            "ultra": 20.0
        },
        performance_metrics={
            "draft": 60.0,
            "standard": 120.0,
            "high": 240.0,
            "ultra": 360.0
        },
        supported_formats=["glb", "obj", "ply"],
        max_resolution=1024,
        supports_quantization=False,
        quantization_levels=[]
    )
}


class TaskRequirements:
    """Requirements for a 3D generation task"""
    
    def __init__(
        self,
        input_type: str = "image",  # "image" or "text"
        quality_preset: str = "standard",
        output_format: str = "glb",
        required_capabilities: Optional[List[ModelCapability]] = None,
        max_generation_time: Optional[float] = None,
        max_memory_gb: Optional[float] = None,
        preferred_model: Optional[str] = None  # Preferred model to use
    ):
        self.input_type = input_type
        self.quality_preset = quality_preset
        self.output_format = output_format
        self.required_capabilities = required_capabilities or []
        self.max_generation_time = max_generation_time
        self.max_memory_gb = max_memory_gb
        self.preferred_model = preferred_model
        
        # Add implicit capabilities
        if input_type == "image":
            if ModelCapability.IMAGE_TO_3D not in self.required_capabilities:
                self.required_capabilities.append(ModelCapability.IMAGE_TO_3D)
        elif input_type == "text":
            if ModelCapability.TEXT_TO_3D not in self.required_capabilities:
                self.required_capabilities.append(ModelCapability.TEXT_TO_3D)


class ModelSelector:
    """Selects optimal model based on requirements"""
    
    def __init__(self, memory_manager: ThreeDMemoryManager):
        self.memory_manager = memory_manager
        
    def select_model(
        self,
        requirements: TaskRequirements,
        available_models: List[ModelType3D]
    ) -> Tuple[Optional[ModelType3D], str]:
        """Select best model for task"""
        
        logger.info(f"Selecting model for requirements: capabilities={[c.value for c in requirements.required_capabilities]}, "
                   f"quality={requirements.quality_preset}, format={requirements.output_format}, "
                   f"preferred={requirements.preferred_model}")
        logger.info(f"Available models: {[m.value for m in available_models]}")
        
        candidates = []
        reasons = []
        
        # If preferred model is specified, check it first
        if requirements.preferred_model:
            preferred_type = None
            for model_type in available_models:
                if model_type.value == requirements.preferred_model:
                    preferred_type = model_type
                    break
                    
            if preferred_type:
                # Move preferred model to front of list
                available_models = [preferred_type] + [m for m in available_models if m != preferred_type]
            else:
                logger.warning(f"Preferred model {requirements.preferred_model} not available")
        
        for model_type in available_models:
            profile = MODEL_PROFILES.get(model_type)
            if not profile:
                reasons.append(f"{model_type.value}: No profile found")
                continue
                
            # Check capabilities
            missing_caps = [cap for cap in requirements.required_capabilities if cap not in profile.capabilities]
            if missing_caps:
                reasons.append(f"{model_type.value}: Missing capabilities {[c.value for c in missing_caps]}")
                continue
                
            # Check format support
            if requirements.output_format not in profile.supported_formats:
                reasons.append(f"{model_type.value}: Format '{requirements.output_format}' not supported")
                continue
                
            # Check memory requirements
            memory_needed = profile.memory_requirements.get(requirements.quality_preset, 10.0)
            if requirements.max_memory_gb and memory_needed > requirements.max_memory_gb:
                continue
                
            # Check if enough memory available
            available_memory = self.memory_manager.get_available_memory()
            if memory_needed > available_memory:
                # Check if can use quantization
                if profile.supports_quantization:
                    # Try different quantization levels
                    quantization_factors = {
                        "Q8_0": 0.5,    # 50% of FP16
                        "Q6_K": 0.375,  # 37.5% of FP16
                        "Q5_K_S": 0.325, # 32.5% of FP16
                        "Q4_K_M": 0.25   # 25% of FP16
                    }
                    
                    # Find best quantization level for available memory
                    can_fit = False
                    for q_level, factor in quantization_factors.items():
                        adjusted_memory = memory_needed * factor
                        if adjusted_memory <= available_memory:
                            can_fit = True
                            memory_needed = adjusted_memory
                            break
                    
                    if not can_fit:
                        reasons.append(f"{model_type.value}: Insufficient memory even with Q4_K_M quantization")
                        continue
                else:
                    reasons.append(f"{model_type.value}: Needs {memory_needed:.1f}GB, only {available_memory:.1f}GB available")
                    continue
                    
            # Check performance
            gen_time = profile.performance_metrics.get(requirements.quality_preset, 120.0)
            if requirements.max_generation_time and gen_time > requirements.max_generation_time:
                continue
                
            # Calculate score
            score = self._calculate_model_score(profile, requirements, available_memory)
            candidates.append((model_type, score, profile))
            
        if not candidates:
            logger.warning(f"No suitable model found. Reasons: {reasons}")
            return None, f"No suitable model found. Issues: {'; '.join(reasons)}"
            
        # Sort by score and select best
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_model, score, profile = candidates[0]
        
        reason = self._get_selection_reason(profile, requirements)
        
        return best_model, reason
        
    def _calculate_model_score(
        self,
        profile: ModelProfile,
        requirements: TaskRequirements,
        available_memory: float
    ) -> float:
        """Calculate model fitness score"""
        
        score = 100.0
        
        # Capability match bonus
        capability_coverage = len([c for c in requirements.required_capabilities if c in profile.capabilities])
        score += capability_coverage * 10
        
        # Memory efficiency
        memory_needed = profile.memory_requirements.get(requirements.quality_preset, 10.0)
        memory_headroom = available_memory - memory_needed
        if memory_headroom > 0:
            score += min(memory_headroom * 5, 20)  # Up to 20 points for headroom
            
        # Performance score
        gen_time = profile.performance_metrics.get(requirements.quality_preset, 120.0)
        if requirements.max_generation_time:
            time_ratio = requirements.max_generation_time / gen_time
            score += min(time_ratio * 20, 40)  # Up to 40 points for speed
            
        # Resolution capability
        if ModelCapability.HIGH_RESOLUTION in profile.capabilities:
            score += 10
            
        # Quantization support
        if profile.supports_quantization:
            score += 5
            
        return score
        
    def _get_selection_reason(
        self,
        profile: ModelProfile,
        requirements: TaskRequirements
    ) -> str:
        """Get human-readable selection reason"""
        
        reasons = []
        
        # Main capability
        if ModelCapability.FAST_GENERATION in profile.capabilities:
            reasons.append("fast generation")
        if ModelCapability.HIGH_RESOLUTION in profile.capabilities:
            reasons.append("high resolution support")
            
        # Memory efficiency
        memory_needed = profile.memory_requirements.get(requirements.quality_preset, 10.0)
        reasons.append(f"{memory_needed:.1f}GB memory requirement")
        
        # Performance
        gen_time = profile.performance_metrics.get(requirements.quality_preset, 120.0)
        reasons.append(f"~{int(gen_time)}s generation time")
        
        return f"{profile.model_type.value}: {', '.join(reasons)}"


class ThreeDOrchestrator:
    """Orchestrates multiple 3D models for generation"""
    
    def __init__(
        self,
        models_dir: Path = Path("models/3d"),
        cache_dir: Path = Path("cache/3d")
    ):
        self.models_dir = models_dir
        self.cache_dir = cache_dir
        
        # Initialize components
        self.memory_manager = ThreeDMemoryManager(cache_dir)
        self.model_swapper = ModelSwapper()
        self.model_selector = ModelSelector(self.memory_manager)
        
        # Available models
        self.available_models = [
            ModelType3D.HUNYUAN3D_21,
            ModelType3D.HUNYUAN3D_2MINI,
            # ModelType3D.HI3DGEN,  # Coming April 2025
            # ModelType3D.SPARC3D   # Future
        ]
        
        # Model loaders
        self.model_loaders = {
            ModelType3D.HUNYUAN3D_21: self._load_hunyuan3d_21,
            ModelType3D.HUNYUAN3D_2MINI: self._load_hunyuan3d_mini,
        }
        
    def generate(
        self,
        input_data: Union[Image.Image, str],
        requirements: Optional[TaskRequirements] = None,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Generate 3D model based on requirements"""
        
        # Validate input
        if input_data is None:
            raise ValueError("Input data cannot be None")
            
        # Check if image is valid
        if isinstance(input_data, Image.Image):
            if input_data.mode is None:
                raise ValueError("Invalid Image object: mode is None")
            logger.info(f"Input image: size={input_data.size}, mode={input_data.mode}")
        elif isinstance(input_data, str):
            if not input_data.strip():
                raise ValueError("Text input cannot be empty")
            logger.info(f"Input text: {input_data[:50]}...")
        else:
            raise ValueError(f"Invalid input type: {type(input_data)}")
        
        # Default requirements
        if requirements is None:
            requirements = TaskRequirements(
                input_type="image" if isinstance(input_data, Image.Image) else "text",
                quality_preset="standard",
                output_format="glb"
            )
            
        # Log memory status
        self.memory_manager.log_memory_status()
        
        # Select optimal model
        selected_model, reason = self.model_selector.select_model(
            requirements,
            self.available_models
        )
        
        if selected_model is None:
            raise RuntimeError(f"No suitable model available: {reason}")
            
        logger.info(f"Selected model: {reason}")
        logger.info(f"[ORCHESTRATOR] Model selection result: {selected_model.value}")
        
        # Load model if not current
        if self.model_swapper.current_model_name != selected_model.value:
            logger.info(f"[ORCHESTRATOR] Current model ({self.model_swapper.current_model_name}) != selected model ({selected_model.value})")
            logger.info(f"[ORCHESTRATOR] Loading new model: {selected_model.value}")
            
            if progress_callback:
                progress_callback(0.0, f"Loading {selected_model.value}...")
                
            loader = self.model_loaders.get(selected_model)
            if not loader:
                raise RuntimeError(f"No loader for {selected_model.value}")
                
            success, msg = self.model_swapper.swap_model(
                selected_model.value,
                loader,
                force_unload=True
            )
            
            if not success:
                raise RuntimeError(f"Failed to load model: {msg}")
                
        # Get current model
        model = self.model_swapper.current_model
        if model is None:
            raise RuntimeError("Model not loaded")
            
        # Check if quantization needed
        profile = MODEL_PROFILES[selected_model]
        memory_needed = profile.memory_requirements.get(requirements.quality_preset, 10.0)
        available_memory = self.memory_manager.get_available_memory()
        
        use_quantization = None
        if memory_needed > available_memory and profile.supports_quantization:
            # Select quantization level based on available memory
            quantization_options = [
                ("Q8_0", 0.5),
                ("Q6_K", 0.375),
                ("Q5_K_S", 0.325),
                ("Q4_K_M", 0.25)
            ]
            
            for q_level, factor in quantization_options:
                if available_memory >= memory_needed * factor:
                    use_quantization = q_level
                    break
            
            if use_quantization is None:
                # Even Q4_K_M doesn't fit, use it anyway and hope for the best
                use_quantization = "Q4_K_M"
                logger.warning(f"Memory very low ({available_memory:.1f}GB), using Q4_K_M quantization anyway")
            else:
                logger.info(f"Using {use_quantization} quantization for memory efficiency")
            
            # Apply quantization settings before generation only if GGUF is available
            if hasattr(model, 'config') and self._check_gguf_available(model.config.model_variant, use_quantization):
                model.config.gguf_quantization = use_quantization
                model.config.use_gguf = True
                
                # If model has pipeline, update its config too
                if hasattr(model, 'pipeline') and hasattr(model.pipeline, 'config'):
                    model.pipeline.config.gguf_quantization = use_quantization
                    model.pipeline.config.use_gguf = True
                    
                # Recalculate memory with quantization
                memory_needed = memory_needed * {
                    "Q8_0": 0.5,
                    "Q6_K": 0.375,
                    "Q5_K_S": 0.325,
                    "Q4_K_M": 0.25
                }.get(use_quantization, 1.0)
                
                logger.info(f"Adjusted memory requirement with {use_quantization}: {memory_needed:.1f}GB")
                
        # Generate with selected model
        try:
            # Prepare parameters for HunYuan3D model
            # The model expects prompt as first arg and image as keyword arg
            if isinstance(input_data, str):
                prompt = input_data
                image = None
            else:
                # For image input, use a generic prompt
                prompt = "a high quality 3D model"
                image = input_data
            
            result = model.generate(
                prompt=prompt,
                image=image,
                quality=requirements.quality_preset,
                format=requirements.output_format,
                progress_callback=progress_callback
            )
            
            # Add metadata
            result["model_used"] = selected_model.value
            result["selection_reason"] = reason
            result["memory_used_gb"] = memory_needed
            result["quantization"] = use_quantization
            
            return result
            
        except Exception as e:
            logger.error(f"Generation failed with {selected_model.value}: {e}")
            logger.error(f"Full error traceback: {traceback.format_exc()}")
            
            # Disable automatic fallback to mini model to avoid confusion
            # The user explicitly requested a specific model, so we should fail clearly
            # rather than silently switching to a different model
            logger.error(f"Generation failed. Not attempting fallback to avoid unexpected model switches.")
            raise RuntimeError(f"3D generation failed with {selected_model.value}: {str(e)}")
                
                
    def _check_gguf_available(self, model_variant: str, quantization: str = "Q8_0") -> bool:
        """Check if GGUF model files are available"""
        from pathlib import Path
        from ...config import MODELS_DIR
        
        # Check for GGUF file in expected location
        gguf_filename = f"hunyuan3d-{quantization}.gguf"
        model_base_path = MODELS_DIR / "3d" / model_variant
        
        # Check different possible paths
        possible_paths = [
            model_base_path / "hunyuan3d-dit-v2-1" / gguf_filename,
            model_base_path / "hunyuan3d-dit-mini" / gguf_filename,
            model_base_path / gguf_filename,
        ]
        
        for path in possible_paths:
            if path.exists():
                logger.info(f"Found GGUF model at: {path}")
                return True
                
        logger.info(f"No GGUF models found for {model_variant}, will use standard model")
        return False
                
    def _load_hunyuan3d_21(self) -> HunYuan3DModel:
        """Load HunYuan3D 2.1 model"""
        logger.info("\n" + "="*80)
        logger.info("[ORCHESTRATOR] Loading HunYuan3D 2.1 model")
        
        # Check if GGUF model is available
        use_gguf = self._check_gguf_available("hunyuan3d-21", "Q8_0")
        
        logger.info(f"[ORCHESTRATOR] Creating HunYuan3DModel with:")
        logger.info(f"  - model_variant: hunyuan3d-21")
        logger.info(f"  - use_gguf: {use_gguf}")
        logger.info(f"  - device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
        
        model = HunYuan3DModel(
            model_variant="hunyuan3d-21",
            use_gguf=use_gguf,
            gguf_quantization="Q8_0" if use_gguf else None,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Load components based on available memory
        components = ["multiview", "reconstruction", "texture"]
        available_memory = self.memory_manager.get_available_memory()
        
        if available_memory < 8.0:
            # Load only essential components
            components = ["multiview", "reconstruction"]
            logger.warning("Limited memory - loading only essential components")
        
        # Add progress callback wrapper
        def load_progress(progress, message):
            logger.info(f"Loading HunYuan3D: {message} ({progress*100:.0f}%)")
            
        success = model.load(components, progress_callback=load_progress)
        if not success:
            raise RuntimeError("Failed to load HunYuan3D model")
            
        # Validate pipeline after loading
        if hasattr(model, 'pipeline') and hasattr(model.pipeline, 'multiview_model'):
            logger.info(f"Post-load validation:")
            logger.info(f"  Model pipeline: {model.pipeline}")
            logger.info(f"  Multiview model: {model.pipeline.multiview_model}")
            logger.info(f"  Multiview pipeline: {getattr(model.pipeline.multiview_model, 'pipeline', 'NO ATTR')}")
            logger.info(f"  Multiview loaded: {getattr(model.pipeline.multiview_model, 'loaded', 'NO ATTR')}")
            
            # Extra validation
            if model.pipeline.multiview_model and hasattr(model.pipeline.multiview_model, 'pipeline'):
                if model.pipeline.multiview_model.pipeline is None:
                    logger.error("WARNING: Multiview pipeline is None after loading!")
                    raise RuntimeError("Multiview pipeline failed to load properly")
        
        return model
        
    def _load_hunyuan3d_mini(self) -> HunYuan3DModel:
        """Load HunYuan3D mini model"""
        # Check if GGUF model is available
        use_gguf = self._check_gguf_available("hunyuan3d-2mini", "Q6_K")
        
        model = HunYuan3DModel(
            model_variant="hunyuan3d-2mini",
            use_gguf=use_gguf,
            gguf_quantization="Q6_K" if use_gguf else None,  # More aggressive quantization for mini
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        success = model.load()
        if not success:
            raise RuntimeError("Failed to load HunYuan3D mini model")
            
        return model
        
    def list_available_models(self) -> List[Dict[str, Any]]:
        """List available models with their profiles"""
        models = []
        
        for model_type in self.available_models:
            profile = MODEL_PROFILES.get(model_type)
            if profile:
                models.append({
                    "name": model_type.value,
                    "capabilities": [cap.value for cap in profile.capabilities],
                    "memory_requirements": profile.memory_requirements,
                    "performance": profile.performance_metrics,
                    "formats": profile.supported_formats,
                    "max_resolution": profile.max_resolution,
                    "quantization_support": profile.supports_quantization
                })
                
        return models
        
    def get_current_model(self) -> Optional[str]:
        """Get currently loaded model"""
        return self.model_swapper.current_model_name
        
    def unload_all(self):
        """Unload all models and free memory"""
        self.model_swapper.unload_current()
        self.memory_manager.free_memory(target_gb=10.0)