"""Multi-model orchestrator for 3D generation

Implements the orchestration framework from Managing Multiple AI Models guide:
- Dynamic model switching based on task requirements
- Memory-aware scheduling
- Model compatibility checks
- Pipeline composition
"""

import logging
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
            ModelCapability.GGUF_SUPPORT
        ],
        memory_requirements={
            "draft": 3.0,
            "standard": 4.0,
            "high": 6.0,
            "ultra": 8.0
        },
        performance_metrics={
            "draft": 15.0,
            "standard": 30.0,
            "high": 60.0,
            "ultra": 90.0
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
        max_memory_gb: Optional[float] = None
    ):
        self.input_type = input_type
        self.quality_preset = quality_preset
        self.output_format = output_format
        self.required_capabilities = required_capabilities or []
        self.max_generation_time = max_generation_time
        self.max_memory_gb = max_memory_gb
        
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
        
        candidates = []
        
        for model_type in available_models:
            profile = MODEL_PROFILES.get(model_type)
            if not profile:
                continue
                
            # Check capabilities
            if not all(cap in profile.capabilities for cap in requirements.required_capabilities):
                continue
                
            # Check format support
            if requirements.output_format not in profile.supported_formats:
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
                    # Reduce memory requirement by 50% for Q8_0
                    memory_needed *= 0.5
                    if memory_needed > available_memory:
                        continue
                else:
                    continue
                    
            # Check performance
            gen_time = profile.performance_metrics.get(requirements.quality_preset, 120.0)
            if requirements.max_generation_time and gen_time > requirements.max_generation_time:
                continue
                
            # Calculate score
            score = self._calculate_model_score(profile, requirements, available_memory)
            candidates.append((model_type, score, profile))
            
        if not candidates:
            return None, "No suitable model found for requirements"
            
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
        
        # Load model if not current
        if self.model_swapper.current_model_name != selected_model.value:
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
            if available_memory > memory_needed * 0.5:
                use_quantization = "Q8_0"
            elif available_memory > memory_needed * 0.4:
                use_quantization = "Q6_K"
            elif available_memory > memory_needed * 0.35:
                use_quantization = "Q5_K_S"
            else:
                use_quantization = "Q4_K_M"
                
            logger.info(f"Using {use_quantization} quantization for memory efficiency")
            
            # Reload model with quantization if needed
            if hasattr(model, 'config') and model.config.gguf_quantization != use_quantization:
                model.config.gguf_quantization = use_quantization
                model.config.use_gguf = True
                # Model will load quantized version on next use
                
        # Generate with selected model
        try:
            result = model.generate(
                input_data,
                quality=requirements.quality_preset,
                output_format=requirements.output_format,
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
            
            # Try fallback model if available
            if selected_model == ModelType3D.HUNYUAN3D_21:
                logger.info("Trying fallback to mini model...")
                requirements.max_memory_gb = 6.0  # Force mini model
                return self.generate(input_data, requirements, progress_callback)
            else:
                raise
                
    def _load_hunyuan3d_21(self) -> HunYuan3DModel:
        """Load HunYuan3D 2.1 model"""
        model = HunYuan3DModel(
            variant="hunyuan3d-21",
            use_gguf=True,
            quantization="Q8_0",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Load components based on available memory
        components = ["multiview", "reconstruction", "texture"]
        available_memory = self.memory_manager.get_available_memory()
        
        if available_memory < 8.0:
            # Load only essential components
            components = ["multiview", "reconstruction"]
            logger.warning("Limited memory - loading only essential components")
            
        model.load(components)
        return model
        
    def _load_hunyuan3d_mini(self) -> HunYuan3DModel:
        """Load HunYuan3D mini model"""
        model = HunYuan3DModel(
            variant="hunyuan3d-2mini",
            use_gguf=True,
            quantization="Q6_K",  # More aggressive quantization for mini
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        model.load()
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