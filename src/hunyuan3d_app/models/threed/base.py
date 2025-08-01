"""Base classes for 3D model generation

Following the architecture from the 3D Implementation Guide:
- Base3DModel: Abstract base for all 3D models
- Base3DPipeline: Common pipeline interface
- IntermediateProcessor: Base for intermediate processing steps
- Quality presets and configurations
"""

import logging
import gc
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
import torch
import numpy as np
from PIL import Image
import trimesh

logger = logging.getLogger(__name__)


class ModelType3D(Enum):
    """Types of 3D models supported"""
    HUNYUAN3D_21 = "hunyuan3d-21"
    HUNYUAN3D_21_TURBO = "hunyuan3d-21-turbo"  # 50% faster inference
    HUNYUAN3D_2MINI = "hunyuan3d-2mini"
    HUNYUAN3D_2MINI_TURBO = "hunyuan3d-2mini-turbo"  # 50% faster inference
    HUNYUAN3D_2MV = "hunyuan3d-2mv"
    HI3DGEN = "hi3dgen"  # Coming April 2025
    HI3DGEN_TURBO = "hi3dgen-turbo"  # Coming April 2025
    SPARC3D = "sparc3d"
    SPARC3D_TURBO = "sparc3d-turbo"  # 50% faster inference


class IntermediateFormat(Enum):
    """Intermediate data formats for 3D pipeline"""
    DEPTH_MAP = "depth_map"
    NORMAL_MAP = "normal_map"
    MULTIVIEW_IMAGES = "multiview_images"
    POINT_CLOUD = "point_cloud"
    VOXEL_GRID = "voxel_grid"
    UV_MAP = "uv_map"
    TEXTURE_ATLAS = "texture_atlas"
    PBR_MAPS = "pbr_maps"


@dataclass
class QualityPreset3D:
    """Quality preset configuration for 3D generation"""
    name: str
    multiview_steps: int
    multiview_count: int
    reconstruction_resolution: int
    texture_resolution: int
    use_pbr: bool
    use_normal_maps: bool
    use_depth_refinement: bool
    memory_efficient: bool
    
    # GGUF quantization support
    supports_quantization: bool = True
    preferred_quantization: Optional[str] = None  # e.g., "Q8_0", "Q6_K"


# Define quality presets based on the guide
QUALITY_PRESETS_3D = {
    "draft": QualityPreset3D(
        name="Draft",
        multiview_steps=20,
        multiview_count=4,
        reconstruction_resolution=128,
        texture_resolution=512,
        use_pbr=False,
        use_normal_maps=False,
        use_depth_refinement=False,
        memory_efficient=True,
        supports_quantization=True,
        preferred_quantization="Q4_K_M"
    ),
    "standard": QualityPreset3D(
        name="Standard",
        multiview_steps=30,
        multiview_count=6,
        reconstruction_resolution=256,
        texture_resolution=1024,
        use_pbr=False,
        use_normal_maps=True,
        use_depth_refinement=False,
        memory_efficient=True,
        supports_quantization=True,
        preferred_quantization="Q6_K"
    ),
    "high": QualityPreset3D(
        name="High Quality",
        multiview_steps=40,
        multiview_count=8,
        reconstruction_resolution=512,
        texture_resolution=2048,
        use_pbr=True,
        use_normal_maps=True,
        use_depth_refinement=True,
        memory_efficient=False,
        supports_quantization=True,
        preferred_quantization="Q8_0"
    ),
    "ultra": QualityPreset3D(
        name="Ultra Quality",
        multiview_steps=50,
        multiview_count=12,
        reconstruction_resolution=1024,
        texture_resolution=4096,
        use_pbr=True,
        use_normal_maps=True,
        use_depth_refinement=True,
        memory_efficient=False,
        supports_quantization=False
    )
}


class IntermediateData:
    """Container for intermediate processing data"""
    def __init__(self):
        self.data: Dict[IntermediateFormat, Any] = {}
        self.metadata: Dict[str, Any] = {}
        
    def add(self, format_type: IntermediateFormat, data: Any, metadata: Optional[Dict] = None):
        """Add intermediate data"""
        self.data[format_type] = data
        if metadata:
            self.metadata[format_type.value] = metadata
            
    def get(self, format_type: IntermediateFormat) -> Optional[Any]:
        """Get intermediate data"""
        return self.data.get(format_type)
        
    def has(self, format_type: IntermediateFormat) -> bool:
        """Check if data exists"""
        return format_type in self.data
        
    def clear(self, format_type: Optional[IntermediateFormat] = None):
        """Clear intermediate data to free memory"""
        if format_type:
            if format_type in self.data:
                del self.data[format_type]
                if format_type.value in self.metadata:
                    del self.metadata[format_type.value]
        else:
            self.data.clear()
            self.metadata.clear()


class Base3DModel(ABC):
    """Abstract base class for all 3D models"""
    
    def __init__(self, model_path: Path, device: str = "cuda", dtype: torch.dtype = torch.float16):
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.loaded = False
        
    @abstractmethod
    def load(self, progress_callback=None) -> bool:
        """Load the model weights"""
        pass
        
    def unload(self):
        """Unload model to free memory
        
        Base implementation following the Managing Multiple AI Models guide
        """
        # Move to CPU if possible
        if hasattr(self, 'model') and self.model is not None:
            if hasattr(self.model, 'to'):
                try:
                    self.model.to('cpu')
                except Exception as e:
                    logger.warning(f"Failed to move model to CPU: {e}")
        
        # Clear any pipeline references
        if hasattr(self, 'pipeline') and self.pipeline is not None:
            if hasattr(self.pipeline, 'to'):
                try:
                    self.pipeline.to('cpu')
                except:
                    pass
            # Don't set to None yet - subclasses may need to do more cleanup
            
        # Clear loaded flag
        self.loaded = False
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
    @abstractmethod
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage in GB"""
        pass
        
    def to(self, device: str):
        """Move model to device"""
        self.device = device
        
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.loaded


class IntermediateProcessor(ABC):
    """Base class for intermediate processing steps"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        
    @abstractmethod
    def process(self, input_data: Any, **kwargs) -> Any:
        """Process input data"""
        pass
        
    def to(self, device: str):
        """Move processor to device"""
        self.device = device


class MultiViewModel(Base3DModel):
    """Base class for multi-view generation models"""
    
    @abstractmethod
    def generate_views(
        self,
        image: Union[Image.Image, torch.Tensor],
        num_views: int,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
        seed: Optional[int] = None,
        progress_callback=None
    ) -> List[Image.Image]:
        """Generate multiple views from a single image"""
        pass


class ReconstructionModel(Base3DModel):
    """Base class for 3D reconstruction models"""
    
    @abstractmethod
    def reconstruct(
        self,
        views: List[Image.Image],
        resolution: int = 256,
        use_depth: bool = False,
        use_normals: bool = False,
        progress_callback=None
    ) -> trimesh.Trimesh:
        """Reconstruct 3D mesh from multiple views"""
        pass


class TextureModel(Base3DModel):
    """Base class for texture generation models"""
    
    @abstractmethod
    def generate_texture(
        self,
        mesh: trimesh.Trimesh,
        views: List[Image.Image],
        resolution: int = 1024,
        use_pbr: bool = False,
        progress_callback=None
    ) -> Dict[str, Image.Image]:
        """Generate texture maps for mesh"""
        pass


class Base3DPipeline(ABC):
    """Base pipeline for 3D generation"""
    
    def __init__(
        self,
        multiview_model: Optional[MultiViewModel] = None,
        reconstruction_model: Optional[ReconstructionModel] = None,
        texture_model: Optional[TextureModel] = None,
        intermediate_processors: Optional[Dict[str, IntermediateProcessor]] = None,
        device: str = "cuda"
    ):
        self.multiview_model = multiview_model
        self.reconstruction_model = reconstruction_model
        self.texture_model = texture_model
        self.intermediate_processors = intermediate_processors or {}
        self.device = device
        
        # Intermediate data storage
        self.intermediate_data = IntermediateData()
        
    @abstractmethod
    def generate(
        self,
        image: Union[Image.Image, torch.Tensor],
        quality_preset: Union[str, QualityPreset3D] = "standard",
        output_format: str = "glb",
        progress_callback=None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate 3D model from image"""
        pass
        
    def add_intermediate_processor(self, name: str, processor: IntermediateProcessor):
        """Add an intermediate processor"""
        self.intermediate_processors[name] = processor
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get total memory usage"""
        usage = {"total": 0.0}
        
        if self.multiview_model and self.multiview_model.is_loaded:
            mv_usage = self.multiview_model.get_memory_usage()
            usage["multiview"] = mv_usage.get("total", 0.0)
            usage["total"] += usage["multiview"]
            
        if self.reconstruction_model and self.reconstruction_model.is_loaded:
            rec_usage = self.reconstruction_model.get_memory_usage()
            usage["reconstruction"] = rec_usage.get("total", 0.0)
            usage["total"] += usage["reconstruction"]
            
        if self.texture_model and self.texture_model.is_loaded:
            tex_usage = self.texture_model.get_memory_usage()
            usage["texture"] = tex_usage.get("total", 0.0)
            usage["total"] += usage["texture"]
            
        return usage
        
    def clear_intermediate_data(self):
        """Clear intermediate data to free memory"""
        self.intermediate_data.clear()
        
    def supports_quantization(self) -> bool:
        """Check if pipeline supports GGUF quantization"""
        # Check if any model supports quantization
        if self.multiview_model and hasattr(self.multiview_model, 'supports_quantization'):
            return self.multiview_model.supports_quantization()
        if self.reconstruction_model and hasattr(self.reconstruction_model, 'supports_quantization'):
            return self.reconstruction_model.supports_quantization()
        if self.texture_model and hasattr(self.texture_model, 'supports_quantization'):
            return self.texture_model.supports_quantization()
        return False
        
    def apply_quantization(self, quantization_level: str = "Q8_0") -> bool:
        """Apply GGUF quantization to supported models
        
        Args:
            quantization_level: Quantization level (Q8_0, Q6_K, Q5_K_S, Q4_K_M)
            
        Returns:
            True if quantization was applied successfully
        """
        success = True
        applied_models = []
        
        # Apply to multiview model
        if self.multiview_model and hasattr(self.multiview_model, 'apply_quantization'):
            if self.multiview_model.apply_quantization(quantization_level):
                applied_models.append("multiview")
            else:
                success = False
        
        # Apply to reconstruction model  
        if self.reconstruction_model and hasattr(self.reconstruction_model, 'apply_quantization'):
            if self.reconstruction_model.apply_quantization(quantization_level):
                applied_models.append("reconstruction")
            else:
                success = False
                
        # Apply to texture model
        if self.texture_model and hasattr(self.texture_model, 'apply_quantization'):
            if self.texture_model.apply_quantization(quantization_level):
                applied_models.append("texture")
            else:
                success = False
        
        if applied_models:
            logger.info(f"Applied {quantization_level} quantization to: {', '.join(applied_models)}")
        
        return success
        
    def get_quantization_info(self) -> Dict[str, Any]:
        """Get current quantization information
        
        Returns:
            Dictionary with quantization status for each model
        """
        info = {
            "supports_quantization": self.supports_quantization(),
            "models": {}
        }
        
        if self.multiview_model and hasattr(self.multiview_model, 'get_quantization_info'):
            info["models"]["multiview"] = self.multiview_model.get_quantization_info()
            
        if self.reconstruction_model and hasattr(self.reconstruction_model, 'get_quantization_info'):
            info["models"]["reconstruction"] = self.reconstruction_model.get_quantization_info()
            
        if self.texture_model and hasattr(self.texture_model, 'get_quantization_info'):
            info["models"]["texture"] = self.texture_model.get_quantization_info()
            
        return info