"""
Model dependency management
"""

from typing import Dict, List, Set, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# Define model dependencies
MODEL_DEPENDENCIES = {
    # HunYuan3D models and their required components
    "hunyuan3d-21": {
        "components": [
            "hunyuan3d-dit-v2-1",
            "hunyuan3d-vae-v2-1", 
            "hunyuan3d-paintpbr-v2-1",
            "dinov2-giant"  # Required for texture generation
        ],
        "optional": ["xatlas", "realesrgan-x4"],
        "description": "Full HunYuan3D 2.1 pipeline with texture generation"
    },
    "hunyuan3d-20": {
        "components": [
            "hunyuan3d-dit-v2-0",
            "hunyuan3d-vae-v2-0",
            "hunyuan3d-paintpbr-v2-0",
            "dinov2-giant"  # Required for texture generation
        ],
        "optional": ["xatlas", "realesrgan-x4"],
        "description": "Full HunYuan3D 2.0 pipeline with texture generation"
    },
    "hunyuan3d-2mini": {
        "components": [
            "hunyuan3d-dit-mini",
            "hunyuan3d-vae-mini"
        ],
        "optional": ["xatlas"],
        "description": "Lightweight HunYuan3D Mini"
    },
    
    # FLUX models and their components
    "flux-1-dev": {
        "components": ["clip-vit-large-patch14"],
        "optional": ["controlnet-canny", "controlnet-depth"],
        "description": "FLUX.1 Development model"
    },
    "flux-1-schnell": {
        "components": ["clip-vit-large-patch14"],
        "optional": ["controlnet-canny", "controlnet-depth"],
        "description": "FLUX.1 Schnell (fast) model"
    },
    
    # Texture components
    "hunyuan3d-paintpbr-v2-1": {
        "components": ["dinov2-giant"],
        "optional": ["realesrgan-x4"],
        "description": "PBR texture generation"
    },
    "hunyuan3d-paintpbr-v2-0": {
        "components": ["dinov2-giant"],
        "optional": ["realesrgan-x4"],
        "description": "PBR texture generation"
    },
    
    # Video models
    "wan2_1_1.3b": {
        "components": [],
        "optional": ["clip-vit-large-patch14"],
        "description": "Wan2.1 1.3B video model"
    },
    "mira-1b-tost": {
        "components": [],
        "optional": ["clip-vit-large-patch14"],
        "description": "Mira 1B video model"
    }
}

# Component download patterns
COMPONENT_PATTERNS = {
    # HunYuan3D components
    "hunyuan3d-dit-v2-1": {
        "patterns": ["hunyuan3d-dit-v2-1/**/*", "*.yaml", "*.json"],
        "repo_id": "Tencent/HunYuan3D-2.1",
        "size": "~6GB"
    },
    "hunyuan3d-vae-v2-1": {
        "patterns": ["hunyuan3d-vae-v2-1/**/*", "*.yaml", "*.json"],
        "repo_id": "Tencent/HunYuan3D-2.1",
        "size": "~2GB"
    },
    "hunyuan3d-paintpbr-v2-1": {
        "patterns": ["hunyuan3d-paintpbr-v2-1/**/*", "*.yaml", "*.json"],
        "repo_id": "Tencent/HunYuan3D-2.1",
        "size": "~4GB"
    },
    
    # DINO v2
    "dinov2-giant": {
        "patterns": ["*.bin", "*.json", "pytorch_model.bin", "config.json"],
        "repo_id": "facebook/dinov2-giant",
        "size": "~4.4GB"
    },
    
    # CLIP
    "clip-vit-large-patch14": {
        "patterns": ["*.bin", "*.json", "pytorch_model.bin", "config.json"],
        "repo_id": "openai/clip-vit-large-patch14",
        "size": "~1.7GB"
    },
    
    # ControlNet
    "controlnet-canny": {
        "patterns": ["*.safetensors", "*.json", "config.json"],
        "repo_id": "InstantX/FLUX.1-dev-Controlnet-Canny",
        "size": "~1.2GB"
    },
    "controlnet-depth": {
        "patterns": ["*.safetensors", "*.json", "config.json"],
        "repo_id": "Shakker-Labs/FLUX.1-dev-ControlNet-Depth",
        "size": "~1.2GB"
    },
    
    # Preprocessing
    "rembg-u2net": {
        "patterns": ["*.onnx", "*.pth"],
        "repo_id": "danielgatis/rembg",
        "size": "~170MB"
    },
    
    # Upscaling
    "realesrgan-x4": {
        "patterns": ["*.pth", "*.onnx"],
        "repo_id": "ai-forever/Real-ESRGAN",
        "size": "~64MB"
    },
    
    # UV unwrapping (local, not from HF)
    "xatlas": {
        "patterns": [],
        "repo_id": "local/xatlas",
        "size": "~10MB",
        "local": True
    }
}


class DependencyChecker:
    """Check and manage model dependencies"""
    
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        
    def check_model_dependencies(self, model_id: str) -> Dict[str, any]:
        """
        Check if all dependencies for a model are satisfied
        
        Returns:
            {
                "satisfied": bool,
                "missing_required": List[str],
                "missing_optional": List[str],
                "components": List[str],
                "description": str
            }
        """
        if model_id not in MODEL_DEPENDENCIES:
            # Model has no dependencies
            return {
                "satisfied": True,
                "missing_required": [],
                "missing_optional": [],
                "components": [],
                "description": "No dependencies"
            }
            
        deps = MODEL_DEPENDENCIES[model_id]
        missing_required = []
        missing_optional = []
        
        # Check required components
        for component in deps.get("components", []):
            if not self._is_component_installed(component):
                missing_required.append(component)
                
        # Check optional components
        for component in deps.get("optional", []):
            if not self._is_component_installed(component):
                missing_optional.append(component)
                
        return {
            "satisfied": len(missing_required) == 0,
            "missing_required": missing_required,
            "missing_optional": missing_optional,
            "components": deps.get("components", []),
            "description": deps.get("description", "")
        }
        
    def _is_component_installed(self, component_id: str) -> bool:
        """Check if a component is installed"""
        import logging
        logger = logging.getLogger(__name__)
        
        # Check in various possible locations
        possible_paths = [
            self.models_dir / "components" / component_id,
            self.models_dir / "pipeline" / component_id,
            self.models_dir / "texture" / component_id,
            self.models_dir / "texture_components" / component_id,
            self.models_dir / "preprocessing" / component_id,
            self.models_dir / "controlnet" / component_id,
            self.models_dir / "depth" / component_id,
            self.models_dir / "3d" / component_id,
        ]
        
        # Special handling for HunYuan3D components
        if component_id.startswith("hunyuan3d-"):
            # Check inside 3D model directories
            for model_dir in (self.models_dir / "3d").iterdir() if (self.models_dir / "3d").exists() else []:
                if model_dir.is_dir():
                    component_path = model_dir / component_id
                    if component_path.exists() and component_path.is_dir() and any(component_path.iterdir()):
                        logger.debug(f"Found component {component_id} in {model_dir}")
                        return True
        
        # Special handling for dinov2-giant - need to verify it actually exists
        if component_id == "dinov2-giant":
            # Check standalone locations first
            dinov2_paths = [
                self.models_dir / "texture" / "dinov2-giant",  # Where we actually download it
                self.models_dir / "texture_components" / "dinov2-giant",
                self.models_dir / "components" / "dinov2-giant",
                self.models_dir / "texture_components" / "dinov2",
            ]
            for path in dinov2_paths:
                if path.exists() and path.is_dir():
                    # Check for actual model files
                    model_files = list(path.glob("*.bin")) + list(path.glob("*.safetensors")) + list(path.glob("*.pth"))
                    if model_files:
                        logger.debug(f"Found dinov2-giant at {path} with model files")
                        return True
            
            # DO NOT assume it's embedded in paintpbr - that uses CLIP, not DINO
            logger.debug(f"dinov2-giant not found - it needs to be downloaded separately")
            return False
        
        # Check all possible paths
        for path in possible_paths:
            if path.exists() and path.is_dir() and any(path.iterdir()):
                logger.debug(f"Found component {component_id} at {path}")
                return True
                
        logger.debug(f"Component {component_id} not found in any location")
        return False
        
    def get_all_missing_dependencies(self) -> Dict[str, List[str]]:
        """Get all missing dependencies across all models"""
        all_missing = {}
        
        for model_id in MODEL_DEPENDENCIES:
            result = self.check_model_dependencies(model_id)
            if result["missing_required"]:
                all_missing[model_id] = result["missing_required"]
                
        return all_missing
        
    def can_generate(self, model_type: str) -> Dict[str, any]:
        """
        Check if generation is possible for a given model type
        
        Returns:
            {
                "can_generate": bool,
                "available_models": List[str],
                "missing_models": List[str],
                "missing_components": Dict[str, List[str]]
            }
        """
        available_models = []
        missing_models = []
        missing_components = {}
        
        # Get models for this type
        if model_type == "3d":
            model_ids = ["hunyuan3d-21", "hunyuan3d-20", "hunyuan3d-2mini"]
        elif model_type == "image":
            model_ids = ["flux-1-dev", "flux-1-schnell"]
        elif model_type == "video":
            model_ids = ["wan2_1_1.3b", "mira-1b-tost"]
        else:
            model_ids = []
            
        # Check each model
        for model_id in model_ids:
            if self._is_model_installed(model_id):
                # Check dependencies
                deps = self.check_model_dependencies(model_id)
                if deps["satisfied"]:
                    available_models.append(model_id)
                else:
                    missing_components[model_id] = deps["missing_required"]
            else:
                missing_models.append(model_id)
                
        return {
            "can_generate": len(available_models) > 0,
            "available_models": available_models,
            "missing_models": missing_models,
            "missing_components": missing_components
        }
        
    def _is_model_installed(self, model_id: str) -> bool:
        """Check if a model is installed"""
        import logging
        logger = logging.getLogger(__name__)
        
        # Map model IDs to types
        if "hunyuan3d" in model_id:
            model_type = "3d"
        elif "flux" in model_id:
            model_type = "image"
        elif "wan" in model_id or "mira" in model_id:
            model_type = "video"
        else:
            model_type = "other"
            
        model_path = self.models_dir / model_type / model_id
        exists = model_path.exists()
        has_files = any(model_path.iterdir()) if exists else False
        
        logger.debug(f"Checking model {model_id} at {model_path}: exists={exists}, has_files={has_files}")
        
        return exists and has_files
        
    def get_component_info(self, component_id: str) -> Optional[Dict[str, any]]:
        """Get information about a component"""
        return COMPONENT_PATTERNS.get(component_id)