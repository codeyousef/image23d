"""Request validation for 3D generation"""

import logging
from typing import Tuple, Optional
from pathlib import Path

from ...models.generation import ThreeDGenerationRequest
from ...models.enhancement import ModelType

logger = logging.getLogger(__name__)


class RequestValidator:
    """Validates 3D generation requests"""
    
    def validate_request(self, request: ThreeDGenerationRequest) -> Tuple[bool, Optional[str]]:
        """Validate 3D generation request
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if either prompt or input image is provided
        if not request.prompt and not request.input_image:
            return False, "Either prompt or input image must be provided"
            
        # Validate model
        valid_models = ["hunyuan3d-21", "hunyuan3d-20", "hunyuan3d-2mini"]
        if request.model not in valid_models:
            return False, f"Invalid model: {request.model}. Must be one of {valid_models}"
            
        # Validate quality preset
        valid_presets = ["draft", "standard", "high", "ultra", "custom"]
        if request.quality_preset not in valid_presets:
            return False, f"Invalid quality preset: {request.quality_preset}"
            
        # Validate numeric parameters
        if request.num_views < 1 or request.num_views > 8:
            return False, "Number of views must be between 1 and 8"
            
        if request.mesh_resolution < 32 or request.mesh_resolution > 512:
            return False, "Mesh resolution must be between 32 and 512"
            
        if request.texture_resolution < 256 or request.texture_resolution > 4096:
            return False, "Texture resolution must be between 256 and 4096"
            
        # Validate export formats
        valid_formats = ["glb", "obj", "ply", "stl", "fbx", "usdz", "gltf"]
        for fmt in request.export_formats:
            if fmt.lower() not in valid_formats:
                return False, f"Invalid export format: {fmt}"
                
        # Validate input image if provided
        if request.input_image:
            if isinstance(request.input_image, str):
                path = Path(request.input_image)
                if not path.exists():
                    return False, f"Input image not found: {request.input_image}"
                    
        return True, None
    
    def get_model_type(self, model_id: str) -> ModelType:
        """Get model type from model ID"""
        model_map = {
            "hunyuan3d-21": ModelType.HUNYUAN_3D_21,
            "hunyuan3d-20": ModelType.HUNYUAN_3D_20,
            "hunyuan3d-2mini": ModelType.HUNYUAN_3D_MINI,
        }
        
        # Default to HUNYUAN_3D_21 for unknown models
        return model_map.get(model_id, ModelType.HUNYUAN_3D_21)