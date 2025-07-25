"""Enhanced Hunyuan3D Studio

Main enhanced studio class that combines all features through mixins.
"""

import logging
from typing import Dict, Any

from .base import Hunyuan3DStudio
from .feature_initialization import FeatureInitializationMixin
from .job_processors import JobProcessorMixin
from .model_operations import ModelOperationsMixin

logger = logging.getLogger(__name__)


class Hunyuan3DStudioEnhanced(
    Hunyuan3DStudio,
    FeatureInitializationMixin,
    JobProcessorMixin,
    ModelOperationsMixin
):
    """Enhanced version with all new features integrated"""
    
    def __init__(self):
        """Initialize enhanced studio with all features"""
        # Initialize base class
        super().__init__()
        
        # Initialize all enhanced features
        self._initialize_features()
        
        # Register job handlers
        self._register_job_handlers()
        
        # Initialize advanced models
        self._initialize_advanced_models()
        
        logger.info("Enhanced Hunyuan3D Studio initialized with all features")
    
    def get_feature_status(self) -> Dict[str, Any]:
        """Get status of all enhanced features
        
        Returns:
            Dictionary with feature status information
        """
        status = {
            "base_initialized": hasattr(self, 'model_manager'),
            "websocket_server": hasattr(self, 'progress_manager') and self.progress_manager is not None,
            "credential_manager": hasattr(self, 'credential_manager'),
            "civitai_integration": hasattr(self, 'civitai_manager'),
            "lora_system": hasattr(self, 'lora_manager') and len(self.available_loras) > 0,
            "queue_system": hasattr(self, 'queue_manager'),
            "history_tracking": hasattr(self, 'history_manager'),
            "model_comparison": hasattr(self, 'model_comparison'),
            "video_generation": hasattr(self, 'video_generator'),
            "character_consistency": hasattr(self, 'character_consistency_manager'),
            "face_swap": hasattr(self, 'face_swap_manager') and self.face_swap_manager.facefusion_loaded
        }
        
        # Count active features
        active_count = sum(1 for v in status.values() if v)
        status["active_features"] = f"{active_count}/{len(status)-1}"
        
        return status