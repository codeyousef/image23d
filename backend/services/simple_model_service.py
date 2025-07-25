"""
Simple model service for testing
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass

@dataclass
class ModelInfo:
    id: str
    name: str
    type: str
    size: str
    vram_required: str
    description: str
    repo_id: str = ""
    is_downloaded: bool = False
    download_progress: float = 0.0
    is_downloading: bool = False
    download_job_id: Optional[str] = None
    files: List[str] = None
    requirements: Dict[str, Any] = None

class ModelService:
    """Simple model service for testing"""
    
    def __init__(self):
        self.download_jobs = {}
        
    async def initialize(self):
        """Initialize model service"""
        pass
    
    async def list_models(self, model_type: Optional[str] = None) -> List[ModelInfo]:
        """List mock models for testing"""
        return [
            ModelInfo(
                id="flux-1-dev",
                name="FLUX.1 [dev]",
                type="image",
                size="5.2GB",
                vram_required="8GB",
                description="High-quality text-to-image model"
            )
        ]
    
    async def get_model(self, model_id: str) -> Optional[ModelInfo]:
        """Get a specific model"""
        models = await self.list_models()
        return next((m for m in models if m.id == model_id), None)