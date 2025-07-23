"""
Model service for managing AI models
"""

import asyncio
from typing import List, Optional, Dict, Any
from pathlib import Path
from dataclasses import dataclass

from core.config import MODELS_DIR, FLUX_MODELS, ALL_3D_MODELS, SPARC3D_MODELS, HI3DGEN_MODELS
from core.services import ModelManager

@dataclass
class ModelInfo:
    id: str
    name: str
    type: str
    size: str
    vram_required: str
    description: str
    repo_id: str
    is_downloaded: bool = False
    download_progress: float = 0.0
    is_downloading: bool = False
    download_job_id: Optional[str] = None
    files: List[str] = None
    requirements: Dict[str, Any] = None

class ModelService:
    """
    Service for managing model downloads and information
    """
    
    def __init__(self):
        self.model_manager = ModelManager(MODELS_DIR)
        self.download_jobs = {}
        
    async def initialize(self):
        """Initialize model service"""
        await self.model_manager.initialize()
        
    async def list_models(self, model_type: Optional[str] = None) -> List[ModelInfo]:
        """List all available models"""
        models = []
        
        # Add image models
        if not model_type or model_type == "image":
            for model_id, config in FLUX_MODELS.items():
                models.append(ModelInfo(
                    id=model_id,
                    name=config["name"],
                    type="image",
                    size=config["size"],
                    vram_required=config["vram_required"],
                    description=config["description"],
                    repo_id=config["repo_id"],
                    is_downloaded=self._check_downloaded("image", model_id),
                    files=self._get_model_files("image", model_id),
                    requirements={
                        "vram_mb": self._parse_vram(config["vram_required"]),
                        "disk_space_gb": self._parse_size(config["size"])
                    }
                ))
                
        # Add 3D models
        if not model_type or model_type == "3d":
            for model_id, config in ALL_3D_MODELS.items():
                models.append(ModelInfo(
                    id=model_id,
                    name=config["name"],
                    type="3d",
                    size=config["size"],
                    vram_required=config["vram_required"],
                    description=config["description"],
                    repo_id=config["repo_id"],
                    is_downloaded=self._check_downloaded("3d", model_id),
                    files=self._get_model_files("3d", model_id),
                    requirements={
                        "vram_mb": self._parse_vram(config["vram_required"]),
                        "disk_space_gb": self._parse_size(config["size"])
                    }
                ))
                
        return models
        
    async def get_model(self, model_id: str) -> Optional[ModelInfo]:
        """Get information about a specific model"""
        # Check in all model configs
        all_models = {**FLUX_MODELS, **ALL_3D_MODELS}
        
        if model_id in all_models:
            config = all_models[model_id]
            model_type = "image" if model_id in FLUX_MODELS else "3d"
            
            return ModelInfo(
                id=model_id,
                name=config["name"],
                type=model_type,
                size=config["size"],
                vram_required=config["vram_required"],
                description=config["description"],
                repo_id=config["repo_id"],
                is_downloaded=self._check_downloaded(model_type, model_id),
                files=self._get_model_files(model_type, model_id),
                requirements={
                    "vram_mb": self._parse_vram(config["vram_required"]),
                    "disk_space_gb": self._parse_size(config["size"])
                }
            )
            
        return None
        
    async def start_download(self, model_id: str, user_id: str) -> str:
        """Start downloading a model"""
        from uuid import uuid4
        job_id = str(uuid4())
        
        self.download_jobs[job_id] = {
            "model_id": model_id,
            "user_id": user_id,
            "progress": 0.0,
            "status": "starting"
        }
        
        return job_id
        
    async def download_model(self, model_id: str, job_id: str):
        """Download a model (background task)"""
        try:
            # Update job status
            self.download_jobs[job_id]["status"] = "downloading"
            
            # Get model info
            model = await self.get_model(model_id)
            if not model:
                raise ValueError(f"Model {model_id} not found")
                
            # Progress callback
            def progress_callback(progress: float):
                self.download_jobs[job_id]["progress"] = progress
                
            # Download via model manager
            await self.model_manager.download_model(
                model_id=model_id,
                model_type=model.type,
                progress_callback=progress_callback
            )
            
            # Update job status
            self.download_jobs[job_id]["status"] = "completed"
            self.download_jobs[job_id]["progress"] = 100.0
            
        except Exception as e:
            self.download_jobs[job_id]["status"] = "failed"
            self.download_jobs[job_id]["error"] = str(e)
            
    async def delete_model(self, model_id: str) -> bool:
        """Delete a downloaded model"""
        model = await self.get_model(model_id)
        if not model:
            return False
            
        model_path = MODELS_DIR / model.type / model_id
        
        if model_path.exists():
            import shutil
            shutil.rmtree(model_path)
            return True
            
        return False
        
    async def get_download_progress(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get download progress for a job"""
        return self.download_jobs.get(job_id)
        
    async def validate_model(self, model_id: str) -> Dict[str, Any]:
        """Validate a downloaded model"""
        model = await self.get_model(model_id)
        if not model:
            return {"is_valid": False, "message": "Model not found"}
            
        # Check if model files exist
        model_path = MODELS_DIR / model.type / model_id
        
        if not model_path.exists():
            return {"is_valid": False, "message": "Model directory not found"}
            
        # Check for required files based on model type
        required_files = []
        if model.type == "image":
            required_files = ["model_index.json", "scheduler/scheduler_config.json"]
        elif model.type == "3d":
            required_files = ["config.yaml"]
            
        missing_files = []
        for file_name in required_files:
            file_path = model_path / file_name
            if not file_path.exists():
                missing_files.append(file_name)
                
        if missing_files:
            return {
                "is_valid": False,
                "message": "Missing required files",
                "missing_files": missing_files
            }
            
        return {"is_valid": True, "message": "Model is valid"}
        
    def _check_downloaded(self, model_type: str, model_id: str) -> bool:
        """Check if a model is downloaded"""
        model_path = MODELS_DIR / model_type / model_id
        return model_path.exists() and any(model_path.iterdir())
        
    def _get_model_files(self, model_type: str, model_id: str) -> List[str]:
        """Get list of files for a downloaded model"""
        model_path = MODELS_DIR / model_type / model_id
        
        if not model_path.exists():
            return []
            
        files = []
        for file_path in model_path.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(model_path)
                files.append(str(relative_path))
                
        return files
        
    def _parse_vram(self, vram_str: str) -> int:
        """Parse VRAM requirement string to MB"""
        import re
        match = re.search(r'(\d+)', vram_str)
        if match:
            gb = int(match.group(1))
            return gb * 1024
        return 8192  # Default 8GB
        
    def _parse_size(self, size_str: str) -> float:
        """Parse size string to GB"""
        import re
        match = re.search(r'(\d+)', size_str)
        if match:
            return float(match.group(1))
        return 10.0  # Default 10GB