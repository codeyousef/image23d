"""Model Operations Mixin

Handles model management, benchmarking, system checks, and related operations.
"""

import logging
import platform
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import torch
import psutil

from ...services.queue import JobPriority
from ...config import MODELS_DIR

logger = logging.getLogger(__name__)


class ModelOperationsMixin:
    """Mixin for model-related operations"""
    
    def submit_generation_job(
        self,
        job_type: str,
        params: Dict[str, Any],
        priority: str = "normal"
    ) -> str:
        """Submit a generation job to the queue
        
        Args:
            job_type: Type of job (image, 3d, full_pipeline)
            params: Job parameters
            priority: Priority level (low, normal, high, urgent)
            
        Returns:
            Job ID
        """
        priority_map = {
            "low": JobPriority.LOW,
            "normal": JobPriority.NORMAL,
            "high": JobPriority.HIGH,
            "urgent": JobPriority.URGENT
        }
        
        job = self.queue_manager.submit_job(
            job_type=job_type,
            params=params,
            priority=priority_map.get(priority, JobPriority.NORMAL)
        )
        
        return job.id
    
    def search_civitai_models(
        self,
        query: str,
        model_type: str = "LORA",
        base_model: str = "FLUX.1"
    ) -> List[Dict[str, Any]]:
        """Search Civitai for models
        
        Args:
            query: Search query
            model_type: Type of model
            base_model: Base model compatibility
            
        Returns:
            List of model info dictionaries
        """
        models = self.civitai_manager.search_models(
            query=query,
            types=[model_type],
            base_models=[base_model],
            limit=20
        )
        
        return [
            {
                "id": model.id,
                "name": model.name,
                "type": model.type,
                "base_model": model.base_model,
                "description": model.description,
                "images": model.images,
                "downloads": model.stats.get("downloads", 0),
                "likes": model.stats.get("likes", 0)
            }
            for model in models
        ]
    
    def download_civitai_model(
        self,
        model_id: int,
        progress_callback: Optional[Any] = None
    ) -> Tuple[bool, str]:
        """Download a model from Civitai
        
        Args:
            model_id: Civitai model ID
            progress_callback: Progress callback
            
        Returns:
            Tuple of (success, message/path)
        """
        model_info = self.civitai_manager.get_model_by_id(model_id)
        if not model_info:
            return False, "Model not found"
            
        return self.civitai_manager.download_model(
            model_info,
            progress_callback=progress_callback
        )
    
    def get_available_loras(self, base_model: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get available LoRAs
        
        Args:
            base_model: Filter by base model compatibility
            
        Returns:
            List of LoRA info dictionaries
        """
        loras = self.available_loras
        
        if base_model:
            loras = self.lora_manager.get_compatible_loras(base_model, loras)
            
        return [
            {
                "name": lora.name,
                "path": str(lora.path),
                "base_model": lora.base_model,
                "trigger_words": lora.trigger_words,
                "weight_default": lora.weight_default,
                "description": lora.description
            }
            for lora in loras
        ]
    
    def benchmark_current_model(
        self,
        test_prompts: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Benchmark the currently loaded model
        
        Args:
            test_prompts: Optional test prompts
            
        Returns:
            Benchmark results
        """
        if not self.image_model:
            return {"error": "No model loaded"}
            
        result = self.model_comparison.benchmark_model(
            model=self.image_model,
            model_name=self.image_model_name,
            model_type="image",
            quantization=self._get_model_quantization(),
            test_prompts=test_prompts
        )
        
        return result.to_dict()
    
    def _get_model_quantization(self) -> Optional[str]:
        """Get quantization level of current model"""
        if not self.image_model_name:
            return None
            
        # Check if it's a GGUF model
        if "Q8" in self.image_model_name:
            return "Q8_0"
        elif "Q6" in self.image_model_name:
            return "Q6_K"
        elif "Q5" in self.image_model_name:
            return "Q5_K_M"
        elif "Q4" in self.image_model_name:
            return "Q4_K_S"
        elif "Q3" in self.image_model_name:
            return "Q3_K_M"
        elif "Q2" in self.image_model_name:
            return "Q2_K"
        elif "FP8" in self.image_model_name:
            return "FP8"
            
        return None
    
    def get_generation_history(
        self,
        limit: int = 50,
        **filters
    ) -> List[Dict[str, Any]]:
        """Get generation history
        
        Args:
            limit: Maximum records
            **filters: Additional filters
            
        Returns:
            List of generation records
        """
        records = self.history_manager.get_history(limit=limit, **filters)
        return [record.to_dict() for record in records]
    
    def check_system_requirements(self) -> Dict[str, Any]:
        """Check system requirements for all features
        
        Returns:
            Dictionary with system check results
        """
        requirements = {
            "overall_status": "ready",
            "errors": [],
            "warnings": [],
            "info": {}
        }
        
        # Check GPU
        if torch.cuda.is_available():
            gpu_props = torch.cuda.get_device_properties(0)
            vram_gb = gpu_props.total_memory / (1024**3)
            requirements["info"]["gpu"] = gpu_props.name
            requirements["info"]["vram"] = f"{vram_gb:.1f} GB"
            
            if vram_gb < 8:
                requirements["errors"].append("Insufficient VRAM: At least 8GB required")
                requirements["overall_status"] = "error"
            elif vram_gb < 12:
                requirements["warnings"].append("Limited VRAM: Some features may be restricted")
                if requirements["overall_status"] != "error":
                    requirements["overall_status"] = "warning"
        else:
            requirements["errors"].append("No CUDA GPU detected")
            requirements["overall_status"] = "error"
            
        # Check RAM
        ram_gb = psutil.virtual_memory().total / (1024**3)
        requirements["info"]["ram"] = f"{ram_gb:.1f} GB"
        if ram_gb < 16:
            requirements["warnings"].append("Limited RAM: Recommended 16GB+")
            
        # Check disk space
        disk_usage = psutil.disk_usage(str(MODELS_DIR.parent))
        free_gb = disk_usage.free / (1024**3)
        requirements["info"]["disk_free"] = f"{free_gb:.1f} GB"
        if free_gb < 50:
            requirements["warnings"].append("Low disk space: Models require significant storage")
            
        # Format as HTML
        html = f"""
        <div style='padding: 15px; background: #f5f5f5; border-radius: 8px;'>
            <h3>System Requirements Check</h3>
            <div style='margin: 10px 0; padding: 10px; background: {'#ffebee' if requirements['overall_status'] == 'error' else '#fff3e0' if requirements['overall_status'] == 'warning' else '#e8f5e9'}; border-radius: 4px;'>
                <strong>Status:</strong> {requirements['overall_status'].upper()}
            </div>
            
            <h4>System Info:</h4>
            <ul>
                <li><strong>GPU:</strong> {requirements['info'].get('gpu', 'Not detected')}</li>
                <li><strong>VRAM:</strong> {requirements['info'].get('vram', 'N/A')}</li>
                <li><strong>RAM:</strong> {requirements['info'].get('ram', 'N/A')}</li>
                <li><strong>Free Disk:</strong> {requirements['info'].get('disk_free', 'N/A')}</li>
                <li><strong>Platform:</strong> {platform.system()} {platform.release()}</li>
            </ul>
            
            {'<h4>Errors:</h4><ul>' + ''.join(f'<li style="color: red;">{e}</li>' for e in requirements["errors"]) + '</ul>' if requirements["errors"] else ''}
            {'<h4>Warnings:</h4><ul>' + ''.join(f'<li style="color: orange;">{w}</li>' for w in requirements["warnings"]) + '</ul>' if requirements["warnings"] else ''}
        </div>
        """
        
        requirements["html"] = html
        return requirements
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get current system statistics
        
        Returns:
            System statistics
        """
        stats = {}
        
        # GPU stats
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            stats["gpu"] = {
                "available": free / (1024**3),
                "total": total / (1024**3),
                "used": (total - free) / (1024**3),
                "percentage": ((total - free) / total) * 100
            }
        
        # CPU/RAM stats
        stats["cpu"] = {
            "percentage": psutil.cpu_percent(interval=1),
            "cores": psutil.cpu_count()
        }
        
        memory = psutil.virtual_memory()
        stats["ram"] = {
            "available": memory.available / (1024**3),
            "total": memory.total / (1024**3),
            "used": memory.used / (1024**3),
            "percentage": memory.percent
        }
        
        # Disk stats
        disk = psutil.disk_usage(str(MODELS_DIR.parent))
        stats["disk"] = {
            "free": disk.free / (1024**3),
            "total": disk.total / (1024**3),
            "used": disk.used / (1024**3),
            "percentage": disk.percent
        }
        
        # Model cache stats
        models_size = sum(
            f.stat().st_size for f in MODELS_DIR.rglob("*") if f.is_file()
        ) / (1024**3)
        
        stats["models"] = {
            "total_size": models_size,
            "image_models": len(self.model_manager.get_downloaded_models("image")),
            "3d_models": len(self.model_manager.get_downloaded_models("3d")),
            "loras": len(self.available_loras)
        }
        
        # Queue stats
        queue_stats = self.queue_manager.get_queue_stats()
        stats["queue"] = {
            "pending": queue_stats["pending"],
            "running": queue_stats["running"],
            "completed": queue_stats["completed"],
            "failed": queue_stats["failed"]
        }
        
        return stats
    
    def cleanup_old_data(self, days: int = 30):
        """Clean up old generation data
        
        Args:
            days: Delete data older than this many days
        """
        import datetime
        
        cutoff = datetime.datetime.now() - datetime.timedelta(days=days)
        
        # Clean history
        deleted_count = self.history_manager.cleanup_old_records(cutoff)
        
        # Clean queue job history
        self.queue_manager.cleanup_old_jobs(days)
        
        # Clean cache directories
        cache_freed = 0
        for cache_dir in [self.output_dir, self.output_dir.parent / "cache"]:
            if cache_dir.exists():
                for file in cache_dir.rglob("*"):
                    if file.is_file() and file.stat().st_mtime < cutoff.timestamp():
                        cache_freed += file.stat().st_size
                        file.unlink()
        
        return {
            "history_deleted": deleted_count,
            "cache_freed_gb": cache_freed / (1024**3)
        }
    
    def generate_3d_direct(
        self,
        prompt: str,
        negative_prompt: str = "",
        input_image: Optional[Any] = None,
        model_name: str = "hunyuan3d-21",
        quality_preset: str = "Standard",
        export_format: str = "glb",
        progress_callback: Optional[Any] = None
    ) -> Tuple[Optional[str], str]:
        """Direct 3D generation without queue
        
        Args:
            prompt: Text prompt
            negative_prompt: Negative prompt
            input_image: Optional input image
            model_name: 3D model name
            quality_preset: Quality preset
            export_format: Export format
            progress_callback: Progress callback
            
        Returns:
            Tuple of (mesh_path, info)
        """
        # Create a simple wrapper for progress
        def wrapped_progress(p, msg):
            if progress_callback:
                progress_callback(p, msg)
        
        # Load model if needed
        if self.threed_model_name != model_name:
            wrapped_progress(0.1, f"Loading model {model_name}...")
            status, model, loaded_name = self.model_manager.load_3d_model(
                model_name, self.threed_model, self.threed_model_name, "cuda"
            )
            if "❌" in status:
                return None, status
            self.threed_model = model
            self.threed_model_name = loaded_name
        
        # Generate
        wrapped_progress(0.3, "Generating 3D model...")
        generation_id = str(uuid.uuid4())
        output_dir = self.output_dir / "3d" / generation_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        mesh_path, info = self.threed_generator.generate_3d(
            self.threed_model,
            model_name,
            prompt,
            negative_prompt,
            input_image,
            quality_preset,
            export_format,
            str(output_dir),
            progress=lambda p, msg: wrapped_progress(0.3 + p * 0.7, msg)
        )
        
        if mesh_path:
            # Save to history
            self.history_manager.add_generation(
                generation_id=generation_id,
                generation_type="3d",
                model_name=model_name,
                prompt=prompt,
                negative_prompt=negative_prompt,
                parameters={
                    "quality_preset": quality_preset,
                    "export_format": export_format,
                    "has_input_image": input_image is not None
                },
                output_paths=[mesh_path],
                metadata={"info": info}
            )
        
        return mesh_path, info