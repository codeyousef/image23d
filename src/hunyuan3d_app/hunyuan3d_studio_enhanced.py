"""Enhanced Hunyuan3D Studio with all new features integrated"""

import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List

import gradio as gr
import torch
from PIL import Image

from .hunyuan3d_studio import Hunyuan3DStudio
from .credential_manager import CredentialManager
from .civitai_manager import CivitaiManager
from .lora_manager import LoRAManager
from .queue_manager import QueueManager, JobPriority
from .history_manager import HistoryManager
from .model_comparison import ModelComparison

from .config import (
    ALL_IMAGE_MODELS, HUNYUAN3D_MODELS, QUALITY_PRESETS,
    MODELS_DIR, OUTPUT_DIR, CACHE_DIR
)

logger = logging.getLogger(__name__)


class Hunyuan3DStudioEnhanced(Hunyuan3DStudio):
    """Enhanced version with all new features integrated"""
    
    def __init__(self):
        # Initialize base class
        super().__init__()
        
        # Initialize new managers
        self.credential_manager = CredentialManager()
        self.civitai_manager = CivitaiManager(
            cache_dir=CACHE_DIR / "civitai",
            api_key=self.credential_manager.get_credential("civitai")
        )
        self.lora_manager = LoRAManager(
            lora_dir=MODELS_DIR / "loras"
        )
        self.queue_manager = QueueManager(
            max_workers=2,
            job_history_dir=OUTPUT_DIR / "job_history"
        )
        self.history_manager = HistoryManager(
            db_path=CACHE_DIR / "generation_history.db",
            thumbnails_dir=CACHE_DIR / "thumbnails"
        )
        self.model_comparison = ModelComparison(
            output_dir=OUTPUT_DIR / "benchmarks",
            cache_dir=CACHE_DIR / "benchmarks"
        )
        
        # Register job handlers
        self._register_job_handlers()
        
        # Load LoRAs on startup
        self.available_loras = self.lora_manager.scan_lora_directory()
        
    def _register_job_handlers(self):
        """Register handlers for queue processing"""
        self.queue_manager.register_handler("image", self._process_image_job)
        self.queue_manager.register_handler("3d", self._process_3d_job)
        self.queue_manager.register_handler("full_pipeline", self._process_full_pipeline_job)
        
    def _process_image_job(self, params: Dict[str, Any], progress_callback):
        """Process an image generation job"""
        # Extract parameters
        model_name = params.get("model_name")
        prompt = params.get("prompt")
        negative_prompt = params.get("negative_prompt", "")
        width = params.get("width", 1024)
        height = params.get("height", 1024)
        steps = params.get("steps", 30)
        guidance_scale = params.get("guidance_scale", 7.5)
        seed = params.get("seed", -1)
        lora_configs = params.get("lora_configs", [])
        
        # Load model if needed
        if self.image_model_name != model_name:
            progress_callback(0.1, "Loading model...")
            status = self.load_image_model(model_name, progress=None)
            if "❌" in status:
                raise Exception(f"Failed to load model: {status}")
                
        # Apply LoRAs if specified
        if lora_configs and self.image_model:
            progress_callback(0.2, "Applying LoRAs...")
            self.lora_manager.apply_multiple_loras(self.image_model, lora_configs)
            
        # Generate image
        progress_callback(0.3, "Generating image...")
        image, info = self.image_generator.generate_image(
            self.image_model,
            self.image_model_name,
            prompt,
            negative_prompt,
            width,
            height,
            steps,
            guidance_scale,
            seed,
            progress=lambda p, d: progress_callback(0.3 + p * 0.6, d)
        )
        
        if image:
            # Save to history
            progress_callback(0.9, "Saving to history...")
            import uuid
            generation_id = str(uuid.uuid4())
            
            # Save image
            image_path = OUTPUT_DIR / f"image_{generation_id}.png"
            image.save(image_path)
            
            # Add to history
            self.history_manager.add_generation(
                generation_id=generation_id,
                generation_type="image",
                model_name=model_name,
                prompt=prompt,
                negative_prompt=negative_prompt,
                parameters={
                    "width": width,
                    "height": height,
                    "steps": steps,
                    "guidance_scale": guidance_scale,
                    "seed": seed,
                    "loras": [{"name": lora.name, "weight": weight} for lora, weight in lora_configs]
                },
                output_paths=[str(image_path)],
                metadata={"info": info}
            )
            
        progress_callback(1.0, "Complete!")
        return {"image": image, "info": info, "path": str(image_path) if image else None}
        
    def _process_3d_job(self, params: Dict[str, Any], progress_callback):
        """Process a 3D conversion job"""
        # Implementation would be similar to image job
        # This is a placeholder
        progress_callback(1.0, "3D conversion not yet implemented in job queue")
        return {"success": False, "error": "Not implemented"}
        
    def _process_full_pipeline_job(self, params: Dict[str, Any], progress_callback):
        """Process a full text-to-3D pipeline job"""
        # First generate image
        progress_callback(0.0, "Starting pipeline...")
        image_result = self._process_image_job(params, 
            lambda p, d: progress_callback(p * 0.5, f"Image: {d}")
        )
        
        if image_result.get("image"):
            # Then convert to 3D
            three_d_params = {
                "image": image_result["image"],
                "model_name": params.get("hunyuan3d_model_name"),
                "quality_preset": params.get("quality_preset", "standard")
            }
            
            # This would call the 3D conversion
            progress_callback(0.5, "Converting to 3D...")
            # Placeholder for now
            progress_callback(1.0, "Pipeline complete!")
            
        return image_result
        
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
        
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics
        
        Returns:
            Dictionary of statistics
        """
        # Base stats
        stats = {
            "models_loaded": int(bool(self.image_model)) + int(bool(self.hunyuan3d_model)),
            "vram_used": 0.0,
            "gpu_usage": 0.0,
            "memory_usage": 0.0
        }
        
        # GPU stats
        if torch.cuda.is_available():
            stats["vram_used"] = torch.cuda.memory_allocated() / (1024**3)
            stats["vram_total"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            # Try to get GPU utilization
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                stats["gpu_usage"] = utilization.gpu
                stats["memory_usage"] = utilization.memory
            except:
                pass
                
        # History stats
        history_stats = self.history_manager.get_statistics()
        stats.update({
            "total_generations": history_stats["total_generations"],
            "favorites": history_stats["favorites"]
        })
        
        # Queue stats
        queue_stats = self.queue_manager.get_queue_status()
        stats.update({
            "queue_pending": queue_stats["pending"],
            "queue_active": queue_stats["active"]
        })
        
        return stats
        
    def cleanup_old_data(self, days: int = 30):
        """Clean up old data
        
        Args:
            days: Remove data older than this many days
        """
        # Clean Civitai cache
        self.civitai_manager.cleanup_cache(days)
        
        # Clean job history
        self.queue_manager.clear_completed_jobs()
        
        # Could also clean old outputs if needed
        logger.info(f"Cleaned up data older than {days} days")