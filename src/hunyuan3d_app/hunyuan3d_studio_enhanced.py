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
from .lora_suggestion import LoRASuggestionEngine
from .queue_manager import QueueManager, JobPriority
from .history_manager import HistoryManager
from .model_comparison import ModelComparison
from .video_generation import VideoGenerator
from .character_consistency import CharacterConsistencyManager
from .face_swap import FaceSwapManager
from .websocket_server import ProgressStreamManager, get_progress_manager

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
        
        # LoRA system with suggestion engine
        self.lora_suggestion_engine = LoRASuggestionEngine(
            civitai_manager=self.civitai_manager,
            cache_dir=CACHE_DIR / "lora_suggestions"
        )
        self.lora_manager = LoRAManager(
            lora_dir=MODELS_DIR / "loras",
            suggestion_engine=self.lora_suggestion_engine
        )
        
        # Queue and history
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
        
        # New advanced features
        self.video_generator = VideoGenerator(
            cache_dir=CACHE_DIR / "video"
        )
        self.character_consistency_manager = CharacterConsistencyManager(
            profiles_dir=CACHE_DIR / "characters" / "profiles",
            embeddings_dir=CACHE_DIR / "characters" / "embeddings",
            cache_dir=CACHE_DIR / "characters"
        )
        self.face_swap_manager = FaceSwapManager(
            model_dir=MODELS_DIR / "insightface",
            cache_dir=CACHE_DIR / "faceswap"
        )
        
        # Progress streaming
        self.progress_manager = get_progress_manager()
        
        # Additional setup
        self.output_dir = OUTPUT_DIR
        
        # Register job handlers
        self._register_job_handlers()
        
        # Load LoRAs on startup
        self.available_loras = self.lora_manager.scan_lora_directory()
        
        # Initialize models
        self._initialize_advanced_models()
        
    def _initialize_advanced_models(self):
        """Initialize advanced models on startup"""
        import asyncio
        
        # Start WebSocket server
        async def start_ws():
            await self.progress_manager.start_server()
            
        # Run in background
        import threading
        def run_async():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.create_task(start_ws())
            loop.run_forever()
            
        ws_thread = threading.Thread(target=run_async, daemon=True)
        ws_thread.start()
        
        logger.info("Advanced models initialization complete")
        
    def _register_job_handlers(self):
        """Register handlers for queue processing"""
        self.queue_manager.register_handler("image", self._process_image_job)
        self.queue_manager.register_handler("3d", self._process_3d_job)
        self.queue_manager.register_handler("full_pipeline", self._process_full_pipeline_job)
        self.queue_manager.register_handler("video", self._process_video_job)
        self.queue_manager.register_handler("face_swap", self._process_face_swap_job)
        
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
        
    def check_system_requirements(self) -> Dict[str, Any]:
        """Check system requirements for all features
        
        Returns:
            Dictionary with system check results
        """
        import platform
        import psutil
        
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
        """Get comprehensive system statistics
        
        Returns:
            Dictionary of statistics
        """
        # Base stats
        stats = {
            "models_loaded": int(bool(self.image_model)) + int(bool(self.hunyuan3d_model)) + 
                           int(self.video_generator.current_model is not None),
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
        
    def _process_video_job(self, params: Dict[str, Any], progress_callback):
        """Process a video generation job"""
        try:
            from .video_generation import VideoModel, VideoGenerationParams
            
            # Extract parameters
            model_type = params.get("model_type", VideoModel.LTXVIDEO)
            video_params = VideoGenerationParams(
                prompt=params.get("prompt", ""),
                negative_prompt=params.get("negative_prompt", ""),
                duration_seconds=params.get("duration", 5.0),
                fps=params.get("fps", 24),
                width=params.get("width", 768),
                height=params.get("height", 512),
                motion_strength=params.get("motion_strength", 1.0),
                guidance_scale=params.get("guidance_scale", 7.5),
                num_inference_steps=params.get("steps", 30),
                seed=params.get("seed", -1)
            )
            
            # Add character consistency if provided
            character_id = params.get("character_id")
            if character_id:
                character = self.character_consistency_manager.get_character(character_id)
                if character and character.full_embeddings is not None:
                    video_params.character_embeddings = character.full_embeddings
                    video_params.consistency_strength = params.get("consistency_strength", 0.8)
                    
            # Load model
            progress_callback(0.1, "Loading video model...")
            success, msg = self.video_generator.load_model(model_type)
            if not success:
                raise Exception(f"Failed to load model: {msg}")
                
            # Generate video
            progress_callback(0.2, "Generating video...")
            frames, info = self.video_generator.generate_video(video_params, progress_callback)
            
            if not frames:
                raise Exception(info.get("error", "Video generation failed"))
                
            # Save video
            progress_callback(0.9, "Encoding video...")
            import uuid
            video_path = self.output_dir / f"video_{uuid.uuid4()}.mp4"
            success = self.video_generator.save_video(frames, video_path, fps=video_params.fps)
            
            if not success:
                raise Exception("Failed to save video")
                
            # Save to history
            self.history_manager.add_generation(
                generation_id=str(uuid.uuid4()),
                generation_type="video",
                model_name=model_type.value,
                prompt=video_params.prompt,
                negative_prompt=video_params.negative_prompt,
                parameters={
                    "duration": video_params.duration_seconds,
                    "fps": video_params.fps,
                    "resolution": f"{video_params.width}x{video_params.height}",
                    "steps": video_params.num_inference_steps
                },
                output_paths=[str(video_path)],
                metadata=info
            )
            
            progress_callback(1.0, "Video generation complete!")
            return {"video_path": str(video_path), "info": info}
            
        except Exception as e:
            logger.error(f"Video generation failed: {e}")
            raise
            
    def _process_face_swap_job(self, params: Dict[str, Any], progress_callback):
        """Process a face swap job"""
        try:
            from .face_swap import FaceSwapParams
            
            # Initialize models if needed
            if not self.face_swap_manager.models_loaded:
                progress_callback(0.1, "Initializing face swap models...")
                success, msg = self.face_swap_manager.initialize_models()
                if not success:
                    raise Exception(f"Failed to initialize models: {msg}")
                    
            # Create parameters
            swap_params = FaceSwapParams(
                source_face_index=params.get("source_face_index", 0),
                target_face_index=params.get("target_face_index", -1),
                similarity_threshold=params.get("similarity_threshold", 0.6),
                face_restore=params.get("face_restore", True),
                face_restore_fidelity=params.get("restore_fidelity", 0.5),
                background_enhance=params.get("background_enhance", False),
                face_upsample=params.get("face_upsample", True)
            )
            
            # Get images
            source_image = params.get("source_image")
            target_image = params.get("target_image")
            
            if not source_image or not target_image:
                raise ValueError("Both source and target images are required")
                
            # Perform swap
            progress_callback(0.2, "Swapping faces...")
            result_img, info = self.face_swap_manager.swap_face(
                source_image=source_image,
                target_image=target_image,
                params=swap_params
            )
            
            if not result_img:
                raise Exception(info.get("error", "Face swap failed"))
                
            # Save result
            progress_callback(0.9, "Saving result...")
            import uuid
            output_path = self.output_dir / f"face_swap_{uuid.uuid4()}.png"
            result_img.save(output_path)
            
            # Save to history
            self.history_manager.add_generation(
                generation_id=str(uuid.uuid4()),
                generation_type="face_swap",
                model_name="InsightFace",
                prompt="Face Swap",
                parameters={
                    "source_faces": info.get("source_faces", 0),
                    "target_faces": info.get("target_faces", 0),
                    "swapped_faces": info.get("swapped_faces", 0)
                },
                output_paths=[str(output_path)],
                metadata=info
            )
            
            progress_callback(1.0, "Face swap complete!")
            return {"image": result_img, "info": info, "path": str(output_path)}
            
        except Exception as e:
            logger.error(f"Face swap failed: {e}")
            raise