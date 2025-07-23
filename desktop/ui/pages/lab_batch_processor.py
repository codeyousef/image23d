"""
Batch processing implementation for the Lab page
"""

import asyncio
import csv
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor
from nicegui import ui

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Handles batch processing for the Lab page"""
    
    def __init__(self, model_manager, output_dir: Path):
        self.model_manager = model_manager
        self.output_dir = output_dir
        self.batch_dir = output_dir / "batch"
        self.batch_dir.mkdir(exist_ok=True)
        
        # Processing state
        self.processing = False
        self.cancel_requested = False
        self.current_task: Optional[asyncio.Task] = None
        
        # Worker pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    async def process_batch(
        self,
        items: List[Dict[str, Any]],
        model_id: str,
        images_per_prompt: int = 1,
        parallel_workers: int = 1,
        progress_callback: Optional[Callable] = None,
        item_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Process a batch of generation requests
        
        Args:
            items: List of batch items with prompts/settings
            model_id: Model to use for generation
            images_per_prompt: Number of images to generate per prompt
            parallel_workers: Number of parallel workers
            progress_callback: Callback for overall progress (value, message)
            item_callback: Callback for item status updates (index, status, result)
            
        Returns:
            Batch processing results
        """
        self.processing = True
        self.cancel_requested = False
        
        # Create batch session directory
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = self.batch_dir / f"batch_{session_id}"
        session_dir.mkdir(exist_ok=True)
        
        # Save batch manifest
        manifest = {
            "session_id": session_id,
            "model_id": model_id,
            "total_items": len(items),
            "images_per_prompt": images_per_prompt,
            "parallel_workers": parallel_workers,
            "items": items,
            "started_at": datetime.now().isoformat()
        }
        
        with open(session_dir / "manifest.json", 'w') as f:
            json.dump(manifest, f, indent=2)
            
        results = {
            "session_id": session_id,
            "session_dir": str(session_dir),
            "total": len(items),
            "completed": 0,
            "failed": 0,
            "results": []
        }
        
        try:
            # Load model if needed
            if progress_callback:
                progress_callback(0.05, f"Loading {model_id}...")
                
            # Determine model type
            model_type = self._get_model_type(model_id)
            
            # Get appropriate processor
            processor = self.model_manager.get_processor_for_model(
                model_id, 
                model_type,
                session_dir
            )
            
            # Process items
            total_tasks = len(items) * images_per_prompt
            completed_tasks = 0
            
            # Process in batches based on parallel workers
            for i in range(0, len(items), parallel_workers):
                if self.cancel_requested:
                    break
                    
                batch_slice = items[i:i+parallel_workers]
                
                # Process batch slice in parallel
                tasks = []
                for j, item in enumerate(batch_slice):
                    item_index = i + j
                    
                    # Update item status
                    if item_callback:
                        item_callback(item_index, 'processing', None)
                        
                    # Create task for each item
                    task = self._process_single_item(
                        processor,
                        model_type,
                        item,
                        item_index,
                        session_dir,
                        images_per_prompt
                    )
                    tasks.append(task)
                    
                # Wait for batch slice to complete
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for j, result in enumerate(batch_results):
                    item_index = i + j
                    
                    if isinstance(result, Exception):
                        logger.error(f"Item {item_index} failed: {result}")
                        results["failed"] += 1
                        if item_callback:
                            item_callback(item_index, 'failed', str(result))
                    else:
                        results["completed"] += 1
                        results["results"].append(result)
                        if item_callback:
                            item_callback(item_index, 'completed', result)
                            
                    completed_tasks += images_per_prompt
                    
                    # Update progress
                    if progress_callback:
                        progress = completed_tasks / total_tasks
                        progress_callback(
                            progress,
                            f"Processed {results['completed']} items, {results['failed']} failed"
                        )
                        
            # Save final results
            results["completed_at"] = datetime.now().isoformat()
            with open(session_dir / "results.json", 'w') as f:
                json.dump(results, f, indent=2)
                
            return results
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            results["error"] = str(e)
            return results
            
        finally:
            self.processing = False
            
    async def _process_single_item(
        self,
        processor,
        model_type: str,
        item: Dict[str, Any],
        index: int,
        session_dir: Path,
        num_images: int
    ) -> Dict[str, Any]:
        """Process a single batch item"""
        item_dir = session_dir / f"item_{index:04d}"
        item_dir.mkdir(exist_ok=True)
        
        result = {
            "index": index,
            "prompt": item.get("prompt", ""),
            "outputs": []
        }
        
        try:
            if model_type == "image":
                # Process image generation
                from core.models.generation import ImageGenerationRequest
                
                for img_idx in range(num_images):
                    request = ImageGenerationRequest(
                        prompt=item.get("prompt", ""),
                        negative_prompt=item.get("negative_prompt", ""),
                        width=item.get("width", 1024),
                        height=item.get("height", 1024),
                        num_inference_steps=item.get("steps", 4),
                        guidance_scale=item.get("guidance_scale", 0.0),
                        seed=item.get("seed", -1) + img_idx if item.get("seed", -1) > 0 else -1
                    )
                    
                    # Generate image
                    response = await processor.generate(request)
                    
                    if response.status == "completed" and response.image_path:
                        output_path = item_dir / f"image_{img_idx:02d}.png"
                        
                        # Copy result to batch directory
                        import shutil
                        shutil.copy(response.image_path, output_path)
                        
                        result["outputs"].append({
                            "path": str(output_path),
                            "seed": response.metadata.get("seed", -1)
                        })
                        
            elif model_type == "3d":
                # Process 3D generation
                from core.models.generation import ThreeDGenerationRequest
                
                request = ThreeDGenerationRequest(
                    prompt=item.get("prompt", ""),
                    input_image=item.get("input_image"),
                    model=item.get("model", "hunyuan3d-21"),
                    quality_preset=item.get("quality_preset", "standard"),
                    export_formats=item.get("export_formats", ["glb"])
                )
                
                # Generate 3D model
                response = await processor.generate(request)
                
                if response.status == "completed" and response.model_path:
                    result["outputs"].append({
                        "path": str(response.model_path),
                        "export_paths": response.export_paths
                    })
                    
        except Exception as e:
            logger.error(f"Failed to process item {index}: {e}")
            result["error"] = str(e)
            
        return result
        
    def _get_model_type(self, model_id: str) -> str:
        """Determine model type from model ID"""
        model_id_lower = model_id.lower()
        
        if any(x in model_id_lower for x in ['flux', 'sdxl', 'stable']):
            return "image"
        elif any(x in model_id_lower for x in ['hunyuan3d', 'sparc3d', 'hi3dgen']):
            return "3d"
        else:
            return "image"  # Default to image
            
    def cancel_processing(self):
        """Cancel current batch processing"""
        self.cancel_requested = True
        if self.current_task and not self.current_task.done():
            self.current_task.cancel()
            
    async def load_csv_batch(self, csv_path: Path) -> List[Dict[str, Any]]:
        """Load batch items from CSV file
        
        Expected CSV format:
        prompt,negative_prompt,steps,guidance_scale,seed,width,height
        """
        items = []
        
        try:
            with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    item = {
                        "prompt": row.get("prompt", ""),
                        "negative_prompt": row.get("negative_prompt", ""),
                        "steps": int(row.get("steps", 4)),
                        "guidance_scale": float(row.get("guidance_scale", 0.0)),
                        "seed": int(row.get("seed", -1)),
                        "width": int(row.get("width", 1024)),
                        "height": int(row.get("height", 1024))
                    }
                    items.append(item)
                    
        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            raise
            
        return items
        
    async def process_image_batch(
        self,
        images: List[Path],
        operation: str,
        settings: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Process a batch of images
        
        Args:
            images: List of image paths
            operation: Operation to perform (upscale, style_transfer, etc.)
            settings: Operation-specific settings
            
        Returns:
            List of results
        """
        results = []
        
        for img_path in images:
            try:
                if operation == "upscale":
                    # Implement upscaling
                    result = await self._upscale_image(img_path, settings)
                elif operation == "style_transfer":
                    # Implement style transfer
                    result = await self._apply_style_transfer(img_path, settings)
                else:
                    result = {"error": f"Unknown operation: {operation}"}
                    
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to process {img_path}: {e}")
                results.append({"error": str(e)})
                
        return results
        
    async def _upscale_image(self, image_path: Path, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Placeholder for image upscaling"""
        # This would integrate with Real-ESRGAN or similar
        return {
            "input": str(image_path),
            "output": str(image_path),  # Placeholder
            "scale": settings.get("scale", 2),
            "status": "not_implemented"
        }
        
    async def _apply_style_transfer(self, image_path: Path, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Placeholder for style transfer"""
        return {
            "input": str(image_path),
            "style": settings.get("style_image"),
            "output": str(image_path),  # Placeholder
            "status": "not_implemented"
        }
        
    def get_batch_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent batch processing history"""
        history = []
        
        try:
            # List batch directories
            batch_dirs = sorted(
                [d for d in self.batch_dir.iterdir() if d.is_dir() and d.name.startswith("batch_")],
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )[:limit]
            
            for batch_dir in batch_dirs:
                manifest_path = batch_dir / "manifest.json"
                if manifest_path.exists():
                    with open(manifest_path, 'r') as f:
                        manifest = json.load(f)
                        
                    # Check for results
                    results_path = batch_dir / "results.json"
                    if results_path.exists():
                        with open(results_path, 'r') as f:
                            results = json.load(f)
                            manifest["results"] = results
                            
                    history.append(manifest)
                    
        except Exception as e:
            logger.error(f"Failed to load batch history: {e}")
            
        return history