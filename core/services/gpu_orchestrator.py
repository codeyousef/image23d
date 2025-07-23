"""
GPU Orchestrator - Intelligently routes work between local and serverless GPUs
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Callable
from datetime import datetime
import psutil
import GPUtil
from enum import Enum

from .serverless_gpu import ServerlessGPUManager, GPUType, MockServerlessGPUManager
from ..config import RUNPOD_CONFIG

logger = logging.getLogger(__name__)

class ExecutionMode(str, Enum):
    """Where to execute the workload"""
    LOCAL = "local"
    SERVERLESS = "serverless"
    AUTO = "auto"

class GPUOrchestrator:
    """
    Intelligently orchestrates GPU workloads between local and serverless
    
    Features:
    - Automatic routing based on GPU availability
    - Cost optimization
    - Fallback handling
    - Queue management
    """
    
    def __init__(self, runpod_api_key: Optional[str] = None):
        self.runpod_api_key = runpod_api_key
        self.serverless_manager = None
        self.local_gpu_available = self._check_local_gpu()
        self.execution_stats = {
            "local_runs": 0,
            "serverless_runs": 0,
            "total_cost": 0.0
        }
        
    def _check_local_gpu(self) -> bool:
        """Check if local GPU is available and has sufficient VRAM"""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                available_memory = gpu.memoryFree
                logger.info(f"Local GPU detected: {gpu.name} with {available_memory}MB free")
                return available_memory > 4000  # Need at least 4GB free
            return False
        except:
            logger.warning("No local GPU detected")
            return False
            
    def _check_system_memory(self) -> float:
        """Check available system RAM in GB"""
        return psutil.virtual_memory().available / (1024 ** 3)
        
    async def execute(
        self,
        task_type: str,
        inputs: Dict[str, Any],
        mode: ExecutionMode = ExecutionMode.AUTO,
        callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Execute a task on the most appropriate GPU
        
        Args:
            task_type: Type of task (image_generation, 3d_generation, etc.)
            inputs: Task inputs
            mode: Execution mode (local, serverless, or auto)
            callback: Progress callback
            
        Returns:
            Task results with execution metadata
        """
        start_time = datetime.utcnow()
        
        # Determine execution location
        use_serverless = self._should_use_serverless(task_type, inputs, mode)
        
        if use_serverless:
            result = await self._execute_serverless(task_type, inputs, callback)
            self.execution_stats["serverless_runs"] += 1
        else:
            result = await self._execute_local(task_type, inputs, callback)
            self.execution_stats["local_runs"] += 1
            
        # Add execution metadata
        duration = (datetime.utcnow() - start_time).total_seconds()
        result["execution_metadata"] = {
            "mode": "serverless" if use_serverless else "local",
            "duration_seconds": duration,
            "task_type": task_type
        }
        
        return result
        
    def _should_use_serverless(
        self, 
        task_type: str, 
        inputs: Dict[str, Any], 
        mode: ExecutionMode
    ) -> bool:
        """Determine whether to use serverless GPU"""
        
        # Respect explicit mode
        if mode == ExecutionMode.LOCAL:
            return False
        elif mode == ExecutionMode.SERVERLESS:
            return True
            
        # Auto mode - make intelligent decision
        
        # Check if we have RunPod configured
        if not self.runpod_api_key:
            return False
            
        # Estimate VRAM requirements
        vram_needed = self._estimate_vram_requirement(task_type, inputs)
        
        # Check local GPU capability
        if not self.local_gpu_available:
            logger.info("No local GPU available, using serverless")
            return True
            
        # Check if local GPU has enough VRAM
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                available_vram = gpus[0].memoryFree
                if available_vram < vram_needed:
                    logger.info(f"Insufficient local VRAM ({available_vram}MB < {vram_needed}MB), using serverless")
                    return True
        except:
            pass
            
        # For heavy workloads, prefer serverless
        if task_type in ["3d_generation", "video_generation"]:
            logger.info(f"Heavy workload '{task_type}', using serverless for better performance")
            return True
            
        # Default to local
        return False
        
    def _estimate_vram_requirement(self, task_type: str, inputs: Dict[str, Any]) -> int:
        """Estimate VRAM requirement in MB"""
        base_requirements = {
            "image_generation": 6000,  # 6GB base
            "3d_generation": 12000,    # 12GB base
            "face_swap": 4000,         # 4GB base
            "video_generation": 16000,  # 16GB base
            "prompt_enhancement": 2000  # 2GB base
        }
        
        vram = base_requirements.get(task_type, 8000)
        
        # Adjust for resolution
        if "width" in inputs and "height" in inputs:
            pixels = inputs["width"] * inputs["height"]
            if pixels > 1024 * 1024:  # Over 1 megapixel
                vram += 2000
                
        # Adjust for model size
        if "model_name" in inputs:
            if "xl" in inputs["model_name"].lower():
                vram += 4000
            elif "flux" in inputs["model_name"].lower():
                vram += 8000
                
        return vram
        
    async def _execute_local(
        self,
        task_type: str,
        inputs: Dict[str, Any],
        callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Execute task on local GPU"""
        logger.info(f"Executing {task_type} locally")
        
        if callback:
            callback(0, "Starting local execution...")
            
        # Use enhanced processors
        if task_type == "image_generation":
            from ..processors.image_processor import ImageProcessor
            from ..processors.prompt_enhancer import PromptEnhancer
            from .model_manager import ModelManager
            from ..models.generation import ImageGenerationRequest
            from ..config import MODELS_DIR, OUTPUT_DIR
            
            # Initialize components
            model_manager = ModelManager(MODELS_DIR)
            await model_manager.initialize()
            
            prompt_enhancer = PromptEnhancer()
            processor = ImageProcessor(model_manager, OUTPUT_DIR, prompt_enhancer)
            
            # Create request
            request = ImageGenerationRequest(
                prompt=inputs["prompt"],
                model=inputs.get("model", "flux-1-schnell"),
                width=inputs.get("width", 1024),
                height=inputs.get("height", 1024),
                steps=inputs.get("steps", 20),
                guidance_scale=inputs.get("guidance_scale", 7.5),
                enhancement_fields=inputs.get("enhancement_fields", {}),
                use_enhancement=inputs.get("use_enhancement", True)
            )
            
            # Progress wrapper
            def progress_wrapper(progress: float, message: str):
                if callback:
                    callback(progress, message)
            
            # Generate image with enhanced processor
            response = await processor.generate(request, progress_wrapper)
            
            return {
                "success": response.status.value == "completed",
                "output_path": str(response.image_path) if response.image_path else None,
                "cost_usd": 0.0,  # Local execution is free
                "enhanced_prompt": response.enhanced_prompt,
                "metadata": response.metadata
            }
            
        elif task_type == "3d_generation":
            from ..processors.threed_processor import ThreeDProcessor
            from ..processors.sparc3d_processor import Sparc3DProcessor
            from ..processors.hi3dgen_processor import Hi3DGenProcessor
            from ..processors.prompt_enhancer import PromptEnhancer
            from .model_manager import ModelManager
            from ..models.generation import ThreeDGenerationRequest
            from ..config import MODELS_DIR, OUTPUT_DIR
            # Initialize components
            model_manager = ModelManager(MODELS_DIR)
            await model_manager.initialize()
            
            prompt_enhancer = PromptEnhancer()
            
            # Select appropriate processor based on model
            model_name = inputs.get("model", "hunyuan3d-2.1")
            if "sparc3d" in model_name.lower():
                processor = Sparc3DProcessor(model_manager, OUTPUT_DIR, prompt_enhancer)
            elif "hi3dgen" in model_name.lower():
                processor = Hi3DGenProcessor(model_manager, OUTPUT_DIR, prompt_enhancer)
            else:
                processor = ThreeDProcessor(model_manager, OUTPUT_DIR, prompt_enhancer)
            
            # Create request
            request = ThreeDGenerationRequest(
                prompt=inputs.get("prompt"),
                input_image=inputs.get("input_image"),
                model=model_name,
                quality_preset=inputs.get("quality_preset", "standard"),
                export_formats=inputs.get("export_formats", ["glb"]),
                enhancement_fields=inputs.get("enhancement_fields", {}),
                use_enhancement=inputs.get("use_enhancement", True)
            )
            
            # Progress wrapper
            def progress_wrapper(progress: float, message: str):
                if callback:
                    callback(progress, message)
            
            # Generate 3D model with enhanced processor
            response = await processor.generate(request, progress_wrapper)
            
            return {
                "success": response.status.value == "completed",
                "output_path": str(response.model_path) if response.model_path else None,
                "preview_images": [str(p) for p in response.preview_images] if response.preview_images else [],
                "export_paths": {k: str(v) for k, v in response.export_paths.items()} if response.export_paths else {},
                "cost_usd": 0.0,
                "metadata": response.metadata
            }
            
        else:
            raise ValueError(f"Unsupported task type for local execution: {task_type}")
            
    async def _execute_serverless(
        self,
        task_type: str,
        inputs: Dict[str, Any],
        callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Execute task on serverless GPU"""
        logger.info(f"Executing {task_type} on RunPod")
        
        # Initialize serverless manager if needed
        if not self.serverless_manager:
            if self.runpod_api_key:
                self.serverless_manager = ServerlessGPUManager(self.runpod_api_key)
            else:
                logger.warning("Using mock serverless manager - no API key provided")
                self.serverless_manager = MockServerlessGPUManager()
                
        # Map task type to model type
        model_type_map = {
            "image_generation": "image",
            "3d_generation": "3d",
            "face_swap": "face",
            "video_generation": "video"
        }
        
        model_type = model_type_map.get(task_type, task_type)
        
        # Determine GPU type based on requirements
        vram_needed = self._estimate_vram_requirement(task_type, inputs)
        if vram_needed > 40000:
            gpu_type = GPUType.A100_80GB
        elif vram_needed > 24000:
            gpu_type = GPUType.A100_40GB
        elif vram_needed > 16000:
            gpu_type = GPUType.RTX_4090
        else:
            gpu_type = GPUType.RTX_3090
            
        # Execute on RunPod
        async with self.serverless_manager as manager:
            result = await manager.run_inference(
                model_type=model_type,
                inputs=inputs,
                gpu_type=gpu_type,
                callback=callback
            )
            
        # Update cost tracking
        self.execution_stats["total_cost"] += result.get("cost_usd", 0)
        
        return result
        
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        return {
            **self.execution_stats,
            "local_gpu_available": self.local_gpu_available,
            "runpod_configured": bool(self.runpod_api_key)
        }
        
    def estimate_cost(
        self, 
        task_type: str, 
        inputs: Dict[str, Any],
        mode: ExecutionMode = ExecutionMode.AUTO
    ) -> Dict[str, Any]:
        """Estimate cost for a task"""
        
        # Determine if serverless would be used
        use_serverless = self._should_use_serverless(task_type, inputs, mode)
        
        if not use_serverless:
            return {
                "estimated_cost_usd": 0.0,
                "execution_mode": "local",
                "reason": "Local GPU available"
            }
            
        # Estimate serverless cost
        vram_needed = self._estimate_vram_requirement(task_type, inputs)
        
        # Time estimates (seconds)
        time_estimates = {
            "image_generation": 20,
            "3d_generation": 180,
            "face_swap": 30,
            "video_generation": 120,
            "prompt_enhancement": 5
        }
        
        est_time = time_estimates.get(task_type, 60)
        
        # Adjust time based on inputs
        if "steps" in inputs:
            est_time *= (inputs["steps"] / 20)  # Normalize to 20 steps
            
        # GPU cost rates (per second)
        gpu_costs = {
            GPUType.A100_80GB: 0.000417,  # $1.50/hour
            GPUType.A100_40GB: 0.000250,  # $0.90/hour
            GPUType.RTX_4090: 0.000122,   # $0.44/hour
            GPUType.RTX_3090: 0.000069,   # $0.25/hour
        }
        
        # Select GPU based on VRAM
        if vram_needed > 40000:
            gpu_type = GPUType.A100_80GB
        elif vram_needed > 24000:
            gpu_type = GPUType.A100_40GB
        elif vram_needed > 16000:
            gpu_type = GPUType.RTX_4090
        else:
            gpu_type = GPUType.RTX_3090
            
        cost_per_second = gpu_costs[gpu_type]
        estimated_cost = est_time * cost_per_second
        
        return {
            "estimated_cost_usd": round(estimated_cost, 4),
            "estimated_seconds": est_time,
            "execution_mode": "serverless",
            "gpu_type": gpu_type.value,
            "vram_required_mb": vram_needed
        }