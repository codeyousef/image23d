"""
Serverless GPU management for cost-effective inference using RunPod
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
import json
import aiohttp
from enum import Enum

from ..config import RUNPOD_CONFIG

logger = logging.getLogger(__name__)

class GPUType(str, Enum):
    """Available GPU types on RunPod"""
    A100_40GB = "A100_40GB"
    A100_80GB = "A100_80GB"
    RTX_4090 = "RTX_4090"
    RTX_3090 = "RTX_3090"
    A6000 = "A6000"

class JobStatus(str, Enum):
    """RunPod job status"""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

class ServerlessGPUManager:
    """
    Manages serverless GPU execution via RunPod API
    
    Cost: $0.00025/second ($0.90/hour) - 93% savings vs fixed GPU
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://api.runpod.ai/v2"
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
            
    async def run_inference(
        self,
        model_type: str,
        inputs: Dict[str, Any],
        gpu_type: GPUType = GPUType.A100_40GB,
        callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Run inference on serverless GPU
        
        Args:
            model_type: Type of model to run
            inputs: Input parameters for the model
            gpu_type: GPU type to use
            callback: Progress callback function
            
        Returns:
            Inference results
        """
        if not self.api_key:
            raise ValueError("RunPod API key not configured")
            
        if not self.session:
            raise RuntimeError("Session not initialized. Use async context manager.")
            
        start_time = datetime.utcnow()
        
        try:
            # Submit job
            job_id = await self._submit_job(model_type, inputs, gpu_type)
            logger.info(f"Submitted RunPod job: {job_id}")
            
            if callback:
                callback(0, "Job submitted to RunPod")
                
            # Poll for completion
            result = await self._poll_job(job_id, callback)
            
            # Calculate cost
            duration = (datetime.utcnow() - start_time).total_seconds()
            cost = duration * RUNPOD_CONFIG["cost_per_second"]
            
            logger.info(f"Job completed in {duration:.1f}s, cost: ${cost:.4f}")
            
            return {
                "result": result,
                "job_id": job_id,
                "duration_seconds": duration,
                "cost_usd": cost,
                "gpu_type": gpu_type.value
            }
            
        except Exception as e:
            logger.error(f"RunPod inference failed: {str(e)}")
            raise
            
    async def _submit_job(self, model_type: str, inputs: Dict[str, Any], gpu_type: GPUType) -> str:
        """Submit a job to RunPod"""
        endpoint = f"{self.base_url}/run"
        
        payload = {
            "input": {
                "model_type": model_type,
                **inputs
            },
            "gpu_type": gpu_type.value,
            "timeout": RUNPOD_CONFIG["timeout"]
        }
        
        async with self.session.post(endpoint, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Failed to submit job: {error_text}")
                
            data = await response.json()
            return data["id"]
            
    async def _poll_job(self, job_id: str, callback: Optional[Callable]) -> Any:
        """Poll job status until completion"""
        endpoint = f"{self.base_url}/status/{job_id}"
        
        poll_interval = 2.0  # seconds
        max_polls = int(RUNPOD_CONFIG["timeout"] / poll_interval)
        
        for i in range(max_polls):
            async with self.session.get(endpoint) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Failed to get job status: {error_text}")
                    
                data = await response.json()
                status = JobStatus(data["status"])
                
                if status == JobStatus.COMPLETED:
                    return data["output"]
                elif status == JobStatus.FAILED:
                    raise Exception(f"Job failed: {data.get('error', 'Unknown error')}")
                elif status == JobStatus.CANCELLED:
                    raise Exception("Job was cancelled")
                    
                # Update progress
                if callback and "progress" in data:
                    progress = data["progress"]
                    message = data.get("message", f"Processing... {progress}%")
                    callback(progress, message)
                    
            await asyncio.sleep(poll_interval)
            
        raise Exception("Job timed out")
        
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job"""
        if not self.session:
            raise RuntimeError("Session not initialized")
            
        endpoint = f"{self.base_url}/cancel/{job_id}"
        
        async with self.session.post(endpoint) as response:
            return response.status == 200
            
    async def get_available_gpus(self) -> List[Dict[str, Any]]:
        """Get list of available GPU types and pricing"""
        # This would normally call the RunPod API
        # For now, return static data
        return [
            {
                "type": GPUType.A100_40GB.value,
                "vram_gb": 40,
                "cost_per_hour": 0.90,
                "availability": "high"
            },
            {
                "type": GPUType.A100_80GB.value,
                "vram_gb": 80,
                "cost_per_hour": 1.50,
                "availability": "medium"
            },
            {
                "type": GPUType.RTX_4090.value,
                "vram_gb": 24,
                "cost_per_hour": 0.44,
                "availability": "high"
            },
            {
                "type": GPUType.RTX_3090.value,
                "vram_gb": 24,
                "cost_per_hour": 0.25,
                "availability": "high"
            },
            {
                "type": GPUType.A6000.value,
                "vram_gb": 48,
                "cost_per_hour": 0.79,
                "availability": "medium"
            }
        ]
        
    def estimate_cost(self, operation: str, gpu_type: GPUType = GPUType.A100_40GB) -> Dict[str, float]:
        """
        Estimate cost for different operations
        
        Returns:
            Dictionary with estimated time and cost
        """
        # Operation time estimates (seconds)
        time_estimates = {
            "image_generation": 20,
            "3d_conversion": 180,
            "face_swap": 30,
            "video_per_second": 60,
            "prompt_enhancement": 5
        }
        
        est_time = time_estimates.get(operation, 60)
        est_cost = est_time * RUNPOD_CONFIG["cost_per_second"]
        
        return {
            "estimated_seconds": est_time,
            "estimated_cost_usd": est_cost,
            "gpu_type": gpu_type.value
        }

class MockServerlessGPUManager(ServerlessGPUManager):
    """Mock implementation for local testing without RunPod"""
    
    def __init__(self):
        super().__init__(api_key="mock")
        self.mock_delay = 2.0
        
    async def run_inference(
        self,
        model_type: str,
        inputs: Dict[str, Any],
        gpu_type: GPUType = GPUType.A100_40GB,
        callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Mock inference execution"""
        logger.info(f"Mock inference for {model_type} on {gpu_type}")
        
        # Simulate progress
        for i in range(0, 101, 20):
            if callback:
                callback(i, f"Processing... {i}%")
            await asyncio.sleep(self.mock_delay / 5)
            
        return {
            "result": {"status": "success", "mock": True},
            "job_id": "mock-job-123",
            "duration_seconds": self.mock_delay,
            "cost_usd": self.mock_delay * RUNPOD_CONFIG["cost_per_second"],
            "gpu_type": gpu_type.value
        }