"""
Job management service for the backend API
"""

import uuid
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import logging

from backend.models.generation import (
    GenerationJob, JobStatus, JobResponse, JobStatusResponse, 
    GenerationRequest, BatchGenerationRequest, BatchJobResponse,
    estimate_credits_cost
)

logger = logging.getLogger(__name__)


class JobService:
    """Service for managing generation jobs"""
    
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.job_queue: List[str] = []
        self.batch_jobs: Dict[str, Dict[str, Any]] = {}
        self._running = False
        
    async def initialize(self):
        """Initialize the job service"""
        self._running = True
        # Start background job processor
        asyncio.create_task(self._process_jobs())
        logger.info("Job service initialized")
        
    def create_job(self, job_type: str, params: Dict[str, Any], user_id: str, 
                   priority: str = "normal") -> Dict[str, Any]:
        """Create a new generation job"""
        job_id = f"job-{uuid.uuid4().hex[:8]}-{int(time.time())}"
        
        # Estimate credits cost
        credits_cost = estimate_credits_cost(job_type, params)
        
        # Create job record
        job = {
            "job_id": job_id,
            "type": job_type,
            "status": JobStatus.QUEUED,
            "params": params,
            "user_id": user_id,
            "priority": priority,
            "created_at": datetime.utcnow(),
            "started_at": None,
            "completed_at": None,
            "progress": 0.0,
            "message": "Queued for processing",
            "result": None,
            "error": None,
            "credits_cost": credits_cost,
            "credits_used": None
        }
        
        self.jobs[job_id] = job
        
        # Add to queue based on priority
        if priority == "high" or priority == "urgent":
            self.job_queue.insert(0, job_id)
        else:
            self.job_queue.append(job_id)
            
        logger.info(f"Created job {job_id} of type {job_type}")
        
        return {
            "job_id": job_id,
            "status": "queued",
            "created_at": job["created_at"],
            "queue_position": self._get_queue_position(job_id),
            "credits_cost": credits_cost
        }
        
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job by ID"""
        return self.jobs.get(job_id)
        
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status"""
        job = self.jobs.get(job_id)
        if not job:
            return None
            
        return {
            "job_id": job_id,
            "status": job["status"],
            "progress": job["progress"],
            "message": job["message"],
            "created_at": job["created_at"],
            "started_at": job.get("started_at"),
            "completed_at": job.get("completed_at"),
            "error": job.get("error"),
            "eta": self._estimate_eta(job_id),
            "result": job.get("result")
        }
        
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job"""
        job = self.jobs.get(job_id)
        if not job:
            return False
            
        if job["status"] in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            return False
            
        job["status"] = JobStatus.CANCELLED
        job["message"] = "Job cancelled by user"
        job["completed_at"] = datetime.utcnow()
        
        # Remove from queue if still queued
        if job_id in self.job_queue:
            self.job_queue.remove(job_id)
            
        logger.info(f"Cancelled job {job_id}")
        return True
        
    def create_batch_job(self, request: BatchGenerationRequest, user_id: str) -> Dict[str, Any]:
        """Create a batch job"""
        batch_id = f"batch-{uuid.uuid4().hex[:8]}-{int(time.time())}"
        
        # Create individual jobs for each item
        job_ids = []
        total_credits = 0
        
        for i, item in enumerate(request.items):
            # Merge shared params with item-specific params
            params = {**request.shared_params, **item}
            
            job_response = self.create_job(
                job_type=request.type,
                params=params,
                user_id=user_id,
                priority=request.priority
            )
            job_ids.append(job_response["job_id"])
            total_credits += job_response.get("credits_cost", 0)
            
        # Store batch info
        batch_job = {
            "batch_id": batch_id,
            "job_ids": job_ids,
            "status": JobStatus.QUEUED,
            "total_jobs": len(job_ids),
            "completed_jobs": 0,
            "failed_jobs": 0,
            "total_credits": total_credits,
            "created_at": datetime.utcnow(),
            "user_id": user_id
        }
        
        self.batch_jobs[batch_id] = batch_job
        
        logger.info(f"Created batch job {batch_id} with {len(job_ids)} jobs")
        
        return batch_job
        
    def get_user_jobs(self, user_id: str, status: Optional[str] = None, 
                     limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
        """Get jobs for a user"""
        user_jobs = []
        
        for job_id, job in self.jobs.items():
            if job["user_id"] == user_id:
                if status is None or job["status"] == status:
                    user_jobs.append({
                        "job_id": job_id,
                        "type": job["type"],
                        "status": job["status"],
                        "created_at": job["created_at"],
                        "completed_at": job.get("completed_at"),
                        "credits_used": job.get("credits_used"),
                        "result": job.get("result")
                    })
                    
        # Sort by creation time (newest first)
        user_jobs.sort(key=lambda x: x["created_at"], reverse=True)
        
        # Apply pagination
        return user_jobs[offset:offset + limit]
        
    def _get_queue_position(self, job_id: str) -> int:
        """Get position in queue"""
        try:
            return self.job_queue.index(job_id) + 1
        except ValueError:
            return 0
            
    def _estimate_eta(self, job_id: str) -> Optional[int]:
        """Estimate time remaining for job"""
        job = self.jobs.get(job_id)
        if not job:
            return None
            
        if job["status"] == JobStatus.COMPLETED:
            return 0
        elif job["status"] == JobStatus.PROCESSING:
            # Estimate based on progress
            if job["progress"] > 0:
                elapsed = (datetime.utcnow() - job["started_at"]).total_seconds()
                total_estimated = elapsed / job["progress"]
                return int(total_estimated - elapsed)
            return 30  # Default estimate
        elif job["status"] == JobStatus.QUEUED:
            # Estimate based on queue position
            position = self._get_queue_position(job_id)
            return position * 30  # 30 seconds per job estimate
        else:
            return None
            
    async def _process_jobs(self):
        """Background job processor"""
        while self._running:
            try:
                if self.job_queue:
                    job_id = self.job_queue.pop(0)
                    await self._process_single_job(job_id)
                else:
                    await asyncio.sleep(1)  # Wait for jobs
            except Exception as e:
                logger.error(f"Error processing jobs: {e}")
                await asyncio.sleep(5)
                
    async def _process_single_job(self, job_id: str):
        """Process a single job"""
        job = self.jobs.get(job_id)
        if not job:
            return
            
        try:
            # Update job status
            job["status"] = JobStatus.PROCESSING
            job["started_at"] = datetime.utcnow()
            job["message"] = "Processing..."
            
            logger.info(f"Processing job {job_id} of type {job['type']}")
            
            # Simulate job processing with progress updates
            await self._simulate_job_processing(job_id)
            
            # Mark as completed
            job["status"] = JobStatus.COMPLETED
            job["completed_at"] = datetime.utcnow()
            job["progress"] = 1.0
            job["message"] = "Generation completed successfully"
            job["credits_used"] = job["credits_cost"]
            
            # Generate mock result
            job["result"] = self._generate_mock_result(job["type"], job["params"])
            
            logger.info(f"Completed job {job_id}")
            
        except Exception as e:
            job["status"] = JobStatus.FAILED
            job["completed_at"] = datetime.utcnow()
            job["error"] = str(e)
            job["message"] = f"Job failed: {e}"
            logger.error(f"Job {job_id} failed: {e}")
            
    async def _simulate_job_processing(self, job_id: str):
        """Simulate job processing with progress updates"""
        job = self.jobs[job_id]
        
        # Simulate different processing times based on job type
        job_type = job["type"]
        if job_type == "text_to_image":
            total_steps = 20
            step_duration = 0.5
        elif job_type == "image_to_3d":
            total_steps = 50
            step_duration = 1.0
        elif job_type == "text_to_video":
            total_steps = 100
            step_duration = 0.8
        else:
            total_steps = 30
            step_duration = 0.6
            
        for step in range(1, total_steps + 1):
            await asyncio.sleep(step_duration)
            
            progress = step / total_steps
            job["progress"] = progress
            
            if step < total_steps * 0.3:
                job["message"] = "Loading models..."
            elif step < total_steps * 0.8:
                job["message"] = "Generating content..."
            else:
                job["message"] = "Finalizing output..."
                
    def _generate_mock_result(self, job_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mock result for testing"""
        base_url = "https://api.example.com/outputs"
        
        if job_type == "text_to_image":
            return {
                "output_url": f"{base_url}/image_{uuid.uuid4().hex[:8]}.png",
                "metadata": {
                    "seed": params.get("seed", 42),
                    "model": params.get("model", "flux-1-dev"),
                    "width": params.get("width", 512),
                    "height": params.get("height", 512),
                    "steps": params.get("num_inference_steps", 20),
                    "generation_time": 15.3
                }
            }
        elif job_type == "image_to_3d":
            return {
                "output_url": f"{base_url}/model_{uuid.uuid4().hex[:8]}.glb",
                "thumbnail_url": f"{base_url}/thumb_{uuid.uuid4().hex[:8]}.png",
                "metadata": {
                    "model": params.get("model", "hunyuan3d-mini"),
                    "quality": params.get("quality_preset", "standard"),
                    "num_views": params.get("num_views", 6),
                    "generation_time": 45.7
                }
            }
        elif job_type == "text_to_video":
            return {
                "output_url": f"{base_url}/video_{uuid.uuid4().hex[:8]}.mp4",
                "thumbnail_url": f"{base_url}/thumb_{uuid.uuid4().hex[:8]}.png",
                "metadata": {
                    "model": params.get("model", "ltx-video"),
                    "duration": params.get("duration", 3.0),
                    "fps": params.get("fps", 24),
                    "resolution": f"{params.get('width', 512)}x{params.get('height', 512)}",
                    "generation_time": 67.2
                }
            }
        else:
            return {
                "output_url": f"{base_url}/output_{uuid.uuid4().hex[:8]}.png",
                "metadata": {
                    "type": job_type,
                    "generation_time": 20.1
                }
            }


# Global job service instance
job_service = JobService()