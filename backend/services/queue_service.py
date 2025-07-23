"""
Queue service for managing generation jobs
"""

import asyncio
import json
from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import uuid4
import redis.asyncio as redis

from backend.models.generation import GenerationJob, JobStatus
from backend.api.websocket import get_connection_manager
from core.services.gpu_orchestrator import GPUOrchestrator, ExecutionMode

class QueueService:
    """
    Service for managing the generation job queue
    """
    
    def __init__(self):
        self.redis = None
        self.gpu_orchestrator = None
        self.processing = False
        self.workers = []
        
    async def initialize(self):
        """Initialize queue service"""
        # Connect to Redis
        try:
            self.redis = await redis.from_url(
                "redis://localhost:6379",
                encoding="utf-8",
                decode_responses=True
            )
            await self.redis.ping()
            print("✅ Connected to Redis")
        except:
            print("⚠️  Redis not available, using in-memory queue")
            self.redis = None
            self._memory_queue = []
            self._memory_jobs = {}
            
        # Initialize GPU orchestrator
        import os
        api_key = os.getenv("RUNPOD_API_KEY")
        self.gpu_orchestrator = GPUOrchestrator(runpod_api_key=api_key)
        
        # Start workers
        self.processing = True
        for i in range(2):  # 2 concurrent workers
            worker = asyncio.create_task(self._process_queue(i))
            self.workers.append(worker)
            
    async def enqueue(self, job: GenerationJob) -> str:
        """Add a job to the queue"""
        job_data = job.dict()
        job_data["created_at"] = datetime.utcnow().isoformat()
        job_data["status"] = JobStatus.PENDING.value
        
        if self.redis:
            # Store job data
            await self.redis.hset(f"job:{job.id}", mapping=job_data)
            # Add to queue
            await self.redis.lpush("job_queue", job.id)
        else:
            # In-memory fallback
            self._memory_jobs[job.id] = job_data
            self._memory_queue.append(job.id)
            
        # Notify via WebSocket
        manager = get_connection_manager()
        await manager.send_to_user(job.user_id, {
            "type": "job_queued",
            "job_id": job.id,
            "position": await self._get_queue_position(job.id)
        })
        
        return job.id
        
    async def get_job(self, job_id: str) -> Optional[GenerationJob]:
        """Get job details"""
        if self.redis:
            job_data = await self.redis.hgetall(f"job:{job_id}")
            if not job_data:
                return None
        else:
            job_data = self._memory_jobs.get(job_id)
            if not job_data:
                return None
                
        return GenerationJob(**job_data)
        
    async def get_user_jobs(
        self,
        user_id: str,
        status: Optional[JobStatus] = None,
        limit: int = 20,
        offset: int = 0
    ) -> List[GenerationJob]:
        """Get user's jobs"""
        jobs = []
        
        if self.redis:
            # Get all job IDs for user
            pattern = f"job:*"
            cursor = 0
            user_jobs = []
            
            while True:
                cursor, keys = await self.redis.scan(
                    cursor, match=pattern, count=100
                )
                for key in keys:
                    job_data = await self.redis.hgetall(key)
                    if job_data.get("user_id") == user_id:
                        if not status or job_data.get("status") == status.value:
                            user_jobs.append(job_data)
                if cursor == 0:
                    break
                    
            # Sort by created_at
            user_jobs.sort(
                key=lambda x: x.get("created_at", ""),
                reverse=True
            )
            
            # Apply pagination
            user_jobs = user_jobs[offset:offset + limit]
            jobs = [GenerationJob(**job) for job in user_jobs]
        else:
            # In-memory fallback
            for job_id, job_data in self._memory_jobs.items():
                if job_data.get("user_id") == user_id:
                    if not status or job_data.get("status") == status.value:
                        jobs.append(GenerationJob(**job_data))
                        
            # Sort and paginate
            jobs.sort(key=lambda x: x.created_at, reverse=True)
            jobs = jobs[offset:offset + limit]
            
        return jobs
        
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a job"""
        job = await self.get_job(job_id)
        if not job:
            return False
            
        # Update status
        job.status = JobStatus.CANCELLED
        job.completed_at = datetime.utcnow()
        
        if self.redis:
            await self.redis.hset(f"job:{job_id}", mapping={
                "status": job.status.value,
                "completed_at": job.completed_at.isoformat()
            })
        else:
            self._memory_jobs[job_id].update({
                "status": job.status.value,
                "completed_at": job.completed_at.isoformat()
            })
            
        # Notify via WebSocket
        manager = get_connection_manager()
        await manager.send_job_update(job_id, {
            "status": "cancelled",
            "message": "Job cancelled by user"
        })
        
        return True
        
    async def estimate_job(self, job: GenerationJob) -> Dict[str, Any]:
        """Estimate job cost and time"""
        estimate = self.gpu_orchestrator.estimate_cost(
            task_type=f"{job.type}_generation",
            inputs=job.request,
            mode=job.execution_mode
        )
        
        # Add queue wait time
        queue_length = await self._get_queue_length()
        avg_processing_time = 60  # seconds, would be calculated from history
        estimate["queue_wait_seconds"] = queue_length * avg_processing_time
        estimate["total_time_seconds"] = (
            estimate["estimated_seconds"] + 
            estimate["queue_wait_seconds"]
        )
        
        return estimate
        
    async def get_queue_position(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get queue position for a job"""
        position = await self._get_queue_position(job_id)
        if position == -1:
            return None
            
        return {
            "position": position,
            "ahead_in_queue": position - 1,
            "estimated_wait": position * 60  # seconds
        }
        
    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        if self.redis:
            total_pending = await self.redis.llen("job_queue")
            # Count running jobs
            total_running = 0
            pattern = f"job:*"
            cursor = 0
            
            while True:
                cursor, keys = await self.redis.scan(
                    cursor, match=pattern, count=100
                )
                for key in keys:
                    status = await self.redis.hget(key, "status")
                    if status == JobStatus.RUNNING.value:
                        total_running += 1
                if cursor == 0:
                    break
        else:
            total_pending = len(self._memory_queue)
            total_running = sum(
                1 for job in self._memory_jobs.values()
                if job.get("status") == JobStatus.RUNNING.value
            )
            
        return {
            "total_pending": total_pending,
            "total_running": total_running,
            "average_wait_time": 60,  # Would calculate from history
            "average_processing_time": 120,  # Would calculate from history
            "gpu_utilization": 0.75,  # Would get from GPU monitor
            "user_stats": {}  # Would aggregate per user
        }
        
    async def retry_job(self, job_id: str) -> str:
        """Retry a failed job"""
        original_job = await self.get_job(job_id)
        if not original_job:
            raise ValueError("Job not found")
            
        # Create new job with same parameters
        new_job = GenerationJob(
            id=str(uuid4()),
            user_id=original_job.user_id,
            type=original_job.type,
            request=original_job.request,
            execution_mode=original_job.execution_mode,
            credits_required=original_job.credits_required
        )
        
        # Enqueue new job
        await self.enqueue(new_job)
        
        return new_job.id
        
    async def process_job(self, job_id: str):
        """Process a specific job (called from background task)"""
        # This is handled by the queue workers
        pass
        
    async def _process_queue(self, worker_id: int):
        """Worker process for handling queued jobs"""
        print(f"🔧 Queue worker {worker_id} started")
        
        while self.processing:
            try:
                # Get next job from queue
                job_id = await self._dequeue()
                if not job_id:
                    await asyncio.sleep(1)
                    continue
                    
                # Process the job
                await self._execute_job(job_id)
                
            except Exception as e:
                print(f"❌ Worker {worker_id} error: {e}")
                await asyncio.sleep(5)
                
    async def _dequeue(self) -> Optional[str]:
        """Get next job from queue"""
        if self.redis:
            job_id = await self.redis.rpop("job_queue")
            return job_id
        else:
            if self._memory_queue:
                return self._memory_queue.pop(0)
            return None
            
    async def _execute_job(self, job_id: str):
        """Execute a job"""
        job = await self.get_job(job_id)
        if not job:
            return
            
        # Update status to running
        job.status = JobStatus.RUNNING
        job.started_at = datetime.utcnow()
        await self._update_job(job)
        
        # Notify start
        manager = get_connection_manager()
        await manager.send_job_update(job_id, {
            "status": "running",
            "message": "Starting generation..."
        })
        
        try:
            # Progress callback
            async def progress_callback(progress: float, message: str):
                await manager.send_job_update(job_id, {
                    "progress": progress,
                    "message": message
                })
                
            # Execute via GPU orchestrator
            result = await self.gpu_orchestrator.execute(
                task_type=f"{job.type}_generation",
                inputs=job.request,
                mode=job.execution_mode,
                callback=lambda p, m: asyncio.create_task(progress_callback(p, m))
            )
            
            # Update job with results
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            job.progress = 100.0
            job.result_url = result.get("output_path", "")
            job.cost_usd = result.get("cost_usd", 0.0)
            
            # Deduct credits
            # await credit_service.deduct_credits(job.user_id, job.credits_required)
            
            await self._update_job(job)
            
            # Notify completion
            await manager.send_job_update(job_id, {
                "status": "completed",
                "message": "Generation complete!",
                "result_url": job.result_url
            })
            
        except Exception as e:
            # Update job with error
            job.status = JobStatus.FAILED
            job.completed_at = datetime.utcnow()
            job.error = str(e)
            await self._update_job(job)
            
            # Notify failure
            await manager.send_job_update(job_id, {
                "status": "failed",
                "message": "Generation failed",
                "error": str(e)
            })
            
    async def _update_job(self, job: GenerationJob):
        """Update job in storage"""
        job_data = job.dict()
        
        if self.redis:
            await self.redis.hset(f"job:{job.id}", mapping=job_data)
        else:
            self._memory_jobs[job.id] = job_data
            
    async def _get_queue_position(self, job_id: str) -> int:
        """Get position of job in queue"""
        if self.redis:
            queue = await self.redis.lrange("job_queue", 0, -1)
            if job_id in queue:
                return queue.index(job_id) + 1
        else:
            if job_id in self._memory_queue:
                return self._memory_queue.index(job_id) + 1
        return -1
        
    async def _get_queue_length(self) -> int:
        """Get total queue length"""
        if self.redis:
            return await self.redis.llen("job_queue")
        else:
            return len(self._memory_queue)