"""
Core queue management service shared between platforms
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
import json
from concurrent.futures import ThreadPoolExecutor
import uuid

from ..models.queue import (
    JobRequest, JobStatus, JobResult, JobPriority, 
    JobType, JobProgress, QueueStats
)

logger = logging.getLogger(__name__)

class QueueManager:
    """
    Manages job queues for both desktop and web applications
    """
    
    def __init__(self, max_workers: int = 2, job_history_dir: Optional[Path] = None):
        self.max_workers = max_workers
        self.job_history_dir = job_history_dir
        if self.job_history_dir:
            self.job_history_dir.mkdir(parents=True, exist_ok=True)
            
        # Job storage
        self._jobs: Dict[str, JobRequest] = {}
        self._job_results: Dict[str, JobResult] = {}
        self._job_progress: Dict[str, JobProgress] = {}
        
        # Queue by priority
        self._queues: Dict[JobPriority, asyncio.Queue] = {
            priority: asyncio.Queue() for priority in JobPriority
        }
        
        # Job handlers
        self._handlers: Dict[JobType, Callable] = {}
        
        # Worker management
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._workers: List[asyncio.Task] = []
        self._running = False
        
        # Progress callbacks
        self._progress_callbacks: Dict[str, List[Callable]] = {}
        
    def register_handler(self, job_type: JobType, handler: Callable):
        """Register a handler for a specific job type"""
        self._handlers[job_type] = handler
        logger.info(f"Registered handler for job type: {job_type}")
        
    async def submit_job(self, job: JobRequest) -> str:
        """Submit a job to the queue"""
        # Store job
        self._jobs[job.id] = job
        self._job_progress[job.id] = JobProgress(
            job_id=job.id,
            status=JobStatus.QUEUED,
            progress=0.0,
            message="Job queued"
        )
        
        # Add to appropriate priority queue
        await self._queues[job.priority].put(job.id)
        
        logger.info(f"Job {job.id} submitted with priority {job.priority}")
        
        # Save job to history if enabled
        if self.job_history_dir:
            self._save_job_to_history(job)
            
        return job.id
        
    async def get_job_progress(self, job_id: str) -> Optional[JobProgress]:
        """Get current progress for a job"""
        return self._job_progress.get(job_id)
        
    async def get_job_result(self, job_id: str) -> Optional[JobResult]:
        """Get result for a completed job"""
        return self._job_results.get(job_id)
        
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a job"""
        if job_id not in self._jobs:
            return False
            
        job = self._jobs[job_id]
        
        # Update status
        if job_id in self._job_progress:
            self._job_progress[job_id].status = JobStatus.CANCELLED
            
        # Create result
        self._job_results[job_id] = JobResult(
            job_id=job_id,
            status=JobStatus.CANCELLED,
            error="Job cancelled by user",
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            duration_seconds=0.0
        )
        
        logger.info(f"Job {job_id} cancelled")
        return True
        
    def register_progress_callback(self, job_id: str, callback: Callable):
        """Register a callback for job progress updates"""
        if job_id not in self._progress_callbacks:
            self._progress_callbacks[job_id] = []
        self._progress_callbacks[job_id].append(callback)
        
    async def start(self):
        """Start the queue processing"""
        if self._running:
            return
            
        self._running = True
        
        # Start worker tasks
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker_loop(i))
            self._workers.append(worker)
            
        logger.info(f"Queue manager started with {self.max_workers} workers")
        
    async def stop(self):
        """Stop the queue processing"""
        self._running = False
        
        # Cancel all workers
        for worker in self._workers:
            worker.cancel()
            
        # Wait for workers to finish
        await asyncio.gather(*self._workers, return_exceptions=True)
        
        # Shutdown executor
        self._executor.shutdown(wait=True)
        
        logger.info("Queue manager stopped")
        
    async def _worker_loop(self, worker_id: int):
        """Worker loop that processes jobs"""
        logger.info(f"Worker {worker_id} started")
        
        while self._running:
            try:
                # Get next job from highest priority queue
                job_id = await self._get_next_job()
                if not job_id:
                    await asyncio.sleep(0.1)
                    continue
                    
                job = self._jobs.get(job_id)
                if not job:
                    continue
                    
                # Update status
                self._update_progress(job_id, JobStatus.RUNNING, 0.0, "Starting job...")
                
                # Get handler
                handler = self._handlers.get(job.type)
                if not handler:
                    logger.error(f"No handler for job type: {job.type}")
                    self._complete_job(job_id, JobStatus.FAILED, error="No handler available")
                    continue
                    
                # Create progress callback
                def progress_callback(progress: float, message: str = ""):
                    self._update_progress(job_id, JobStatus.RUNNING, progress, message)
                    
                # Execute job
                started_at = datetime.utcnow()
                try:
                    result = await handler(job.params, progress_callback)
                    
                    # Complete job
                    self._complete_job(
                        job_id,
                        JobStatus.COMPLETED,
                        result=result,
                        started_at=started_at
                    )
                    
                except Exception as e:
                    logger.error(f"Job {job_id} failed: {str(e)}")
                    self._complete_job(
                        job_id,
                        JobStatus.FAILED,
                        error=str(e),
                        started_at=started_at
                    )
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {str(e)}")
                
        logger.info(f"Worker {worker_id} stopped")
        
    async def _get_next_job(self) -> Optional[str]:
        """Get next job from highest priority queue"""
        for priority in [JobPriority.URGENT, JobPriority.HIGH, JobPriority.NORMAL, JobPriority.LOW]:
            queue = self._queues[priority]
            if not queue.empty():
                try:
                    return await asyncio.wait_for(queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue
        return None
        
    def _update_progress(self, job_id: str, status: JobStatus, progress: float, message: str):
        """Update job progress"""
        if job_id not in self._job_progress:
            return
            
        self._job_progress[job_id] = JobProgress(
            job_id=job_id,
            status=status,
            progress=progress,
            message=message
        )
        
        # Call registered callbacks
        if job_id in self._progress_callbacks:
            for callback in self._progress_callbacks[job_id]:
                try:
                    callback(self._job_progress[job_id])
                except Exception as e:
                    logger.error(f"Progress callback error: {e}")
                    
    def _complete_job(self, job_id: str, status: JobStatus, result: Any = None, error: str = None, started_at: datetime = None):
        """Mark a job as complete"""
        if not started_at:
            started_at = datetime.utcnow()
            
        completed_at = datetime.utcnow()
        duration = (completed_at - started_at).total_seconds()
        
        # Create result
        job_result = JobResult(
            job_id=job_id,
            status=status,
            result=result,
            error=error,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=duration
        )
        
        self._job_results[job_id] = job_result
        
        # Update progress
        self._update_progress(
            job_id,
            status,
            100.0 if status == JobStatus.COMPLETED else 0.0,
            "Job completed" if status == JobStatus.COMPLETED else f"Job failed: {error}"
        )
        
        # Save to history
        if self.job_history_dir:
            self._save_result_to_history(job_id, job_result)
            
        # Clean up callbacks
        if job_id in self._progress_callbacks:
            del self._progress_callbacks[job_id]
            
    def _save_job_to_history(self, job: JobRequest):
        """Save job to history file"""
        if not self.job_history_dir:
            return
            
        job_file = self.job_history_dir / f"job_{job.id}.json"
        with open(job_file, 'w') as f:
            json.dump(job.dict(), f, indent=2, default=str)
            
    def _save_result_to_history(self, job_id: str, result: JobResult):
        """Save job result to history file"""
        if not self.job_history_dir:
            return
            
        result_file = self.job_history_dir / f"result_{job_id}.json"
        with open(result_file, 'w') as f:
            json.dump(result.dict(), f, indent=2, default=str)
            
    def get_queue_stats(self) -> QueueStats:
        """Get current queue statistics"""
        stats = QueueStats()
        
        # Count jobs by status
        for job_id, progress in self._job_progress.items():
            if progress.status == JobStatus.QUEUED:
                stats.total_queued += 1
            elif progress.status == JobStatus.RUNNING:
                stats.total_running += 1
                
        # Count by priority
        for priority, queue in self._queues.items():
            stats.by_priority[priority] = queue.qsize()
            
        # Count by type
        for job in self._jobs.values():
            if job.id in self._job_progress:
                status = self._job_progress[job.id].status
                if status in [JobStatus.QUEUED, JobStatus.RUNNING]:
                    stats.by_type[job.type] = stats.by_type.get(job.type, 0) + 1
                    
        return stats