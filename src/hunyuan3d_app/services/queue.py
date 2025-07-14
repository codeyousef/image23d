"""Batch processing and queue management for generation tasks"""

import asyncio
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from queue import PriorityQueue, Queue
from typing import Dict, List, Optional, Any, Callable, Tuple

import gradio as gr

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Status of a generation job"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobPriority(Enum):
    """Priority levels for jobs"""
    LOW = 3
    NORMAL = 2
    HIGH = 1
    URGENT = 0


@dataclass
class GenerationJob:
    """Represents a single generation job"""
    id: str
    type: str  # "image", "3d", "full_pipeline"
    params: Dict[str, Any]
    priority: JobPriority = JobPriority.NORMAL
    status: JobStatus = JobStatus.PENDING
    created_at: float = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    progress: float = 0.0
    progress_message: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
        if self.metadata is None:
            self.metadata = {}
            
    def __lt__(self, other):
        """For priority queue comparison"""
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        return self.created_at < other.created_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        # Custom serialization to handle PIL Images in params
        data = {
            'id': self.id,
            'type': self.type,
            'priority': self.priority.name,
            'status': self.status.name,
            'created_at': self.created_at,
            'started_at': self.started_at,
            'completed_at': self.completed_at,
            'error': self.error,
            'progress': self.progress,
            'progress_message': self.progress_message,
            'metadata': self.metadata
        }
        
        # Handle params serialization
        serialized_params = {}
        for key, value in self.params.items():
            if hasattr(value, 'save') and hasattr(value, 'format'):
                # Skip PIL Images in params - they're not needed for history
                serialized_params[f'{key}_type'] = 'PIL.Image'
            else:
                try:
                    import json
                    json.dumps(value)  # Test if serializable
                    serialized_params[key] = value
                except (TypeError, ValueError):
                    serialized_params[key] = str(value)
        data['params'] = serialized_params
        
        # Handle result serialization - PIL Images can't be serialized to JSON
        if self.result and isinstance(self.result, dict):
            serialized_result = {}
            for key, value in self.result.items():
                if hasattr(value, 'save') and hasattr(value, 'format'):
                    # Save PIL Image to file and store path instead
                    try:
                        from pathlib import Path
                        import tempfile
                        import uuid
                        
                        # Create temp file for the image
                        temp_dir = Path(tempfile.gettempdir()) / "hunyuan3d_job_images"
                        temp_dir.mkdir(parents=True, exist_ok=True)
                        
                        image_path = temp_dir / f"job_{self.id}_{key}_{uuid.uuid4().hex[:8]}.png"
                        value.save(image_path)
                        serialized_result[f'{key}_path'] = str(image_path)
                    except Exception as e:
                        logger.warning(f"Failed to save image for job {self.id}: {e}")
                        # Skip the image if we can't save it
                        continue
                else:
                    try:
                        # Test if value is JSON serializable
                        import json
                        json.dumps(value)
                        serialized_result[key] = value
                    except (TypeError, ValueError):
                        # Skip non-serializable values
                        logger.warning(f"Skipping non-serializable value for key {key} in job {self.id}")
                        continue
            data['result'] = serialized_result
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GenerationJob':
        """Create from dictionary"""
        data['priority'] = JobPriority[data['priority']]
        data['status'] = JobStatus[data['status']]
        return cls(**data)


class QueueManager:
    """Manages generation job queues and worker threads"""
    
    def __init__(
        self,
        max_workers: int = 2,
        max_queue_size: int = 100,
        job_history_dir: Optional[Path] = None
    ):
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        
        # Job storage
        self.jobs: Dict[str, GenerationJob] = {}
        self.job_queue = PriorityQueue(maxsize=max_queue_size)
        self.active_jobs: Dict[str, GenerationJob] = {}
        
        # Worker management
        self.workers: List[threading.Thread] = []
        self.stop_event = threading.Event()
        self.job_handlers: Dict[str, Callable] = {}
        
        # History storage
        self.job_history_dir = job_history_dir or Path("outputs/job_history")
        self.job_history_dir.mkdir(parents=True, exist_ok=True)
        
        # Progress callbacks
        self.progress_callbacks: Dict[str, Callable] = {}
        
        # Start workers
        self._start_workers()
        
    def _start_workers(self):
        """Start worker threads"""
        for i in range(self.max_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"QueueWorker-{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
            
        logger.info(f"Started {self.max_workers} queue workers")
        
    def _worker_loop(self):
        """Main worker loop"""
        while not self.stop_event.is_set():
            try:
                # Get job from queue with timeout
                job = self.job_queue.get(timeout=1.0)
                
                if job is None:  # Shutdown signal
                    break
                    
                # Process job
                self._process_job(job)
                
            except:
                # Timeout or empty queue, continue
                continue
                
    def _process_job(self, job: GenerationJob):
        """Process a single job"""
        logger.info(f"Processing job {job.id} (type: {job.type})")
        
        # Update job status
        job.status = JobStatus.RUNNING
        job.started_at = time.time()
        self.active_jobs[job.id] = job
        
        try:
            # Get handler for job type
            handler = self.job_handlers.get(job.type)
            if not handler:
                raise ValueError(f"No handler registered for job type: {job.type}")
                
            # Create progress callback
            def update_progress(progress: float, message: str = ""):
                job.progress = progress
                job.progress_message = message
                
                # Call registered progress callback
                if job.id in self.progress_callbacks:
                    self.progress_callbacks[job.id](job)
                    
            # Add job ID to params for handlers that need it
            params_with_id = job.params.copy()
            params_with_id["job_id"] = job.id
            
            # Execute job
            result = handler(params_with_id, update_progress)
            
            # Update job with result
            job.status = JobStatus.COMPLETED
            job.result = result
            job.completed_at = time.time()
            # Don't set progress to 1.0 here - let the final callback do it
            job.progress_message = "Completed"
            
            # Debug log the result
            if result:
                logger.info(f"Job {job.id} completed successfully with result keys: {list(result.keys())}")
            else:
                logger.warning(f"Job {job.id} completed but result is None/empty")
            logger.info(f"Job {job.id} completed successfully")
            
        except Exception as e:
            # Handle job failure
            logger.error(f"Job {job.id} failed: {str(e)}")
            job.status = JobStatus.FAILED
            job.error = str(e)
            job.completed_at = time.time()
            
        finally:
            # Remove from active jobs
            self.active_jobs.pop(job.id, None)
            
            # Save job history
            self._save_job_history(job)
            
            # Set progress to 1.0 only after everything is complete
            job.progress = 1.0
            
            # Call final progress update
            if job.id in self.progress_callbacks:
                self.progress_callbacks[job.id](job)
                
    def _save_job_history(self, job: GenerationJob):
        """Save job to history"""
        try:
            history_file = self.job_history_dir / f"{job.id}.json"
            job_dict = job.to_dict()
            with open(history_file, 'w') as f:
                json.dump(job_dict, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving job history: {e}")
            logger.debug(f"Job type: {job.type}, Job status: {job.status}")
            if job.result:
                logger.debug(f"Result keys: {list(job.result.keys()) if isinstance(job.result, dict) else type(job.result)}")
            
    def register_handler(self, job_type: str, handler: Callable):
        """Register a handler for a job type
        
        Args:
            job_type: Type of job (e.g., "image", "3d")
            handler: Function that processes the job
        """
        self.job_handlers[job_type] = handler
        logger.info(f"Registered handler for job type: {job_type}")
        
    def submit_job(
        self,
        job_type: str,
        params: Dict[str, Any],
        priority: JobPriority = JobPriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None
    ) -> GenerationJob:
        """Submit a new job to the queue
        
        Args:
            job_type: Type of job
            params: Job parameters
            priority: Job priority
            metadata: Optional metadata
            
        Returns:
            The created job
        """
        # Create job
        job = GenerationJob(
            id=str(uuid.uuid4()),
            type=job_type,
            params=params,
            priority=priority,
            metadata=metadata or {}
        )
        
        # Store job
        self.jobs[job.id] = job
        
        # Add to queue
        self.job_queue.put(job)
        
        logger.info(f"Submitted job {job.id} with priority {priority.name}")
        return job
        
    def submit_batch(
        self,
        jobs: List[Tuple[str, Dict[str, Any], JobPriority]]
    ) -> List[GenerationJob]:
        """Submit multiple jobs at once
        
        Args:
            jobs: List of (job_type, params, priority) tuples
            
        Returns:
            List of created jobs
        """
        created_jobs = []
        
        for job_type, params, priority in jobs:
            job = self.submit_job(job_type, params, priority)
            created_jobs.append(job)
            
        logger.info(f"Submitted batch of {len(created_jobs)} jobs")
        return created_jobs
        
    def get_job(self, job_id: str) -> Optional[GenerationJob]:
        """Get a job by ID"""
        return self.jobs.get(job_id)
        
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status"""
        pending_jobs = [
            job for job in self.jobs.values()
            if job.status == JobStatus.PENDING
        ]
        
        return {
            "total_jobs": len(self.jobs),
            "pending": len(pending_jobs),
            "active": len(self.active_jobs),
            "completed": len([j for j in self.jobs.values() if j.status == JobStatus.COMPLETED]),
            "failed": len([j for j in self.jobs.values() if j.status == JobStatus.FAILED]),
            "queue_size": self.job_queue.qsize(),
            "max_workers": self.max_workers,
            "active_workers": len([w for w in self.workers if w.is_alive()])
        }
        
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending job
        
        Args:
            job_id: Job ID to cancel
            
        Returns:
            True if cancelled, False if not found or already running
        """
        job = self.jobs.get(job_id)
        if not job:
            return False
            
        if job.status == JobStatus.PENDING:
            job.status = JobStatus.CANCELLED
            job.completed_at = time.time()
            self._save_job_history(job)
            logger.info(f"Cancelled job {job_id}")
            return True
            
        return False
        
    def clear_completed_jobs(self):
        """Clear completed and failed jobs from memory"""
        to_remove = []
        
        for job_id, job in self.jobs.items():
            if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                to_remove.append(job_id)
                
        for job_id in to_remove:
            del self.jobs[job_id]
            
        logger.info(f"Cleared {len(to_remove)} completed jobs")
        
    def register_progress_callback(self, job_id: str, callback: Callable):
        """Register a callback for job progress updates"""
        self.progress_callbacks[job_id] = callback
        
    def unregister_progress_callback(self, job_id: str):
        """Unregister a progress callback"""
        self.progress_callbacks.pop(job_id, None)
        
    def shutdown(self):
        """Shutdown the queue manager"""
        logger.info("Shutting down queue manager...")
        
        # Signal workers to stop
        self.stop_event.set()
        
        # Add None to queue to wake up workers
        for _ in range(self.max_workers):
            try:
                self.job_queue.put(None, block=False)
            except:
                pass
                
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5.0)
            
        logger.info("Queue manager shutdown complete")
        
    def create_ui_component(self) -> gr.Group:
        """Create Gradio UI component for queue management
        
        Returns:
            Gradio Group component
        """
        with gr.Group() as queue_group:
            gr.Markdown("### 📋 Generation Queue")
            
            # Queue status
            with gr.Row():
                status_display = gr.JSON(
                    value=self.get_queue_status(),
                    label="Queue Status"
                )
                
            # Job list
            with gr.Row():
                with gr.Column():
                    job_list = gr.Dataframe(
                        headers=["ID", "Type", "Status", "Priority", "Progress", "Created"],
                        label="Jobs",
                        interactive=False
                    )
                    
                with gr.Column():
                    selected_job_info = gr.JSON(
                        label="Selected Job Details",
                        value={}
                    )
                    
                    # Add result preview
                    result_preview = gr.Image(
                        label="Result Preview",
                        visible=False,
                        elem_id="job_result_preview"
                    )
                    
            # Controls
            with gr.Row():
                refresh_btn = gr.Button("🔄 Refresh", size="sm")
                clear_btn = gr.Button("🗑️ Clear Completed", size="sm")
                cancel_btn = gr.Button("❌ Cancel Selected", size="sm")
                
            # Auto-refresh
            auto_refresh = gr.Checkbox(
                label="Auto-refresh (every 2 seconds)",
                value=True
            )
            
            # Update functions
            def update_job_list():
                """Update job list display"""
                rows = []
                for job in sorted(self.jobs.values(), key=lambda j: j.created_at, reverse=True):
                    rows.append([
                        job.id[:8] + "...",
                        job.type,
                        job.status.value,
                        job.priority.name,
                        f"{job.progress*100:.1f}%",
                        datetime.fromtimestamp(job.created_at).strftime("%H:%M:%S")
                    ])
                return rows[:50]  # Limit to 50 most recent
                
            def update_status():
                """Update queue status"""
                return self.get_queue_status()
                
            def select_job(evt: gr.SelectData):
                """Handle job selection with result preview"""
                if evt.index[0] < len(self.jobs):
                    job_list_sorted = sorted(
                        self.jobs.values(),
                        key=lambda j: j.created_at,
                        reverse=True
                    )
                    if evt.index[0] < len(job_list_sorted):
                        job = job_list_sorted[evt.index[0]]
                        job_dict = job.to_dict()
                        
                        # Check if job has image result
                        preview_update = gr.update(visible=False)
                        if job.result and 'path' in job.result and job.result['path']:
                            from pathlib import Path
                            if Path(job.result['path']).exists():
                                preview_update = gr.update(
                                    visible=True,
                                    value=job.result['path']
                                )
                        
                        return job_dict, preview_update
                return {}, gr.update(visible=False)
                
            def clear_completed():
                """Clear completed jobs"""
                self.clear_completed_jobs()
                return update_job_list(), update_status()
                
            def cancel_selected(selected_job_info):
                """Cancel selected job"""
                if selected_job_info and "id" in selected_job_info:
                    self.cancel_job(selected_job_info["id"])
                return update_job_list(), update_status()
                
            # Wire up events
            refresh_btn.click(
                lambda: [update_job_list(), update_status()],
                outputs=[job_list, status_display]
            )
            
            clear_btn.click(
                clear_completed,
                outputs=[job_list, status_display]
            )
            
            cancel_btn.click(
                cancel_selected,
                inputs=[selected_job_info],
                outputs=[job_list, status_display]
            )
            
            job_list.select(
                select_job,
                outputs=[selected_job_info, result_preview]
            )
            
            # Auto-refresh timer
            def auto_update(should_refresh):
                if should_refresh:
                    return update_job_list(), update_status()
                return gr.update(), gr.update()
                
            # Set up periodic refresh using a button (since .load() doesn't work on Group)
            refresh_btn = gr.Button("🔄 Refresh Queue", variant="secondary")
            refresh_btn.click(
                lambda: [update_job_list(), update_status()],
                outputs=[job_list, status_display]
            )
            
            # This would need a timer component in real implementation for auto-refresh
            # For now, users need to click refresh manually
            
        return queue_group