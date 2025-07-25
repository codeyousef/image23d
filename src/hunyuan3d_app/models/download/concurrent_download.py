"""Concurrent download management"""

import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, Callable, Tuple, List

from .progress_manager import DownloadInfo, ProgressManager
from .download_worker import DownloadWorker
from .websocket_integration import WebSocketProgressEmitter

logger = logging.getLogger(__name__)


class ConcurrentDownloadManager:
    """Manages concurrent downloads with queuing"""
    
    def __init__(
        self,
        max_concurrent: int = 3,
        hf_token: Optional[str] = None,
        websocket_emitter: Optional[WebSocketProgressEmitter] = None
    ):
        self.max_concurrent = max_concurrent
        self.hf_token = hf_token
        self.websocket_emitter = websocket_emitter
        
        # Threading primitives
        self.download_semaphore = threading.Semaphore(max_concurrent)
        self.lock = threading.Lock()
        
        # Download tracking
        self.active_downloads: Dict[str, DownloadInfo] = {}
        self.download_threads: Dict[str, threading.Thread] = {}
        self.download_queue: List[DownloadInfo] = []
        
        # Progress manager
        self.progress_manager = ProgressManager()
    
    def is_model_downloading(self, model_name: str) -> bool:
        """Check if a specific model is currently downloading."""
        with self.lock:
            for download_info in self.active_downloads.values():
                if download_info.model_name == model_name:
                    return True
            for queued_info in self.download_queue:
                if queued_info.model_name == model_name:
                    return True
        return False
    
    def download_model_concurrent(
        self,
        repo_id: str,
        model_name: str,
        model_type: str,
        target_dir: Path,
        progress_callback: Optional[Callable] = None,
        completion_callback: Optional[Callable] = None,
        specific_files: Optional[List[str]] = None,
        direct_url: Optional[str] = None,
        priority: int = 0
    ) -> Tuple[bool, str, str]:
        """Start a concurrent download
        
        Returns:
            Tuple of (success, message, download_id)
        """
        # Generate download ID
        download_id = str(uuid.uuid4())
        
        # Create download info
        download_info = self.progress_manager.create_download_info(
            download_id=download_id,
            model_name=model_name,
            model_type=model_type,
            repo_id=repo_id,
            target_dir=str(target_dir),
            priority=priority,
            user_callback=progress_callback
        )
        
        # Store additional info
        download_info.specific_files = specific_files
        download_info.direct_url = direct_url
        download_info.completion_callback = completion_callback
        
        # Check if already downloading
        with self.lock:
            # Check active downloads
            for active_info in self.active_downloads.values():
                if (active_info.model_name == model_name and 
                    active_info.model_type == model_type):
                    return False, f"{model_name} is already downloading", ""
            
            # Check queue
            for queued_info in self.download_queue:
                if (queued_info.model_name == model_name and 
                    queued_info.model_type == model_type):
                    return False, f"{model_name} is already in queue", ""
            
            # Add to queue
            self.download_queue.append(download_info)
            # Sort by priority (higher priority first)
            self.download_queue.sort(key=lambda x: x.priority, reverse=True)
        
        # Process queue
        self._process_download_queue()
        
        return True, f"{model_name} added to download queue", download_id
    
    def _process_download_queue(self):
        """Process the download queue"""
        with self.lock:
            # Check if we can start more downloads
            if (len(self.active_downloads) >= self.max_concurrent or 
                not self.download_queue):
                return
            
            # Get next download from queue
            download_info = self.download_queue.pop(0)
            download_id = download_info.download_id
            
            # Add to active downloads
            self.active_downloads[download_id] = download_info
            download_info.status = "starting"
        
        # Start download thread
        thread = threading.Thread(
            target=self._concurrent_download_worker,
            args=(download_id,),
            daemon=True
        )
        
        with self.lock:
            self.download_threads[download_id] = thread
        
        thread.start()
        
        # Update queue status
        self._emit_queue_update()
    
    def _concurrent_download_worker(self, download_id: str):
        """Worker thread for concurrent download"""
        download_info = None
        
        try:
            # Acquire semaphore
            with self.download_semaphore:
                # Get download info
                with self.lock:
                    download_info = self.active_downloads.get(download_id)
                    if not download_info:
                        logger.error(f"Download {download_id} not found")
                        return
                
                # Create download worker
                worker = DownloadWorker(self.hf_token)
                
                # Create progress callback wrapper
                def wrapped_progress(progress):
                    self._update_concurrent_progress(
                        download_id, 
                        progress, 
                        download_info.user_callback
                    )
                
                # Download the model
                target_dir = Path(download_info.target_dir)
                
                progress = worker.download_model(
                    repo_id=download_info.repo_id,
                    target_dir=target_dir,
                    model_name=download_info.model_name,
                    specific_files=getattr(download_info, 'specific_files', None),
                    direct_url=getattr(download_info, 'direct_url', None),
                    progress_callback=wrapped_progress
                )
                
                # Success
                self._handle_concurrent_completion(
                    download_id,
                    True,
                    f"Successfully downloaded {download_info.model_name}"
                )
                
        except InterruptedError:
            self._handle_concurrent_completion(
                download_id,
                False,
                "Download cancelled"
            )
        except Exception as e:
            logger.error(f"Download error for {download_id}: {e}")
            self._handle_concurrent_completion(
                download_id,
                False,
                f"Download failed: {str(e)}"
            )
        finally:
            # Clean up and process next in queue
            with self.lock:
                if download_id in self.active_downloads:
                    del self.active_downloads[download_id]
                if download_id in self.download_threads:
                    del self.download_threads[download_id]
            
            # Process next download
            self._process_download_queue()
    
    def _update_concurrent_progress(
        self,
        download_id: str,
        progress: Any,
        user_callback: Optional[Callable]
    ):
        """Update progress for concurrent download"""
        with self.lock:
            download_info = self.active_downloads.get(download_id)
            if not download_info:
                return
            
            # Update progress
            self.progress_manager.update_concurrent_progress(
                download_info,
                progress,
                user_callback
            )
        
        # Emit WebSocket update if available
        if self.websocket_emitter and download_info:
            self.websocket_emitter.emit_progress(
                download_info.model_name,
                progress,
                "downloading"
            )
    
    def _handle_concurrent_completion(
        self,
        download_id: str,
        success: bool,
        message: str
    ):
        """Handle download completion"""
        with self.lock:
            download_info = self.active_downloads.get(download_id)
            if not download_info:
                return
            
            # Update status
            download_info.status = "completed" if success else "failed"
            download_info.error = None if success else message
        
        # Call completion callback
        if download_info and hasattr(download_info, 'completion_callback'):
            completion_callback = download_info.completion_callback
            if completion_callback:
                try:
                    completion_callback(success, message)
                except Exception as e:
                    logger.error(f"Error in completion callback: {e}")
        
        # Emit WebSocket completion
        if self.websocket_emitter and download_info:
            self.websocket_emitter.emit_status(
                download_info.model_name,
                "completed" if success else "error",
                message
            )
    
    def _emit_queue_update(self):
        """Emit download queue status update"""
        with self.lock:
            queue_info = {
                "active": len(self.active_downloads),
                "queued": len(self.download_queue),
                "max_concurrent": self.max_concurrent,
                "downloads": []
            }
            
            # Add active downloads
            for download_info in self.active_downloads.values():
                queue_info["downloads"].append({
                    "id": download_info.download_id,
                    "model": download_info.model_name,
                    "status": download_info.status,
                    "progress": download_info.progress.percentage
                })
            
            # Add queued downloads
            for download_info in self.download_queue[:5]:  # Show first 5 in queue
                queue_info["downloads"].append({
                    "id": download_info.download_id,
                    "model": download_info.model_name,
                    "status": "queued",
                    "progress": 0
                })
        
        # Emit update
        if self.websocket_emitter:
            self.websocket_emitter.emit_queue_update(queue_info)
    
    def cancel_download(self, download_id: str) -> bool:
        """Cancel a specific download
        
        Returns:
            True if cancelled, False if not found
        """
        with self.lock:
            # Check if in queue
            for i, download_info in enumerate(self.download_queue):
                if download_info.download_id == download_id:
                    self.download_queue.pop(i)
                    self._emit_queue_update()
                    return True
            
            # Check if active
            if download_id in self.active_downloads:
                # Mark for cancellation
                download_info = self.active_downloads[download_id]
                download_info.status = "cancelling"
                # The worker will handle the actual cancellation
                return True
        
        return False
    
    def get_download_status(self, download_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific download"""
        with self.lock:
            # Check active
            if download_id in self.active_downloads:
                info = self.active_downloads[download_id]
                return {
                    "status": info.status,
                    "progress": info.progress.percentage,
                    "model": info.model_name,
                    "error": info.error
                }
            
            # Check queue
            for info in self.download_queue:
                if info.download_id == download_id:
                    return {
                        "status": "queued",
                        "progress": 0,
                        "model": info.model_name,
                        "position": self.download_queue.index(info) + 1
                    }
        
        return None