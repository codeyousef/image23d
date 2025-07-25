"""Base Model Downloader Class

Core model downloading functionality.
"""

import logging
import os
import threading
from pathlib import Path
from typing import Optional, Dict, Any, Callable, Tuple, List

from huggingface_hub import HfApi
import shutil

from ..base import DownloadProgress
from ...services.websocket import ProgressStreamManager
from .progress_manager import ProgressManager
from .websocket_integration import WebSocketProgressEmitter
from .model_verification import ModelVerifier
from .download_worker import DownloadWorker
from .concurrent_download import ConcurrentDownloadManager

logger = logging.getLogger(__name__)


class ModelDownloader:
    """Handles model downloading from Hugging Face Hub."""
    
    def __init__(
        self,
        cache_dir: Path,
        hf_token: Optional[str] = None,
        websocket_server: Optional[ProgressStreamManager] = None
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hf_token = hf_token
        self.api = HfApi()
        self.websocket_server = websocket_server
        
        # Initialize components
        self.progress_manager = ProgressManager()
        self.websocket_emitter = WebSocketProgressEmitter(websocket_server)
        self.verifier = ModelVerifier(hf_token)
        self.concurrent_manager = ConcurrentDownloadManager(
            max_concurrent=3,
            hf_token=hf_token,
            websocket_emitter=self.websocket_emitter
        )
        
        # Legacy compatibility
        self.download_in_progress = False
        self.current_download_model = None
        self.stop_download_flag = False
        self.download_thread = None
        self.download_progress = DownloadProgress()
        self.current_task_id = None
        
        # Re-export some properties for compatibility
        self.max_concurrent_downloads = self.concurrent_manager.max_concurrent
        self.download_semaphore = self.concurrent_manager.download_semaphore
        self.active_downloads = self.concurrent_manager.active_downloads
        self.download_threads = self.concurrent_manager.download_threads
        self.download_queue = self.concurrent_manager.download_queue
        self.file_sizes = self.progress_manager.file_sizes
        self.lock = self.concurrent_manager.lock
    
    def set_token(self, token: str):
        """Set Hugging Face token."""
        self.hf_token = token
        if token:
            os.environ["HF_TOKEN"] = token
        
        # Update components
        self.verifier = ModelVerifier(token)
        self.concurrent_manager.hf_token = token
    
    def is_downloading(self) -> bool:
        """Check if any download is in progress."""
        return len(self.concurrent_manager.active_downloads) > 0 or self.download_in_progress
    
    def get_active_downloads(self) -> Dict[str, Any]:
        """Get information about all active downloads."""
        return self.concurrent_manager.get_active_downloads()
    
    def get_download_queue(self) -> List[Dict[str, Any]]:
        """Get the current download queue."""
        return self.concurrent_manager.get_download_queue()
    
    def is_model_downloading(self, model_name: str) -> bool:
        """Check if a specific model is currently downloading."""
        return self.concurrent_manager.is_model_downloading(model_name)
    
    def stop_download(self):
        """Stop current download."""
        self.stop_download_flag = True
        if self.download_thread and self.download_thread.is_alive():
            self.download_thread.join(timeout=5)
    
    def _emit_websocket_progress(self, progress_type: str = "progress"):
        """Emit progress update via WebSocket if available."""
        if self.current_download_model:
            self.websocket_emitter.set_task_id(self.current_task_id or "download")
            self.websocket_emitter.emit_progress(
                self.current_download_model,
                self.download_progress,
                "downloading"
            )
    
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
        return self.concurrent_manager.download_model_concurrent(
            repo_id=repo_id,
            model_name=model_name,
            model_type=model_type,
            target_dir=target_dir,
            progress_callback=progress_callback,
            completion_callback=completion_callback,
            specific_files=specific_files,
            direct_url=direct_url,
            priority=priority
        )
    
    def download_model(
        self,
        repo_id: str,
        model_name: str,
        model_type: str,
        target_dir: Path,
        progress_callback: Optional[Callable] = None,
        specific_files: Optional[List[str]] = None,
        direct_url: Optional[str] = None
    ) -> Tuple[bool, str]:
        """Download a model (legacy synchronous method)
        
        Returns:
            Tuple of (success, message)
        """
        # Check if already downloading
        if self.is_model_downloading(model_name):
            return False, f"{model_name} is already downloading"
        
        # Set up state
        self.download_in_progress = True
        self.current_download_model = model_name
        self.stop_download_flag = False
        self.download_progress = DownloadProgress()
        
        # Generate task ID for WebSocket
        import uuid
        self.current_task_id = str(uuid.uuid4())
        self.websocket_emitter.set_task_id(self.current_task_id)
        
        # Start download in thread
        result = {"success": False, "message": ""}
        
        def download_worker():
            try:
                # Create worker
                worker = DownloadWorker(self.hf_token)
                
                # Set up stop handling
                def check_stop():
                    if self.stop_download_flag:
                        worker.stop()
                
                # Create progress wrapper
                def wrapped_progress(progress):
                    check_stop()
                    self.download_progress = progress
                    self._update_progress(progress.to_dict())
                    self._emit_websocket_progress()
                    if progress_callback:
                        progress_callback(progress)
                
                # Download
                progress = worker.download_model(
                    repo_id=repo_id,
                    target_dir=target_dir,
                    model_name=model_name,
                    specific_files=specific_files,
                    direct_url=direct_url,
                    progress_callback=wrapped_progress
                )
                
                # Success
                result["success"] = True
                result["message"] = f"Successfully downloaded {model_name}"
                
                # Emit completion
                self.websocket_emitter.emit_status(
                    model_name,
                    "completed",
                    result["message"]
                )
                
            except InterruptedError:
                result["message"] = "Download cancelled by user"
                self.websocket_emitter.emit_status(
                    model_name,
                    "cancelled",
                    result["message"]
                )
            except Exception as e:
                result["message"] = f"Download failed: {str(e)}"
                self.websocket_emitter.emit_status(
                    model_name,
                    "error",
                    result["message"]
                )
                logger.error(f"Download error: {e}")
            finally:
                self.download_in_progress = False
                self.current_download_model = None
                self.stop_download_flag = False
        
        # Run download
        self.download_thread = threading.Thread(target=download_worker, daemon=True)
        self.download_thread.start()
        self.download_thread.join()  # Wait for completion
        
        return result["success"], result["message"]
    
    def _update_progress(self, progress_data: Dict[str, Any]):
        """Update download progress from raw data"""
        self.progress_manager.update_progress(progress_data)
        self.download_progress = self.progress_manager.download_progress
    
    def check_model_exists(self, repo_id: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Check if a model exists on Hugging Face Hub
        
        Returns:
            Tuple of (exists, model_info)
        """
        return self.verifier.check_model_exists(repo_id)
    
    def get_model_size(self, repo_id: str) -> Optional[float]:
        """Get the total size of a model in bytes
        
        Returns:
            Total size in bytes or None
        """
        return self.verifier.get_model_size(repo_id)
    
    def delete_model(self, model_path: Path) -> Tuple[bool, str]:
        """Delete a downloaded model
        
        Args:
            model_path: Path to the model directory
            
        Returns:
            Tuple of (success, message)
        """
        try:
            if not model_path.exists():
                return False, "Model directory does not exist"
            
            # Remove the directory
            shutil.rmtree(model_path)
            
            # Also check for any symlinks or cache files
            cache_patterns = [
                model_path.parent / f".{model_path.name}.lock",
                model_path.parent / f"{model_path.name}.json"
            ]
            
            for cache_file in cache_patterns:
                if cache_file.exists():
                    cache_file.unlink()
            
            return True, f"Successfully deleted {model_path.name}"
            
        except Exception as e:
            logger.error(f"Failed to delete model at {model_path}: {e}")
            return False, f"Failed to delete model: {str(e)}"
    
    # Compatibility methods for concurrent downloads
    def _concurrent_download_worker(self, download_id: str):
        """Compatibility wrapper for concurrent download worker"""
        self.concurrent_manager._concurrent_download_worker(download_id)
    
    def _update_concurrent_progress(
        self,
        download_id: str,
        progress: DownloadProgress,
        user_callback: Optional[Callable]
    ):
        """Compatibility wrapper for concurrent progress update"""
        self.concurrent_manager._update_concurrent_progress(
            download_id, progress, user_callback
        )
    
    def _handle_concurrent_completion(self, download_id: str, success: bool, message: str):
        """Compatibility wrapper for concurrent completion handling"""
        self.concurrent_manager._handle_concurrent_completion(
            download_id, success, message
        )
    
    def _process_download_queue(self):
        """Compatibility wrapper for queue processing"""
        self.concurrent_manager._process_download_queue()