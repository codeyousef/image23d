"""Model downloading functionality."""

import logging
import os
import shutil
import threading
import time
import asyncio
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, Callable, Tuple, List

from huggingface_hub import snapshot_download, HfApi
from huggingface_hub.utils import RepositoryNotFoundError

from .base import DownloadProgress
from ..services.websocket import ProgressStreamManager, ProgressUpdate, MessageType

logger = logging.getLogger(__name__)


class ModelDownloader:
    """Handles model downloading from Hugging Face Hub."""
    
    def __init__(self, cache_dir: Path, hf_token: Optional[str] = None, websocket_server: Optional[ProgressStreamManager] = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hf_token = hf_token
        self.api = HfApi()
        self.websocket_server = websocket_server
        
        # Download state
        self.max_concurrent_downloads = 3
        self.download_semaphore = threading.Semaphore(self.max_concurrent_downloads)
        
        # Download tracking
        self.active_downloads = {}  # {download_id: download_info}
        self.download_threads = {}  # {download_id: thread}
        self.download_queue = []  # Queue of pending downloads
        self.file_sizes = {}
        self.lock = threading.Lock()
        
        # Legacy compatibility
        self.download_in_progress = False
        self.current_download_model = None
        self.stop_download_flag = False
        self.download_thread = None
        self.download_progress = DownloadProgress()
        self.current_task_id = None
    
    def set_token(self, token: str):
        """Set Hugging Face token."""
        self.hf_token = token
        if token:
            os.environ["HF_TOKEN"] = token
    
    def is_downloading(self) -> bool:
        """Check if any download is in progress."""
        with self.lock:
            return len(self.active_downloads) > 0 or self.download_in_progress
    
    def get_active_downloads(self) -> Dict[str, Any]:
        """Get information about all active downloads."""
        with self.lock:
            return self.active_downloads.copy()
    
    def get_download_queue(self) -> List[Dict[str, Any]]:
        """Get the current download queue."""
        with self.lock:
            return self.download_queue.copy()
    
    def is_model_downloading(self, model_name: str) -> bool:
        """Check if a specific model is currently downloading."""
        with self.lock:
            for download_info in self.active_downloads.values():
                if download_info.get('model_name') == model_name:
                    return True
            for queued in self.download_queue:
                if queued.get('model_name') == model_name:
                    return True
        return False
    
    def stop_download(self):
        """Stop current download."""
        self.stop_download_flag = True
        if self.download_thread and self.download_thread.is_alive():
            self.download_thread.join(timeout=5)
    
    def _emit_websocket_progress(self, progress_type: str = "progress"):
        """Emit progress update via WebSocket if available."""
        if not self.websocket_server or not self.current_task_id:
            return
            
        try:
            # Create progress update
            progress_data = {
                "task_type": "download",
                "model": self.current_download_model or "Unknown",
                "status": "downloading" if self.download_in_progress else "idle",
                "percentage": self.download_progress.percentage,
                "downloaded_gb": self.download_progress.downloaded_size / (1024**3),
                "total_gb": self.download_progress.total_size / (1024**3),
                "speed_mbps": self.download_progress.speed / (1024**2) if self.download_progress.speed else 0,
                "eta_minutes": self.download_progress.eta / 60 if self.download_progress.eta else 0,
                "current_file": self.download_progress.current_file,
                "completed_files": self.download_progress.completed_files,
                "total_files": self.download_progress.total_files
            }
            
            # Map progress type to MessageType
            msg_type = MessageType.PROGRESS
            if progress_type == "error":
                msg_type = MessageType.ERROR
            elif progress_type == "success":
                msg_type = MessageType.SUCCESS
            
            update = ProgressUpdate(
                task_id=self.current_task_id,
                type=msg_type,
                data=progress_data
            )
            
            # Send via WebSocket in a thread-safe way
            if hasattr(self.websocket_server, 'stream_progress'):
                # Create a new event loop for the thread if needed
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                # Schedule the coroutine
                asyncio.run_coroutine_threadsafe(
                    self.websocket_server.stream_progress(self.current_task_id, update),
                    loop
                )
        except Exception as e:
            logger.debug(f"Failed to emit WebSocket progress: {e}")
    
    def download_model_concurrent(
        self,
        repo_id: str,
        model_name: str,
        model_type: str,
        target_dir: Path,
        progress_callback: Optional[Callable] = None,
        specific_files: Optional[List[str]] = None,
        priority: int = 0,
        direct_url: Optional[str] = None
    ) -> Tuple[bool, str, str]:
        """Download a model with support for concurrent downloads.
        
        Args:
            repo_id: HuggingFace repo ID or "direct" for direct URL downloads
            model_name: Name of the model
            model_type: Type of model
            target_dir: Target directory
            progress_callback: Progress callback
            specific_files: Specific files to download
            priority: Download priority
            direct_url: Direct URL for non-HuggingFace downloads
        
        Returns:
            Tuple of (success, message, download_id)
        """
        # Check if already downloading
        if self.is_model_downloading(model_name):
            return False, f"{model_name} is already downloading or queued", ""
        
        # Generate unique download ID
        download_id = str(uuid.uuid4())
        
        # Create download info
        download_info = {
            'download_id': download_id,
            'repo_id': repo_id,
            'model_name': model_name,
            'model_type': model_type,
            'target_dir': target_dir,
            'progress_callback': progress_callback,
            'specific_files': specific_files,
            'priority': priority,
            'status': 'queued',
            'progress': DownloadProgress(),
            'start_time': time.time(),
            'direct_url': direct_url
        }
        
        with self.lock:
            # Check if we can start immediately
            if len(self.active_downloads) < self.max_concurrent_downloads:
                # Start download immediately
                self.active_downloads[download_id] = download_info
                download_info['status'] = 'downloading'
                
                # Start download thread
                thread = threading.Thread(
                    target=self._concurrent_download_worker,
                    args=(download_id,),
                    name=f"Download-{model_name}"
                )
                self.download_threads[download_id] = thread
                thread.start()
                
                return True, f"Started downloading {model_name}", download_id
            else:
                # Add to queue
                self.download_queue.append(download_info)
                # Sort queue by priority
                self.download_queue.sort(key=lambda x: x['priority'], reverse=True)
                
                return True, f"{model_name} added to download queue (position {len(self.download_queue)})", download_id
    
    def _concurrent_download_worker(self, download_id: str):
        """Worker thread for concurrent downloads."""
        download_info = None
        
        try:
            # Acquire semaphore
            self.download_semaphore.acquire()
            
            with self.lock:
                download_info = self.active_downloads.get(download_id)
                if not download_info:
                    logger.error(f"Download {download_id} not found in active downloads")
                    return
            
            # Extract download parameters
            repo_id = download_info['repo_id']
            model_name = download_info['model_name']
            model_type = download_info['model_type']
            target_dir = download_info['target_dir']
            progress_callback = download_info['progress_callback']
            specific_files = download_info['specific_files']
            direct_url = download_info.get('direct_url')
            
            logger.info(f"Starting concurrent download of {model_name} (ID: {download_id})")
            
            # Use the existing download logic
            self._download_worker(
                repo_id=repo_id,
                model_name=model_name,
                model_type=model_type,
                target_dir=target_dir,
                progress_callback=lambda p: self._update_concurrent_progress(download_id, p, progress_callback),
                completion_callback=lambda s, m: self._handle_concurrent_completion(download_id, s, m),
                specific_files=specific_files,
                direct_url=direct_url
            )
            
        except Exception as e:
            logger.error(f"Error in concurrent download worker: {e}")
            self._handle_concurrent_completion(download_id, False, str(e))
        finally:
            # Release semaphore
            self.download_semaphore.release()
            
            # Remove from active downloads
            with self.lock:
                if download_id in self.active_downloads:
                    del self.active_downloads[download_id]
                if download_id in self.download_threads:
                    del self.download_threads[download_id]
            
            # Process queue
            self._process_download_queue()
    
    def _update_concurrent_progress(self, download_id: str, progress: DownloadProgress, user_callback: Optional[Callable]):
        """Update progress for a concurrent download."""
        with self.lock:
            if download_id in self.active_downloads:
                self.active_downloads[download_id]['progress'] = progress
        
        # Call user callback if provided
        if user_callback:
            user_callback(progress)
    
    def _handle_concurrent_completion(self, download_id: str, success: bool, message: str):
        """Handle completion of a concurrent download."""
        with self.lock:
            if download_id in self.active_downloads:
                self.active_downloads[download_id]['status'] = 'completed' if success else 'failed'
                self.active_downloads[download_id]['message'] = message
        
        logger.info(f"Download {download_id} completed: {success} - {message}")
    
    def _process_download_queue(self):
        """Process the download queue when a slot becomes available."""
        with self.lock:
            # Check if we have capacity and queued downloads
            if len(self.active_downloads) < self.max_concurrent_downloads and self.download_queue:
                # Get next download from queue
                download_info = self.download_queue.pop(0)
                download_id = download_info['download_id']
                
                # Move to active downloads
                self.active_downloads[download_id] = download_info
                download_info['status'] = 'downloading'
                
                # Start download thread
                thread = threading.Thread(
                    target=self._concurrent_download_worker,
                    args=(download_id,),
                    name=f"Download-{download_info['model_name']}"
                )
                self.download_threads[download_id] = thread
                thread.start()
                
                logger.info(f"Started queued download: {download_info['model_name']}")
    
    def download_model(
        self,
        repo_id: str,
        model_name: str,
        model_type: str,
        target_dir: Path,
        progress_callback: Optional[Callable] = None,
        completion_callback: Optional[Callable] = None,
        specific_files: Optional[List[str]] = None,
        direct_url: Optional[str] = None
    ) -> Tuple[bool, str]:
        """Download a model from Hugging Face Hub or direct URL.
        
        Args:
            repo_id: Hugging Face repository ID or "direct" for direct URL
            model_name: Name of the model
            model_type: Type of model (image, 3d, etc.)
            target_dir: Directory to download to
            progress_callback: Optional callback for progress updates
            completion_callback: Optional callback when download completes
            specific_files: Optional list of specific files to download (for GGUF models)
            direct_url: Direct URL for non-HuggingFace downloads
            
        Returns:
            Tuple of (success, message)
        """
        if self.download_in_progress:
            return False, "Another download is already in progress"
        
        # Reset download state
        self.download_in_progress = True
        self.current_download_model = model_name
        self.stop_download_flag = False
        self.download_progress = DownloadProgress()
        self.current_task_id = str(uuid.uuid4())  # Generate unique task ID
        
        # Create internal completion callback that ensures state is updated
        def internal_completion_callback(success, message):
            # Update download state
            self.download_progress.is_complete = True
            self.download_progress.error = None if success else message
            
            # Call user's completion callback if provided
            if completion_callback:
                completion_callback(success, message)
        
        # Start download in thread
        self.download_thread = threading.Thread(
            target=self._download_worker,
            args=(repo_id, model_name, model_type, target_dir, progress_callback, internal_completion_callback, specific_files, direct_url),
            name=f"ModelDownload-{model_name}",
            daemon=True
        )
        logger.info(f"Starting download thread for {model_name}")
        self.download_thread.start()
        
        return True, f"Started downloading {model_name}"
    
    def _download_worker(
        self,
        repo_id: str,
        model_name: str,
        model_type: str,
        target_dir: Path,
        progress_callback: Optional[Callable],
        completion_callback: Optional[Callable],
        specific_files: Optional[List[str]] = None,
        direct_url: Optional[str] = None
    ):
        """Worker thread for downloading models."""
        logger.info(f"Download worker started for {model_name}")
        try:
            # Create target directory
            target_dir = Path(target_dir)
            target_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Target directory: {target_dir}")
            
            # Clean up any stuck downloads
            cache_dir = target_dir / ".cache"
            if cache_dir.exists():
                logger.info("Cleaning up previous download locks...")
                import shutil
                try:
                    shutil.rmtree(cache_dir)
                    logger.info("Cleaned up cache directory")
                except Exception as e:
                    logger.warning(f"Could not clean cache: {e}")
            
            # Create progress tracking callback
            def hf_progress_callback(progress_data):
                if self.stop_download_flag:
                    raise InterruptedError("Download cancelled by user")
                
                # Update progress tracking
                self._update_progress(progress_data)
                
                # Emit WebSocket progress
                self._emit_websocket_progress()
                
                # Call user callback if provided
                if progress_callback:
                    progress_callback(self.download_progress)
            
            # Download the model
            logger.info(f"Starting download of {model_name} from {repo_id} to {target_dir}")
            
            # Handle direct URL downloads
            if direct_url and repo_id == "direct":
                logger.info(f"Using direct URL download: {direct_url}")
                
                # Extract filename from URL
                import os
                from urllib.parse import urlparse
                parsed_url = urlparse(direct_url)
                filename = os.path.basename(parsed_url.path)
                if not filename:
                    filename = f"{model_name}.model"
                
                # Initialize progress
                self.download_progress.total_files = 1
                self.download_progress.total_size = 0  # Will be updated from response headers
                self.download_progress.start_time = time.time()
                self.download_progress.current_file = filename
                
                # Download file
                local_path = target_dir / filename
                logger.info(f"Downloading {filename} to {local_path}")
                
                try:
                    import requests
                    
                    # Get file size
                    response = requests.head(direct_url, timeout=10)
                    file_size = int(response.headers.get('content-length', 0))
                    self.download_progress.total_size = file_size
                    
                    # Download with progress
                    self._download_file_with_progress(
                        repo_id="direct",
                        filename=filename,
                        target_dir=target_dir,
                        file_size=file_size,
                        progress_callback=progress_callback,
                        direct_url=direct_url
                    )
                    
                    # Mark as complete
                    self.download_progress.completed_files = 1
                    self.download_progress.percentage = 100
                    self.download_progress.is_complete = True
                    
                    if progress_callback:
                        progress_callback(self.download_progress)
                    
                    logger.info(f"Successfully downloaded {model_name}")
                    
                    if completion_callback:
                        completion_callback(True, f"Successfully downloaded {model_name}")
                    
                    return
                    
                except Exception as e:
                    error_msg = f"Failed to download from direct URL: {str(e)}"
                    logger.error(error_msg)
                    self.download_progress.error = error_msg
                    if completion_callback:
                        completion_callback(False, error_msg)
                    return
            
            elif specific_files:
                # Download only specific files (for GGUF models)
                logger.info(f"Downloading specific files: {specific_files}")
                
                # Get file sizes for progress tracking
                from huggingface_hub import get_hf_file_metadata
                file_sizes = {}
                total_size = 0
                
                for file_name in specific_files:
                    try:
                        metadata = get_hf_file_metadata(
                            url=f"https://huggingface.co/{repo_id}/resolve/main/{file_name}",
                            token=self.hf_token
                        )
                        file_sizes[file_name] = metadata.size or 0
                        total_size += metadata.size or 0
                        logger.info(f"File {file_name}: {metadata.size / (1024**3):.2f} GB")
                    except Exception as e:
                        logger.warning(f"Could not get size for {file_name}: {e}")
                        file_sizes[file_name] = 0
                
                # Initialize progress tracking
                self.download_progress.total_files = len(specific_files)
                self.download_progress.total_size = total_size
                self.download_progress.start_time = time.time()
                self.download_progress.downloaded_size = 0
                
                logger.info(f"Total GGUF download size: {total_size / (1024**3):.2f} GB")
                
                # Download each file with progress tracking
                for idx, file_name in enumerate(specific_files):
                    try:
                        if self.stop_download_flag:
                            raise InterruptedError("Download cancelled by user")
                        
                        file_size = file_sizes.get(file_name, 0)
                        logger.info(f"Downloading GGUF file ({idx+1}/{len(specific_files)}): {file_name}")
                        
                        # Update progress info
                        self.download_progress.current_file = file_name
                        self.download_progress.current_file_size = file_size
                        self.download_progress.current_file_downloaded = 0
                        self.download_progress.completed_files = idx
                        
                        # Use the detailed progress download method
                        self._download_file_with_progress(
                            repo_id=repo_id,
                            filename=file_name,
                            target_dir=target_dir,
                            file_size=file_size,
                            progress_callback=progress_callback
                        )
                        
                        # Update total downloaded
                        self.download_progress.downloaded_size += file_size
                        self.download_progress.percentage = int((self.download_progress.downloaded_size / total_size) * 100) if total_size > 0 else 100
                        
                        logger.info(f"Successfully downloaded {file_name}")
                        
                    except Exception as e:
                        if "does not exist" in str(e).lower():
                            logger.warning(f"File {file_name} not found in repository, skipping")
                            continue
                        else:
                            logger.error(f"Failed to download {file_name}: {e}")
                            raise
                
                # Mark as complete
                self.download_progress.completed_files = len(specific_files)
                self.download_progress.percentage = 100
                if progress_callback:
                    progress_callback(self.download_progress)
                    
                local_dir = target_dir
            else:
                # Download entire repository with proper progress tracking
                max_retries = 3
                retry_count = 0
                
                while retry_count < max_retries:
                    try:
                        logger.info(f"Download attempt {retry_count + 1} of {max_retries}")
                        
                        # Import required functions
                        from huggingface_hub import list_repo_files, hf_hub_download, get_hf_file_metadata
                        import requests
                        
                        # Get list of all files in the repository
                        logger.info(f"Fetching file list from {repo_id}...")
                        all_files = list(list_repo_files(repo_id=repo_id, token=self.hf_token))
                        logger.info(f"Found {len(all_files)} files in repository")
                        
                        # Filter files (skip .git files)
                        files_to_download = [f for f in all_files if not f.startswith('.git')]
                        logger.info(f"Will download {len(files_to_download)} files")
                        
                        # Get file sizes for accurate progress
                        total_size = 0
                        file_sizes = {}
                        for filename in files_to_download:
                            try:
                                metadata = get_hf_file_metadata(
                                    url=f"https://huggingface.co/{repo_id}/resolve/main/{filename}",
                                    token=self.hf_token
                                )
                                file_sizes[filename] = metadata.size or 0
                                total_size += metadata.size or 0
                            except:
                                file_sizes[filename] = 0
                        
                        # Initialize progress tracking
                        self.download_progress.total_files = len(files_to_download)
                        self.download_progress.total_size = total_size
                        self.download_progress.start_time = time.time()
                        self.download_progress.downloaded_size = 0
                        
                        logger.info(f"Total download size: {total_size / (1024**3):.2f} GB")
                        
                        # Download each file with detailed progress
                        for idx, filename in enumerate(files_to_download):
                            if self.stop_download_flag:
                                raise InterruptedError("Download cancelled by user")
                            
                            file_size = file_sizes.get(filename, 0)
                            logger.info(f"Downloading ({idx+1}/{len(files_to_download)}): {filename} ({file_size / (1024**2):.1f} MB)")
                            
                            # Update current file info
                            self.download_progress.current_file = filename
                            self.download_progress.current_file_size = file_size
                            self.download_progress.current_file_downloaded = 0
                            self.download_progress.completed_files = idx
                            
                            # Calculate overall progress
                            if total_size > 0:
                                self.download_progress.percentage = int((self.download_progress.downloaded_size / total_size) * 100)
                            
                            # Update progress
                            if progress_callback:
                                progress_callback(self.download_progress)
                            
                            try:
                                # Check if file already exists and is complete
                                local_file_path = target_dir / filename
                                if local_file_path.exists() and local_file_path.stat().st_size == file_size:
                                    logger.info(f"File {filename} already downloaded, skipping...")
                                    self.download_progress.downloaded_size += file_size
                                    continue
                                
                                # Download with custom progress tracking
                                self._download_file_with_progress(
                                    repo_id=repo_id,
                                    filename=filename,
                                    target_dir=target_dir,
                                    file_size=file_size,
                                    progress_callback=progress_callback
                                )
                                
                                # Update total downloaded size
                                self.download_progress.downloaded_size += file_size
                                
                            except Exception as file_error:
                                logger.warning(f"Failed to download {filename}: {file_error}")
                                # Continue with other files
                                continue
                        
                        # Mark as complete
                        self.download_progress.completed_files = len(files_to_download)
                        self.download_progress.percentage = 100
                        if progress_callback:
                            progress_callback(self.download_progress)
                        
                        local_dir = target_dir
                        logger.info(f"Downloaded all files successfully")
                        break
                        
                    except Exception as download_error:
                        retry_count += 1
                        if retry_count < max_retries:
                            logger.warning(f"Download attempt {retry_count} failed: {str(download_error)}")
                            logger.info(f"Retrying in 5 seconds... (attempt {retry_count + 1} of {max_retries})")
                            time.sleep(5)
                        else:
                            logger.error(f"All download attempts failed after {max_retries} tries")
                            raise
            
            # Download complete
            self.download_progress.is_complete = True
            self.download_progress.percentage = 100
            self.download_progress.completed_files = self.download_progress.total_files
            logger.info(f"Successfully downloaded {model_name} to {local_dir}")
            
            # Final progress update
            if progress_callback:
                progress_callback(self.download_progress)
            
            if completion_callback:
                completion_callback(True, f"Successfully downloaded {model_name}")
            
            # Mark download as no longer in progress
            self.download_in_progress = False
                
        except InterruptedError as e:
            logger.info(f"Download of {model_name} was cancelled")
            self.download_progress.error = "Download cancelled"
            if completion_callback:
                completion_callback(False, "Download cancelled by user")
                
        except RepositoryNotFoundError:
            error_msg = f"Repository {repo_id} not found. This may be a gated model requiring authentication."
            logger.error(error_msg)
            self.download_progress.error = error_msg
            if completion_callback:
                completion_callback(False, error_msg)
                
        except Exception as e:
            error_msg = f"Error downloading {model_name}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.download_progress.error = error_msg
            if completion_callback:
                completion_callback(False, error_msg)
                
        finally:
            self.download_in_progress = False
            self.current_download_model = None
            self.stop_download_flag = False
            logger.info(f"Download worker finished for {model_name}")
    
    def _download_file_with_progress(
        self,
        repo_id: str,
        filename: str,
        target_dir: Path,
        file_size: int,
        progress_callback: Optional[Callable],
        direct_url: Optional[str] = None
    ):
        """Download a single file with detailed progress tracking."""
        import requests
        from huggingface_hub import hf_hub_url
        
        # Get the download URL
        if direct_url:
            url = direct_url
        else:
            url = hf_hub_url(repo_id=repo_id, filename=filename)
        
        # Set up the local file path
        local_path = target_dir / filename
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Set up headers for resume
        headers = {"Authorization": f"Bearer {self.hf_token}"} if self.hf_token else {}
        resume_pos = 0
        
        # Check if partial download exists
        if local_path.exists():
            resume_pos = local_path.stat().st_size
            if resume_pos >= file_size:
                # File already complete
                logger.info(f"File {filename} already complete ({resume_pos / (1024**3):.2f} GB)")
                self.download_progress.current_file_downloaded = file_size
                return
            logger.info(f"Resuming download of {filename} from {resume_pos / (1024**3):.2f} GB / {file_size / (1024**3):.2f} GB")
            headers["Range"] = f"bytes={resume_pos}-"
        
        # Update progress at start of file
        self.download_progress.current_file_downloaded = resume_pos
        if progress_callback:
            progress_callback(self.download_progress)
        
        # Download with progress and retry logic
        max_retries = 3
        retry_count = 0
        response = None
        
        while retry_count < max_retries:
            try:
                # Increase timeout for large files
                response = requests.get(url, headers=headers, stream=True, timeout=60)
                response.raise_for_status()
                break
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                retry_count += 1
                if retry_count < max_retries:
                    logger.warning(f"Download timeout/connection error (attempt {retry_count}/{max_retries}): {e}")
                    logger.info("Retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    logger.error(f"Failed to download after {max_retries} attempts")
                    raise
            except Exception as e:
                logger.error(f"Download error: {e}")
                raise
        
        if not response:
            raise RuntimeError("Failed to establish download connection")
        
        # Write to file and track progress
        chunk_size = 8192
        mode = 'ab' if resume_pos > 0 else 'wb'
        
        with open(local_path, mode) as f:
            downloaded = resume_pos
            last_update = time.time()
            last_downloaded = resume_pos
            
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Update progress
                    self.download_progress.current_file_downloaded = downloaded
                    
                    # Calculate speed every 0.5 seconds
                    current_time = time.time()
                    if current_time - last_update > 0.5:
                        # Calculate download speed based on actual bytes downloaded since last update
                        time_diff = current_time - last_update
                        bytes_diff = downloaded - last_downloaded
                        current_speed = bytes_diff / time_diff if time_diff > 0 else 0
                        
                        # Update download rate history for smoothing
                        self.download_progress.download_rate_history.append(current_speed)
                        if len(self.download_progress.download_rate_history) > 10:
                            self.download_progress.download_rate_history.pop(0)
                        
                        # Use average of recent speeds for smoother display
                        if self.download_progress.download_rate_history:
                            self.download_progress.speed = sum(self.download_progress.download_rate_history) / len(self.download_progress.download_rate_history)
                        
                        # Calculate overall stats
                        elapsed = current_time - self.download_progress.start_time
                        if elapsed > 0 and self.download_progress.speed > 0:
                            # Calculate ETA based on total remaining bytes
                            # Note: self.download_progress.downloaded_size tracks overall progress across all files
                            total_downloaded = self.download_progress.downloaded_size + (downloaded - resume_pos)
                            remaining = self.download_progress.total_size - total_downloaded
                            self.download_progress.eta = remaining / self.download_progress.speed
                        
                        # Update UI and WebSocket
                        if progress_callback:
                            progress_callback(self.download_progress)
                        
                        # Emit WebSocket progress
                        self._emit_websocket_progress()
                        
                        last_update = current_time
                        last_downloaded = downloaded
                
                # Check if cancelled
                if self.stop_download_flag:
                    raise InterruptedError("Download cancelled")
    
    def _update_progress(self, progress_data: Dict[str, Any]):
        """Update download progress from HuggingFace callback data."""
        # Extract progress information
        if isinstance(progress_data, dict):
            # Handle different progress data formats
            if "downloaded" in progress_data and "total" in progress_data:
                self.download_progress.update(
                    downloaded=progress_data["downloaded"],
                    total=progress_data["total"],
                    filename=progress_data.get("filename", ""),
                    speed=progress_data.get("speed", 0),
                    eta=progress_data.get("eta", 0)
                )
    
    def check_model_exists(self, repo_id: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Check if a model exists on Hugging Face Hub.
        
        Args:
            repo_id: Repository ID to check
            
        Returns:
            Tuple of (exists, model_info)
        """
        try:
            model_info = self.api.model_info(repo_id, token=self.hf_token)
            return True, {
                "id": model_info.id,
                "author": model_info.author,
                "downloads": model_info.downloads,
                "likes": model_info.likes,
                "tags": model_info.tags,
                "created_at": model_info.created_at,
                "last_modified": model_info.last_modified,
                "private": model_info.private,
                "gated": model_info.gated
            }
        except Exception as e:
            logger.error(f"Error checking model {repo_id}: {e}")
            return False, None
    
    def get_model_size(self, repo_id: str) -> Optional[float]:
        """Get the total size of a model in GB.
        
        Args:
            repo_id: Repository ID
            
        Returns:
            Size in GB or None if error
        """
        try:
            model_info = self.api.model_info(repo_id, token=self.hf_token, files_metadata=True)
            
            total_size = 0
            if hasattr(model_info, 'siblings'):
                for file in model_info.siblings:
                    if hasattr(file, 'size') and file.size:
                        total_size += file.size
            
            return total_size / (1024**3)  # Convert to GB
            
        except Exception as e:
            logger.error(f"Error getting model size for {repo_id}: {e}")
            return None
    
    def delete_model(self, model_path: Path) -> Tuple[bool, str]:
        """Delete a downloaded model.
        
        Args:
            model_path: Path to the model directory
            
        Returns:
            Tuple of (success, message)
        """
        try:
            if model_path.exists():
                shutil.rmtree(model_path)
                return True, f"Successfully deleted model at {model_path}"
            else:
                return False, f"Model path {model_path} does not exist"
        except Exception as e:
            error_msg = f"Error deleting model: {str(e)}"
            logger.error(error_msg)
            return False, error_msg