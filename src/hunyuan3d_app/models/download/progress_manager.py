"""Progress management for downloads"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Callable

from ..base import DownloadProgress

logger = logging.getLogger(__name__)


@dataclass
class DownloadInfo:
    """Information about an active download"""
    download_id: str
    model_name: str
    model_type: str
    repo_id: str
    target_dir: str
    progress: DownloadProgress
    start_time: float = field(default_factory=time.time)
    priority: int = 0
    status: str = "queued"
    error: Optional[str] = None
    user_callback: Optional[Callable] = None


class ProgressManager:
    """Manages download progress tracking and updates"""
    
    def __init__(self):
        # Progress tracking
        self.download_progress = DownloadProgress()
        self.file_sizes = {}
        self.current_download_model = None
        self.download_in_progress = False
    
    def create_download_info(
        self,
        download_id: str,
        model_name: str,
        model_type: str,
        repo_id: str,
        target_dir: str,
        priority: int = 0,
        user_callback: Optional[Callable] = None
    ) -> DownloadInfo:
        """Create a new download info object"""
        return DownloadInfo(
            download_id=download_id,
            model_name=model_name,
            model_type=model_type,
            repo_id=repo_id,
            target_dir=target_dir,
            progress=DownloadProgress(),
            priority=priority,
            user_callback=user_callback
        )
    
    def update_progress(self, progress_data: Dict[str, Any]):
        """Update download progress from raw data"""
        self.download_progress.current_file = progress_data.get('current_file', '')
        self.download_progress.percentage = progress_data.get('percentage', 0)
        self.download_progress.downloaded_size = progress_data.get('downloaded_size', 0)
        self.download_progress.total_size = progress_data.get('total_size', 0)
        self.download_progress.speed = progress_data.get('speed', 0)
        self.download_progress.eta = progress_data.get('eta', 0)
        
        # Log progress
        if self.download_progress.percentage > 0:
            logger.debug(
                f"Download progress: {self.download_progress.percentage:.1f}% "
                f"({self.download_progress.downloaded_size / (1024**3):.2f} GB / "
                f"{self.download_progress.total_size / (1024**3):.2f} GB) "
                f"Speed: {self.download_progress.speed / (1024**2):.1f} MB/s"
            )
    
    def update_concurrent_progress(
        self,
        download_info: DownloadInfo,
        progress: DownloadProgress,
        user_callback: Optional[Callable]
    ):
        """Update progress for a concurrent download"""
        # Update the download info's progress
        download_info.progress = progress
        download_info.status = "downloading"
        
        # Call user callback if provided
        if user_callback:
            try:
                user_callback(progress)
            except Exception as e:
                logger.error(f"Error in user progress callback: {e}")
    
    def calculate_overall_progress(self, active_downloads: Dict[str, DownloadInfo]) -> Dict[str, Any]:
        """Calculate overall progress across all active downloads"""
        if not active_downloads:
            return {
                "total_percentage": 0,
                "active_count": 0,
                "total_downloaded": 0,
                "total_size": 0,
                "overall_speed": 0
            }
        
        total_downloaded = 0
        total_size = 0
        total_speed = 0
        
        for download_info in active_downloads.values():
            progress = download_info.progress
            total_downloaded += progress.downloaded_size
            total_size += progress.total_size
            total_speed += progress.speed
        
        overall_percentage = (total_downloaded / total_size * 100) if total_size > 0 else 0
        
        return {
            "total_percentage": overall_percentage,
            "active_count": len(active_downloads),
            "total_downloaded": total_downloaded,
            "total_size": total_size,
            "overall_speed": total_speed
        }
    
    def format_progress_message(self, progress: DownloadProgress) -> str:
        """Format a progress message for display"""
        if progress.total_size > 0:
            percentage = progress.percentage
            downloaded_gb = progress.downloaded_size / (1024**3)
            total_gb = progress.total_size / (1024**3)
            speed_mb = progress.speed / (1024**2)
            
            message = f"Progress: {percentage:.1f}% ({downloaded_gb:.2f}/{total_gb:.2f} GB)"
            if speed_mb > 0:
                message += f" - {speed_mb:.1f} MB/s"
            if progress.eta > 0:
                eta_min = progress.eta / 60
                message += f" - ETA: {eta_min:.1f} min"
            
            return message
        else:
            return f"Downloading {progress.current_file}..."
    
    def reset_progress(self):
        """Reset progress tracking"""
        self.download_progress = DownloadProgress()
        self.current_download_model = None
        self.download_in_progress = False