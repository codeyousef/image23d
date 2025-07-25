"""WebSocket integration for download progress"""

import logging
from typing import Optional, Dict, Any

from ...services.websocket import ProgressStreamManager, ProgressUpdate, MessageType
from ..base import DownloadProgress

logger = logging.getLogger(__name__)


class WebSocketProgressEmitter:
    """Handles WebSocket progress emission for downloads"""
    
    def __init__(self, websocket_server: Optional[ProgressStreamManager] = None):
        self.websocket_server = websocket_server
        self.current_task_id = None
    
    def set_task_id(self, task_id: str):
        """Set the current task ID for progress updates"""
        self.current_task_id = task_id
    
    def emit_progress(
        self,
        model_name: str,
        progress: DownloadProgress,
        status: str = "downloading"
    ):
        """Emit download progress via WebSocket"""
        if not self.websocket_server or not self.current_task_id:
            return
        
        try:
            # Create progress update
            progress_data = {
                "task_type": "download",
                "model": model_name,
                "status": status,
                "percentage": progress.percentage,
                "current_file": progress.current_file,
                "downloaded_size": progress.downloaded_size,
                "total_size": progress.total_size,
                "speed": progress.speed,
                "eta": progress.eta
            }
            
            # Format message
            if progress.total_size > 0:
                downloaded_gb = progress.downloaded_size / (1024**3)
                total_gb = progress.total_size / (1024**3)
                speed_mb = progress.speed / (1024**2) if progress.speed > 0 else 0
                
                message = f"Downloading {model_name}: {progress.percentage:.1f}% ({downloaded_gb:.2f}/{total_gb:.2f} GB)"
                if speed_mb > 0:
                    message += f" - {speed_mb:.1f} MB/s"
            else:
                message = f"Downloading {model_name}: {progress.current_file}"
            
            # Create progress update
            update = ProgressUpdate(
                task_id=self.current_task_id,
                progress=progress.percentage / 100.0,  # Convert to 0-1 range
                message=message,
                details=progress_data
            )
            
            # Send update
            self.websocket_server.send_progress_update(
                MessageType.PROGRESS,
                update
            )
            
        except Exception as e:
            logger.error(f"Failed to emit WebSocket progress: {e}")
    
    def emit_status(
        self,
        model_name: str,
        status: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """Emit status update via WebSocket"""
        if not self.websocket_server or not self.current_task_id:
            return
        
        try:
            # Create status data
            status_data = {
                "task_type": "download",
                "model": model_name,
                "status": status,
                "message": message
            }
            
            if details:
                status_data.update(details)
            
            # Determine message type
            if status == "completed":
                msg_type = MessageType.COMPLETE
                progress = 1.0
            elif status == "error":
                msg_type = MessageType.ERROR
                progress = 0.0
            else:
                msg_type = MessageType.STATUS
                progress = 0.0
            
            # Create update
            update = ProgressUpdate(
                task_id=self.current_task_id,
                progress=progress,
                message=message,
                details=status_data
            )
            
            # Send update
            self.websocket_server.send_progress_update(msg_type, update)
            
        except Exception as e:
            logger.error(f"Failed to emit WebSocket status: {e}")
    
    def emit_queue_update(self, queue_info: Dict[str, Any]):
        """Emit download queue update"""
        if not self.websocket_server:
            return
        
        try:
            # Create queue update
            update = ProgressUpdate(
                task_id="download_queue",
                progress=0.0,
                message=f"Download queue: {queue_info.get('queued', 0)} waiting, {queue_info.get('active', 0)} active",
                details=queue_info
            )
            
            # Send as status update
            self.websocket_server.send_progress_update(
                MessageType.STATUS,
                update
            )
            
        except Exception as e:
            logger.error(f"Failed to emit queue update: {e}")