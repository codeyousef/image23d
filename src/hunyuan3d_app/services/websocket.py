"""WebSocket server for real-time verbose progress streaming"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import deque
import uuid
import socket
import threading

import websockets
from websockets.server import WebSocketServerProtocol

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of progress messages"""
    STATUS = "status"
    PROGRESS = "progress"
    LOG = "log"
    PREVIEW = "preview"
    ERROR = "error"
    SUCCESS = "success"
    WARNING = "warning"
    INFO = "info"
    HEARTBEAT = "heartbeat"


class LogLevel(Enum):
    """Log levels for messages"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"


@dataclass
class ProgressUpdate:
    """A progress update message"""
    task_id: str
    type: MessageType
    data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    level: LogLevel = LogLevel.INFO
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "task_id": self.task_id,
            "type": self.type.value,
            "data": self.data,
            "timestamp": self.timestamp,
            "level": self.level.value
        }


@dataclass
class TaskProgress:
    """Track progress for a specific task"""
    task_id: str
    task_type: str
    started_at: float = field(default_factory=time.time)
    current_step: str = ""
    progress: float = 0.0
    total_steps: int = 0
    completed_steps: int = 0
    status: str = "running"
    logs: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def add_log(self, message: str, level: LogLevel = LogLevel.INFO):
        """Add a log message"""
        self.logs.append({
            "message": message,
            "level": level.value,
            "timestamp": time.time()
        })


def is_port_available(port: int, host: str = "localhost") -> bool:
    """Check if a port is available for binding"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


class ProgressStreamManager:
    """Manages WebSocket connections and progress streaming"""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8765,
        buffer_size: int = 10,
        batch_interval: float = 0.1
    ):
        self.host = host
        self.port = port
        self.buffer_size = buffer_size
        self.batch_interval = batch_interval
        
        # Connected clients
        self.clients: Set[WebSocketServerProtocol] = set()
        
        # Task tracking
        self.active_tasks: Dict[str, TaskProgress] = {}
        
        # Message buffer for batching
        self.message_buffer: List[ProgressUpdate] = []
        self.last_batch_time = time.time()
        
        # Server reference
        self.server = None
        self.server_task = None
        
    async def start_server(self):
        """Start the WebSocket server"""
        # Try to find an available port if the default is in use
        ports_to_try = [self.port, 8766, 8767, 8768, 8769]
        
        # First check which ports are available
        available_port = None
        for port in ports_to_try:
            if is_port_available(port, self.host):
                available_port = port
                break
        
        if not available_port:
            logger.error("No available ports for WebSocket server")
            raise OSError("No available ports for WebSocket server")
        
        try:
            self.server = await websockets.serve(
                self.handle_client,
                self.host,
                available_port
            )
            
            self.port = available_port  # Update to the successful port
            logger.info(f"WebSocket server started on ws://{self.host}:{available_port}")
            
            # Start batch sender
            asyncio.create_task(self._batch_sender())
            
            # Start heartbeat
            asyncio.create_task(self._heartbeat_sender())
            
        except Exception as e:
            logger.error(f"Failed to start WebSocket server on port {available_port}: {e}")
            raise
            
    async def stop_server(self):
        """Stop the WebSocket server"""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            
        # Close all client connections
        for client in list(self.clients):
            await client.close()
            
        logger.info("WebSocket server stopped")
        
    async def handle_client(self, websocket: WebSocketServerProtocol, path: str):
        """Handle a new client connection"""
        self.clients.add(websocket)
        client_id = str(uuid.uuid4())[:8]
        
        logger.info(f"Client {client_id} connected from {websocket.remote_address}")
        
        try:
            # Send initial state
            await self._send_initial_state(websocket)
            
            # Handle messages from client
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._handle_client_message(websocket, data)
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON from client {client_id}")
                except Exception as e:
                    logger.error(f"Error handling message from {client_id}: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.discard(websocket)
            logger.info(f"Client {client_id} disconnected")
            
    async def _send_initial_state(self, websocket: WebSocketServerProtocol):
        """Send initial state to newly connected client"""
        # Send active tasks
        for task_id, task in self.active_tasks.items():
            message = {
                "type": "task_state",
                "data": {
                    "task_id": task_id,
                    "task_type": task.task_type,
                    "progress": task.progress,
                    "status": task.status,
                    "current_step": task.current_step,
                    "logs": list(task.logs)
                }
            }
            await websocket.send(json.dumps(message))
            
    async def _handle_client_message(
        self,
        websocket: WebSocketServerProtocol,
        data: Dict[str, Any]
    ):
        """Handle message from client"""
        msg_type = data.get("type")
        
        if msg_type == "subscribe":
            # Client wants to subscribe to specific task
            task_id = data.get("task_id")
            if task_id and task_id in self.active_tasks:
                # Send current state of the task
                task = self.active_tasks[task_id]
                await websocket.send(json.dumps({
                    "type": "task_state",
                    "data": {
                        "task_id": task_id,
                        "progress": task.progress,
                        "logs": list(task.logs)
                    }
                }))
                
        elif msg_type == "ping":
            # Respond to ping
            await websocket.send(json.dumps({"type": "pong"}))
            
    async def stream_progress(
        self,
        task_id: str,
        update: ProgressUpdate
    ):
        """Stream a progress update"""
        # Update task tracking
        if task_id not in self.active_tasks:
            self.active_tasks[task_id] = TaskProgress(
                task_id=task_id,
                task_type=update.data.get("task_type", "unknown")
            )
            
        task = self.active_tasks[task_id]
        
        # Update task state
        if update.type == MessageType.PROGRESS:
            task.progress = update.data.get("progress", 0)
            task.current_step = update.data.get("message", "")
        elif update.type in [MessageType.LOG, MessageType.INFO, MessageType.WARNING, MessageType.ERROR]:
            task.add_log(update.data.get("message", ""), update.level)
        elif update.type == MessageType.SUCCESS:
            task.status = "completed"
            task.progress = 1.0
        elif update.type == MessageType.ERROR:
            task.status = "failed"
            
        # Add to buffer
        self.message_buffer.append(update)
        
        # Send immediately if buffer is full or high priority
        if (len(self.message_buffer) >= self.buffer_size or 
            update.level in [LogLevel.ERROR, LogLevel.SUCCESS] or
            update.type == MessageType.PREVIEW):
            await self._flush_buffer()
            
    async def _flush_buffer(self):
        """Flush message buffer to all clients"""
        if not self.message_buffer or not self.clients:
            return
            
        # Prepare batch message
        messages = [msg.to_dict() for msg in self.message_buffer]
        batch = {
            "type": "batch",
            "messages": messages
        }
        
        # Send to all clients
        disconnected = []
        for client in self.clients:
            try:
                await client.send(json.dumps(batch))
            except websockets.exceptions.ConnectionClosed:
                disconnected.append(client)
            except Exception as e:
                logger.error(f"Error sending to client: {e}")
                disconnected.append(client)
                
        # Remove disconnected clients
        for client in disconnected:
            self.clients.discard(client)
            
        # Clear buffer
        self.message_buffer.clear()
        self.last_batch_time = time.time()
        
    async def _batch_sender(self):
        """Periodically flush buffer"""
        while True:
            await asyncio.sleep(self.batch_interval)
            
            # Flush if we have messages and enough time has passed
            if self.message_buffer and (time.time() - self.last_batch_time) >= self.batch_interval:
                await self._flush_buffer()
                
    async def _heartbeat_sender(self):
        """Send periodic heartbeats to keep connections alive"""
        while True:
            await asyncio.sleep(30)  # Every 30 seconds
            
            if self.clients:
                heartbeat = json.dumps({
                    "type": "heartbeat",
                    "timestamp": time.time()
                })
                
                disconnected = []
                for client in self.clients:
                    try:
                        await client.send(heartbeat)
                    except:
                        disconnected.append(client)
                        
                for client in disconnected:
                    self.clients.discard(client)
                    
    def create_progress_callback(
        self,
        task_id: str,
        task_type: str = "generation"
    ) -> Callable[[float, str], None]:
        """Create a progress callback function for a task
        
        Args:
            task_id: Unique task ID
            task_type: Type of task
            
        Returns:
            Progress callback function
        """
        def callback(progress: float, message: str):
            # Run in event loop
            asyncio.create_task(
                self.stream_progress(
                    task_id,
                    ProgressUpdate(
                        task_id=task_id,
                        type=MessageType.PROGRESS,
                        data={
                            "progress": progress,
                            "message": message,
                            "task_type": task_type
                        }
                    )
                )
            )
            
        return callback
        
    async def log_message(
        self,
        task_id: str,
        message: str,
        level: LogLevel = LogLevel.INFO
    ):
        """Log a message for a task"""
        await self.stream_progress(
            task_id,
            ProgressUpdate(
                task_id=task_id,
                type=MessageType.LOG,
                data={"message": message},
                level=level
            )
        )
        
    async def send_preview(
        self,
        task_id: str,
        preview_data: Dict[str, Any]
    ):
        """Send a preview update"""
        await self.stream_progress(
            task_id,
            ProgressUpdate(
                task_id=task_id,
                type=MessageType.PREVIEW,
                data=preview_data
            )
        )
        
    def send_progress_update(
        self,
        task_id: str,
        progress: float,
        message: str,
        task_type: str = "generation"
    ):
        """Send a progress update (synchronous wrapper for async stream_progress)"""
        try:
            update = ProgressUpdate(
                task_id=task_id,
                type=MessageType.PROGRESS,
                data={
                    "progress": progress,
                    "message": message,
                    "task_type": task_type
                }
            )
            
            # If we have an event loop, use it; otherwise schedule for later
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Create a task to run the async method
                    asyncio.create_task(self.stream_progress(task_id, update))
                else:
                    # Run in new event loop
                    loop.run_until_complete(self.stream_progress(task_id, update))
            except RuntimeError:
                # No event loop, run in thread
                def run_async():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(self.stream_progress(task_id, update))
                    loop.close()
                
                import threading
                thread = threading.Thread(target=run_async, daemon=True)
                thread.start()
                
        except Exception as e:
            logger.warning(f"Failed to send progress update: {e}")

    def complete_task(self, task_id: str, result: Optional[Dict[str, Any]] = None):
        """Mark a task as completed"""
        asyncio.create_task(
            self.stream_progress(
                task_id,
                ProgressUpdate(
                    task_id=task_id,
                    type=MessageType.SUCCESS,
                    data={"result": result} if result else {},
                    level=LogLevel.SUCCESS
                )
            )
        )
        
        # Clean up after a delay
        async def cleanup():
            await asyncio.sleep(60)  # Keep for 1 minute
            self.active_tasks.pop(task_id, None)
            
        asyncio.create_task(cleanup())
        
    def fail_task(self, task_id: str, error: str):
        """Mark a task as failed"""
        asyncio.create_task(
            self.stream_progress(
                task_id,
                ProgressUpdate(
                    task_id=task_id,
                    type=MessageType.ERROR,
                    data={"error": error},
                    level=LogLevel.ERROR
                )
            )
        )
        
    def get_active_tasks(self) -> Dict[str, Dict[str, Any]]:
        """Get information about active tasks"""
        return {
            task_id: {
                "task_type": task.task_type,
                "progress": task.progress,
                "status": task.status,
                "started_at": task.started_at,
                "current_step": task.current_step
            }
            for task_id, task in self.active_tasks.items()
        }


# Global instance for easy access
_progress_manager: Optional[ProgressStreamManager] = None
_manager_lock = threading.Lock()


def get_progress_manager() -> ProgressStreamManager:
    """Get the global progress manager instance (thread-safe singleton)"""
    global _progress_manager
    
    if _progress_manager is not None:
        return _progress_manager
        
    with _manager_lock:
        # Double-check pattern
        if _progress_manager is None:
            _progress_manager = ProgressStreamManager()
            logger.info("Created global ProgressStreamManager instance")
            
    return _progress_manager


async def start_progress_server(host: str = "localhost", port: int = 8765):
    """Start the progress server"""
    manager = get_progress_manager()
    manager.host = host
    manager.port = port
    await manager.start_server()
    
    # Keep server running
    try:
        await asyncio.Future()  # Run forever
    except KeyboardInterrupt:
        await manager.stop_server()