"""WebSocket-based real-time progress component for Gradio."""

import json
import logging
import asyncio
import threading
from typing import Optional, Callable, Dict, Any
import gradio as gr
import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

logger = logging.getLogger(__name__)


class WebSocketProgressClient:
    """WebSocket client for real-time progress updates."""
    
    def __init__(self, 
                 host: str = "localhost",
                 port: int = 8765,
                 on_message: Optional[Callable] = None,
                 on_error: Optional[Callable] = None,
                 on_connect: Optional[Callable] = None,
                 on_disconnect: Optional[Callable] = None):
        self.host = host
        self.port = port
        self.ws_url = f"ws://{host}:{port}"
        
        # Callbacks
        self.on_message = on_message or (lambda msg: None)
        self.on_error = on_error or (lambda err: logger.error(f"WebSocket error: {err}"))
        self.on_connect = on_connect or (lambda: logger.info("WebSocket connected"))
        self.on_disconnect = on_disconnect or (lambda: logger.info("WebSocket disconnected"))
        
        # Connection state
        self.websocket = None
        self.running = False
        self.connection_thread = None
        self.loop = None
        
    async def _connect(self):
        """Connect to WebSocket server."""
        retry_count = 0
        max_retries = 5
        
        while self.running and retry_count < max_retries:
            try:
                logger.info(f"Attempting to connect to WebSocket at {self.ws_url}...")
                self.websocket = await websockets.connect(self.ws_url)
                logger.info("WebSocket connection established")
                self.on_connect()
                
                # Listen for messages
                await self._listen()
                
            except ConnectionRefusedError:
                retry_count += 1
                logger.warning(f"WebSocket connection refused, retry {retry_count}/{max_retries}")
                await asyncio.sleep(2 ** retry_count)  # Exponential backoff
                
            except ConnectionClosed:
                logger.info("WebSocket connection closed")
                self.on_disconnect()
                if self.running:
                    await asyncio.sleep(5)  # Wait before reconnecting
                    
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                self.on_error(str(e))
                await asyncio.sleep(5)
                
    async def _listen(self):
        """Listen for WebSocket messages."""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    self.on_message(data)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON received: {message}")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    
        except WebSocketException as e:
            logger.error(f"WebSocket exception: {e}")
            self.on_error(str(e))
            
    def _run_event_loop(self):
        """Run the asyncio event loop in a separate thread."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._connect())
        
    def start(self):
        """Start the WebSocket client."""
        if self.running:
            logger.warning("WebSocket client already running")
            return
            
        self.running = True
        self.connection_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.connection_thread.start()
        logger.info("WebSocket client started")
        
    def stop(self):
        """Stop the WebSocket client."""
        self.running = False
        
        if self.websocket and not self.websocket.closed:
            asyncio.run_coroutine_threadsafe(self.websocket.close(), self.loop)
            
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)
            
        if self.connection_thread:
            self.connection_thread.join(timeout=5)
            
        logger.info("WebSocket client stopped")
        
    async def send_message(self, message: Dict[str, Any]):
        """Send a message to the server."""
        if self.websocket and not self.websocket.closed:
            await self.websocket.send(json.dumps(message))
            

class WebSocketProgressComponent:
    """Gradio component for WebSocket-based progress updates."""
    
    def __init__(self, 
                 host: str = "localhost",
                 port: int = 8765,
                 task_filter: Optional[str] = None):
        self.host = host
        self.port = port
        self.task_filter = task_filter
        self.client = None
        self.current_progress = {}
        self.update_callback = None
        
    def create_component(self) -> gr.HTML:
        """Create the Gradio HTML component."""
        progress_html = gr.HTML(
            value="<div style='padding: 10px; border: 1px solid #ddd; border-radius: 5px;'>"
                  "<p>Waiting for connection...</p></div>",
            elem_id="websocket_progress"
        )
        
        # Set up the update callback
        def update_progress_display():
            return self._format_progress_html()
            
        self.update_callback = update_progress_display
        
        return progress_html
        
    def _format_progress_html(self) -> str:
        """Format current progress as HTML."""
        if not self.current_progress:
            return "<div style='padding: 10px;'><p>No active tasks</p></div>"
            
        html_parts = ["<div style='padding: 10px;'>"]
        
        for task_id, progress in self.current_progress.items():
            task_type = progress.get('type', 'unknown')
            status = progress.get('status', 'unknown')
            
            # Filter tasks if needed
            if self.task_filter and task_type != self.task_filter:
                continue
                
            if task_type == 'download':
                html_parts.append(self._format_download_progress(progress))
            elif task_type == 'generation':
                html_parts.append(self._format_generation_progress(progress))
            else:
                html_parts.append(self._format_generic_progress(progress))
                
        html_parts.append("</div>")
        return "".join(html_parts)
        
    def _format_download_progress(self, progress: Dict[str, Any]) -> str:
        """Format download progress."""
        model = progress.get('model', 'Unknown')
        percentage = progress.get('percentage', 0)
        downloaded_gb = progress.get('downloaded_gb', 0)
        total_gb = progress.get('total_gb', 0)
        speed_mbps = progress.get('speed_mbps', 0)
        eta_minutes = progress.get('eta_minutes', 0)
        current_file = progress.get('current_file', '')
        
        return f"""
        <div style='margin-bottom: 15px; padding: 10px; background: #f5f5f5; border-radius: 5px;'>
            <h4 style='margin: 0 0 10px 0;'>📥 Downloading: {model}</h4>
            
            <div style='margin-bottom: 10px;'>
                <div style='display: flex; justify-content: space-between; margin-bottom: 5px;'>
                    <span>{downloaded_gb:.1f} / {total_gb:.1f} GB</span>
                    <span>{percentage:.1f}%</span>
                </div>
                <div style='background: #e0e0e0; height: 20px; border-radius: 10px; overflow: hidden;'>
                    <div style='background: #4CAF50; height: 100%; width: {percentage}%; transition: width 0.3s;'></div>
                </div>
            </div>
            
            {f'<p style="font-size: 0.9em; color: #666;">Current file: {current_file}</p>' if current_file else ''}
            
            <div style='display: flex; justify-content: space-between; font-size: 0.9em; color: #666;'>
                <span>⚡ {speed_mbps:.1f} MB/s</span>
                <span>⏱️ ETA: {eta_minutes:.1f} min</span>
            </div>
        </div>
        """
        
    def _format_generation_progress(self, progress: Dict[str, Any]) -> str:
        """Format generation progress."""
        task = progress.get('task', 'Generation')
        step = progress.get('current_step', 0)
        total_steps = progress.get('total_steps', 0)
        percentage = (step / total_steps * 100) if total_steps > 0 else 0
        
        return f"""
        <div style='margin-bottom: 15px; padding: 10px; background: #f5f5f5; border-radius: 5px;'>
            <h4 style='margin: 0 0 10px 0;'>🎨 {task}</h4>
            
            <div style='margin-bottom: 10px;'>
                <div style='display: flex; justify-content: space-between; margin-bottom: 5px;'>
                    <span>Step {step} / {total_steps}</span>
                    <span>{percentage:.1f}%</span>
                </div>
                <div style='background: #e0e0e0; height: 20px; border-radius: 10px; overflow: hidden;'>
                    <div style='background: #2196F3; height: 100%; width: {percentage}%; transition: width 0.3s;'></div>
                </div>
            </div>
        </div>
        """
        
    def _format_generic_progress(self, progress: Dict[str, Any]) -> str:
        """Format generic progress."""
        task = progress.get('task', 'Processing')
        message = progress.get('message', '')
        
        return f"""
        <div style='margin-bottom: 15px; padding: 10px; background: #f5f5f5; border-radius: 5px;'>
            <h4 style='margin: 0;'>⚙️ {task}</h4>
            {f'<p style="margin: 5px 0 0 0; color: #666;">{message}</p>' if message else ''}
        </div>
        """
        
    def _on_websocket_message(self, data: Dict[str, Any]):
        """Handle incoming WebSocket messages."""
        msg_type = data.get('type')
        task_id = data.get('task_id')
        
        if msg_type == 'progress':
            # Update progress for task
            self.current_progress[task_id] = data.get('data', {})
            self.current_progress[task_id]['type'] = data.get('task_type', 'unknown')
            
        elif msg_type == 'success':
            # Task completed
            if task_id in self.current_progress:
                del self.current_progress[task_id]
                
        elif msg_type == 'error':
            # Task failed
            if task_id in self.current_progress:
                del self.current_progress[task_id]
                
        # Trigger UI update if callback is set
        if self.update_callback:
            # This would need to be connected to Gradio's update mechanism
            pass
            
    def start(self):
        """Start the WebSocket client."""
        self.client = WebSocketProgressClient(
            host=self.host,
            port=self.port,
            on_message=self._on_websocket_message,
            on_connect=lambda: logger.info("Progress WebSocket connected"),
            on_disconnect=lambda: logger.info("Progress WebSocket disconnected")
        )
        self.client.start()
        
    def stop(self):
        """Stop the WebSocket client."""
        if self.client:
            self.client.stop()
            self.client = None
            

def create_websocket_progress(host: str = "localhost", 
                            port: int = 8765,
                            task_filter: Optional[str] = None) -> WebSocketProgressComponent:
    """Create a WebSocket progress component."""
    component = WebSocketProgressComponent(host, port, task_filter)
    component.start()
    return component