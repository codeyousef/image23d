"""Base Create Page Class

Core CreatePage class with initialization and main structure.
"""

import asyncio
import threading
from typing import Optional, Dict, Any
from pathlib import Path
from nicegui import ui

from core.models.enhancement import ModelType
from core.models.generation import (
    ImageGenerationRequest, 
    ThreeDGenerationRequest,
    VideoGenerationRequest
)
from core.processors.image_processor import ImageProcessor
from core.processors.threed_processor import ThreeDProcessor
from core.processors.sparc3d_processor import Sparc3DProcessor
from core.processors.hi3dgen_processor import Hi3DGenProcessor
from core.processors.prompt_enhancer import PromptEnhancer
from core.services.model_manager import ModelManager
from ...components.enhancement_panel_fullwidth import EnhancementPanel
from ...components.progress_pipeline_card import ProgressPipeline, PipelineStep

from .ui_renderer import UIRendererMixin
from .mode_controls import ModeControlsMixin
from .generation_handlers import GenerationHandlersMixin
from .model_management import ModelManagementMixin
from .output_handlers import OutputHandlersMixin
from .initialization import InitializationMixin


class CreatePage(
    UIRendererMixin,
    ModeControlsMixin,
    GenerationHandlersMixin,
    ModelManagementMixin,
    OutputHandlersMixin,
    InitializationMixin
):
    """Main creation page with better layout"""
    
    def __init__(self, output_dir: Path, model_manager=None):
        self.output_dir = output_dir
        
        # Use passed model manager or create new one
        if model_manager:
            self.model_manager = model_manager
        else:
            from core.config import MODELS_DIR
            self.model_manager = ModelManager(MODELS_DIR)
        
        self.prompt_enhancer = PromptEnhancer()
        
        # Threading state for initialization
        self.model_manager_initialized = False
        self.initialization_in_progress = False
        self.initialization_error = None
        self.initialization_lock = threading.Lock()
        
        # Enhanced processors - will be initialized after model manager
        self.image_processor = None
        self.threed_processor = None
        self.sparc3d_processor = None
        self.hi3dgen_processor = None
        
        # Upload handling
        self._uploaded_image_path = None
        
        # UI components
        self.enhancement_panel = EnhancementPanel(on_change=self._on_enhancement_change)
        self.progress_pipeline = ProgressPipeline()
        
        # State - start with 3D since we have 3D models
        self.current_mode = "3d"
        self.generation_task: Optional[asyncio.Task] = None
        self._uploaded_image_path = None
        
        # Notification system for background tasks
        self._notification_message = None
        self._notification_type = None
        
        # Initialize UI elements that will be created in render()
        self.model_select = None
        self.mode_tabs = None
        self.image_tab = None
        self.threed_tab = None
        self.video_tab = None
        self.prompt_input = None
        self.negative_input = None
        self.mode_controls_container = None
        self.generate_button = None
        self.cancel_button = None
        self.preview_container = None
        self.export_button = None
        self.save_button = None
        self._update_timer = None
        self._pipeline_update_queue = []
        
        # Mode-specific controls
        self.width_input = None
        self.height_input = None
        self.steps_slider = None
        self.guidance_slider = None
        self.image_model_select = None
        self.format_select = None
        self.img2img_toggle = None
        self.image_upload = None
        self.uploaded_image_preview = None
        self.num_views_slider = None
        self.mesh_res_slider = None
        self.texture_res_slider = None
        
        # Advanced 3D performance controls
        self.inference_steps_slider = None
        self.guidance_scale_slider = None
        self.mesh_decode_resolution_slider = None
        self.mesh_decode_batch_size_slider = None
        self.paint_max_views_slider = None
        self.paint_resolution_slider = None
        self.render_size_slider = None
        self.texture_size_slider = None
        self.time_estimate_card = None
        self.time_estimate_label = None
        self.time_breakdown_label = None
        self.performance_tips = None
        
        # Get models directory
        from core.config import MODELS_DIR
        self.models_dir = MODELS_DIR
        
        # Generation output tracking
        self._current_output = None
        self._current_output_type = None
        
    def _on_enhancement_change(self, values: Dict[str, Any]):
        """Handle enhancement panel changes"""
        # This will be called when enhancement values change
        # Currently just storing values, could add validation or preview here
        pass
    
    @ui.refreshable
    def _show_notification(self):
        """Refreshable notification display for background tasks"""
        if self._notification_message:
            ui.notify(self._notification_message, type=self._notification_type, position='top')
            # Clear the notification after showing
            self._notification_message = None
            self._notification_type = None
    
    def _notify_from_background(self, message: str, notification_type: str = 'info'):
        """Queue a notification from a background task"""
        self._notification_message = message
        self._notification_type = notification_type
        self._show_notification.refresh()