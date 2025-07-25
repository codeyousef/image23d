"""Initialization Mixin

Handles background initialization and status checking.
"""

import asyncio
import threading
from nicegui import ui

from core.processors.image_processor import ImageProcessor
from core.processors.threed_processor import ThreeDProcessor
from core.processors.sparc3d_processor import Sparc3DProcessor
from core.processors.hi3dgen_processor import Hi3DGenProcessor


class InitializationMixin:
    """Mixin for initialization operations"""
    
    def _start_background_initialization(self):
        """Start background thread to initialize model manager"""
        with self.initialization_lock:
            if self.initialization_in_progress:
                return
            self.initialization_in_progress = True
        
        # Start background thread
        init_thread = threading.Thread(target=self._initialize_in_background, daemon=True)
        init_thread.start()
    
    def _initialize_in_background(self):
        """Initialize model manager in background thread"""
        try:
            # Initialize model manager (this is async, so we need to run it in event loop)
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.model_manager.initialize())
            finally:
                loop.close()
            
            # Initialize enhanced processors
            with self.initialization_lock:
                self.image_processor = ImageProcessor(
                    self.model_manager, 
                    self.output_dir / "images", 
                    self.prompt_enhancer
                )
                self.threed_processor = ThreeDProcessor(
                    self.model_manager, 
                    self.output_dir / "3d", 
                    self.prompt_enhancer
                )
                self.sparc3d_processor = Sparc3DProcessor(
                    self.model_manager, 
                    self.output_dir / "3d", 
                    self.prompt_enhancer
                )
                self.hi3dgen_processor = Hi3DGenProcessor(
                    self.model_manager, 
                    self.output_dir / "3d", 
                    self.prompt_enhancer
                )
                
                self.model_manager_initialized = True
                self.initialization_in_progress = False
            
            print("Model manager and processors initialized successfully")
            
        except Exception as e:
            with self.initialization_lock:
                self.initialization_error = str(e)
                self.initialization_in_progress = False
            print(f"Error initializing model manager: {e}")
    
    def _check_initialization_status(self):
        """Check if model manager is initialized and show appropriate status"""
        with self.initialization_lock:
            if self.initialization_error:
                ui.notify(
                    f'Model manager initialization failed: {self.initialization_error}',
                    type='negative',
                    position='top',
                    timeout=0  # Don't auto-hide error messages
                )
                return False
            
            if self.initialization_in_progress:
                ui.notify(
                    'Model manager is initializing, please wait...',
                    type='info',
                    position='top'
                )
                return False
            
            if not self.model_manager_initialized:
                ui.notify(
                    'Model manager not initialized',
                    type='warning',
                    position='top'
                )
                return False
            
            return True