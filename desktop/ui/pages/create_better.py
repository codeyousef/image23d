"""
Better Create page with improved layout
"""

import asyncio
import threading
from typing import Optional, Dict, Any
from pathlib import Path
from nicegui import ui
import time

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
from ..components.enhancement_panel_fullwidth import EnhancementPanel
from ..components.progress_pipeline_card import ProgressPipeline, PipelineStep

class CreatePage:
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
        
        # State
        self.current_mode = "image"
        self.generation_task: Optional[asyncio.Task] = None
        self._uploaded_image_path = None
        
        # Initialize UI elements that will be created in render()
        self.model_select = None
        
        # Get models directory
        from core.config import MODELS_DIR
        self.models_dir = MODELS_DIR
        
    def render(self):
        """Render the create page with better layout"""
        with ui.column().classes('w-full h-full'):
            # Header with mode selector
            with ui.card().classes('w-full').style('background-color: #1F1F1F; border: 1px solid #333333'):
                with ui.row().classes('items-center justify-between'):
                    ui.label('Create').classes('text-2xl font-bold')
                    
                    # Mode selector tabs
                    self.mode_tabs = ui.tabs().classes('bg-transparent')
                    # Properly handle tab change events
                    def handle_tab_change():
                        # Get the current tab value directly from the tabs widget
                        if self.mode_tabs.value:
                            self._on_mode_change(self.mode_tabs.value)
                    
                    self.mode_tabs.on('update:model-value', handle_tab_change)
                    
                    with self.mode_tabs:
                        self.image_tab = ui.tab('image', label='Image', icon='image')
                        self.threed_tab = ui.tab('3d', label='3D Model', icon='view_in_ar')
                        self.video_tab = ui.tab('video', label='Video', icon='movie').props('disable')
                        
                    # Set initial tab
                    self.mode_tabs.value = 'image'
                        
            # Main content area with better proportions
            with ui.row().classes('w-full flex-grow gap-4 mt-4'):
                # Left: Main generation controls (60% width)
                with ui.column().classes('flex-grow').style('flex: 3'):
                    # Prompt section
                    with ui.card().classes('w-full mb-4').style('background-color: #1F1F1F; border: 1px solid #333333'):
                        ui.label('Prompt').classes('text-lg font-semibold mb-2')
                        self.prompt_input = ui.textarea(
                            placeholder='Describe what you want to create in detail...',
                            value=''
                        ).classes('w-full').props('rows=4 outlined dense').style('font-size: 14px')
                        
                        # Negative prompt (collapsible)
                        with ui.expansion('Negative Prompt', icon='remove_circle').classes('mt-2'):
                            self.negative_input = ui.textarea(
                                placeholder='What to avoid...',
                                value=''
                            ).classes('w-full').props('rows=2 outlined dense')
                    
                    # Model and settings
                    with ui.card().classes('w-full mb-4').style('background-color: #1F1F1F; border: 1px solid #333333'):
                        ui.label('Generation Settings').classes('text-lg font-semibold mb-2')
                        
                        # Model selector with refresh
                        with ui.row().classes('w-full gap-2 mb-4'):
                            self.model_select = ui.select(
                                label='Model',
                                options={},
                                value=None,
                                on_change=self._on_model_select_change
                            ).classes('flex-grow').props('outlined dense')
                            
                            ui.button(
                                icon='refresh',
                                on_click=self._load_available_models
                            ).props('flat dense round')
                        
                        # Mode-specific controls
                        self.mode_controls_container = ui.column().classes('w-full gap-3')
                        self._render_mode_controls()
                        
                    # Generation button
                    with ui.row().classes('w-full gap-4'):
                        self.generate_button = ui.button(
                            'Generate',
                            icon='auto_awesome'
                        ).props('unelevated size=lg').classes('flex-grow').style(
                            'background-color: #7C3AED; color: white; height: 48px'
                        ).on('click', self._start_generation)
                        
                        self.cancel_button = ui.button(
                            'Cancel',
                            icon='cancel'
                        ).props('flat size=lg').classes('hidden').style(
                            'height: 48px'
                        ).on('click', self._cancel_generation)
                
                # Right: Enhancement and progress
                with ui.column().style('flex: 2; max-height: 600px; overflow-y: auto;'):
                    # Enhancement panel with full width
                    with ui.card().classes('w-full mb-4').style('background-color: #1F1F1F; border: 1px solid #333333'):
                        # Create a scrollable container for enhancement fields
                        with ui.column().classes('w-full').style('max-height: 350px; overflow-y: auto'):
                            self.enhancement_panel.render()
                    
                    # Progress pipeline
                    with ui.card().classes('w-full').style('background-color: #1F1F1F; border: 1px solid #333333'):
                        self.progress_pipeline.render()
                        
            # Bottom: Output preview
            with ui.card().classes('w-full mt-4').style('background-color: #1F1F1F; border: 1px solid #333333; min-height: 400px;'):
                with ui.row().classes('items-center justify-between mb-4'):
                    ui.label('Output').classes('text-lg font-semibold')
                    
                    # Export options
                    with ui.row().classes('gap-2'):
                        self.export_button = ui.button('Export', icon='download').props('flat')
                        self.export_button.on('click', self._export_output)
                        self.export_button.visible = False
                        
                        self.save_button = ui.button('Save to Library', icon='save').props('flat')
                        self.save_button.on('click', self._save_to_library)
                        self.save_button.visible = False
                        
                # Preview area
                self.preview_container = ui.column().classes('w-full items-center justify-center p-4')
                with self.preview_container:
                    ui.icon('image', size='4rem').classes('text-gray-600 mb-4')
                    ui.label('Your generated content will appear here').classes('text-gray-500')
                    
        # Start background initialization thread
        self._start_background_initialization()
        
        # Load available models after UI elements are created
        self._load_available_models()
        
        # Ensure 3D controls are rendered if starting with 3D mode
        if self.current_mode == "3d":
            self._render_mode_controls()
        
        # Use ui.timer for periodic updates from a queue and initialization status
        self._pipeline_update_queue = []
        self._update_timer = ui.timer(0.1, lambda: self._process_pipeline_updates())
        
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
                self.image_processor = ImageProcessor(self.model_manager, self.output_dir / "images", self.prompt_enhancer)
                self.threed_processor = ThreeDProcessor(self.model_manager, self.output_dir / "3d", self.prompt_enhancer)
                self.sparc3d_processor = Sparc3DProcessor(self.model_manager, self.output_dir / "3d", self.prompt_enhancer)
                self.hi3dgen_processor = Hi3DGenProcessor(self.model_manager, self.output_dir / "3d", self.prompt_enhancer)
                
                self.model_manager_initialized = True
                self.initialization_in_progress = False
            
            print("Model manager and processors initialized successfully")
            
        except Exception as e:
            with self.initialization_lock:
                self.initialization_error = str(e)
                self.initialization_in_progress = False
            print(f"Error initializing model manager: {e}")
        
    def _render_mode_controls(self):
        """Render controls specific to current mode with consistent sizing"""
        self.mode_controls_container.clear()
        
        with self.mode_controls_container:
            if self.current_mode == "image":
                # Resolution controls in a row
                with ui.row().classes('w-full gap-4'):
                    self.width_input = ui.number(
                        label='Width',
                        value=1024,
                        min=256,
                        max=2048,
                        step=64
                    ).classes('flex-1').props('outlined dense')
                    
                    self.height_input = ui.number(
                        label='Height',
                        value=1024,
                        min=256,
                        max=2048,
                        step=64
                    ).classes('flex-1').props('outlined dense')
                    
                    # Aspect ratio buttons
                    with ui.column().classes('gap-1'):
                        ui.label('Aspect').classes('text-xs text-gray-500')
                        with ui.row().classes('gap-1'):
                            ui.button('1:1', on_click=lambda: self._set_aspect(1, 1)).props('flat dense size=sm')
                            ui.button('16:9', on_click=lambda: self._set_aspect(16, 9)).props('flat dense size=sm')
                            ui.button('9:16', on_click=lambda: self._set_aspect(9, 16)).props('flat dense size=sm')
                    
                # Steps and guidance in a row
                with ui.row().classes('w-full gap-4'):
                    with ui.column().classes('flex-1'):
                        ui.label('Steps').classes('text-sm text-gray-400')
                        self.steps_slider = ui.slider(
                            min=1,
                            max=150,
                            value=20
                        ).props('label-always')
                    
                    with ui.column().classes('flex-1'):
                        ui.label('Guidance Scale').classes('text-sm text-gray-400')
                        self.guidance_slider = ui.slider(
                            min=1.0,
                            max=20.0,
                            value=7.5,
                            step=0.5
                        ).props('label-always')
                
            elif self.current_mode == "3d":
                # Image model selection for text-to-3D
                self.image_model_select = ui.select(
                    label='Image Generation Model',
                    options={},  # Will be populated by _load_available_models
                    value=None
                ).classes('w-full').props('outlined dense')
                ui.label('Select the image model to use for text-to-3D generation').classes('text-xs text-gray-400 -mt-2 mb-2')
                
                # Quality preset with better styling
                self.quality_select = ui.select(
                    label='Quality Preset',
                    options={
                        'draft': 'Draft (Fast, ~2 min)',
                        'standard': 'Standard (Balanced, ~5 min)',
                        'high': 'High Quality (Slower, ~10 min)',
                        'ultra': 'Ultra (Best quality, ~20 min)'
                    },
                    value='standard'
                ).classes('w-full').props('outlined dense')
                
                # Input image upload with preview
                self.image_upload = ui.upload(
                    label='Input Image (optional)',
                    max_files=1,
                    on_upload=self._handle_image_upload
                ).classes('w-full').props('outlined accept="image/*"')
                ui.label('Upload an image for direct image-to-3D conversion').classes('text-xs text-gray-400 -mt-2')
                
                # Advanced 3D Settings
                with ui.expansion('Advanced Settings', icon='settings').classes('w-full'):
                    # Resolution Settings
                    ui.label('Resolution Settings').classes('font-semibold')
                    with ui.row().classes('w-full gap-4'):
                        with ui.column().classes('flex-1'):
                            ui.label('Mesh Resolution').classes('text-sm')
                            self.mesh_resolution = ui.slider(
                                min=256, max=1024, value=512, step=128
                            ).props('label-always')
                        
                        with ui.column().classes('flex-1'):
                            ui.label('Texture Resolution').classes('text-sm')
                            self.texture_resolution = ui.slider(
                                min=512, max=4096, value=2048, step=256
                            ).props('label-always')
                    
                    # View Generation Settings
                    ui.label('View Generation Settings').classes('font-semibold mt-4')
                    with ui.row().classes('w-full gap-4'):
                        with ui.column().classes('flex-1'):
                            ui.label('Number of Views').classes('text-sm')
                            self.num_views = ui.slider(
                                min=4, max=12, value=6, step=1
                            ).props('label-always')
                        
                        with ui.column().classes('flex-1'):
                            ui.label('Camera Distance').classes('text-sm')
                            self.view_distance = ui.slider(
                                min=1.0, max=3.0, value=2.0, step=0.1
                            ).props('label-always')
                    
                    # Processing Options
                    ui.label('Processing Options').classes('font-semibold mt-4')
                    with ui.column().classes('w-full gap-2'):
                        with ui.row().classes('w-full gap-4'):
                            self.remove_bg_switch = ui.switch('Remove Background', value=True)
                            self.auto_center_switch = ui.switch('Auto Center Object', value=True)
                        
                        with ui.row().classes('w-full gap-4'):
                            self.pbr_materials_switch = ui.switch('PBR Materials', value=False)
                            self.depth_enhancement_switch = ui.switch('Depth Enhancement', value=True)
                        
                        with ui.row().classes('w-full gap-4'):
                            self.normal_enhancement_switch = ui.switch('Normal Map Enhancement', value=True)
                            self.multiview_consistency_switch = ui.switch('Multi-view Consistency', value=True)
                
                # Export format selection
                self.export_formats = ui.select(
                    label='Export Formats',
                    options={
                        'glb': 'GLB (Recommended)',
                        'obj': 'OBJ (Universal)', 
                        'ply': 'PLY (Point Cloud)',
                        'stl': 'STL (3D Printing)',
                        'fbx': 'FBX (Animation)',
                        'usdz': 'USDZ (AR)'
                    },
                    multiple=True,
                    value=['glb']
                ).classes('w-full').props('outlined dense chips')
                    
    def _set_aspect(self, w: int, h: int):
        """Set aspect ratio"""
        base = 1024
        if w > h:
            self.width_input.value = base
            self.height_input.value = int(base * h / w)
        else:
            self.height_input.value = base
            self.width_input.value = int(base * w / h)
            
    def _check_model_downloaded(self, model_id: str, model_type: str) -> bool:
        """Check if a model is downloaded"""
        # Check for model directory
        if model_type == 'image':
            model_path = self.models_dir / 'image' / model_id
        elif model_type == '3d':
            model_path = self.models_dir / '3d' / model_id
        else:
            model_path = self.models_dir / model_type / model_id
            
        return model_path.exists() and any(model_path.iterdir()) if model_path.exists() else False

    def _load_available_models(self, show_notifications=True):
        """Load available models from the actual system"""
        # Check if UI elements exist
        if not hasattr(self, 'model_select') or self.model_select is None:
            return
            
        try:
            if self.current_mode == "image":
                # Get downloaded image models from the actual model manager
                options = {}
                downloaded_models = []
                
                # Check if model manager is initialized
                if self.model_manager and hasattr(self.model_manager, 'get_downloaded_models'):
                    try:
                        # Get actual downloaded models
                        detected_models = self.model_manager.get_downloaded_models('image')
                        for model_id in detected_models:
                            # Create a display name
                            display_name = model_id.replace('-', ' ').replace('_', ' ')
                            if 'Q8' in model_id:
                                display_name += ' (High Quality)'
                            elif 'Q6' in model_id:
                                display_name += ' (Balanced)'
                            elif 'FP8' in model_id:
                                display_name += ' (Fast)'
                            options[model_id] = display_name
                            downloaded_models.append(model_id)
                    except Exception as e:
                        print(f"Error getting downloaded models: {e}")
                
                # Fallback to config if no models detected
                if not downloaded_models:
                    from core.config import FLUX_MODELS
                    all_image_models = FLUX_MODELS
                    
                    for model_id, config in all_image_models.items():
                        if self._check_model_downloaded(model_id, 'image'):
                            options[model_id] = f"{config['name']} ({config['vram_required']})"
                            downloaded_models.append(model_id)
                    
                self.model_select.options = options
                
                if options:
                    # Default to a fast model if available
                    if "flux-1-schnell" in downloaded_models:
                        self.model_select.value = "flux-1-schnell"
                    elif "flux-1-dev" in downloaded_models:
                        self.model_select.value = "flux-1-dev"
                    else:
                        self.model_select.value = downloaded_models[0] if downloaded_models else None
                else:
                    self.model_select.value = None
                    self.model_select.options = {'': 'No models downloaded'}
                    if show_notifications:
                        ui.notify('No image models downloaded. Please download models from the Models page.', type='warning')
                    
            elif self.current_mode == "3d":
                # Get only downloaded 3D models
                options = {}
                downloaded_models = []
                
                print(f"Loading 3D models...")  # Debug
                
                # Import the 3D models config
                from core.config import ALL_3D_MODELS
                
                for model_id, config in ALL_3D_MODELS.items():
                    model_path = self.models_dir / '3d' / model_id
                    is_downloaded = self._check_model_downloaded(model_id, '3d')
                    print(f"  Checking {model_id}: path={model_path}, exists={model_path.exists()}, downloaded={is_downloaded}")  # Debug
                    
                    if is_downloaded:
                        options[model_id] = f"{config['name']} ({config['vram_required']})"
                        downloaded_models.append(model_id)
                
                print(f"Found {len(options)} 3D models: {list(options.keys())}")  # Debug
                    
                self.model_select.options = options
                
                if options:
                    if "hunyuan3d-21" in downloaded_models:
                        self.model_select.value = "hunyuan3d-21"
                    else:
                        self.model_select.value = downloaded_models[0] if downloaded_models else None
                else:
                    self.model_select.value = None
                    self.model_select.options = {'': 'No models downloaded'}
                    if show_notifications:
                        ui.notify('No 3D models downloaded. Please download models from the Models page.', type='warning')
                
                # Also load image models for text-to-3D
                if hasattr(self, 'image_model_select'):
                    image_options = {}
                    image_models = []
                    
                    # Check if model manager is initialized
                    if self.model_manager and hasattr(self.model_manager, 'get_downloaded_models'):
                        try:
                            # Get actual downloaded image models
                            detected_models = self.model_manager.get_downloaded_models('image')
                            for model_id in detected_models:
                                # Create a display name
                                display_name = model_id.replace('-', ' ').replace('_', ' ')
                                if 'Q8' in model_id:
                                    display_name += ' (High Quality)'
                                elif 'Q6' in model_id:
                                    display_name += ' (Balanced)'
                                elif 'FP8' in model_id:
                                    display_name += ' (Fast)'
                                elif 'schnell' in model_id.lower():
                                    display_name += ' (4x Faster)'
                                image_options[model_id] = display_name
                                image_models.append(model_id)
                        except Exception as e:
                            print(f"Error getting downloaded image models: {e}")
                    
                    # Fallback to config if no models detected
                    if not image_models:
                        from core.config import FLUX_MODELS
                        for model_id, config in FLUX_MODELS.items():
                            if self._check_model_downloaded(model_id, 'image'):
                                image_options[model_id] = f"{config['name']} ({config['vram_required']})"
                                image_models.append(model_id)
                    
                    self.image_model_select.options = image_options
                    
                    if image_options:
                        # Default to a fast model if available
                        if "FLUX.1-schnell" in image_models:
                            self.image_model_select.value = "FLUX.1-schnell"
                        elif "FLUX.1-dev" in image_models:
                            self.image_model_select.value = "FLUX.1-dev"
                        else:
                            self.image_model_select.value = image_models[0] if image_models else None
                    else:
                        self.image_model_select.value = None
                        self.image_model_select.options = {'': 'No image models downloaded'}
                    
        except Exception as e:
            if show_notifications:
                ui.notify(f'Failed to load models: {str(e)}', type='negative')
            else:
                print(f'Failed to load models: {str(e)}')
            
    def _on_mode_change(self, tab_value):
        """Handle mode change"""
        # Extract the actual tab name from the value
        if isinstance(tab_value, str):
            self.current_mode = tab_value
        elif hasattr(tab_value, 'name'):
            self.current_mode = tab_value.name
        else:
            # Try to extract from string representation
            tab_str = str(tab_value)
            if '3d' in tab_str.lower():
                self.current_mode = '3d'
            elif 'video' in tab_str.lower():
                self.current_mode = 'video'
            else:
                self.current_mode = 'image'
        
        print(f"Mode changed to: {self.current_mode}")  # Debug log
        
        # Update enhancement panel model type based on selected model
        if self.current_mode == "image":
            self.enhancement_panel.set_model_type(ModelType.FLUX_1_DEV)
        elif self.current_mode == "3d":
            # Default to HunYuan3D, but this will be updated when model is selected
            self.enhancement_panel.set_model_type(ModelType.HUNYUAN_3D_21)
            
        # Update controls
        self._render_mode_controls()
        
        # Load appropriate models
        self._load_available_models()
        
        # Reset pipeline
        self.progress_pipeline.reset()
        
    def _on_enhancement_change(self, values: Dict[str, Any]):
        """Handle enhancement panel changes"""
        # Could preview enhanced prompt here
        pass
        
    def _on_model_select_change(self, e):
        """Handle model selection change"""
        model_id = e.value
        if not model_id:
            return
            
        # Update enhancement panel based on selected model
        if self.current_mode == "3d":
            # Map model ID to ModelType
            if "sparc3d" in model_id.lower():
                self.enhancement_panel.set_model_type(ModelType.SPARC3D)
            elif "hi3dgen" in model_id.lower():
                self.enhancement_panel.set_model_type(ModelType.HI3DGEN)
            elif "mini" in model_id.lower():
                self.enhancement_panel.set_model_type(ModelType.HUNYUAN_3D_MINI)
            elif "2.0" in model_id or "2mv" in model_id or "2standard" in model_id:
                self.enhancement_panel.set_model_type(ModelType.HUNYUAN_3D_20)
            else:
                self.enhancement_panel.set_model_type(ModelType.HUNYUAN_3D_21)
        
    def _handle_image_upload(self, e):
        """Handle image upload for 3D mode"""
        if e.content:
            try:
                # Save uploaded file temporarily
                temp_dir = self.output_dir / "temp"
                temp_dir.mkdir(exist_ok=True)
                temp_path = temp_dir / e.name
                temp_path.write_bytes(e.content.read())
                
                # Store the path for use in generation
                self._uploaded_image_path = str(temp_path)
                
                ui.notify(f'Image "{e.name}" uploaded successfully', type='positive')
                
                # Optional: Add preview logic here in the future
                # self._show_image_preview(temp_path)
                
            except Exception as ex:
                ui.notify(f'Failed to upload image: {str(ex)}', type='negative')
                self._uploaded_image_path = None
            
    async def _start_generation(self):
        """Start the generation process"""
        if self.generation_task and not self.generation_task.done():
            self._pipeline_update_queue.append(('notify', ('Generation already in progress', 'warning')))
            return
            
        # Check if processors are initialized
        with self.initialization_lock:
            if not self.model_manager_initialized:
                if self.initialization_in_progress:
                    self._pipeline_update_queue.append(('notify', ('Models are still initializing, please wait...', 'warning')))
                elif self.initialization_error:
                    self._pipeline_update_queue.append(('notify', (f'Model initialization failed: {self.initialization_error}', 'negative')))
                else:
                    self._pipeline_update_queue.append(('notify', ('Models not initialized. Please restart the application.', 'negative')))
                return
            
        # Validate inputs
        if self.current_mode == "image" and not self.prompt_input.value:
            self._pipeline_update_queue.append(('notify', ('Please enter a prompt', 'warning')))
            return
        elif self.current_mode == "3d" and not self.prompt_input.value and not getattr(self, '_uploaded_image_path', None):
            self._pipeline_update_queue.append(('notify', ('Please enter a prompt or upload an image', 'warning')))
            return
            
        if not self.model_select.value:
            self._pipeline_update_queue.append(('notify', ('Please select a model', 'warning')))
            return
            
        # Queue all UI updates
        self._pipeline_update_queue.append(('start_generation', None))
        self._pipeline_update_queue.append(('clear_preview', 'generating'))
        
        # Start generation task
        self.generation_task = asyncio.create_task(self._run_generation())
        
    async def _run_generation(self):
        """Run the actual generation process"""
        try:
            # Get enhancement values
            enhancement = self.enhancement_panel.get_values()
            
            # Reset and start pipeline
            self.progress_pipeline.start()
            
            if self.current_mode == "image":
                await self._generate_image(enhancement)
            elif self.current_mode == "3d":
                await self._generate_3d(enhancement)
                
        except Exception as e:
            # Queue error UI updates instead of direct manipulation
            self._pipeline_update_queue.append(('error', str(e)))
            
        finally:
            # Queue UI state reset
            self._pipeline_update_queue.append(('reset_ui', None))
            
    async def _generate_image(self, enhancement: Dict[str, Any]):
        """Generate an image"""
        # Check if image processor is available
        if not self.image_processor:
            raise ValueError("Image processor not initialized")
            
        # Set pipeline steps
        self.progress_pipeline.set_steps([
            PipelineStep("enhance", "Enhance Prompt", "psychology", 5),
            PipelineStep("load", "Load Model", "memory", 10),
            PipelineStep("generate", "Generate Image", "brush", 20),
            PipelineStep("save", "Save Output", "save", 2)
        ])
        
        # Create request
        request = ImageGenerationRequest(
            prompt=self.prompt_input.value,
            model=self.model_select.value,
            negative_prompt=self.negative_input.value if hasattr(self, 'negative_input') else None,
            width=int(self.width_input.value),
            height=int(self.height_input.value),
            steps=int(self.steps_slider.value),
            guidance_scale=float(self.guidance_slider.value),
            enhancement_fields=enhancement['fields'],
            use_enhancement=enhancement['use_llm']
        )
        
        # Progress callback - sync function that uses queue for safe updates
        def progress_callback(progress, message):
            # Queue updates instead of direct UI manipulation
            self._pipeline_update_queue.append((progress, message))
                    
        # Generate
        response = await self.image_processor.generate(request, progress_callback)
        
        # Show result
        if response.image_path:
            # Queue success UI updates
            self._pipeline_update_queue.append(('image_result', response.image_path))
            self._pipeline_update_queue.append(('notify', ('Image generated successfully!', 'positive')))
        else:
            raise ValueError(response.error or "No image generated")
            
    async def _generate_3d(self, enhancement: Dict[str, Any]):
        """Generate a 3D model"""
        # Check if 3D processors are available
        if not self.threed_processor or not self.sparc3d_processor or not self.hi3dgen_processor:
            raise ValueError("3D processors not initialized")
            
        # Set pipeline steps
        self.progress_pipeline.set_steps([
            PipelineStep("prepare", "Prepare Input", "build", 5),
            PipelineStep("generate", "Generate 3D Model", "view_in_ar", 60),
            PipelineStep("export", "Export Model", "save", 10)
        ])
        
        # Handle input image if uploaded
        input_image = getattr(self, '_uploaded_image_path', None)
            
        # Create enhanced request with all the new settings
        request = ThreeDGenerationRequest(
            prompt=self.prompt_input.value or "",
            model=self.model_select.value,
            image_model=self.image_model_select.value if hasattr(self, 'image_model_select') else None,
            input_image=input_image,
            quality_preset=self.quality_select.value,
            num_views=int(self.num_views.value) if hasattr(self, 'num_views') else 6,
            mesh_resolution=int(self.mesh_resolution.value) if hasattr(self, 'mesh_resolution') else 512,
            texture_resolution=int(self.texture_resolution.value) if hasattr(self, 'texture_resolution') else 2048,
            export_formats=self.export_formats.value or ['glb'],
            remove_background=self.remove_bg_switch.value if hasattr(self, 'remove_bg_switch') else True,
            
            # Enhanced processing options
            auto_center=self.auto_center_switch.value if hasattr(self, 'auto_center_switch') else True,
            view_distance=float(self.view_distance.value) if hasattr(self, 'view_distance') else 2.0,
            pbr_materials=self.pbr_materials_switch.value if hasattr(self, 'pbr_materials_switch') else False,
            depth_enhancement=self.depth_enhancement_switch.value if hasattr(self, 'depth_enhancement_switch') else True,
            normal_enhancement=self.normal_enhancement_switch.value if hasattr(self, 'normal_enhancement_switch') else True,
            multiview_consistency=self.multiview_consistency_switch.value if hasattr(self, 'multiview_consistency_switch') else True,
            
            enhancement_fields=enhancement['fields'],
            use_enhancement=enhancement['use_llm'] and not input_image
        )
        
        # Progress callback - sync function for 3D generation
        def progress_callback(progress, message):
            # Queue updates as tuples
            self._pipeline_update_queue.append((progress, message))
                    
        # Select appropriate processor based on model
        model_name = self.model_select.value
        if "sparc3d" in model_name.lower():
            processor = self.sparc3d_processor
        elif "hi3dgen" in model_name.lower():
            processor = self.hi3dgen_processor
        else:
            processor = self.threed_processor
            
        # Generate
        response = await processor.generate(request, progress_callback)
        
        # Show result
        if response.model_path:
            # Queue success UI updates
            self._pipeline_update_queue.append(('3d_result', (response.model_path, response.preview_images)))
            self._pipeline_update_queue.append(('notify', ('3D model generated successfully!', 'positive')))
        else:
            raise ValueError(response.error or "No 3D model generated")
            
    async def _cancel_generation(self):
        """Cancel ongoing generation"""
        if self.generation_task and not self.generation_task.done():
            self.generation_task.cancel()
            # Queue cancellation UI updates
            self._pipeline_update_queue.append(('cancel', None))
            self._pipeline_update_queue.append(('notify', ('Generation cancelled', 'info')))
            
    def _show_image_result(self, image_path: Path):
        """Display generated image"""
        self._current_output_path = image_path
        
        # Clear and update - this is called from UI thread via queue
        self.preview_container.clear()
        with self.preview_container:
            # Show image with proper containment
            ui.image(str(image_path)).classes('w-full h-auto').style(
                'max-width: 100%; max-height: 500px; object-fit: contain; border-radius: 8px;'
            )
            
            # Show metadata
            with ui.row().classes('gap-4 mt-4'):
                ui.label(f'Resolution: {self.width_input.value}x{self.height_input.value}').classes('text-sm text-gray-400')
                ui.label(f'Model: {self.model_select.value}').classes('text-sm text-gray-400')
                ui.label(f'Steps: {self.steps_slider.value}').classes('text-sm text-gray-400')
            
        self.export_button.visible = True
        self.save_button.visible = True
            
    def _show_3d_result(self, model_path: Path, preview_images: list):
        """Display generated 3D model"""
        self._current_output_path = model_path
        
        # Clear and update - this is called from UI thread via queue
        self.preview_container.clear()
        with self.preview_container:
            # Show preview images in a nice grid
            if preview_images:
                ui.label('Preview Renders').classes('text-lg font-semibold mb-2')
                with ui.row().classes('gap-2 flex-wrap'):
                    for img_path in preview_images[:6]:  # Show up to 6 previews
                        ui.image(str(img_path)).classes('w-48 h-48 rounded').style('object-fit: cover')
                        
            # Model info
            with ui.column().classes('mt-4 p-4 bg-gray-800 rounded'):
                ui.label(f'3D Model Generated').classes('text-lg font-semibold')
                with ui.row().classes('gap-4 mt-2'):
                    ui.label(f'File: {model_path.name}').classes('text-sm')
                    ui.label(f'Size: {model_path.stat().st_size / 1024 / 1024:.1f} MB').classes('text-sm')
                    ui.label(f'Format: {model_path.suffix.upper()[1:]}').classes('text-sm')
            
        self.export_button.visible = True
        self.save_button.visible = True
            
    def _process_pipeline_updates(self):
        """Process queued pipeline updates in the UI thread"""
        try:
            # Check for initialization completion
            self._check_initialization_status()
            
            # Process pipeline updates
            while self._pipeline_update_queue:
                update = self._pipeline_update_queue.pop(0)
                
                if isinstance(update, tuple):
                    update_type, data = update
                    
                    # Handle different update types
                    if update_type == 'error':
                        # Handle error display
                        self.progress_pipeline.fail(f'Error: {data}')
                        ui.notify(f'Generation failed: {data}', type='negative')
                        self.preview_container.clear()
                        with self.preview_container:
                            ui.icon('error', size='4rem').classes('text-red-500 mb-4')
                            ui.label('Generation Failed').classes('text-xl mb-2')
                            ui.label(str(data)).classes('text-gray-500')
                            
                    elif update_type == 'start_generation':
                        # Update UI state when starting generation
                        self.generate_button.props('loading')
                        self.cancel_button.classes(remove='hidden')
                        self.export_button.visible = False
                        self.save_button.visible = False
                        
                    elif update_type == 'clear_preview':
                        # Clear preview and show generating state
                        self.preview_container.clear()
                        with self.preview_container:
                            ui.spinner(size='lg')
                            ui.label('Generating...').classes('mt-4')
                            
                    elif update_type == 'reset_ui':
                        # Reset UI state
                        self.generate_button.props(remove='loading')
                        self.cancel_button.classes('hidden')
                        
                    elif update_type == 'image_result':
                        # Show image result
                        self._show_image_result(data)
                        self.progress_pipeline.complete()
                        
                    elif update_type == '3d_result':
                        # Show 3D result
                        model_path, preview_images = data
                        self._show_3d_result(model_path, preview_images)
                        self.progress_pipeline.complete()
                        
                    elif update_type == 'notify':
                        # Show notification
                        message, notify_type = data
                        ui.notify(message, type=notify_type)
                        
                    elif update_type == 'cancel':
                        # Handle cancellation
                        self.progress_pipeline.fail('Generation cancelled')
                        self.preview_container.clear()
                        with self.preview_container:
                            ui.icon('cancel', size='4rem').classes('text-gray-600 mb-4')
                            ui.label('Generation cancelled').classes('text-gray-500')
                            
                    elif isinstance(data, (int, float)):
                        # Handle progress updates (backward compatibility)
                        progress = update_type
                        message = data
                        if hasattr(self, 'progress_pipeline') and self.progress_pipeline:
                            # Map progress to pipeline steps
                            if progress < 10 and self.progress_pipeline.current_step_index < 0:
                                self.progress_pipeline.advance_step("Enhancing prompt...")
                            elif progress < 30 and self.progress_pipeline.current_step_index < 1:
                                self.progress_pipeline.advance_step("Loading model...")
                            elif progress < 90 and self.progress_pipeline.current_step_index < 2:
                                self.progress_pipeline.advance_step(message or "Generating...")
                            elif progress >= 90 and self.progress_pipeline.current_step_index < 3:
                                self.progress_pipeline.advance_step("Saving output...")
                else:
                    # Handle legacy string messages
                    if hasattr(self, 'progress_pipeline') and self.progress_pipeline:
                        self.progress_pipeline.advance_step(update)
        except Exception as e:
            # Silently handle any UI update errors
            pass
    
    def _check_initialization_status(self):
        """Check initialization status and update UI accordingly"""
        with self.initialization_lock:
            # Check if initialization just completed
            if (self.model_manager_initialized and 
                not getattr(self, '_initialization_ui_updated', False)):
                
                self._initialization_ui_updated = True
                # Update models list now that processors are ready
                self._load_available_models(show_notifications=False)
                
            # Check for initialization errors
            elif (self.initialization_error and 
                  not getattr(self, '_initialization_error_shown', False)):
                
                self._initialization_error_shown = True
                ui.notify(f"Error initializing models: {self.initialization_error}", type='warning')
                
        # Initialize tracking flags if not present (done via getattr with defaults above)
    
    def _export_output(self):
        """Export/download the current output"""
        if hasattr(self, '_current_output_path') and self._current_output_path:
            # In a real desktop app, this would open a file dialog
            ui.notify(f'Output saved to: {self._current_output_path}', type='positive', timeout=5000)
            
    def _save_to_library(self):
        """Save output to library"""
        if hasattr(self, '_current_output_path') and self._current_output_path:
            # This would add to the library database
            ui.notify('Saved to library!', type='positive')
            # TODO: Actually implement library saving