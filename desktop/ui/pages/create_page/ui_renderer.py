"""UI Renderer Mixin

Handles main UI rendering for the create page.
"""

from nicegui import ui


class UIRendererMixin:
    """Mixin for UI rendering methods"""
    
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
                    self._render_prompt_section()
                    
                    # Model and settings
                    self._render_settings_section()
                    
                    # Generation button
                    self._render_generation_buttons()
                
                # Right: Enhancement and progress
                with ui.column().style('flex: 2; max-height: 600px; overflow-y: auto;'):
                    # Enhancement panel with full width
                    self._render_enhancement_section()
                    
                    # Progress pipeline
                    self._render_progress_section()
                        
            # Bottom: Output preview
            self._render_output_section()
                    
        # Start background initialization thread
        self._start_background_initialization()
        
        # Load available models after UI elements are created
        self._load_available_models()
        
        # Ensure 3D controls are rendered if starting with 3D mode
        if self.current_mode == "3d":
            self._render_mode_controls()
        
        # Initialize notification system for background tasks
        self._show_notification()
        
        # Use ui.timer for periodic updates from a queue and initialization status
        self._pipeline_update_queue = []
        self._update_timer = ui.timer(0.1, lambda: self._process_pipeline_updates())
    
    def _render_prompt_section(self):
        """Render the prompt input section"""
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
    
    def _render_settings_section(self):
        """Render the model and settings section"""
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
    
    def _render_generation_buttons(self):
        """Render the generation control buttons"""
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
    
    def _render_enhancement_section(self):
        """Render the enhancement panel section"""
        with ui.card().classes('w-full mb-4').style('background-color: #1F1F1F; border: 1px solid #333333'):
            # Create a scrollable container for enhancement fields
            with ui.column().classes('w-full').style('max-height: 350px; overflow-y: auto'):
                self.enhancement_panel.render()
    
    def _render_progress_section(self):
        """Render the progress pipeline section"""
        with ui.card().classes('w-full').style('background-color: #1F1F1F; border: 1px solid #333333'):
            self.progress_pipeline.render()
    
    def _render_output_section(self):
        """Render the output preview section"""
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