"""
Create page - Main generation interface
"""

import asyncio
from typing import Optional, Dict, Any, List
from pathlib import Path
from nicegui import ui

from core.models.enhancement import ModelType
from core.models.generation import (
    ImageGenerationRequest, 
    ThreeDGenerationRequest,
    VideoGenerationRequest
)
from core.processors import ImageProcessor, ThreeDProcessor, VideoProcessor
from ..components import EnhancementPanel, ProgressPipeline, PipelineStep

class CreatePage:
    """Main creation page with generation controls"""
    
    def __init__(self, model_manager, output_dir: Path):
        self.model_manager = model_manager
        self.output_dir = output_dir
        
        # Processors
        self.image_processor = ImageProcessor(model_manager, output_dir / "images")
        self.threed_processor = ThreeDProcessor(model_manager, output_dir / "3d")
        self.video_processor = VideoProcessor(model_manager, output_dir / "videos")
        
        # UI components
        self.enhancement_panel = EnhancementPanel(on_change=self._on_enhancement_change)
        self.progress_pipeline = ProgressPipeline()
        
        # State
        self.current_mode = "image"
        self.generation_task: Optional[asyncio.Task] = None
        
    def render(self):
        """Render the create page"""
        with ui.column().classes('w-full h-full p-6'):
            # Header
            with ui.row().classes('items-center justify-between mb-6'):
                ui.label('Create').classes('text-2xl font-bold')
                
                # Mode selector
                self.mode_tabs = ui.tabs().classes('bg-surface')
                self.mode_tabs.on('update:model-value', self._on_mode_change)
                
                with self.mode_tabs:
                    ui.tab('image', label='Image', icon='image')
                    ui.tab('3d', label='3D Model', icon='view_in_ar')
                    ui.tab('video', label='Video', icon='movie')
                    
            # Main content area
            with ui.row().classes('gap-6 w-full flex-grow'):
                # Left: Input and controls
                with ui.column().classes('flex-1'):
                    self._render_input_section()
                    
                # Right: Enhancement and progress
                with ui.column().classes('w-96'):
                    self.enhancement_panel.render()
                    ui.space().classes('h-4')
                    self.progress_pipeline.render()
                    
            # Bottom: Output preview
            self._render_output_section()
            
    def _render_input_section(self):
        """Render input controls section"""
        with ui.card().classes('card'):
            # Prompt input
            self.prompt_input = ui.textarea(
                label='Prompt',
                placeholder='Describe what you want to create...',
                value=''
            ).classes('w-full').props('rows=3')
            
            # Model selector
            with ui.row().classes('gap-4 mt-4'):
                self.model_select = ui.select(
                    label='Model',
                    options=[],
                    value=None
                ).classes('flex-1')
                
                # Load models button
                ui.button(
                    'Refresh',
                    icon='refresh',
                    on_click=self._load_available_models
                ).props('flat')
                
            # Mode-specific controls
            self.mode_controls_container = ui.column().classes('mt-4')
            self._render_mode_controls()
            
            # Generation controls
            with ui.row().classes('gap-4 mt-6'):
                self.generate_button = ui.button(
                    'Generate',
                    icon='auto_awesome',
                    on_click=self._start_generation
                ).props('unelevated').classes('flex-1')
                
                self.cancel_button = ui.button(
                    'Cancel',
                    icon='cancel',
                    on_click=self._cancel_generation
                ).props('flat').classes('hidden')
                
    def _render_mode_controls(self):
        """Render controls specific to current mode"""
        self.mode_controls_container.clear()
        
        with self.mode_controls_container:
            if self.current_mode == "image":
                with ui.row().classes('gap-4'):
                    self.width_input = ui.number(
                        label='Width',
                        value=1024,
                        min=256,
                        max=2048,
                        step=64
                    ).classes('flex-1')
                    
                    self.height_input = ui.number(
                        label='Height',
                        value=1024,
                        min=256,
                        max=2048,
                        step=64
                    ).classes('flex-1')
                    
                # Steps slider with label
                ui.label('Steps').classes('text-sm font-medium')
                self.steps_slider = ui.slider(
                    min=1,
                    max=150,
                    value=20
                ).classes('w-full')
                
            elif self.current_mode == "3d":
                # Quality preset
                self.quality_select = ui.select(
                    label='Quality Preset',
                    options=['draft', 'standard', 'high', 'ultra'],
                    value='standard'
                ).classes('w-full')
                
                # Input image upload
                self.image_upload = ui.upload(
                    label='Input Image (optional)',
                    accept='image/*',
                    max_files=1
                ).classes('w-full mt-4')
                
            elif self.current_mode == "video":
                with ui.row().classes('gap-4'):
                    self.duration_input = ui.number(
                        label='Duration (s)',
                        value=4.0,
                        min=1.0,
                        max=10.0,
                        step=0.5
                    ).classes('flex-1')
                    
                    self.fps_input = ui.number(
                        label='FPS',
                        value=24,
                        min=8,
                        max=60,
                        step=1
                    ).classes('flex-1')
                    
    def _render_output_section(self):
        """Render output preview section"""
        with ui.card().classes('card mt-6'):
            with ui.row().classes('items-center justify-between mb-4'):
                ui.label('Output').classes('text-lg font-semibold')
                
                # Export options
                self.export_menu = ui.menu()
                with ui.button(icon='download').props('flat'):
                    with self.export_menu:
                        ui.menu_item('Save to Library', on_click=self._save_to_library)
                        ui.menu_item('Export...', on_click=self._show_export_dialog)
                        
            # Preview area
            self.preview_container = ui.column().classes('min-h-[300px] items-center justify-center')
            with self.preview_container:
                ui.label('Output will appear here').classes('text-gray-500')
                
    async def _on_mode_change(self, event):
        """Handle mode change"""
        self.current_mode = event.value
        
        # Update enhancement panel model type
        if self.current_mode == "image":
            self.enhancement_panel.set_model_type(ModelType.FLUX_1_DEV)
        elif self.current_mode == "3d":
            self.enhancement_panel.set_model_type(ModelType.HUNYUAN_3D_21)
            
        # Update controls
        self._render_mode_controls()
        
        # Load appropriate models
        await self._load_available_models()
        
        # Reset pipeline
        self.progress_pipeline.reset()
        
    async def _load_available_models(self):
        """Load available models for current mode"""
        await self.model_manager.initialize()
        
        model_type_map = {
            "image": "image",
            "3d": "3d", 
            "video": "video"
        }
        
        models = self.model_manager.list_available_models(model_type_map[self.current_mode])
        
        # Update model selector
        options = {m['id']: m['name'] for m in models if m['is_available']}
        self.model_select.options = options
        
        if options:
            self.model_select.value = list(options.keys())[0]
            
    def _on_enhancement_change(self, values: Dict[str, Any]):
        """Handle enhancement panel changes"""
        # Could preview enhanced prompt here
        pass
        
    async def _start_generation(self):
        """Start the generation process"""
        if self.generation_task and not self.generation_task.done():
            ui.notify('Generation already in progress', type='warning')
            return
            
        # Validate inputs
        if not self.prompt_input.value and self.current_mode != "3d":
            ui.notify('Please enter a prompt', type='warning')
            return
            
        if not self.model_select.value:
            ui.notify('Please select a model', type='warning')
            return
            
        # Update UI state
        self.generate_button.props('loading')
        self.cancel_button.classes(remove='hidden')
        
        # Start generation task
        self.generation_task = asyncio.create_task(self._run_generation())
        
    async def _run_generation(self):
        """Run the generation process"""
        try:
            # Get enhancement values
            enhancement = self.enhancement_panel.get_values()
            
            # Reset and start pipeline
            await self.progress_pipeline.start()
            
            if self.current_mode == "image":
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
                    width=int(self.width_input.value),
                    height=int(self.height_input.value),
                    steps=int(self.steps_slider.value),
                    enhancement_fields=enhancement['fields'],
                    use_enhancement=enhancement['use_llm']
                )
                
                # Generate
                def progress_callback(progress, message):
                    if progress < 25:
                        step = 0
                    elif progress < 50:
                        step = 1
                    elif progress < 90:
                        step = 2
                    else:
                        step = 3
                        
                    # Advance to correct step
                    while self.progress_pipeline.current_step_index < step:
                        asyncio.create_task(self.progress_pipeline.advance_step(message))
                        
                response = await self.image_processor.generate(request, progress_callback)
                
                # Show result
                if response.image_path:
                    self._show_image_result(response.image_path)
                    
            elif self.current_mode == "3d":
                # Similar for 3D generation
                self.progress_pipeline.set_steps([
                    PipelineStep("prepare", "Prepare Input", "build", 5),
                    PipelineStep("multiview", "Generate Views", "view_carousel", 30),
                    PipelineStep("reconstruct", "3D Reconstruction", "view_in_ar", 60),
                    PipelineStep("texture", "Generate Textures", "texture", 40),
                    PipelineStep("export", "Export Model", "save", 10)
                ])
                
                # Handle input image if uploaded
                input_image = None
                if hasattr(self, 'image_upload') and self.image_upload.value:
                    # Save uploaded file
                    input_image = str(self.output_dir / "temp" / self.image_upload.value[0].name)
                    
                request = ThreeDGenerationRequest(
                    prompt=self.prompt_input.value or "Generate 3D model from image",
                    model=self.model_select.value,
                    input_image=input_image,
                    quality_preset=self.quality_select.value,
                    enhancement_fields=enhancement['fields'],
                    use_enhancement=enhancement['use_llm'] and not input_image
                )
                
                # Generate with progress
                response = await self.threed_processor.generate(request, self._3d_progress_callback)
                
                if response.model_path:
                    self._show_3d_result(response.model_path, response.preview_images)
                    
            # Mark complete
            await self.progress_pipeline.complete()
            ui.notify('Generation complete!', type='positive')
            
        except Exception as e:
            await self.progress_pipeline.fail(f'Error: {str(e)}')
            ui.notify(f'Generation failed: {str(e)}', type='negative')
            
        finally:
            # Reset UI state
            self.generate_button.props(remove='loading')
            self.cancel_button.classes('hidden')
            
    def _3d_progress_callback(self, progress, message):
        """Progress callback for 3D generation"""
        # Map progress to pipeline steps
        if progress < 10:
            step = 0
        elif progress < 40:
            step = 1
        elif progress < 60:
            step = 2
        elif progress < 85:
            step = 3
        else:
            step = 4
            
        # Advance to correct step
        while self.progress_pipeline.current_step_index < step:
            asyncio.create_task(self.progress_pipeline.advance_step(message))
            
    async def _cancel_generation(self):
        """Cancel ongoing generation"""
        if self.generation_task and not self.generation_task.done():
            self.generation_task.cancel()
            await self.progress_pipeline.fail('Generation cancelled')
            ui.notify('Generation cancelled', type='info')
            
    def _show_image_result(self, image_path: Path):
        """Display generated image"""
        self.preview_container.clear()
        with self.preview_container:
            ui.image(str(image_path)).classes('max-w-full max-h-[500px] rounded')
            
    def _show_3d_result(self, model_path: Path, preview_images: List[Path]):
        """Display generated 3D model"""
        self.preview_container.clear()
        with self.preview_container:
            # Show preview images in a grid
            with ui.row().classes('gap-2'):
                for img_path in preview_images[:4]:
                    ui.image(str(img_path)).classes('w-32 h-32 rounded')
                    
            ui.label(f'3D Model: {model_path.name}').classes('mt-4')
            
    async def _save_to_library(self):
        """Save output to library"""
        ui.notify('Saved to library', type='positive')
        
    async def _show_export_dialog(self):
        """Show export options dialog"""
        with ui.dialog() as dialog, ui.card():
            ui.label('Export Options').classes('text-lg font-semibold mb-4')
            # Add export options here
            ui.button('Export', on_click=dialog.close)
            
        dialog.open()