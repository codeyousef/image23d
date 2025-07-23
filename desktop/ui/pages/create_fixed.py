"""
Fixed Create page that uses the actual generation system
"""

import asyncio
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
from core.processors.image_processor_fixed import ImageProcessor  
from core.processors.threed_processor_fixed import ThreeDProcessor
from ..components import EnhancementPanel, ProgressPipeline, PipelineStep

class CreatePage:
    """Main creation page with real generation"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        
        # Use fixed processors
        self.image_processor = ImageProcessor(output_dir / "images")
        self.threed_processor = ThreeDProcessor(output_dir / "3d")
        
        # UI components
        self.enhancement_panel = EnhancementPanel(on_change=self._on_enhancement_change)
        self.progress_pipeline = ProgressPipeline()
        
        # State
        self.current_mode = "image"
        self.generation_task: Optional[asyncio.Task] = None
        
        # Get available models from the actual system
        self._load_available_models()
        
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
                    ui.tab('video', label='Video', icon='movie').props('disable')  # Video not implemented
                    
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
                    options={},
                    value=None
                ).classes('flex-1')
                
                # Refresh models button
                ui.button(
                    icon='refresh',
                    on_click=self._load_available_models
                ).props('flat round')
                
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
                ui.label('Steps').classes('text-sm font-medium mt-4')
                self.steps_slider = ui.slider(
                    min=1,
                    max=150,
                    value=20
                ).classes('w-full')
                
                # Guidance scale
                ui.label('Guidance Scale').classes('text-sm font-medium mt-4')
                self.guidance_slider = ui.slider(
                    min=1.0,
                    max=20.0,
                    value=7.5,
                    step=0.5
                ).classes('w-full')
                
            elif self.current_mode == "3d":
                # Quality preset
                self.quality_select = ui.select(
                    label='Quality Preset',
                    options={
                        'draft': 'Draft (Fast)',
                        'standard': 'Standard',
                        'high': 'High Quality',
                        'ultra': 'Ultra (Slow)'
                    },
                    value='standard'
                ).classes('w-full')
                
                # Input image upload
                self.image_upload = ui.upload(
                    label='Input Image (optional)',
                    accept='image/*',
                    max_files=1,
                    on_upload=self._handle_image_upload
                ).classes('w-full mt-4')
                
                # Export formats
                self.export_formats = ui.select(
                    label='Export Formats',
                    options=['glb', 'obj', 'ply', 'stl'],
                    multiple=True,
                    value=['glb']
                ).classes('w-full mt-4')
                
    def _render_output_section(self):
        """Render output preview section"""
        with ui.card().classes('card mt-6'):
            with ui.row().classes('items-center justify-between mb-4'):
                ui.label('Output').classes('text-lg font-semibold')
                
                # Export options
                self.export_button = ui.button(icon='download').props('flat')
                self.export_button.on('click', self._export_output)
                self.export_button.visible = False
                        
            # Preview area
            self.preview_container = ui.column().classes('min-h-[300px] items-center justify-center')
            with self.preview_container:
                ui.label('Output will appear here').classes('text-gray-500')
                
    def _load_available_models(self):
        """Load available models from the actual system"""
        try:
            from src.hunyuan3d_app.config import IMAGE_MODELS, HUNYUAN3D_MODELS
            
            if self.current_mode == "image":
                # Get image models
                options = {}
                for model_id, config in IMAGE_MODELS.items():
                    options[model_id] = config.name
                    
                self.model_select.options = options
                if options:
                    self.model_select.value = list(options.keys())[0]
                    
            elif self.current_mode == "3d":
                # Get 3D models
                options = {}
                for model_id, config in HUNYUAN3D_MODELS.items():
                    options[model_id] = config["name"]
                    
                self.model_select.options = options
                if options:
                    self.model_select.value = "hunyuan3d-21"  # Default to 2.1
                    
        except Exception as e:
            ui.notify(f'Failed to load models: {str(e)}', type='negative')
            
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
        self._load_available_models()
        
        # Reset pipeline
        self.progress_pipeline.reset()
        
    def _on_enhancement_change(self, values: Dict[str, Any]):
        """Handle enhancement panel changes"""
        # Could preview enhanced prompt here
        pass
        
    def _handle_image_upload(self, e):
        """Handle image upload for 3D mode"""
        if e.content:
            ui.notify('Image uploaded successfully', type='positive')
            
    async def _start_generation(self):
        """Start the generation process"""
        if self.generation_task and not self.generation_task.done():
            ui.notify('Generation already in progress', type='warning')
            return
            
        # Validate inputs
        if self.current_mode == "image" and not self.prompt_input.value:
            ui.notify('Please enter a prompt', type='warning')
            return
        elif self.current_mode == "3d" and not self.prompt_input.value and not hasattr(self, '_uploaded_image_path'):
            ui.notify('Please enter a prompt or upload an image', type='warning')
            return
            
        if not self.model_select.value:
            ui.notify('Please select a model', type='warning')
            return
            
        # Update UI state
        self.generate_button.props('loading')
        self.cancel_button.classes(remove='hidden')
        self.export_button.visible = False
        
        # Start generation task
        self.generation_task = asyncio.create_task(self._run_generation())
        
    async def _run_generation(self):
        """Run the actual generation process"""
        try:
            # Get enhancement values
            enhancement = self.enhancement_panel.get_values()
            
            # Reset and start pipeline
            await self.progress_pipeline.start()
            
            if self.current_mode == "image":
                await self._generate_image(enhancement)
            elif self.current_mode == "3d":
                await self._generate_3d(enhancement)
                
        except Exception as e:
            await self.progress_pipeline.fail(f'Error: {str(e)}')
            ui.notify(f'Generation failed: {str(e)}', type='negative')
            
        finally:
            # Reset UI state
            self.generate_button.props(remove='loading')
            self.cancel_button.classes('hidden')
            
    async def _generate_image(self, enhancement: Dict[str, Any]):
        """Generate an image"""
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
            guidance_scale=float(self.guidance_slider.value),
            enhancement_fields=enhancement['fields'],
            use_enhancement=enhancement['use_llm']
        )
        
        # Progress callback
        async def progress_callback(progress, message):
            # Map progress to pipeline steps
            if progress < 10:
                await self.progress_pipeline.advance_step("Enhancing prompt...")
            elif progress < 30:
                if self.progress_pipeline.current_step_index < 1:
                    await self.progress_pipeline.advance_step("Loading model...")
            elif progress < 90:
                if self.progress_pipeline.current_step_index < 2:
                    await self.progress_pipeline.advance_step(message)
            else:
                if self.progress_pipeline.current_step_index < 3:
                    await self.progress_pipeline.advance_step("Saving output...")
                    
        # Generate
        response = await self.image_processor.generate(request, progress_callback)
        
        # Show result
        if response.image_path:
            self._show_image_result(response.image_path)
            await self.progress_pipeline.complete()
            ui.notify('Image generated successfully!', type='positive')
        else:
            raise ValueError(response.error or "No image generated")
            
    async def _generate_3d(self, enhancement: Dict[str, Any]):
        """Generate a 3D model"""
        # Set pipeline steps
        self.progress_pipeline.set_steps([
            PipelineStep("prepare", "Prepare Input", "build", 5),
            PipelineStep("generate", "Generate 3D Model", "view_in_ar", 60),
            PipelineStep("export", "Export Model", "save", 10)
        ])
        
        # Handle input image if uploaded
        input_image = None
        if hasattr(self, 'image_upload') and self.image_upload.value:
            # Save uploaded file temporarily
            upload_data = self.image_upload.value[0]
            temp_path = self.output_dir / "temp" / upload_data.name
            temp_path.parent.mkdir(exist_ok=True)
            temp_path.write_bytes(upload_data.content.read())
            input_image = str(temp_path)
            self._uploaded_image_path = input_image
            
        # Create request
        request = ThreeDGenerationRequest(
            prompt=self.prompt_input.value or "",
            model=self.model_select.value,
            input_image=input_image,
            quality_preset=self.quality_select.value,
            export_formats=self.export_formats.value or ['glb'],
            enhancement_fields=enhancement['fields'],
            use_enhancement=enhancement['use_llm'] and not input_image,
            remove_background=True
        )
        
        # Progress callback
        async def progress_callback(progress, message):
            # Update UI progress
            if progress < 10:
                await self.progress_pipeline.advance_step("Preparing input...")
            elif progress < 90:
                if self.progress_pipeline.current_step_index < 1:
                    await self.progress_pipeline.advance_step(message)
            else:
                if self.progress_pipeline.current_step_index < 2:
                    await self.progress_pipeline.advance_step("Exporting model...")
                    
        # Generate
        response = await self.threed_processor.generate(request, progress_callback)
        
        # Show result
        if response.model_path:
            self._show_3d_result(response.model_path, response.preview_images)
            await self.progress_pipeline.complete()
            ui.notify('3D model generated successfully!', type='positive')
        else:
            raise ValueError(response.error or "No 3D model generated")
            
    async def _cancel_generation(self):
        """Cancel ongoing generation"""
        if self.generation_task and not self.generation_task.done():
            self.generation_task.cancel()
            await self.progress_pipeline.fail('Generation cancelled')
            ui.notify('Generation cancelled', type='info')
            
    def _show_image_result(self, image_path: Path):
        """Display generated image"""
        self.preview_container.clear()
        self._current_output_path = image_path
        
        with self.preview_container:
            ui.image(str(image_path)).classes('max-w-full max-h-[500px] rounded')
            
        self.export_button.visible = True
            
    def _show_3d_result(self, model_path: Path, preview_images: list):
        """Display generated 3D model"""
        self.preview_container.clear()
        self._current_output_path = model_path
        
        with self.preview_container:
            if preview_images:
                # Show preview images
                with ui.row().classes('gap-2'):
                    for img_path in preview_images[:4]:
                        ui.image(str(img_path)).classes('w-32 h-32 rounded')
                        
            ui.label(f'3D Model: {model_path.name}').classes('mt-4 text-lg')
            ui.label(f'Size: {model_path.stat().st_size / 1024 / 1024:.1f} MB').classes('text-sm text-gray-500')
            
        self.export_button.visible = True
            
    def _export_output(self):
        """Export/download the current output"""
        if hasattr(self, '_current_output_path') and self._current_output_path:
            # In a real desktop app, this would open a file dialog
            # For now, just show the path
            ui.notify(f'Output saved to: {self._current_output_path}', type='positive', timeout=5000)