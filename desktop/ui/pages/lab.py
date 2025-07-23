"""
Lab page - Advanced tools and experimental features
"""

import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List
from nicegui import ui
import time
import json
import logging

from .lab_batch_processor import BatchProcessor

logger = logging.getLogger(__name__)

class LabPage:
    """Lab page for advanced tools and experiments"""
    
    def __init__(self, output_dir: Path, model_manager=None):
        self.output_dir = output_dir
        self.model_manager = model_manager
        self.current_tool = 'batch'
        self.batch_items: List[Dict[str, Any]] = []
        self.processing_task: Optional[asyncio.Task] = None
        
        # Initialize batch processor
        self.batch_processor = BatchProcessor(model_manager, output_dir) if model_manager else None
        
    def render(self):
        """Render the lab page"""
        with ui.column().classes('w-full h-full'):
            # Header
            with ui.card().classes('w-full mb-4').style('background-color: #1F1F1F; border: 1px solid #333333'):
                with ui.row().classes('items-center justify-between'):
                    ui.label('Lab').classes('text-2xl font-bold')
                    ui.label('Experimental Features & Advanced Tools').classes('text-gray-400')
                    
            # Tool selector
            with ui.tabs().classes('w-full') as tabs:
                self.batch_tab = ui.tab('Batch Processing', icon='queue')
                self.upscale_tab = ui.tab('AI Upscaling', icon='zoom_out_map')
                self.style_tab = ui.tab('Style Transfer', icon='palette')
                self.animation_tab = ui.tab('Animation', icon='animation')
                self.analysis_tab = ui.tab('Model Analysis', icon='analytics')
                
            with ui.tab_panels(tabs, value=self.batch_tab).classes('w-full flex-grow'):
                # Batch Processing
                with ui.tab_panel(self.batch_tab):
                    self._render_batch_processing()
                    
                # AI Upscaling
                with ui.tab_panel(self.upscale_tab):
                    self._render_upscaling()
                    
                # Style Transfer
                with ui.tab_panel(self.style_tab):
                    self._render_style_transfer()
                    
                # Animation
                with ui.tab_panel(self.animation_tab):
                    self._render_animation()
                    
                # Model Analysis
                with ui.tab_panel(self.analysis_tab):
                    self._render_model_analysis()
                    
    def _render_batch_processing(self):
        """Render batch processing tool"""
        with ui.column().classes('w-full gap-4'):
            # Instructions
            with ui.card().classes('w-full').style('background-color: #0A0A0A; border: 1px solid #333333'):
                ui.label('Batch Processing').classes('text-lg font-semibold mb-2')
                ui.label(
                    'Process multiple prompts or images at once. Perfect for generating variations '
                    'or testing different parameters.'
                ).classes('text-sm text-gray-400')
                
            # Batch input area
            with ui.row().classes('w-full gap-4'):
                # Left: Input configuration
                with ui.card().classes('flex-1').style('background-color: #0A0A0A; border: 1px solid #333333'):
                    ui.label('Batch Configuration').classes('font-semibold mb-4')
                    
                    # Mode selection
                    self.batch_mode = ui.toggle(
                        ['text', 'csv', 'images'],
                        value='text'
                    ).props('dense').on('change', self._update_batch_input)
                    
                    # Input area
                    self.batch_input_container = ui.column().classes('w-full mt-4')
                    self._update_batch_input()
                    
                    # Common settings
                    with ui.column().classes('w-full gap-3 mt-4'):
                        self.batch_model = ui.select(
                            label='Model',
                            options={
                                'flux-1-schnell': 'FLUX.1 Schnell (Fast)',
                                'flux-1-dev': 'FLUX.1 Dev (High Quality)',
                                'hunyuan3d-21': 'HunYuan3D 2.1',
                                'sparc3d-base': 'Sparc3D (High Resolution)',
                                'hi3dgen-base': 'Hi3DGen (High Fidelity)'
                            },
                            value='flux-1-schnell'
                        ).classes('w-full').props('outlined dense')
                        
                        with ui.row().classes('w-full gap-4'):
                            self.batch_count = ui.number(
                                label='Images per prompt',
                                value=1,
                                min=1,
                                max=10
                            ).classes('flex-1').props('outlined dense')
                            
                            self.batch_workers = ui.number(
                                label='Parallel workers',
                                value=1,
                                min=1,
                                max=4
                            ).classes('flex-1').props('outlined dense')
                            
                # Right: Queue preview
                with ui.card().classes('flex-1').style('background-color: #0A0A0A; border: 1px solid #333333'):
                    ui.label('Processing Queue').classes('font-semibold mb-4')
                    
                    # Queue stats
                    with ui.row().classes('w-full justify-between mb-4'):
                        self.queue_stats = ui.label('0 items in queue').classes('text-sm text-gray-400')
                        ui.button(
                            'Clear Queue',
                            icon='clear_all',
                            on_click=self._clear_batch_queue
                        ).props('flat dense')
                        
                    # Queue list
                    self.queue_container = ui.column().classes('w-full gap-2 max-h-96 overflow-auto')
                    self._update_queue_display()
                    
            # Action buttons
            with ui.row().classes('w-full gap-4 mt-4'):
                self.add_batch_button = ui.button(
                    'Add to Queue',
                    icon='add',
                    on_click=self._add_to_batch
                ).props('unelevated').style('background-color: #7C3AED')
                
                self.process_batch_button = ui.button(
                    'Process Batch',
                    icon='play_arrow',
                    on_click=self._process_batch
                ).props('unelevated').bind_enabled_from(self, 'batch_items')
                
                self.stop_batch_button = ui.button(
                    'Stop Processing',
                    icon='stop',
                    on_click=self._stop_batch
                ).props('flat color=negative').visible = False
                
    def _render_upscaling(self):
        """Render AI upscaling tool"""
        with ui.column().classes('w-full gap-4'):
            # Info card
            with ui.card().classes('w-full').style('background-color: #0A0A0A; border: 1px solid #333333'):
                ui.label('AI Upscaling').classes('text-lg font-semibold mb-2')
                ui.label(
                    'Enhance image resolution using AI models. Supports 2x, 4x, and 8x upscaling '
                    'with various enhancement options.'
                ).classes('text-sm text-gray-400')
                
            # Main content
            with ui.row().classes('w-full gap-4'):
                # Left: Upload and settings
                with ui.card().classes('flex-1').style('background-color: #0A0A0A; border: 1px solid #333333'):
                    ui.label('Upload Image').classes('font-semibold mb-4')
                    
                    self.upscale_upload = ui.upload(
                        label='Drop image here or click to browse',
                        max_files=1,
                        on_upload=self._handle_upscale_upload
                    ).classes('w-full').props('outlined accept="image/*"')
                    
                    # Preview
                    self.upscale_preview = ui.column().classes('w-full mt-4 items-center')
                    
                    # Settings
                    with ui.column().classes('w-full gap-3 mt-4'):
                        self.upscale_factor = ui.select(
                            label='Upscale Factor',
                            options={
                                '2': '2x (Good for most uses)',
                                '4': '4x (High quality)',
                                '8': '8x (Maximum, slow)'
                            },
                            value='2'
                        ).classes('w-full').props('outlined dense')
                        
                        self.upscale_model = ui.select(
                            label='Model',
                            options={
                                'realesrgan': 'Real-ESRGAN (General)',
                                'realesrgan-anime': 'Real-ESRGAN (Anime)',
                                'gfpgan': 'GFPGAN (Faces)',
                                'codeformer': 'CodeFormer (Faces)'
                            },
                            value='realesrgan'
                        ).classes('w-full').props('outlined dense')
                        
                        # Enhancement options
                        ui.label('Enhancement Options').classes('font-medium mt-2')
                        self.denoise_switch = ui.switch('Denoise', value=True)
                        self.sharpen_switch = ui.switch('Sharpen', value=False)
                        self.face_enhance_switch = ui.switch('Face Enhancement', value=False)
                        
                # Right: Result preview
                with ui.card().classes('flex-1').style('background-color: #0A0A0A; border: 1px solid #333333'):
                    ui.label('Result').classes('font-semibold mb-4')
                    
                    self.upscale_result = ui.column().classes('w-full items-center justify-center h-96')
                    with self.upscale_result:
                        ui.icon('image', size='4rem').classes('text-gray-600')
                        ui.label('Upscaled image will appear here').classes('text-gray-500 mt-2')
                        
            # Process button
            self.upscale_button = ui.button(
                'Upscale Image',
                icon='zoom_out_map',
                on_click=self._process_upscale
            ).props('unelevated size=lg').classes('mt-4').style('background-color: #7C3AED')
            
    def _render_style_transfer(self):
        """Render style transfer tool"""
        with ui.column().classes('w-full gap-4'):
            # Info
            with ui.card().classes('w-full').style('background-color: #0A0A0A; border: 1px solid #333333'):
                ui.label('Style Transfer').classes('text-lg font-semibold mb-2')
                ui.label(
                    'Apply artistic styles from one image to another. Combines the content of one image '
                    'with the style of another using neural networks.'
                ).classes('text-sm text-gray-400')
                
            # Input areas
            with ui.row().classes('w-full gap-4'):
                # Content image
                with ui.card().classes('flex-1').style('background-color: #0A0A0A; border: 1px solid #333333'):
                    ui.label('Content Image').classes('font-semibold mb-2')
                    ui.label('The image to apply style to').classes('text-sm text-gray-400 mb-4')
                    
                    self.content_upload = ui.upload(
                        max_files=1
                    ).classes('w-full').props('outlined accept="image/*"')
                    
                    self.content_preview = ui.column().classes('w-full mt-4 items-center')
                    
                # Style image
                with ui.card().classes('flex-1').style('background-color: #0A0A0A; border: 1px solid #333333'):
                    ui.label('Style Image').classes('font-semibold mb-2')
                    ui.label('The artistic style to apply').classes('text-sm text-gray-400 mb-4')
                    
                    self.style_upload = ui.upload(
                        max_files=1
                    ).classes('w-full').props('outlined accept="image/*"')
                    
                    self.style_preview = ui.column().classes('w-full mt-4 items-center')
                    
            # Settings
            with ui.card().classes('w-full mt-4').style('background-color: #0A0A0A; border: 1px solid #333333'):
                ui.label('Transfer Settings').classes('font-semibold mb-4')
                
                with ui.row().classes('w-full gap-4'):
                    with ui.column().classes('flex-1'):
                        ui.label('Style Strength').classes('text-sm text-gray-400')
                        self.style_strength = ui.slider(
                            min=0,
                            max=100,
                            value=75
                        ).props('label-always')
                        
                    with ui.column().classes('flex-1'):
                        ui.label('Content Preservation').classes('text-sm text-gray-400')
                        self.content_weight = ui.slider(
                            min=0,
                            max=100,
                            value=50
                        ).props('label-always')
                        
            # Process button
            ui.button(
                'Apply Style Transfer',
                icon='brush',
                on_click=self._process_style_transfer
            ).props('unelevated size=lg').classes('mt-4').style('background-color: #7C3AED')
            
    def _render_animation(self):
        """Render animation tool"""
        with ui.column().classes('w-full items-center justify-center h-96'):
            ui.icon('animation', size='4rem').classes('text-gray-600 mb-4')
            ui.label('Animation Tools').classes('text-xl font-semibold mb-2')
            ui.label('Coming soon: Image-to-video and interpolation tools').classes('text-gray-500')
            
            with ui.row().classes('gap-4 mt-6'):
                ui.badge('Image Morphing').props('color=purple')
                ui.badge('Keyframe Animation').props('color=purple')
                ui.badge('Motion Transfer').props('color=purple')
                
    def _render_model_analysis(self):
        """Render model analysis tool"""
        with ui.column().classes('w-full gap-4'):
            # Info
            with ui.card().classes('w-full').style('background-color: #0A0A0A; border: 1px solid #333333'):
                ui.label('Model Analysis').classes('text-lg font-semibold mb-2')
                ui.label(
                    'Analyze and compare AI models. Benchmark performance, memory usage, '
                    'and quality metrics.'
                ).classes('text-sm text-gray-400')
                
            # Analysis options
            with ui.row().classes('w-full gap-4'):
                # Model selection
                with ui.card().classes('flex-1').style('background-color: #0A0A0A; border: 1px solid #333333'):
                    ui.label('Select Models to Analyze').classes('font-semibold mb-4')
                    
                    # Model checkboxes
                    self.analysis_models = {}
                    models = [
                        'SDXL-Turbo',
                        'FLUX-1-schnell',
                        'FLUX-1-dev',
                        'hunyuan3d-21',
                        'hunyuan3d-20',
                        'hunyuan3d-mini'
                    ]
                    
                    for model in models:
                        self.analysis_models[model] = ui.checkbox(model).props('dense')
                        
                # Benchmark settings
                with ui.card().classes('flex-1').style('background-color: #0A0A0A; border: 1px solid #333333'):
                    ui.label('Benchmark Settings').classes('font-semibold mb-4')
                    
                    self.benchmark_samples = ui.number(
                        label='Test Samples',
                        value=5,
                        min=1,
                        max=50
                    ).classes('w-full').props('outlined dense')
                    
                    self.benchmark_resolution = ui.select(
                        label='Resolution',
                        options={
                            '512': '512x512',
                            '768': '768x768',
                            '1024': '1024x1024'
                        },
                        value='512'
                    ).classes('w-full').props('outlined dense')
                    
                    ui.label('Metrics to Measure').classes('font-medium mt-4 mb-2')
                    self.measure_speed = ui.checkbox('Generation Speed', value=True).props('dense')
                    self.measure_memory = ui.checkbox('Memory Usage', value=True).props('dense')
                    self.measure_quality = ui.checkbox('Quality Score', value=False).props('dense')
                    
            # Results area
            self.analysis_results = ui.card().classes('w-full mt-4').style(
                'background-color: #0A0A0A; border: 1px solid #333333; min-height: 200px'
            )
            
            with self.analysis_results:
                with ui.column().classes('w-full items-center justify-center p-8'):
                    ui.icon('analytics', size='3rem').classes('text-gray-600')
                    ui.label('Analysis results will appear here').classes('text-gray-500 mt-2')
                    
            # Run button
            ui.button(
                'Run Analysis',
                icon='play_arrow',
                on_click=self._run_model_analysis
            ).props('unelevated size=lg').classes('mt-4').style('background-color: #7C3AED')
            
    def _update_batch_input(self):
        """Update batch input area based on mode"""
        self.batch_input_container.clear()
        
        with self.batch_input_container:
            if self.batch_mode.value == 'text':
                ui.label('Enter prompts (one per line)').classes('text-sm text-gray-400 mb-2')
                self.batch_text = ui.textarea(
                    placeholder='A beautiful sunset over mountains\nA futuristic city at night\nA magical forest with glowing plants',
                    value=''
                ).classes('w-full').props('rows=8 outlined dense')
                
            elif self.batch_mode.value == 'csv':
                ui.label('Upload CSV file').classes('text-sm text-gray-400 mb-2')
                ui.label('Format: prompt, negative_prompt, steps, guidance_scale').classes('text-xs text-gray-500 mb-2')
                self.batch_csv = ui.upload(
                    max_files=1
                ).classes('w-full').props('outlined accept=".csv"')
                
            elif self.batch_mode.value == 'images':
                ui.label('Upload images for batch processing').classes('text-sm text-gray-400 mb-2')
                self.batch_images = ui.upload(
                    multiple=True
                ).classes('w-full').props('outlined accept="image/*"')
                
    def _add_to_batch(self):
        """Add items to batch queue"""
        if self.batch_mode.value == 'text' and hasattr(self, 'batch_text'):
            prompts = [p.strip() for p in self.batch_text.value.split('\n') if p.strip()]
            for prompt in prompts:
                self.batch_items.append({
                    'type': 'text',
                    'prompt': prompt,
                    'status': 'pending'
                })
            self.batch_text.value = ''
            
        elif self.batch_mode.value == 'csv' and hasattr(self, 'batch_csv'):
            # Handle CSV upload
            if self.batch_processor:
                try:
                    # TODO: Implement actual CSV file handling from upload
                    ui.notify('CSV upload handling in progress', type='info')
                except Exception as e:
                    ui.notify(f'Failed to load CSV: {str(e)}', type='negative')
            else:
                ui.notify('Batch processor not available', type='negative')
            
        elif self.batch_mode.value == 'images':
            # Handle image batch
            ui.notify('Image batch processing not implemented yet', type='info')
            
        self._update_queue_display()
        ui.notify(f'Added {len(prompts)} items to queue', type='positive')
        
    def _update_queue_display(self):
        """Update queue display"""
        self.queue_container.clear()
        self.queue_stats.text = f'{len(self.batch_items)} items in queue'
        
        with self.queue_container:
            if not self.batch_items:
                ui.label('Queue is empty').classes('text-gray-500 text-center py-8')
            else:
                for i, item in enumerate(self.batch_items):
                    with ui.row().classes('w-full items-center gap-2 p-2 rounded').style(
                        'background-color: #0A0A0A'
                    ):
                        # Status icon
                        if item['status'] == 'pending':
                            ui.icon('schedule', size='sm').classes('text-gray-500')
                        elif item['status'] == 'processing':
                            ui.spinner(size='sm')
                        elif item['status'] == 'completed':
                            ui.icon('check_circle', size='sm').classes('text-green-500')
                        elif item['status'] == 'failed':
                            ui.icon('error', size='sm').classes('text-red-500')
                            
                        # Content
                        ui.label(f"{i+1}. {item['prompt'][:50]}...").classes('flex-grow text-sm')
                        
                        # Remove button
                        if item['status'] == 'pending':
                            ui.button(
                                icon='close',
                                on_click=lambda idx=i: self._remove_from_queue(idx)
                            ).props('flat dense round size=sm')
                            
    def _remove_from_queue(self, index: int):
        """Remove item from queue"""
        if 0 <= index < len(self.batch_items):
            self.batch_items.pop(index)
            self._update_queue_display()
            
    def _clear_batch_queue(self):
        """Clear the batch queue"""
        self.batch_items = [item for item in self.batch_items if item['status'] != 'pending']
        self._update_queue_display()
        
    async def _process_batch(self):
        """Process batch queue"""
        if not self.batch_items or not self.batch_processor:
            if not self.batch_processor:
                ui.notify('Batch processor not initialized', type='negative')
            return
            
        # Update UI
        self.process_batch_button.props('loading')
        self.stop_batch_button.visible = True
        
        try:
            # Get pending items
            pending_items = [item for item in self.batch_items if item['status'] == 'pending']
            
            if not pending_items:
                ui.notify('No pending items to process', type='info')
                return
                
            # Define callbacks
            def progress_callback(value, message):
                # Update progress in UI if needed
                logger.info(f"Batch progress: {value:.0%} - {message}")
                
            def item_callback(index, status, result):
                # Update item status
                if index < len(self.batch_items):
                    self.batch_items[index]['status'] = status
                    if result and isinstance(result, dict):
                        self.batch_items[index]['result'] = result
                    self._update_queue_display()
                    
            # Process batch
            results = await self.batch_processor.process_batch(
                items=pending_items,
                model_id=self.batch_model.value,
                images_per_prompt=int(self.batch_count.value),
                parallel_workers=int(self.batch_workers.value),
                progress_callback=progress_callback,
                item_callback=item_callback
            )
            
            # Show results summary
            ui.notify(
                f'Batch completed: {results["completed"]} successful, {results["failed"]} failed',
                type='positive' if results["failed"] == 0 else 'warning'
            )
            
            # Optionally show batch results location
            if results.get("session_dir"):
                ui.notify(f'Results saved to: {Path(results["session_dir"]).name}', type='info')
            
        except asyncio.CancelledError:
            ui.notify('Batch processing cancelled', type='info')
            
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            ui.notify(f'Batch processing failed: {str(e)}', type='negative')
            
        finally:
            self.process_batch_button.props(remove='loading')
            self.stop_batch_button.visible = False
            
    def _stop_batch(self):
        """Stop batch processing"""
        if self.processing_task and not self.processing_task.done():
            self.processing_task.cancel()
            
    def _handle_upscale_upload(self, e):
        """Handle upscale image upload"""
        if e.content:
            self.upscale_preview.clear()
            with self.upscale_preview:
                ui.image(e.content).classes('max-w-full max-h-64')
                ui.label(e.name).classes('text-sm text-gray-400 mt-2')
                
    async def _process_upscale(self):
        """Process upscaling"""
        ui.notify('Upscaling feature not implemented yet', type='info')
        
    async def _process_style_transfer(self):
        """Process style transfer"""
        ui.notify('Style transfer feature not implemented yet', type='info')
        
    async def _run_model_analysis(self):
        """Run model analysis"""
        # Get selected models
        selected = [model for model, cb in self.analysis_models.items() if cb.value]
        
        if not selected:
            ui.notify('Please select at least one model to analyze', type='warning')
            return
            
        # Clear results
        self.analysis_results.clear()
        
        with self.analysis_results:
            ui.label('Running Analysis...').classes('text-lg font-semibold mb-4')
            progress = ui.linear_progress().classes('w-full mb-4')
            
            # Simulate analysis
            results_container = ui.column().classes('w-full gap-4')
            
            with results_container:
                for i, model in enumerate(selected):
                    progress.value = (i + 1) / len(selected)
                    
                    # Create result card
                    with ui.card().classes('w-full').style('background-color: #1A1A1A'):
                        ui.label(model).classes('font-semibold mb-2')
                        
                        with ui.row().classes('gap-4'):
                            # Mock results
                            with ui.column():
                                ui.label('Speed:').classes('text-sm text-gray-400')
                                ui.label(f'{2.5 + i * 0.5:.1f}s/image').classes('text-sm')
                                
                            with ui.column():
                                ui.label('Memory:').classes('text-sm text-gray-400')
                                ui.label(f'{4.2 + i * 1.1:.1f} GB').classes('text-sm')
                                
                            with ui.column():
                                ui.label('Quality:').classes('text-sm text-gray-400')
                                ui.label(f'{85 + i * 2}/100').classes('text-sm')
                                
                    await asyncio.sleep(0.5)  # Simulate processing
                    
            progress.visible = False
            ui.notify('Analysis complete!', type='positive')