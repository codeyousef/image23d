"""Generation Handlers Mixin

Handles generation-related async operations.
"""

import asyncio
from typing import Dict, Any
from pathlib import Path
from nicegui import ui

from core.models.enhancement import ModelType
from core.models.generation import (
    ImageGenerationRequest, 
    ThreeDGenerationRequest,
    VideoGenerationRequest
)
from ...components.progress_pipeline_card import PipelineStep

# Import dependency checker
try:
    from src.hunyuan3d_app.models.dependencies import DependencyChecker
except ImportError:
    from hunyuan3d_app.models.dependencies import DependencyChecker


class GenerationHandlersMixin:
    """Mixin for generation handling methods"""
    
    def _check_model_dependencies(self) -> Dict[str, Any]:
        """Check if all required models and components are installed"""
        # Initialize dependency checker
        from core.config import MODELS_DIR
        checker = DependencyChecker(MODELS_DIR)
        # Check dependencies for current mode
        
        # Map current mode to model type
        if self.current_mode == "image":
            model_type = "image"
        elif self.current_mode == "3d":
            model_type = "3d"
        elif self.current_mode == "video":
            model_type = "video"
        else:
            model_type = "other"
            
        # Check if generation is possible
        result = checker.can_generate(model_type)
        # Check if generation is possible
        
        # Also check specific model if selected
        if hasattr(self, 'model_select') and self.model_select and self.model_select.value:
            model_id = self.model_select.value
            deps = checker.check_model_dependencies(model_id)
            result['selected_model_dependencies'] = deps
            # Check selected model dependencies
            
        return result
        
    def _show_dependency_dialog(self, dependency_check: Dict[str, Any]):
        """Show dialog with missing dependencies"""
        with ui.dialog() as dialog, ui.card().classes('w-[600px]'):
            with ui.row().classes('items-center justify-between mb-4'):
                ui.label('Missing Dependencies').classes('text-xl font-bold')
                ui.button(icon='close', on_click=dialog.close).props('flat round')
                
            # Show what's missing
            if not dependency_check['available_models']:
                ui.label(f'No {self.current_mode} models are installed.').classes('text-red-500 mb-2')
                ui.label('Please download at least one model from the Model Manager.').classes('text-sm')
            else:
                # Show missing components for available models
                if dependency_check.get('missing_components'):
                    ui.label('Some models are missing required components:').classes('mb-2')
                    
                    for model_id, missing in dependency_check['missing_components'].items():
                        with ui.expansion(f'{model_id} (missing {len(missing)} components)', icon='warning').classes('mb-2'):
                            ui.label('Missing components:').classes('font-semibold')
                            for component in missing:
                                with ui.row().classes('items-center gap-2 ml-4'):
                                    ui.icon('close', size='sm').classes('text-red-500')
                                    ui.label(component).classes('text-sm')
                                    
            # If selected model has issues
            if 'selected_model_dependencies' in dependency_check:
                deps = dependency_check['selected_model_dependencies']
                if not deps['satisfied']:
                    ui.separator().classes('my-4')
                    ui.label(f'Selected model "{self.model_select.value}" is missing:').classes('font-semibold')
                    for component in deps['missing_required']:
                        with ui.row().classes('items-center gap-2 ml-4'):
                            ui.icon('close', size='sm').classes('text-red-500')
                            ui.label(component).classes('text-sm')
                            
            # Action buttons
            with ui.row().classes('gap-2 mt-6'):
                ui.button('Go to Model Manager', 
                         on_click=lambda: [dialog.close(), self._navigate_to_models()]).props('unelevated')
                ui.button('Cancel', on_click=dialog.close).props('flat')
                
        dialog.open()
        
    def _navigate_to_models(self):
        """Navigate to models page"""
        # This would be implemented by the parent page to switch tabs
        ui.notify('Please go to the Model Manager to download required models', type='info')
    
    async def _start_generation(self):
        """Start the generation process"""
        # Check if model manager is initialized
        if not self._check_initialization_status():
            return
        
        # Check model dependencies
        dependency_check = self._check_model_dependencies()
        if not dependency_check['can_generate']:
            self._show_dependency_dialog(dependency_check)
            return
        
        # Validate inputs
        if not self.prompt_input.value or not self.prompt_input.value.strip():
            ui.notify('Please enter a prompt', type='warning', position='top')
            return
        
        # Cancel any existing generation
        if self.generation_task and not self.generation_task.done():
            await self._cancel_generation()
        
        # Update UI state
        self.generate_button.disable()
        self.generate_button.text = 'Generating...'
        self.generate_button.props('icon=hourglass_empty')  # Change icon to hourglass
        self.generate_button.update()
        self.cancel_button.visible = True
        self.cancel_button.classes(remove='hidden')
        
        # Clear previous results
        self.preview_container.clear()
        self.export_button.visible = False
        self.save_button.visible = False
        self._current_output = None
        self._current_output_type = None
        
        # Initialize progress pipeline for the current mode
        if self.current_mode == "3d":
            # Get estimated time for current parameters
            estimated_time = self._get_current_time_estimate()
            
            # Set up 3D generation pipeline steps with weights and time estimates
            from ...components.progress_pipeline_card import PipelineStep
            steps = [
                PipelineStep('enhance', 'Enhance Prompt', 'text_increase', int(estimated_time * 0.02), weight=0.02),   # 2% of total
                PipelineStep('model', 'Load Model', 'download', int(estimated_time * 0.05), weight=0.05),             # 5% of total  
                PipelineStep('generate', 'Generate Views', 'photo_camera', int(estimated_time * 0.70), weight=0.70),  # 70% of total - Most GPU-intensive
                PipelineStep('postprocess', 'Process 3D', 'view_in_ar', int(estimated_time * 0.20), weight=0.20),     # 20% of total - Texture generation
                PipelineStep('save', 'Save & Export', 'save', int(estimated_time * 0.03), weight=0.03)                # 3% of total
            ]
            self.progress_pipeline.set_steps(steps)
            self.progress_pipeline.start()
        elif self.current_mode == "image":
            # Set up image generation pipeline steps with weights
            from ...components.progress_pipeline_card import PipelineStep
            steps = [
                PipelineStep('enhance', 'Enhance Prompt', 'text_increase', 5, weight=0.10),   # 10% of total
                PipelineStep('model', 'Load Model', 'download', 5, weight=0.15),              # 15% of total
                PipelineStep('generate', 'Generate Image', 'image', 20, weight=0.70),         # 70% of total
                PipelineStep('save', 'Save Result', 'save', 5, weight=0.05)                  # 5% of total
            ]
            self.progress_pipeline.set_steps(steps)
            self.progress_pipeline.start()
        
        # Start generation
        self.generation_task = asyncio.create_task(self._run_generation())
        
        # Start a timer to check for results (runs in UI thread)
        if not hasattr(self, '_result_checker_timer') or self._result_checker_timer is None:
            self._result_checker_timer = ui.timer(0.1, self._check_for_results)
    
    async def _run_generation(self):
        """Run the actual generation process"""
        try:
            # Get enhancement values
            enhancement = self.enhancement_panel.get_values()
            
            if self.current_mode == "image":
                await self._generate_image(enhancement)
            elif self.current_mode == "3d":
                await self._generate_3d(enhancement)
            elif self.current_mode == "video":
                # Notify about video generation
                self._notify_from_background('Video generation not yet implemented', 'info')
        
        except asyncio.CancelledError:
            # Notify about cancellation
            self._notify_from_background('Generation cancelled', 'info')
            self.progress_pipeline.reset()
        except Exception as e:
            # Notify about error
            error_msg = str(e)
            self._notify_from_background(f'Generation failed: {error_msg}', 'negative')
            print(f"Generation error: {e}")
            import traceback
            traceback.print_exc()
            self.progress_pipeline.fail(error_msg)
        finally:
            # Reset UI state
            self.generate_button.enable()
            self.generate_button.text = 'Generate'
            self.generate_button.props('icon=auto_awesome')  # Reset icon
            self.generate_button.update()
            self.cancel_button.visible = False
    
    async def _generate_image(self, enhancement: Dict[str, Any]):
        """Generate an image"""
        # Validate model selection
        if not self.model_select.value:
            self._notify_from_background('Please select a model', 'warning')
            return
        
        # Create generation request
        request = ImageGenerationRequest(
            prompt=self.prompt_input.value,
            negative_prompt=self.negative_input.value or "",
            width=int(self.width_input.value),
            height=int(self.height_input.value),
            steps=int(self.steps_slider.value),
            guidance_scale=float(self.guidance_slider.value),
            seed=-1,  # Random seed
            model=self.model_select.value
        )
        
        # Apply enhancements
        request.enhancement_fields = enhancement
        request.use_enhancement = enhancement.get("use_llm", True)
        
        # Process request
        result = await self.image_processor.generate(
            request,
            progress_callback=self._update_progress_safe
        )
        
        if result and result.image_path:
            self._show_image_result(result.image_path)
    
    async def _generate_3d(self, enhancement: Dict[str, Any]):
        """Generate a 3D model"""
        # Get selected 3D model
        if not self.model_select.value:
            self._notify_from_background('Please select a 3D model', 'warning')
            return
        
        # Check if using image input
        input_image = self._uploaded_image_path if hasattr(self, 'img2img_toggle') and self.img2img_toggle.value else None
        
        # If text-to-3D, need an image model
        if not input_image and (not hasattr(self, 'image_model_select') or not self.image_model_select.value):
            self._notify_from_background('Please select an image generation model for text-to-3D', 'warning')
            return
        
        # Create generation request with advanced performance parameters
        request = ThreeDGenerationRequest(
            prompt=self.prompt_input.value,
            negative_prompt=self.negative_input.value or "",
            input_image=str(input_image) if input_image else None,
            num_views=int(self.num_views_slider.value),
            mesh_resolution=int(getattr(self, 'mesh_res_slider', type('obj', (), {'value': 256})).value),
            texture_resolution=int(getattr(self, 'texture_res_slider', type('obj', (), {'value': 1024})).value),
            export_formats=[self.format_select.value.lower()],
            model=self.model_select.value,
            image_model=self.image_model_select.value if not input_image else None,
            # Advanced performance parameters
            mesh_decode_resolution=int(getattr(self, 'mesh_decode_resolution_slider', type('obj', (), {'value': 64})).value),
            mesh_decode_batch_size=getattr(self, 'mesh_decode_batch_size_slider', type('obj', (), {'value': 0})).value or None,
            paint_max_num_view=int(getattr(self, 'paint_max_views_slider', type('obj', (), {'value': 6})).value),
            paint_resolution=int(getattr(self, 'paint_resolution_slider', type('obj', (), {'value': 512})).value),
            render_size=int(getattr(self, 'render_size_slider', type('obj', (), {'value': 1024})).value),
            texture_size=int(getattr(self, 'texture_size_slider', type('obj', (), {'value': 1024})).value)
        )
        
        # Apply enhancements
        request.enhancement_fields = enhancement
        request.use_enhancement = enhancement.get("use_llm", True)
        
        # Select processor based on model
        processor = self._get_3d_processor(self.model_select.value)
        
        # Process request
        result = await processor.generate(
            request,
            progress_callback=self._update_progress_safe
        )
        
        if result and result.model_path:
            # 3D generation completed successfully
            
            self._show_3d_result(
                result.model_path,
                result.preview_images if hasattr(result, 'preview_images') else []
            )
        else:
            # 3D generation failed or returned no result
            pass
    
    def _get_3d_processor(self, model_id: str):
        """Get the appropriate 3D processor for the model"""
        if 'sparc3d' in model_id.lower():
            return self.sparc3d_processor
        elif 'hi3dgen' in model_id.lower():
            return self.hi3dgen_processor
        else:
            return self.threed_processor
    
    def _check_for_results(self):
        """Check for pending results and update UI (runs in UI thread)"""
        # Check for 3D results
        if hasattr(self, '_update_3d_result_ui'):
            self._update_3d_result_ui()
        
        # Stop the timer if generation is complete
        if hasattr(self, 'generation_task') and self.generation_task and self.generation_task.done():
            # Generation complete, stop the timer
            if hasattr(self, '_result_checker_timer') and self._result_checker_timer:
                self._result_checker_timer.active = False
                self._result_checker_timer = None
    
    async def _cancel_generation(self):
        """Cancel the current generation"""
        if self.generation_task and not self.generation_task.done():
            self.generation_task.cancel()
            # Wait for cancellation
            try:
                await self.generation_task
            except asyncio.CancelledError:
                pass
    
    def _update_progress_safe(self, step: str, progress: float, message: str = ""):
        """Thread-safe progress update"""
        # Ensure message is always a string (safety check for tensor arguments)
        if hasattr(message, 'shape'):
            # This is likely a tensor, convert to safe string
            message = f"Processing tensor with shape {message.shape}"
        else:
            message = str(message) if message is not None else ""
        
        # Queue the update for the main thread
        self._pipeline_update_queue.append((step, progress, message))
    
    def _process_pipeline_updates(self):
        """Process queued pipeline updates in the main thread"""
        # Check initialization status
        self._check_initialization_status()
        
        # Process all queued updates
        while self._pipeline_update_queue:
            step, progress, message = self._pipeline_update_queue.pop(0)
            
            # Ensure progress is a float
            try:
                progress_float = float(progress) if progress is not None else 0.0
            except (ValueError, TypeError):
                progress_float = 0.0
            
            # Map step names to pipeline step IDs
            step_id_mapping = {
                'enhance_prompt': 'enhance',
                'load_model': 'model', 
                'generate': 'generate',
                'postprocess': 'postprocess',
                'save': 'save',
                # Handle sub-steps within generate
                'diffusion_sampling': 'generate',
                'volume_decoding': 'postprocess',
                'mesh_generation': 'postprocess',
                'texture_generation': 'postprocess',
            }
            
            pipeline_step_id = step_id_mapping.get(step, None)
            if not pipeline_step_id:
                return  # Unknown step
            
            # Find the step index in our pipeline
            step_index = -1
            for i, pipeline_step in enumerate(self.progress_pipeline.steps):
                if pipeline_step.id == pipeline_step_id:
                    step_index = i
                    break
            
            if step_index == -1:
                return  # Step not found
            
            # Advance to this step if needed
            if step_index > self.progress_pipeline.current_step_index:
                # Need to advance to this step
                while self.progress_pipeline.current_step_index < step_index:
                    self.progress_pipeline.advance_step()
            
            # Update progress within the current step with clean rounding
            if step_index == self.progress_pipeline.current_step_index:
                # Extract sub-step name from message if applicable
                sub_message = ""
                if step in ['diffusion_sampling', 'volume_decoding', 'mesh_generation', 'texture_generation']:
                    sub_message = step.replace('_', ' ').title()
                elif message and any(keyword in message.lower() for keyword in ['diffusion', 'sampling', 'volume', 'decoding', 'mesh', 'texture']):
                    sub_message = message
                
                # Apply clean progress rounding and update
                self.progress_pipeline.update_step_progress(progress_float, sub_message)
            
            # If step is complete, ensure it's marked as finished
            if progress_float >= 1.0 and step_index == self.progress_pipeline.current_step_index:
                self.progress_pipeline.advance_step(message)
    
    def _get_current_time_estimate(self) -> float:
        """Get current time estimate based on selected parameters"""
        try:
            from desktop.ui.utils.time_estimation import estimate_3d_generation_time
            
            # Get current parameter values
            params = {
                'num_inference_steps': getattr(self, 'inference_steps_slider', type('obj', (), {'value': 50})).value,
                'guidance_scale': getattr(self, 'guidance_scale_slider', type('obj', (), {'value': 7.5})).value,
                'mesh_decode_resolution': getattr(self, 'mesh_decode_resolution_slider', type('obj', (), {'value': 64})).value,
                'mesh_decode_batch_size': getattr(self, 'mesh_decode_batch_size_slider', type('obj', (), {'value': 0})).value or None,
                'paint_max_num_view': getattr(self, 'paint_max_views_slider', type('obj', (), {'value': 6})).value,
                'paint_resolution': getattr(self, 'paint_resolution_slider', type('obj', (), {'value': 512})).value,
                'render_size': getattr(self, 'render_size_slider', type('obj', (), {'value': 1024})).value,
                'texture_size': getattr(self, 'texture_size_slider', type('obj', (), {'value': 1024})).value,
                'num_views': getattr(self, 'num_views_slider', type('obj', (), {'value': 6})).value,
                'enable_texture': True,
                'model': getattr(self, 'model_select', type('obj', (), {'value': 'hunyuan3d-21'})).value or 'hunyuan3d-21'
            }
            
            # Calculate time estimate
            estimate = estimate_3d_generation_time(**params)
            return estimate['total_time']
            
        except ImportError:
            # Fallback to simple estimate
            return 60.0  # Default 1 minute estimate