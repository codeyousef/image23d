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


class GenerationHandlersMixin:
    """Mixin for generation handling methods"""
    
    async def _start_generation(self):
        """Start the generation process"""
        # Check if model manager is initialized
        if not self._check_initialization_status():
            return
        
        # Validate inputs
        if not self.prompt_input.value.strip():
            self._notify_from_background('Please enter a prompt', 'warning')
            return
        
        # Cancel any existing generation
        if self.generation_task and not self.generation_task.done():
            await self._cancel_generation()
        
        # Update UI state
        self.generate_button.visible = False
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
            # Set up 3D generation pipeline steps
            from ...components.progress_pipeline_card import PipelineStep
            steps = [
                PipelineStep('enhance', 'Enhance Prompt', 'text_increase', 5),
                PipelineStep('model', 'Load Model', 'download', 10),
                PipelineStep('generate', 'Generate Views', 'photo_camera', 30),
                PipelineStep('postprocess', 'Process 3D', 'view_in_ar', 25),
                PipelineStep('save', 'Save & Export', 'save', 10)
            ]
            self.progress_pipeline.set_steps(steps)
            self.progress_pipeline.start()
        elif self.current_mode == "image":
            # Set up image generation pipeline steps
            from ...components.progress_pipeline_card import PipelineStep
            steps = [
                PipelineStep('enhance', 'Enhance Prompt', 'text_increase', 5),
                PipelineStep('model', 'Load Model', 'download', 5),
                PipelineStep('generate', 'Generate Image', 'image', 20),
                PipelineStep('save', 'Save Result', 'save', 5)
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
            self.generate_button.visible = True
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
        
        # Create generation request
        request = ThreeDGenerationRequest(
            prompt=self.prompt_input.value,
            negative_prompt=self.negative_input.value or "",
            input_image=str(input_image) if input_image else None,
            num_views=int(self.num_views_slider.value),
            mesh_resolution=int(self.mesh_res_slider.value),
            texture_resolution=int(self.texture_res_slider.value),
            export_formats=[self.format_select.value.lower()],
            model=self.model_select.value,
            image_model=self.image_model_select.value if not input_image else None
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
            self._show_3d_result(
                result.model_path,
                result.preview_images if hasattr(result, 'preview_images') else []
            )
    
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
            
            # Map step names to pipeline steps
            step_mapping = {
                'enhance_prompt': PipelineStep('enhance', 'Enhancing Prompt', message, 'running' if progress_float < 1.0 else 'completed'),
                'load_model': PipelineStep('model', 'Loading Model', message, 'running' if progress_float < 1.0 else 'completed'),
                'generate': PipelineStep('generate', 'Generating', message, 'running' if progress_float < 1.0 else 'completed'),
                'postprocess': PipelineStep('postprocess', 'Post-processing', message, 'running' if progress_float < 1.0 else 'completed'),
                'save': PipelineStep('save', 'Saving', message, 'running' if progress_float < 1.0 else 'completed'),
            }
            
            # For now, just advance the pipeline step instead of adding new ones
            # The current ProgressPipeline implementation doesn't support add_step
            if progress_float >= 1.0:
                # Step completed
                self.progress_pipeline.advance_step(message)
            # else: step still running, don't advance yet