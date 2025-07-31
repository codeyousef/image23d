"""Mode Controls Mixin

Handles rendering of mode-specific controls.
"""

from nicegui import ui


class ModeControlsMixin:
    """Mixin for mode-specific control rendering"""
    
    def _render_mode_controls(self):
        """Render controls specific to current mode with consistent sizing"""
        self.mode_controls_container.clear()
        
        with self.mode_controls_container:
            if self.current_mode == "image":
                self._render_image_controls()
            elif self.current_mode == "3d":
                self._render_3d_controls()
            elif self.current_mode == "video":
                self._render_video_controls()
    
    def _render_image_controls(self):
        """Render image generation controls"""
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
    
    def _render_3d_controls(self):
        """Render 3D generation controls"""
        # Image model selection for text-to-3D
        self.image_model_select = ui.select(
            label='Image Generation Model',
            options={},  # Will be populated by _load_available_models
            value=None
        ).classes('w-full mb-4').props('outlined dense')
        
        # Output format
        self.format_select = ui.select(
            label='Output Format',
            options=['GLB', 'USDZ', 'PLY'],
            value='GLB'
        ).classes('w-full mb-4').props('outlined dense')
        
        # Img2Img toggle
        self.img2img_toggle = ui.switch(
            text='Use Image Input',
            value=False,
            on_change=self._on_img2img_toggle
        ).classes('mb-4')
        
        # Image upload (hidden by default)
        self.image_upload = ui.upload(
            label='Upload Reference Image',
            on_upload=self._handle_image_upload,
            auto_upload=True,
            max_files=1
        ).classes('w-full mb-4').props('accept=".jpg,.jpeg,.png,.webp"')
        self.image_upload.visible = False
        
        # Preview uploaded image
        self.uploaded_image_preview = ui.column().classes('w-full mb-4 items-center')
        self.uploaded_image_preview.visible = False
        
        # Performance preset buttons
        with ui.row().classes('w-full gap-2 mb-4'):
            ui.label('Quick Presets:').classes('text-sm text-gray-400 self-center')
            ui.button('⚡ Speed', on_click=lambda: self._apply_preset('speed')).props('flat dense color=green')
            ui.button('⚖️ Balanced', on_click=lambda: self._apply_preset('balanced')).props('flat dense color=blue')
            ui.button('⭐ Quality', on_click=lambda: self._apply_preset('quality')).props('flat dense color=orange')
            ui.button('🔥 Ultra', on_click=lambda: self._apply_preset('ultra')).props('flat dense color=red')
        
        # Time estimation display
        self.time_estimate_card = ui.card().classes('w-full mb-4 p-4 bg-blue-50 border-l-4 border-blue-400')
        with self.time_estimate_card:
            with ui.row().classes('items-center gap-2'):
                ui.icon('schedule', color='blue')
                self.time_estimate_label = ui.label('Estimated Time: Calculating...').classes('text-lg font-semibold text-blue-800')
            self.time_breakdown_label = ui.label('').classes('text-sm text-blue-600')
            self.performance_tips = ui.column().classes('mt-2')
        
        # Basic 3D generation parameters
        with ui.column().classes('w-full gap-4'):
            # Inference steps
            with ui.column().classes('w-full'):
                ui.label('Inference Steps (Quality vs Speed)').classes('text-sm text-gray-400')
                self.inference_steps_slider = ui.slider(
                    min=20,
                    max=100,
                    value=50,
                    step=5,
                    on_change=self._update_time_estimate
                ).props('label-always')
            
            # Guidance scale
            with ui.column().classes('w-full'):
                ui.label('Guidance Scale (Prompt Adherence)').classes('text-sm text-gray-400')
                self.guidance_scale_slider = ui.slider(
                    min=1.0,
                    max=15.0,
                    value=7.5,
                    step=0.5,
                    on_change=self._update_time_estimate
                ).props('label-always')
            
            # Number of views
            with ui.column().classes('w-full'):
                ui.label('Number of Views').classes('text-sm text-gray-400')
                self.num_views_slider = ui.slider(
                    min=4,
                    max=12,
                    value=6,
                    step=2,
                    on_change=self._update_time_estimate
                ).props('label-always')
        
        # Advanced Performance Parameters (Collapsible)
        with ui.expansion('Advanced Performance Settings', icon='tune').classes('w-full mb-4'):
            with ui.column().classes('w-full gap-4 p-4'):
                # Mesh decode resolution
                with ui.column().classes('w-full'):
                    ui.label('Mesh Decode Resolution (32-128) - Higher = Better Quality, Slower').classes('text-sm text-gray-400')
                    self.mesh_decode_resolution_slider = ui.slider(
                        min=32,
                        max=128,
                        value=64,
                        step=16,
                        on_change=self._update_time_estimate
                    ).props('label-always')
                
                # Mesh decode batch size
                with ui.column().classes('w-full'):
                    ui.label('Mesh Decode Batch Size (Auto if 0) - Higher = Faster, More VRAM').classes('text-sm text-gray-400')
                    self.mesh_decode_batch_size_slider = ui.slider(
                        min=0,
                        max=65536,
                        value=0,
                        step=8192,
                        on_change=self._update_time_estimate
                    ).props('label-always')
                
                # Paint pipeline parameters
                with ui.column().classes('w-full'):
                    ui.label('Paint Max Views (4-9) - More Views = Better Texture, Slower').classes('text-sm text-gray-400')
                    self.paint_max_views_slider = ui.slider(
                        min=4,
                        max=9,
                        value=6,
                        step=1,
                        on_change=self._update_time_estimate
                    ).props('label-always')
                
                with ui.column().classes('w-full'):
                    ui.label('Paint Resolution (256-768) - Texture Detail').classes('text-sm text-gray-400')
                    self.paint_resolution_slider = ui.slider(
                        min=256,
                        max=768,
                        value=512,
                        step=64,
                        on_change=self._update_time_estimate
                    ).props('label-always')
                
                with ui.column().classes('w-full'):
                    ui.label('Render Size (512-2048) - Rendering Resolution').classes('text-sm text-gray-400')
                    self.render_size_slider = ui.slider(
                        min=512,
                        max=2048,
                        value=1024,
                        step=256,
                        on_change=self._update_time_estimate
                    ).props('label-always')
                
                with ui.column().classes('w-full'):
                    ui.label('Final Texture Size (512-4096) - Output Texture Resolution').classes('text-sm text-gray-400')
                    self.texture_size_slider = ui.slider(
                        min=512,
                        max=4096,
                        value=1024,
                        step=512,
                        on_change=self._update_time_estimate
                    ).props('label-always')
        
        # Legacy sliders for backward compatibility (hidden)
        self.mesh_res_slider = self.mesh_decode_resolution_slider
        self.texture_res_slider = self.texture_size_slider
        
        # Initialize time estimate
        self._update_time_estimate()
    
    def _render_video_controls(self):
        """Render video generation controls"""
        # Placeholder for video controls
        ui.label('Video generation coming soon...').classes('text-gray-500')
    
    def _set_aspect(self, w: int, h: int):
        """Set aspect ratio for image generation"""
        # Calculate new dimensions maintaining aspect ratio
        current_pixels = self.width_input.value * self.height_input.value
        ratio = w / h
        
        # Calculate new dimensions
        new_height = int((current_pixels / ratio) ** 0.5)
        new_width = int(new_height * ratio)
        
        # Round to nearest 64
        new_width = (new_width // 64) * 64
        new_height = (new_height // 64) * 64
        
        # Clamp to valid range
        new_width = max(256, min(2048, new_width))
        new_height = max(256, min(2048, new_height))
        
        self.width_input.value = new_width
        self.height_input.value = new_height
    
    def _on_img2img_toggle(self, e):
        """Handle img2img toggle change"""
        self.image_upload.visible = e.value
        if not e.value:
            # Clear uploaded image
            self._uploaded_image_path = None
            self.uploaded_image_preview.visible = False
            self.uploaded_image_preview.clear()
    
    def _on_mode_change(self, tab_value):
        """Handle mode change"""
        # Store the current mode directly
        self.current_mode = tab_value
        
        # Clear any uploaded images when switching modes
        self._uploaded_image_path = None
        if hasattr(self, 'uploaded_image_preview') and self.uploaded_image_preview:
            self.uploaded_image_preview.visible = False
            self.uploaded_image_preview.clear()
        
        # Re-render mode controls
        self._render_mode_controls()
        
        # Update dependency status
        if hasattr(self, '_update_dependency_status'):
            self._update_dependency_status()
        
        # Clear current model selection to force refresh
        if hasattr(self, 'model_select') and self.model_select:
            self.model_select.value = None
            self.model_select.options = {}
        
        # Load models for the new mode
        self._load_available_models(show_notifications=False)
        
        # Clear preview
        self.preview_container.clear()
        with self.preview_container:
            if self.current_mode == "image":
                ui.icon('image', size='4rem').classes('text-gray-600 mb-4')
                ui.label('Your generated image will appear here').classes('text-gray-500')
            elif self.current_mode == "3d":
                ui.icon('view_in_ar', size='4rem').classes('text-gray-600 mb-4')
                ui.label('Your generated 3D model will appear here').classes('text-gray-500')
            elif self.current_mode == "video":
                ui.icon('movie', size='4rem').classes('text-gray-600 mb-4')
                ui.label('Your generated video will appear here').classes('text-gray-500')
        
        # Hide export buttons
        self.export_button.visible = False
        self.save_button.visible = False
    
    def _apply_preset(self, preset_name: str):
        """Apply performance preset to 3D controls"""
        if not hasattr(self, 'inference_steps_slider'):
            return
            
        # Import presets from time estimation utility
        try:
            from desktop.ui.utils.time_estimation import PERFORMANCE_PRESETS
            
            if preset_name not in PERFORMANCE_PRESETS:
                return
                
            preset = PERFORMANCE_PRESETS[preset_name]
            params = preset['params']
            
            # Apply basic parameters
            if hasattr(self, 'inference_steps_slider'):
                self.inference_steps_slider.value = params.get('steps', 50)
            if hasattr(self, 'guidance_scale_slider'):
                self.guidance_scale_slider.value = params.get('guidance_scale', 7.5)
            
            # Apply advanced parameters
            if hasattr(self, 'mesh_decode_resolution_slider'):
                self.mesh_decode_resolution_slider.value = params.get('mesh_decode_resolution', 64)
            if hasattr(self, 'paint_resolution_slider'):
                self.paint_resolution_slider.value = params.get('paint_resolution', 512)
            if hasattr(self, 'render_size_slider'):
                self.render_size_slider.value = params.get('render_size', 1024)
            if hasattr(self, 'texture_size_slider'):
                self.texture_size_slider.value = params.get('texture_size', 1024)
            
            # Update time estimate
            self._update_time_estimate()
            
            # Show notification
            ui.notify(f"Applied {preset['name']} preset - {preset['description']}", type='positive')
            
        except ImportError:
            ui.notify("Could not load performance presets", type='warning')
    
    def _update_time_estimate(self, *args):
        """Update real-time time estimation based on current parameters"""
        if not hasattr(self, 'time_estimate_label'):
            return
            
        try:
            from desktop.ui.utils.time_estimation import estimate_3d_generation_time, format_time
            
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
                'model': 'hunyuan3d-21'
            }
            
            # Calculate time estimate
            estimate = estimate_3d_generation_time(**params)
            
            # Update display
            formatted_time = format_time(estimate['total_time'])
            self.time_estimate_label.text = f"Estimated Time: {formatted_time}"
            
            # Update breakdown
            shape_time = format_time(estimate['breakdown']['shape_generation'])
            texture_time = format_time(estimate['breakdown']['texture_generation'])
            memory_gb = estimate['details']['estimated_gpu_memory_gb']
            self.time_breakdown_label.text = f"Shape: {shape_time} | Texture: {texture_time} | VRAM: {memory_gb}GB | {estimate['quality_level']}"
            
            # Update performance tips
            self.performance_tips.clear()
            if estimate['performance_tips']:
                with self.performance_tips:
                    for tip in estimate['performance_tips'][:3]:  # Show max 3 tips
                        ui.label(tip).classes('text-xs text-gray-600 mb-1')
                        
        except ImportError:
            self.time_estimate_label.text = "Time estimation unavailable"
        except Exception as e:
            self.time_estimate_label.text = f"Estimation error: {str(e)[:50]}"