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
        
        # 3D specific sliders with proper labels
        with ui.column().classes('w-full gap-4'):
            # Number of views
            with ui.column().classes('w-full'):
                ui.label('Number of Views').classes('text-sm text-gray-400')
                self.num_views_slider = ui.slider(
                    min=4,
                    max=12,
                    value=8,
                    step=2
                ).props('label-always')
            
            # Mesh resolution
            with ui.column().classes('w-full'):
                ui.label('Mesh Resolution').classes('text-sm text-gray-400')
                self.mesh_res_slider = ui.slider(
                    min=64,
                    max=512,
                    value=256,
                    step=64
                ).props('label-always')
            
            # Texture resolution
            with ui.column().classes('w-full'):
                ui.label('Texture Resolution').classes('text-sm text-gray-400')
                self.texture_res_slider = ui.slider(
                    min=512,
                    max=4096,
                    value=1024,
                    step=512
                ).props('label-always')
    
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