"""Output Handlers Mixin

Handles output display, export, and saving operations.
"""

import shutil
from pathlib import Path
from nicegui import ui


class OutputHandlersMixin:
    """Mixin for output handling operations"""
    
    def _handle_image_upload(self, e):
        """Handle image upload for img2img"""
        if e.content:
            # Save uploaded file
            upload_dir = self.output_dir / "uploads"
            upload_dir.mkdir(exist_ok=True)
            
            # Generate unique filename
            import time
            filename = f"upload_{int(time.time())}.png"
            file_path = upload_dir / filename
            
            # Save file
            with open(file_path, 'wb') as f:
                f.write(e.content.read())
            
            self._uploaded_image_path = file_path
            
            # Show preview
            self.uploaded_image_preview.clear()
            self.uploaded_image_preview.visible = True
            with self.uploaded_image_preview:
                ui.image(str(file_path)).classes('max-w-full h-48 object-contain')
                ui.label(f'Uploaded: {filename}').classes('text-sm text-gray-500')
    
    def _show_image_result(self, image_path: Path):
        """Display generated image result"""
        self.preview_container.clear()
        with self.preview_container:
            # Display the image
            ui.image(str(image_path)).classes('max-w-full max-h-96 object-contain')
            
            # Image info
            ui.label(f'Generated image: {image_path.name}').classes('text-sm text-gray-500 mt-2')
        
        # Store output reference
        self._current_output = image_path
        self._current_output_type = 'image'
        
        # Show export buttons
        self.export_button.visible = True
        self.save_button.visible = True
        
        # Success notification
        ui.notify('Image generated successfully!', type='positive', position='top')
    
    def _show_3d_result(self, model_path: Path, preview_images: list):
        """Store 3D generation result for UI update"""
        # Store 3D generation result for UI update
        # Validate preview images
        if preview_images and not isinstance(preview_images, list):
            preview_images = [preview_images]
        
        # Store the result data for UI update
        self._pending_3d_result = {
            'model_path': model_path,
            'preview_images': preview_images,
            'completed': True
        }
        
        # Store output reference
        self._current_output = model_path
        self._current_output_type = '3d'
        
        # Result stored for processing in UI thread
    
    def _update_3d_result_ui(self):
        """Update UI with 3D generation result (called from UI thread)"""
        # Update UI with pending 3D result if available
        
        if not hasattr(self, '_pending_3d_result') or not self._pending_3d_result:
            # No pending result to process
            return
            
        result = self._pending_3d_result
        # Process the pending result
        
        if not result.get('completed'):
            # Result not ready yet
            return
            
        # Clear the pending result
        self._pending_3d_result = None
        # Clear pending result and update UI
        
        # Update UI
        model_path = result['model_path']
        preview_images = result['preview_images']
        
        # Update UI with model and preview images
        
        self.preview_container.clear()
        with self.preview_container:
            if preview_images and len(preview_images) > 0:
                # Show preview images in a grid
                # Show preview images in a grid
                with ui.grid(columns=2).classes('gap-2 max-w-2xl'):
                    for i, img_path in enumerate(preview_images[:4]):  # Show up to 4 previews
                        if isinstance(img_path, (str, Path)) and Path(img_path).exists():
                            ui.image(str(img_path)).classes('w-full h-48 object-contain')
                        else:
                            ui.label(f'Preview {i+1} (missing)').classes('text-red-500')
            else:
                # No preview images, show 3D icon instead
                # Show 3D icon if no previews
                ui.icon('view_in_ar', size='4rem').classes('text-green-500 mb-4')
                ui.label('3D Model Generated').classes('text-lg font-semibold text-center')
                
            # Model info
            ui.label(f'Generated 3D model: {model_path.name}').classes('text-sm text-gray-500 mt-2')
            ui.label(f'Format: {model_path.suffix.upper()}').classes('text-sm text-gray-400')
            
            # Add model file size if available
            try:
                if model_path.exists():
                    size_mb = model_path.stat().st_size / (1024 * 1024)
                    ui.label(f'Size: {size_mb:.1f} MB').classes('text-sm text-gray-400')
                    # File size calculated successfully
                else:
                    ui.label('File not found').classes('text-sm text-red-500')
                    # Model file not found
            except Exception as e:
                # Error reading file info
                ui.label('Error reading file').classes('text-sm text-red-500')
        
        # Show export buttons
        self.export_button.visible = True
        self.save_button.visible = True
        
        # Success notification
        ui.notify('3D model generated successfully!', type='positive', position='top')
    
    def _export_output(self):
        """Export the current output"""
        if not self._current_output or not self._current_output.exists():
            ui.notify('No output to export', type='warning')
            return
        
        # Create export dialog
        with ui.dialog() as dialog, ui.card():
            ui.label('Export Options').classes('text-lg font-semibold mb-4')
            
            # Export location
            export_path = ui.input(
                'Export Path',
                value=str(Path.home() / 'Downloads' / self._current_output.name)
            ).classes('w-96')
            
            # Export format options for 3D
            format_select = None
            if self._current_output_type == '3d':
                format_select = ui.select(
                    'Format',
                    options=['GLB', 'USDZ', 'PLY', 'OBJ', 'STL'],
                    value=self._current_output.suffix[1:].upper()
                ).classes('w-full mt-4')
            
            # Action buttons
            with ui.row().classes('w-full justify-end gap-2 mt-4'):
                ui.button('Cancel', on_click=dialog.close).props('flat')
                
                async def do_export():
                    try:
                        export_file = Path(export_path.value)
                        
                        # Convert format if needed
                        if format_select and format_select.value.lower() != self._current_output.suffix[1:]:
                            # TODO: Implement format conversion
                            ui.notify('Format conversion not yet implemented', type='info')
                            export_file = export_file.with_suffix(f'.{self._current_output.suffix[1:]}')
                        
                        # Copy file
                        export_file.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(self._current_output, export_file)
                        
                        ui.notify(f'Exported to {export_file}', type='positive')
                        dialog.close()
                        
                    except Exception as e:
                        ui.notify(f'Export failed: {str(e)}', type='negative')
                
                ui.button('Export', on_click=do_export).props('unelevated')
        
        dialog.open()
    
    def _save_to_library(self):
        """Save output to user library"""
        if not self._current_output or not self._current_output.exists():
            ui.notify('No output to save', type='warning')
            return
        
        # Create save dialog
        with ui.dialog() as dialog, ui.card():
            ui.label('Save to Library').classes('text-lg font-semibold mb-4')
            
            # Name input
            name_input = ui.input(
                'Name',
                value=self._current_output.stem
            ).classes('w-96')
            
            # Tags input
            tags_input = ui.input(
                'Tags (comma separated)',
                value=''
            ).classes('w-96 mt-4')
            
            # Description
            desc_input = ui.textarea(
                'Description',
                value=''
            ).classes('w-96 mt-4').props('rows=3')
            
            # Action buttons
            with ui.row().classes('w-full justify-end gap-2 mt-4'):
                ui.button('Cancel', on_click=dialog.close).props('flat')
                
                async def do_save():
                    try:
                        # Create library entry
                        library_dir = self.output_dir / 'library' / self._current_output_type
                        library_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Generate unique ID
                        import time
                        import json
                        entry_id = f"{int(time.time())}_{name_input.value.replace(' ', '_')}"
                        entry_dir = library_dir / entry_id
                        entry_dir.mkdir(exist_ok=True)
                        
                        # Copy output file
                        dest_file = entry_dir / self._current_output.name
                        shutil.copy2(self._current_output, dest_file)
                        
                        # Save metadata
                        metadata = {
                            'id': entry_id,
                            'name': name_input.value,
                            'tags': [tag.strip() for tag in tags_input.value.split(',') if tag.strip()],
                            'description': desc_input.value,
                            'type': self._current_output_type,
                            'file': self._current_output.name,
                            'created': time.time(),
                            'prompt': self.prompt_input.value,
                            'negative_prompt': self.negative_input.value,
                            'model': self.model_select.value
                        }
                        
                        with open(entry_dir / 'metadata.json', 'w') as f:
                            json.dump(metadata, f, indent=2)
                        
                        ui.notify('Saved to library!', type='positive')
                        dialog.close()
                        
                    except Exception as e:
                        ui.notify(f'Save failed: {str(e)}', type='negative')
                
                ui.button('Save', on_click=do_save).props('unelevated')
        
        dialog.open()