"""Model Management Mixin

Handles model loading and management operations.
"""

from typing import Dict, Any
from pathlib import Path
from nicegui import ui


class ModelManagementMixin:
    """Mixin for model management operations"""
    
    def _check_model_downloaded(self, model_id: str, model_type: str) -> bool:
        """Check if a model is downloaded"""
        model_path = self.models_dir / model_type / model_id
        return model_path.exists() and any(model_path.iterdir())
    
    def _load_available_models(self, show_notifications=True):
        """Load available models based on current mode"""
        try:
            if self.current_mode == "image":
                # Get available image models from model manager
                models = self.model_manager.get_available_models('image')
                
                # Format for select widget - show download status
                options = {}
                for model_id, info in models.items():
                    is_downloaded = self._check_model_downloaded(model_id, 'image')
                    label = f"{info['name']} {'✓' if is_downloaded else '⬇'}"
                    if not is_downloaded:
                        label += " (not downloaded)"
                    options[model_id] = label
                
                # Update model select
                if self.model_select:
                    self.model_select.options = options
                    # Select first downloaded model if available
                    downloaded = [k for k, v in options.items() if '✓' in v]
                    if downloaded and not self.model_select.value:
                        self.model_select.value = downloaded[0]
                        
            elif self.current_mode == "3d":
                # Get available 3D models
                models_3d = self.model_manager.get_available_models('threed')
                models_sparc3d = self.model_manager.get_available_models('sparc3d')
                models_hi3dgen = self.model_manager.get_available_models('hi3dgen')
                
                # Combine all 3D models
                all_3d_models = {}
                all_3d_models.update(models_3d)
                all_3d_models.update(models_sparc3d)
                all_3d_models.update(models_hi3dgen)
                
                # Format for select widget
                options = {}
                for model_id, info in all_3d_models.items():
                    # Determine model type folder
                    if model_id in models_sparc3d:
                        model_folder = 'sparc3d'
                    elif model_id in models_hi3dgen:
                        model_folder = 'hi3dgen'
                    else:
                        model_folder = '3d'
                    
                    is_downloaded = self._check_model_downloaded(model_id, model_folder)
                    label = f"{info['name']} {'✓' if is_downloaded else '⬇'}"
                    if not is_downloaded:
                        label += " (not downloaded)"
                    options[model_id] = label
                
                # Update 3D model select
                if self.model_select:
                    self.model_select.options = options
                    # Select first downloaded model if available
                    downloaded = [k for k, v in options.items() if '✓' in v]
                    if downloaded and not self.model_select.value:
                        self.model_select.value = downloaded[0]
                
                # Also update image model select for text-to-3D
                if hasattr(self, 'image_model_select') and self.image_model_select:
                    image_models = self.model_manager.get_available_models('image')
                    image_options = {}
                    for model_id, info in image_models.items():
                        is_downloaded = self._check_model_downloaded(model_id, 'image')
                        if is_downloaded:  # Only show downloaded image models
                            image_options[model_id] = info['name']
                    
                    self.image_model_select.options = image_options
                    if image_options and not self.image_model_select.value:
                        self.image_model_select.value = list(image_options.keys())[0]
                        
            if show_notifications:
                ui.notify('Models refreshed', type='positive', position='top')
                
        except Exception as e:
            print(f"Error loading models: {e}")
            if show_notifications:
                ui.notify(f'Error loading models: {str(e)}', type='negative', position='top')
    
    def _on_model_select_change(self, e):
        """Handle model selection change"""
        if e.value and ' (not downloaded)' in self.model_select.options.get(e.value, ''):
            # Model not downloaded, show warning
            ui.notify(
                'This model needs to be downloaded first. Go to Models page to download.',
                type='warning',
                position='top',
                timeout=5000
            )
            
            # Disable generate button for non-downloaded models
            if self.generate_button:
                self.generate_button.props('disable')
        else:
            # Enable generate button for downloaded models
            if self.generate_button:
                self.generate_button.props(remove='disable')