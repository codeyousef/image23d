"""
Models page - Download and manage AI models
"""

import os
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from nicegui import ui
import shutil

logger = logging.getLogger(__name__)

class ModelsPage:
    """Models page for downloading and managing AI models"""
    
    def __init__(self, models_dir: Path, model_manager=None):
        self.models_dir = models_dir
        self.model_manager = model_manager
        self.downloading_models: Dict[str, float] = {}  # model_id -> progress
        self.download_tasks: Dict[str, asyncio.Task] = {}
        
    def render(self):
        """Render the models page"""
        with ui.column().classes('w-full h-full'):
            # Header
            with ui.card().classes('w-full mb-4').style('background-color: #1F1F1F; border: 1px solid #333333'):
                with ui.row().classes('items-center justify-between'):
                    ui.label('Model Manager').classes('text-2xl font-bold')
                    
                    # Storage stats
                    stats = self._get_storage_stats()
                    with ui.row().classes('gap-4'):
                        ui.label(f"Total Models: {stats['count']}").classes('text-sm text-gray-400')
                        ui.label(f"Storage Used: {stats['size']:.1f} GB").classes('text-sm text-gray-400')
                        ui.label(f"Available: {stats['available']:.1f} GB").classes('text-sm text-gray-400')
                        
            # Model categories
            with ui.tabs().classes('w-full') as tabs:
                self.image_tab = ui.tab('Image Models', icon='image')
                self.threed_tab = ui.tab('3D Models', icon='view_in_ar')
                self.pipeline_tab = ui.tab('Pipeline Components', icon='schema')
                self.video_tab = ui.tab('Video Models', icon='movie')
                self.gguf_tab = ui.tab('GGUF Models', icon='memory')
                
            with ui.tab_panels(tabs, value=self.image_tab).classes('w-full flex-grow'):
                # Image Models Tab
                with ui.tab_panel(self.image_tab):
                    self._render_image_models()
                    
                # 3D Models Tab
                with ui.tab_panel(self.threed_tab):
                    self._render_3d_models()
                    
                # Pipeline Components Tab
                with ui.tab_panel(self.pipeline_tab):
                    self._render_pipeline_components()
                    
                # Video Models Tab
                with ui.tab_panel(self.video_tab):
                    self._render_video_models()
                    
                # GGUF Models Tab
                with ui.tab_panel(self.gguf_tab):
                    self._render_gguf_models()
                    
    def _get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        total_size = 0
        model_count = 0
        
        if self.models_dir.exists():
            for path in self.models_dir.rglob('*'):
                if path.is_file():
                    total_size += path.stat().st_size
                    if path.suffix in ['.bin', '.safetensors', '.gguf', '.pt', '.pth']:
                        model_count += 1
                        
        # Get available disk space
        import psutil
        disk_usage = psutil.disk_usage(str(self.models_dir))
        
        return {
            'count': model_count,
            'size': total_size / (1024**3),  # GB
            'available': disk_usage.free / (1024**3)  # GB
        }
        
    def _render_image_models(self):
        """Render image models section"""
        from core.config import FLUX_MODELS
        
        with ui.column().classes('w-full gap-4'):
            # FLUX models
            ui.label('FLUX Image Models').classes('text-lg font-semibold')
            ui.label('State-of-the-art text-to-image generation with enhanced features').classes('text-sm text-gray-400 mb-2')
            
            for model_id, config in FLUX_MODELS.items():
                self._render_model_card(
                    model_id=model_id,
                    name=config['name'],
                    description=config['description'],
                    size=config['size'],
                    vram_required=config['vram_required'],
                    repo_id=config['repo_id'],
                    model_type='image'
                )
                
            # GGUF quantization note
            ui.separator().classes('my-6')
            ui.label('GGUF Quantization').classes('text-lg font-semibold')
            ui.label('GGUF quantization is automatically handled by the enhanced FLUX processors for memory efficiency.').classes('text-sm text-gray-400 mb-2')
                    
    def _render_3d_models(self):
        """Render 3D models section"""
        from core.config import HUNYUAN3D_MODELS, SPARC3D_MODELS, HI3DGEN_MODELS
        
        with ui.column().classes('w-full gap-4'):
            # HunYuan3D Models
            ui.label('HunYuan3D Models').classes('text-lg font-semibold')
            ui.label('State-of-the-art text/image to 3D generation').classes('text-sm text-gray-400 mb-2')
            
            for model_id, config in HUNYUAN3D_MODELS.items():
                self._render_model_card(
                    model_id=model_id,
                    name=config["name"],
                    description=config["description"],
                    size=config["size"],
                    vram_required=config["vram_required"],
                    repo_id=config["repo_id"],
                    model_type='3d'
                )
            
            # Sparc3D Models
            if SPARC3D_MODELS:
                ui.separator().classes('my-6')
                ui.label('Sparc3D Models').classes('text-lg font-semibold')
                ui.label('Ultra high-resolution 3D reconstruction with sparse representation').classes('text-sm text-gray-400 mb-2')
                
                for model_id, config in SPARC3D_MODELS.items():
                    self._render_model_card(
                        model_id=model_id,
                        name=config["name"],
                        description=config["description"],
                        size=config["size"],
                        vram_required=config["vram_required"],
                        repo_id=config["repo_id"],
                        model_type='3d',
                        is_experimental=True
                    )
            
            # Hi3DGen Models
            if HI3DGEN_MODELS:
                ui.separator().classes('my-6')
                ui.label('Hi3DGen Models').classes('text-lg font-semibold')
                ui.label('High-fidelity 3D geometry via normal map bridging').classes('text-sm text-gray-400 mb-2')
                
                for model_id, config in HI3DGEN_MODELS.items():
                    self._render_model_card(
                        model_id=model_id,
                        name=config["name"],
                        description=config["description"],
                        size=config["size"],
                        vram_required=config["vram_required"],
                        repo_id=config["repo_id"],
                        model_type='3d',
                        is_experimental=True
                    )
                
    def _render_pipeline_components(self):
        """Render pipeline components section"""
        with ui.column().classes('w-full gap-4'):
            # Introduction
            with ui.card().classes('w-full').style('background-color: #0A0A0A; border: 1px solid #333333'):
                ui.label('Pipeline Components').classes('text-lg font-semibold mb-2')
                ui.label(
                    'Specialized models used by the enhanced processing pipelines for depth estimation, '
                    'normal mapping, control, and other intermediate processing steps.'
                ).classes('text-sm text-gray-400')
            
            # ControlNet Models
            ui.label('ControlNet Models').classes('text-lg font-semibold')
            ui.label('Precision control for FLUX image generation').classes('text-sm text-gray-400 mb-2')
            
            controlnet_models = [
                {
                    'id': 'controlnet-canny',
                    'name': 'ControlNet Canny Edge',
                    'description': 'Edge-guided image generation using Canny edge detection',
                    'size': '1.2 GB',
                    'vram': '4GB+',
                    'repo': 'InstantX/FLUX.1-dev-Controlnet-Canny'
                },
                {
                    'id': 'controlnet-depth',
                    'name': 'ControlNet Depth',
                    'description': 'Depth-guided image generation for spatial control',
                    'size': '1.2 GB',
                    'vram': '4GB+',
                    'repo': 'Shakker-Labs/FLUX.1-dev-ControlNet-Depth'
                },
                {
                    'id': 'controlnet-pose',
                    'name': 'ControlNet Pose',
                    'description': 'Human pose-guided image generation',
                    'size': '1.2 GB',
                    'vram': '4GB+',
                    'repo': 'InstantX/FLUX.1-dev-Controlnet-Pose'
                }
            ]
            
            for model in controlnet_models:
                self._render_model_card(
                    model_id=model['id'],
                    name=model['name'],
                    description=model['description'],
                    size=model['size'],
                    vram_required=model['vram'],
                    repo_id=model['repo'],
                    model_type='controlnet'
                )
                
            # Depth & Normal Estimation
            ui.separator().classes('my-6')
            ui.label('Depth & Normal Estimation').classes('text-lg font-semibold')
            ui.label('Models for 3D geometry understanding and processing').classes('text-sm text-gray-400 mb-2')
            
            depth_models = [
                {
                    'id': 'depth-anything-v2',
                    'name': 'Depth Anything V2',
                    'description': 'State-of-the-art monocular depth estimation',
                    'size': '400 MB',
                    'vram': '2GB+',
                    'repo': 'depth-anything/Depth-Anything-V2-Small'
                },
                {
                    'id': 'midas-depth',
                    'name': 'MiDaS Depth Estimation',
                    'description': 'Robust depth estimation for multi-view consistency',
                    'size': '350 MB', 
                    'vram': '2GB+',
                    'repo': 'intel-isl/MiDaS'
                },
                {
                    'id': 'normal-estimator',
                    'name': 'Surface Normal Estimator',
                    'description': 'High-quality surface normal estimation for Hi3DGen',
                    'size': '200 MB',
                    'vram': '1GB+',
                    'repo': 'baegwangbin/surface-normal-uncertainty'
                }
            ]
            
            for model in depth_models:
                self._render_model_card(
                    model_id=model['id'],
                    name=model['name'],
                    description=model['description'],
                    size=model['size'],
                    vram_required=model['vram'],
                    repo_id=model['repo'],
                    model_type='depth'
                )
                
            # Background Removal & Preprocessing
            ui.separator().classes('my-6')
            ui.label('Preprocessing Models').classes('text-lg font-semibold')
            ui.label('Background removal and image preprocessing components').classes('text-sm text-gray-400 mb-2')
            
            preprocessing_models = [
                {
                    'id': 'rembg-u2net',
                    'name': 'RemBG U2Net',
                    'description': 'High-quality background removal for 3D processing',
                    'size': '170 MB',
                    'vram': '1GB+',
                    'repo': 'danielgatis/rembg'
                },
                {
                    'id': 'sam-segment',
                    'name': 'Segment Anything Model',
                    'description': 'Precision object segmentation and masking',
                    'size': '2.4 GB',
                    'vram': '4GB+',
                    'repo': 'facebook/sam-vit-large'
                }
            ]
            
            for model in preprocessing_models:
                self._render_model_card(
                    model_id=model['id'],
                    name=model['name'],
                    description=model['description'],
                    size=model['size'],
                    vram_required=model['vram'],
                    repo_id=model['repo'],
                    model_type='preprocessing'
                )
                
            # Texture Components
            ui.separator().classes('my-6')
            ui.label('Texture Components').classes('text-lg font-semibold')
            ui.label('Models for high-quality texture generation and enhancement').classes('text-sm text-gray-400 mb-2')
            
            texture_models = [
                {
                    'id': 'dinov2-giant',
                    'name': 'DINOv2 Giant',
                    'description': 'Vision transformer for texture feature extraction',
                    'size': '4.4 GB',
                    'vram': '8GB+',
                    'repo': 'facebook/dinov2-giant'
                },
                {
                    'id': 'realesrgan-x4',
                    'name': 'Real-ESRGAN x4',
                    'description': 'AI-powered texture upscaling to 4x resolution',
                    'size': '64 MB',
                    'vram': '2GB+',
                    'repo': 'ai-forever/Real-ESRGAN'
                },
                {
                    'id': 'xatlas',
                    'name': 'xatlas UV Unwrapper',
                    'description': 'Advanced UV unwrapping for texture mapping',
                    'size': '10 MB',
                    'vram': '1GB+',
                    'repo': 'local/xatlas'
                }
            ]
            
            for model in texture_models:
                self._render_model_card(
                    model_id=model['id'],
                    name=model['name'],
                    description=model['description'],
                    size=model['size'],
                    vram_required=model['vram'],
                    repo_id=model['repo'],
                    model_type='texture'
                )
                
    def _render_video_models(self):
        """Render video models section"""
        from core.config import VIDEO_MODELS
        
        with ui.column().classes('w-full gap-4'):
            ui.label('Text-to-Video Models').classes('text-lg font-semibold')
            ui.label('State-of-the-art video generation models').classes('text-sm text-gray-400 mb-2')
            
            for model_id, config in VIDEO_MODELS.items():
                self._render_model_card(
                    model_id=model_id,
                    name=config['name'],
                    description=config['description'],
                    size=config['size'],
                    vram_required=config['vram_required'],
                    repo_id=config['repo_id'],
                    model_type='video'
                )
            
    def _render_gguf_models(self):
        """Render GGUF models section"""
        with ui.column().classes('w-full gap-4'):
            ui.label('GGUF Quantized Models').classes('text-lg font-semibold')
            
            # Explanation card
            with ui.card().classes('w-full mb-4').style('background-color: #0A0A0A; border: 1px solid #333333'):
                ui.label('What are GGUF models?').classes('font-semibold mb-2')
                ui.label(
                    'GGUF (GPT-Generated Unified Format) models are quantized versions that use less memory '
                    'while maintaining good quality. Perfect for systems with limited VRAM.'
                ).classes('text-sm text-gray-400')
                
                with ui.row().classes('gap-4 mt-2'):
                    with ui.column():
                        ui.label('✓ 50-75% less VRAM usage').classes('text-sm text-green-400')
                        ui.label('✓ Faster loading times').classes('text-sm text-green-400')
                    with ui.column():
                        ui.label('✓ Minimal quality loss').classes('text-sm text-green-400')
                        ui.label('✓ CPU fallback support').classes('text-sm text-green-400')
                        
            # GGUF integration information
            with ui.card().classes('w-full').style('background-color: #0A0A0A; border: 1px solid #333333'):
                ui.label('Automatic GGUF Integration').classes('font-semibold mb-2')
                ui.label(
                    'All FLUX models automatically support GGUF quantization through the enhanced processors. '
                    'The system will automatically select the optimal quantization level (Q8_0, Q6_K, or Q4_K) '
                    'based on available VRAM and quality requirements.'
                ).classes('text-sm text-gray-400')
                
    def _render_model_card(self, model_id: str, name: str, description: str, 
                           size: str, vram_required: str, repo_id: str, 
                           model_type: str, is_gguf: bool = False, is_experimental: bool = False):
        """Render a model card"""
        is_downloaded = self._check_model_downloaded(model_id, model_type)
        is_downloading = model_id in self.downloading_models
        
        # Check if model is empty (downloaded but no files)
        is_empty = False
        if is_downloaded and model_type in ['3d', 'image']:
            model_path = self.models_dir / model_type / model_id
            if model_path.exists():
                # Check if directory has actual model files
                model_files = list(model_path.glob('*.ckpt')) + \
                             list(model_path.glob('*.safetensors')) + \
                             list(model_path.glob('*.bin')) + \
                             list(model_path.glob('*.gguf'))
                # Also check for component directories
                component_dirs = [d for d in model_path.iterdir() if d.is_dir() and any(d.iterdir())]
                is_empty = len(model_files) == 0 and len(component_dirs) == 0
        
        with ui.card().classes('w-full').style('background-color: #0A0A0A; border: 1px solid #333333'):
            with ui.row().classes('items-start justify-between'):
                # Model info
                with ui.column().classes('flex-grow'):
                    with ui.row().classes('items-center gap-2'):
                        ui.label(name).classes('text-lg font-semibold')
                        if is_gguf:
                            ui.badge('GGUF').props('color=purple')
                        if is_experimental:
                            ui.badge('Experimental').props('color=orange')
                        if is_downloaded and not is_empty:
                            ui.icon('check_circle', size='sm').classes('text-green-500')
                        elif is_empty:
                            ui.icon('warning', size='sm').classes('text-yellow-500')
                            
                    ui.label(description).classes('text-sm text-gray-400 mt-1')
                    
                    if is_empty:
                        ui.label('⚠️ Model directory exists but files are missing. Please re-download.').classes('text-sm text-yellow-400 mt-1')
                    
                    # Technical details
                    with ui.row().classes('gap-4 mt-2'):
                        with ui.row().classes('items-center gap-1'):
                            ui.icon('storage', size='xs').classes('text-gray-500')
                            ui.label(size).classes('text-xs text-gray-500')
                        with ui.row().classes('items-center gap-1'):
                            ui.icon('memory', size='xs').classes('text-gray-500')
                            ui.label(vram_required).classes('text-xs text-gray-500')
                        with ui.row().classes('items-center gap-1'):
                            ui.icon('source', size='xs').classes('text-gray-500')
                            ui.label(repo_id).classes('text-xs text-gray-500')
                            
                # Actions
                with ui.column().classes('gap-2'):
                    if is_downloading:
                        # Show download progress
                        progress = self.downloading_models.get(model_id, 0)
                        with ui.column().classes('items-end'):
                            ui.label(f'{progress:.0f}%').classes('text-sm')
                            ui.linear_progress(progress / 100).classes('w-32')
                            ui.button(
                                'Cancel',
                                on_click=lambda m=model_id: self._cancel_download(m)
                            ).props('flat dense size=sm color=negative')
                    elif is_downloaded:
                        # Model is downloaded
                        with ui.row().classes('gap-2'):
                            ui.button(
                                'Delete',
                                icon='delete',
                                on_click=lambda m=model_id, t=model_type: self._delete_model(m, t)
                            ).props('flat dense color=negative')
                            ui.button(
                                'Info',
                                icon='info',
                                on_click=lambda m=model_id, t=model_type: self._show_model_info(m, t)
                            ).props('flat dense')
                    else:
                        # Model not downloaded
                        ui.button(
                            'Download',
                            icon='download',
                            on_click=lambda m=model_id, t=model_type, r=repo_id: self._download_model(m, t, r)
                        ).props('unelevated dense').style('background-color: #7C3AED')
                        
    def _check_model_downloaded(self, model_id: str, model_type: str) -> bool:
        """Check if a model is downloaded"""
        try:
            # Use the model manager's is_model_available method if available
            if self.model_manager:
                return self.model_manager.is_model_available(model_id)
        except Exception as e:
            logger.debug(f"Model manager check failed for {model_id}, falling back to filesystem check: {e}")
            
            # Fallback to filesystem check with improved logic
            if model_type == 'image':
                # Check multiple possible locations for image models
                possible_paths = [
                    self.models_dir / 'image' / model_id,
                    self.models_dir / 'gguf' / model_id
                ]
                for path in possible_paths:
                    if path.exists():
                        # Check for common model files
                        if any(path.glob('*.safetensors')) or any(path.glob('*.gguf')) or \
                           (path / 'model_index.json').exists() or \
                           any((path / subdir).exists() for subdir in ['transformer', 'vae', 'text_encoder']):
                            return True
                return False
            elif model_type == '3d':
                model_path = self.models_dir / '3d' / model_id
                if model_path.exists():
                    # Check for HunYuan3D specific structure
                    expected_dirs = ['hunyuan3d-dit-v2-1', 'hunyuan3d-vae-v2-1', 'hunyuan3d-paintpbr-v2-1']
                    return any((model_path / d).exists() for d in expected_dirs)
                return False
            else:
                # Generic check for other model types
                model_path = self.models_dir / model_type / model_id
                return model_path.exists() and any(model_path.iterdir()) if model_path.exists() else False
        
    async def _download_model(self, model_id: str, model_type: str, repo_id: str):
        """Start downloading a model"""
        if model_id in self.downloading_models:
            ui.notify('Already downloading this model', type='warning')
            return
            
        # Show download instructions for 3D models
        if model_type == '3d':
            ui.notify(
                f'To download {model_id}, run:\n'
                f'python scripts/download_hunyuan3d.py {model_id}\n'
                f'Or download from: https://huggingface.co/{repo_id}',
                type='info',
                timeout=10000
            )
            return
            
        ui.notify(f'Starting download: {model_id}', type='info')
        
        # Create download task
        task = asyncio.create_task(self._download_model_async(model_id, model_type, repo_id))
        self.download_tasks[model_id] = task
        
    async def _download_model_async(self, model_id: str, model_type: str, repo_id: str):
        """Async model download with progress"""
        try:
            # Add to downloading list
            self.downloading_models[model_id] = 0
            
            # Simulate download progress (in real app, this would use HuggingFace Hub)
            for i in range(101):
                if model_id not in self.downloading_models:
                    # Cancelled
                    break
                    
                self.downloading_models[model_id] = i
                await asyncio.sleep(0.1)  # Simulate download time
                
                # Trigger UI update
                if hasattr(self, 'update_trigger'):
                    self.update_trigger.emit()
                    
            # Download complete
            if model_id in self.downloading_models:
                del self.downloading_models[model_id]
                ui.notify(f'Downloaded: {model_id}', type='positive')
                
                # In real app, would trigger a refresh of the model list
                
        except Exception as e:
            ui.notify(f'Download failed: {str(e)}', type='negative')
            if model_id in self.downloading_models:
                del self.downloading_models[model_id]
        finally:
            if model_id in self.download_tasks:
                del self.download_tasks[model_id]
                
    def _cancel_download(self, model_id: str):
        """Cancel a download"""
        if model_id in self.download_tasks:
            self.download_tasks[model_id].cancel()
            
        if model_id in self.downloading_models:
            del self.downloading_models[model_id]
            
        ui.notify(f'Cancelled download: {model_id}', type='info')
        
    def _delete_model(self, model_id: str, model_type: str):
        """Delete a model with confirmation"""
        with ui.dialog() as dialog, ui.card():
            ui.label(f'Delete model: {model_id}?').classes('text-lg mb-4')
            ui.label('This will free up disk space but you\'ll need to download it again to use it.').classes('text-sm text-gray-400')
            
            with ui.row().classes('gap-2 mt-4'):
                ui.button('Cancel', on_click=dialog.close).props('flat')
                ui.button(
                    'Delete',
                    on_click=lambda: self._confirm_delete_model(model_id, model_type, dialog)
                ).props('unelevated color=negative')
                
        dialog.open()
        
    def _confirm_delete_model(self, model_id: str, model_type: str, dialog):
        """Confirm model deletion"""
        try:
            # Get model path
            if model_type == 'image':
                model_path = self.models_dir / 'image' / model_id
            elif model_type == '3d':
                model_path = self.models_dir / '3d' / model_id
            elif model_type == 'gguf':
                model_path = self.models_dir / 'gguf' / model_id
            elif model_type == 'controlnet':
                model_path = self.models_dir / 'controlnet' / model_id
            elif model_type == 'depth':
                model_path = self.models_dir / 'depth' / model_id
            elif model_type == 'preprocessing':
                model_path = self.models_dir / 'preprocessing' / model_id
            elif model_type == 'texture':
                model_path = self.models_dir / 'texture' / model_id
            elif model_type == 'video':
                model_path = self.models_dir / 'video' / model_id
            else:
                model_path = self.models_dir / model_type / model_id
                
            # Delete the directory
            if model_path.exists():
                shutil.rmtree(model_path)
                
            ui.notify(f'Deleted model: {model_id}', type='positive')
            dialog.close()
            
            # Refresh the page
            # In a real app, would trigger a refresh
            
        except Exception as e:
            ui.notify(f'Failed to delete: {str(e)}', type='negative')
            
    def _show_model_info(self, model_id: str, model_type: str):
        """Show detailed model information"""
        with ui.dialog() as dialog, ui.card().classes('w-[600px]'):
            with ui.row().classes('items-center justify-between mb-4'):
                ui.label(f'Model: {model_id}').classes('text-xl font-semibold')
                ui.button(icon='close', on_click=dialog.close).props('flat round')
                
            # Model details
            with ui.column().classes('gap-2'):
                # Get model path and size
                if model_type == 'image':
                    model_path = self.models_dir / 'image' / model_id
                elif model_type == '3d':
                    model_path = self.models_dir / '3d' / model_id
                elif model_type == 'gguf':
                    model_path = self.models_dir / 'gguf' / model_id
                elif model_type == 'controlnet':
                    model_path = self.models_dir / 'controlnet' / model_id
                elif model_type == 'depth':
                    model_path = self.models_dir / 'depth' / model_id
                elif model_type == 'preprocessing':
                    model_path = self.models_dir / 'preprocessing' / model_id
                elif model_type == 'texture':
                    model_path = self.models_dir / 'texture' / model_id
                elif model_type == 'video':
                    model_path = self.models_dir / 'video' / model_id
                else:
                    model_path = self.models_dir / model_type / model_id
                    
                if model_path.exists():
                    # Calculate total size
                    total_size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
                    
                    ui.label('Storage Information').classes('font-semibold mt-2')
                    ui.label(f'Path: {model_path}').classes('text-sm font-mono')
                    ui.label(f'Total Size: {total_size / (1024**3):.2f} GB').classes('text-sm')
                    
                    # List files
                    ui.label('Model Files').classes('font-semibold mt-4')
                    with ui.column().classes('gap-1 max-h-64 overflow-auto'):
                        for file in sorted(model_path.rglob('*')):
                            if file.is_file():
                                size_mb = file.stat().st_size / (1024**2)
                                ui.label(f'{file.name} ({size_mb:.1f} MB)').classes('text-xs font-mono')
                                
        dialog.open()