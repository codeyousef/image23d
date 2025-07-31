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
from nicegui import ui, app
import sys
import os
# Add project root to path to import from main app
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))
import shutil
from nicegui import run

# Import download manager
try:
    from src.hunyuan3d_app.models.download_manager import DownloadManager, DownloadTask, DownloadStatus
except ImportError:
    from hunyuan3d_app.models.download_manager import DownloadManager, DownloadTask, DownloadStatus

logger = logging.getLogger(__name__)

class ModelsPage:
    """Models page for downloading and managing AI models"""
    
    def __init__(self, models_dir: Path, model_manager=None):
        self.models_dir = models_dir
        self.model_manager = model_manager
        self.downloading_models: Dict[str, float] = {}  # model_id -> progress
        self.download_tasks: Dict[str, asyncio.Task] = {}
        self.progress_cards: Dict[str, Any] = {}  # model_id -> UI elements
        self._page_container = None  # Store reference to page container
        self.download_states: Dict[str, Dict[str, Any]] = {}  # model_id -> {status, progress, speed, eta, model_type, repo_id}
        self.refreshable_cards: Dict[str, Any] = {}  # model_id -> refreshable function
        self.downloads_container = None  # Container for downloads list
        self._ui_needs_update = False  # Flag for UI updates from background threads
        
        # Initialize download manager
        self.download_manager = DownloadManager(models_dir)
        
    def render(self):
        """Render the models page"""
        with ui.column().classes('w-full h-full') as self._page_container:
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
                self.downloads_tab = ui.tab('Downloads', icon='download')
                
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
                    
                # Downloads Tab
                with ui.tab_panel(self.downloads_tab):
                    self._render_downloads_tab()
            
            # Add a timer to refresh model cards when needed
            def refresh_cards_if_needed():
                """Refresh model cards if downloads state has changed"""
                if self._ui_needs_update and hasattr(self, 'refreshable_cards'):
                    self._ui_needs_update = False
                    # Refresh cards for models that are downloading
                    for model_id in list(self.download_states.keys()):
                        if model_id in self.refreshable_cards and callable(self.refreshable_cards[model_id]):
                            try:
                                self.refreshable_cards[model_id].refresh()
                            except Exception as e:
                                logger.debug(f"Failed to refresh card {model_id}: {e}")
                # Also refresh if there are active downloads even without the flag
                elif hasattr(self, 'refreshable_cards') and self.download_states:
                    for model_id in list(self.download_states.keys()):
                        status = self.download_states[model_id].get('status')
                        if status in ['starting', 'downloading']:
                            if model_id in self.refreshable_cards and callable(self.refreshable_cards[model_id]):
                                try:
                                    self.refreshable_cards[model_id].refresh()
                                except Exception as e:
                                    logger.debug(f"Failed to refresh active download card {model_id}: {e}")
                                
            ui.timer(0.5, refresh_cards_if_needed)
                    
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
        try:
            from core.config import FLUX_MODELS
        except ImportError:
            from src.hunyuan3d_app.config import IMAGE_MODELS as FLUX_MODELS
        
        with ui.column().classes('w-full gap-4'):
            # FLUX models
            ui.label('FLUX Image Models').classes('text-lg font-semibold')
            ui.label('State-of-the-art text-to-image generation with enhanced features').classes('text-sm text-gray-400 mb-2')
            
            for model_id, config in FLUX_MODELS.items():
                self._create_model_card_wrapper(
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
        try:
            from core.config import HUNYUAN3D_MODELS, SPARC3D_MODELS, HI3DGEN_MODELS
        except ImportError:
            from src.hunyuan3d_app.config import HUNYUAN3D_MODELS, SPARC3D_MODELS, HI3DGEN_MODELS
        
        with ui.column().classes('w-full gap-4'):
            # HunYuan3D Models
            ui.label('HunYuan3D Models').classes('text-lg font-semibold')
            ui.label('State-of-the-art text/image to 3D generation').classes('text-sm text-gray-400 mb-2')
            
            for model_id, config in HUNYUAN3D_MODELS.items():
                self._create_model_card_wrapper(
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
                    self._create_model_card_wrapper(
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
                    self._create_model_card_wrapper(
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
                self._create_model_card_wrapper(
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
                self._create_model_card_wrapper(
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
                self._create_model_card_wrapper(
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
                self._create_model_card_wrapper(
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
        try:
            from core.config import VIDEO_MODELS
        except ImportError:
            from src.hunyuan3d_app.config import VIDEO_MODELS
        
        with ui.column().classes('w-full gap-4'):
            ui.label('Text-to-Video Models').classes('text-lg font-semibold')
            ui.label('State-of-the-art video generation models').classes('text-sm text-gray-400 mb-2')
            
            for model_id, config in VIDEO_MODELS.items():
                self._create_model_card_wrapper(
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
                
    def _create_model_card_wrapper(self, model_id: str, name: str, description: str,
                                  size: str, vram_required: str, repo_id: str,
                                  model_type: str, is_gguf: bool = False, is_experimental: bool = False):
        """Create a refreshable model card wrapper"""
        # Initialize refreshable cards dict if needed
        if not hasattr(self, 'refreshable_cards'):
            self.refreshable_cards = {}
            
        @ui.refreshable
        def card():
            self._render_model_card(
                model_id=model_id,
                name=name,
                description=description,
                size=size,
                vram_required=vram_required,
                repo_id=repo_id,
                model_type=model_type,
                is_gguf=is_gguf,
                is_experimental=is_experimental
            )
        
        # Store the refreshable function itself, not the result
        self.refreshable_cards[model_id] = card
        
        # Render the card and return the container
        return card()
                
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
                        with ui.column().classes('items-end') as progress_col:
                            # Check if we have file-based progress
                            task = self.download_manager.get_task(model_id)
                            if task and task.total_files > 0:
                                # Show both file progress and percentage
                                current_file = task.current_file_index if task.current_file_index > 0 else 1
                                if task.completed_files < task.total_files:
                                    progress_label = ui.label(f'File {current_file}/{task.total_files} - {progress:.0f}%').classes('text-sm')
                                else:
                                    progress_label = ui.label(f'{task.total_files}/{task.total_files} files - 100%').classes('text-sm')
                            else:
                                # Percentage-based progress only
                                progress_label = ui.label(f'{progress:.0f}%').classes('text-sm')
                            progress_bar = ui.linear_progress(progress / 100).classes('w-32')
                            ui.button(
                                'Cancel',
                                on_click=lambda m=model_id: self._cancel_download(m)
                            ).props('flat dense size=sm color=negative')
                            
                            # Store references for updates
                            self.progress_cards[model_id] = {
                                'label': progress_label,
                                'bar': progress_bar,
                                'container': progress_col
                            }
                            
                            # Set up timer to update progress
                            timer = ui.timer(0.5, lambda: self._check_progress(model_id))
                            # Store timer reference to stop it later
                            if 'timer' not in self.progress_cards[model_id]:
                                self.progress_cards[model_id]['timer'] = timer
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
                        # Check if it's empty (downloaded but files missing)
                        if is_empty:
                            download_btn = ui.button(
                                'Re-download',
                                icon='refresh'
                            ).props('unelevated dense').style('background-color: #FF6B6B')
                        else:
                            download_btn = ui.button(
                                'Download',
                                icon='download'
                            ).props('unelevated dense').style('background-color: #7C3AED')
                        
                        # Use closure to capture current values
                        download_btn.on('click', lambda e, m=model_id, t=model_type, r=repo_id: self.start_download(m, t, r))
                        
    def _check_model_downloaded(self, model_id: str, model_type: str) -> bool:
        """Check if a model is downloaded"""
        try:
            # Use the model manager's is_model_available method if available
            if self.model_manager:
                return self.model_manager.is_model_available(model_id)
        except Exception as e:
            logger.debug(f"Model manager check failed for {model_id}, falling back to filesystem check: {e}")
            
            # Import component info
            try:
                from src.hunyuan3d_app.models.dependencies import COMPONENT_PATTERNS
            except ImportError:
                from hunyuan3d_app.models.dependencies import COMPONENT_PATTERNS
            
            # Check if it's a component
            if model_id in COMPONENT_PATTERNS:
                # Components can be in various locations
                possible_paths = [
                    self.models_dir / 'components' / model_id,
                    self.models_dir / 'pipeline' / model_id,
                    self.models_dir / model_type / model_id,
                    self.models_dir / 'texture' / model_id,
                    self.models_dir / 'preprocessing' / model_id,
                    self.models_dir / 'controlnet' / model_id,
                    self.models_dir / 'depth' / model_id,
                ]
                
                # Also check inside 3D model directories for HunYuan3D components
                if model_id.startswith('hunyuan3d-'):
                    for model_dir in (self.models_dir / '3d').iterdir() if (self.models_dir / '3d').exists() else []:
                        if model_dir.is_dir():
                            possible_paths.append(model_dir / model_id)
                
                for path in possible_paths:
                    if path.exists() and any(path.iterdir()):
                        return True
                return False
            
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
                    if any((model_path / d).exists() for d in expected_dirs):
                        return True
                    # Also check for generic 3D model files
                    if any(model_path.glob('*.ckpt')) or any(model_path.glob('*.safetensors')):
                        return True
                return False
            else:
                # Generic check for other model types
                model_path = self.models_dir / model_type / model_id
                return model_path.exists() and any(model_path.iterdir()) if model_path.exists() else False
                
    def start_download(self, model_id: str, model_type: str, repo_id: str):
        """Start downloading a model with immediate UI feedback"""
        if model_id in self.downloading_models:
            ui.notify('Already downloading this model', type='warning')
            return
            
        # Add to downloading state immediately
        self.downloading_models[model_id] = 0
        self.download_states[model_id] = {
            'status': 'starting',
            'progress': 0,
            'speed': 0,
            'eta': 'calculating...',
            'model_type': model_type,
            'repo_id': repo_id
        }
        
        # Show notification
        ui.notify(f'Starting download: {model_id}', type='info')
        
        # Mark UI needs update and force immediate refresh
        self._ui_needs_update = True
        
        # Try to refresh the card immediately
        if hasattr(self, 'refreshable_cards') and model_id in self.refreshable_cards:
            if callable(self.refreshable_cards[model_id]):
                try:
                    self.refreshable_cards[model_id].refresh()
                except Exception as e:
                    logger.error(f"Failed to refresh card immediately: {e}")
        
        # Start the actual async download using the correct method
        asyncio.create_task(self._download_model_async(model_id, model_type, repo_id))
        
    async def _download_model(self, model_id: str, model_type: str, repo_id: str):
        """Async download handler"""
        # Show download instructions for 3D models
        if model_type == '3d':
            self.download_states[model_id]['status'] = 'failed'
            self.download_states[model_id]['error'] = 'Manual download required'
            ui.notify(
                f'To download {model_id}, run:\n'
                f'python scripts/download_hunyuan3d.py {model_id}\n'
                f'Or download from: https://huggingface.co/{repo_id}',
                type='info',
                timeout=10000
            )
            return
            
        # Update status
        self.download_states[model_id]['status'] = 'downloading'
        
        # Create download task using asyncio.create_task
        logger.info(f"Creating async task for {model_id}")
        try:
            # Ensure we're in an async context
            loop = asyncio.get_event_loop()
            task = asyncio.create_task(self._download_model_async(model_id, model_type, repo_id))
            self.download_tasks[model_id] = task
            logger.info(f"Async task created successfully for {model_id}")
        except Exception as e:
            logger.error(f"Failed to create async task for {model_id}: {e}")
            # Try alternative approach
            asyncio.ensure_future(self._download_model_async(model_id, model_type, repo_id))
        
    async def _download_model_async(self, model_id: str, model_type: str, repo_id: str):
        """Async model download with progress"""
        logger.info(f"Starting async download for {model_id} from {repo_id}")
        try:
            # Update download state
            self.download_states[model_id]['status'] = 'downloading'
            self._ui_needs_update = True
            
            # Determine download patterns based on model type and component info
            allow_patterns = None
            subfolder = None
            
            # Import component patterns
            try:
                from src.hunyuan3d_app.models.dependencies import COMPONENT_PATTERNS
            except ImportError:
                from hunyuan3d_app.models.dependencies import COMPONENT_PATTERNS
            
            # Check if this is a component download
            if model_id in COMPONENT_PATTERNS:
                component_info = COMPONENT_PATTERNS[model_id]
                allow_patterns = component_info.get('patterns', [])
                # Override repo_id if component has its own
                if 'repo_id' in component_info and component_info['repo_id'] != repo_id:
                    repo_id = component_info['repo_id']
            elif model_type == '3d':
                # HunYuan3D models have specific structure
                if 'hunyuan3d' in model_id:
                    # Download all components for HunYuan3D
                    allow_patterns = [
                        'hunyuan3d-dit-v*/**/*',
                        'hunyuan3d-vae-v*/**/*', 
                        'hunyuan3d-paintpbr-v*/**/*',
                        '*.yaml', '*.json', '*.txt'
                    ]
                else:
                    # Generic 3D model patterns
                    allow_patterns = ['*.ckpt', '*.safetensors', '*.bin', '*.pth', '*.pt', 
                                     '*.yaml', '*.json', 'config.json', 'model_index.json']
            elif model_type == 'image':
                # FLUX models
                allow_patterns = ['*.safetensors', '*.bin', '*.json', 'model_index.json', 
                                 'transformer/**/*', 'vae/**/*', 'text_encoder/**/*', 'tokenizer/**/*']
            elif model_type == 'video':
                # Video models
                allow_patterns = ['*.safetensors', '*.bin', '*.ckpt', '*.json', 'model_index.json']
            elif model_type in ['controlnet', 'depth', 'preprocessing', 'texture']:
                # Component models
                allow_patterns = ['*.safetensors', '*.bin', '*.onnx', '*.pth', '*.json', 'config.json']
            
            # Add progress callback
            def progress_callback(task: DownloadTask):
                """Update download progress (called from background thread)"""
                # Only update data, don't touch UI from background thread
                # Round progress to integer but never round up to 100% unless actually complete
                raw_progress = task.progress * 100
                if raw_progress >= 99.5 and task.status != DownloadStatus.COMPLETED:
                    # Cap at 99% if not actually completed to avoid false 100%
                    progress_percent = 99
                else:
                    progress_percent = round(raw_progress)
                    
                logger.debug(f"Progress callback: task.progress={task.progress:.3f}, raw={raw_progress:.1f}, final={progress_percent}")
                self.downloading_models[model_id] = progress_percent
                self.download_states[model_id].update({
                    'status': task.status.value,
                    'progress': progress_percent,
                    'speed': task.speed,
                    'eta': task.eta_formatted,
                    'error': task.error
                })
                
                # Mark that UI needs update
                self._ui_needs_update = True
                    
            # Register progress callback
            self.download_manager.add_progress_callback(model_id, progress_callback)
            
            # Start actual download
            logger.info(f"Calling download_manager.download_model for {model_id}")
            logger.info(f"Parameters: model_type={model_type}, repo_id={repo_id}, allow_patterns={allow_patterns}")
            success = await self.download_manager.download_model(
                model_type=model_type,
                model_id=model_id,
                repo_id=repo_id,
                allow_patterns=allow_patterns
            )
            logger.info(f"Download manager returned success={success} for {model_id}")
            
            if success:
                await self._download_complete(model_id)
            else:
                task = self.download_manager.get_task(model_id)
                error_msg = task.error if task else "Download failed"
                await self._download_failed(model_id, error_msg)
                    
        except Exception as e:
            await self._download_failed(model_id, str(e))
        finally:
            if model_id in self.download_tasks:
                del self.download_tasks[model_id]
            if model_id in self.progress_cards:
                del self.progress_cards[model_id]
                
    async def _update_progress_ui(self, model_id: str, progress: float):
        """Update progress UI elements - deprecated, using flag-based updates instead"""
        # This method is no longer used - UI updates happen through the timer
        pass
            
    async def _download_complete(self, model_id: str):
        """Handle download completion"""
        if model_id in self.downloading_models:
            del self.downloading_models[model_id]
        if model_id in self.download_states:
            del self.download_states[model_id]  # Remove from download states completely
        
        # Mark UI needs update and force immediate refresh
        self._ui_needs_update = True
        
        # Force immediate card refresh to show download completion
        if hasattr(self, 'refreshable_cards') and model_id in self.refreshable_cards:
            if callable(self.refreshable_cards[model_id]):
                try:
                    self.refreshable_cards[model_id].refresh()
                except Exception as e:
                    logger.error(f"Failed to refresh card after completion: {e}")
        
        # Also refresh the dependency checking system
        try:
            # Force refresh of dependency detection for newly downloaded model
            if hasattr(self, 'dependency_manager'):
                await asyncio.sleep(0.1)  # Small delay to ensure files are written
                # Clear any cached dependency results
                if hasattr(self.dependency_manager, '_cache'):
                    self.dependency_manager._cache.clear()
                
            # If this was a component model, refresh all cards that might depend on it
            if model_id in ['dinov2-giant', 'realesrgan-x4'] or model_id.startswith('hunyuan3d-'):
                logger.info(f"Component {model_id} downloaded, refreshing dependent model cards")
                # Force refresh of all cards since components affect multiple models
                self._ui_needs_update = True
                
        except Exception as e:
            logger.debug(f"Could not refresh dependency manager: {e}")
        
        logger.info(f'Download complete: {model_id}')
        ui.notify(f'Download complete: {model_id}', type='positive')
        
        # Schedule additional refreshes to ensure UI updates
        async def delayed_refresh():
            await asyncio.sleep(0.5)  # Wait for file system to stabilize
            self._ui_needs_update = True
            # Refresh all cards that might be affected
            if hasattr(self, 'refreshable_cards'):
                for card_id, card_refresh in self.refreshable_cards.items():
                    if callable(card_refresh):
                        try:
                            card_refresh.refresh()
                        except Exception as e:
                            logger.debug(f"Could not refresh card {card_id}: {e}")
        
        # Run delayed refresh
        asyncio.create_task(delayed_refresh())
        
    async def _download_failed(self, model_id: str, error: str):
        """Handle download failure"""
        if model_id in self.downloading_models:
            del self.downloading_models[model_id]
        if model_id in self.download_states:
            self.download_states[model_id]['status'] = 'failed'
            self.download_states[model_id]['error'] = error
        # Mark UI needs update
        self._ui_needs_update = True
        logger.error(f'Download failed for {model_id}: {error}')
                
    def _cancel_download(self, model_id: str):
        """Cancel a download"""
        # Cancel via download manager
        self.download_manager.cancel_download(model_id)
        
        if model_id in self.download_tasks:
            self.download_tasks[model_id].cancel()
            
        if model_id in self.downloading_models:
            del self.downloading_models[model_id]
            
        if model_id in self.download_states:
            del self.download_states[model_id]
            
        ui.notify(f'Cancelled download: {model_id}', type='info')
        
        # Mark UI needs update
        self._ui_needs_update = True
        
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
        
    def _render_downloads_tab(self):
        """Render downloads management tab"""
        with ui.column().classes('w-full gap-4'):
            # Header
            with ui.row().classes('items-center justify-between mb-4'):
                ui.label('Download Manager').classes('text-lg font-semibold')
                with ui.row().classes('gap-2'):
                    active_count = len([d for d in self.download_states.values() if d.get('status') in ['downloading', 'starting']])
                    ui.label(f'{active_count} Active Downloads').classes('text-sm text-gray-400')
                    
            # Downloads list container
            with ui.column().classes('w-full gap-2') as downloads_container:
                self.downloads_container = downloads_container
                self._update_downloads_list()
                
            # Auto-refresh downloads list
            def check_and_update():
                """Check if UI needs update and refresh if needed"""
                if self._ui_needs_update:
                    self._ui_needs_update = False
                    self._update_downloads_list()
                # Also update if there are active downloads
                elif any(d.get('status') in ['downloading', 'starting'] for d in self.download_states.values()):
                    self._update_downloads_list()
                    
            ui.timer(0.5, check_and_update)
            
    def _update_downloads_list(self):
        """Update the downloads list UI"""
        if not self.downloads_container:
            return
            
        self.downloads_container.clear()
        
        if not self.download_states:
            with self.downloads_container:
                with ui.card().classes('w-full p-8 text-center').style('background-color: #0A0A0A; border: 1px solid #333333'):
                    ui.icon('download_done', size='3rem').classes('text-gray-600 mb-4')
                    ui.label('No active downloads').classes('text-gray-500')
                    ui.label('Download models from other tabs to see them here').classes('text-sm text-gray-600')
            return
            
        # Show each download
        with self.downloads_container:
            for model_id, state in self.download_states.items():
                self._render_download_item(model_id, state)
                
    def _render_download_item(self, model_id: str, state: Dict[str, Any]):
        """Render a single download item"""
        with ui.card().classes('w-full').style('background-color: #0A0A0A; border: 1px solid #333333'):
            with ui.row().classes('items-center justify-between'):
                # Model info
                with ui.column().classes('flex-grow'):
                    with ui.row().classes('items-center gap-2'):
                        ui.label(model_id).classes('font-semibold')
                        status_color = {
                            'starting': 'blue',
                            'downloading': 'green',
                            'paused': 'orange',
                            'failed': 'red',
                            'completed': 'green'
                        }.get(state.get('status', 'starting'), 'gray')
                        ui.badge(state.get('status', 'starting').upper()).props(f'color={status_color}')
                        
                    # Progress info
                    progress = state.get('progress', 0)
                    ui.linear_progress(progress / 100).classes('w-full mt-2')
                    
                    with ui.row().classes('gap-4 mt-1'):
                        # Check if we have file-based progress
                        task = self.download_manager.get_task(model_id)
                        if task and task.total_files > 0:
                            # Show detailed file progress
                            current_file = task.current_file_index if task.current_file_index > 0 else 1
                            if task.completed_files < task.total_files:
                                ui.label(f'File {current_file}/{task.total_files}').classes('text-sm')
                                ui.label(f'{progress:.0f}%').classes('text-sm font-semibold')
                            else:
                                ui.label(f'{task.total_files}/{task.total_files} files').classes('text-sm')
                                ui.label('100%').classes('text-sm font-semibold')
                        else:
                            # Percentage-based progress only
                            ui.label(f'{progress:.0f}%').classes('text-sm font-semibold')
                        if state.get('speed'):
                            ui.label(f'{state["speed"]:.1f} MB/s').classes('text-sm text-gray-400')
                        if state.get('eta'):
                            ui.label(f'ETA: {state["eta"]}').classes('text-sm text-gray-400')
                            
                # Actions
                with ui.row().classes('gap-2'):
                    if state.get('status') == 'downloading':
                        ui.button(
                            'Pause',
                            icon='pause',
                            on_click=lambda m=model_id: self._pause_download(m)
                        ).props('flat dense')
                        ui.button(
                            'Cancel',
                            icon='close',
                            on_click=lambda m=model_id: self._cancel_download(m)
                        ).props('flat dense color=negative')
                    elif state.get('status') == 'paused':
                        ui.button(
                            'Resume',
                            icon='play_arrow',
                            on_click=lambda m=model_id: self._resume_download(m)
                        ).props('flat dense color=primary')
                        ui.button(
                            'Cancel',
                            icon='close',
                            on_click=lambda m=model_id: self._cancel_download(m)
                        ).props('flat dense color=negative')
                    elif state.get('status') == 'failed':
                        ui.button(
                            'Retry',
                            icon='refresh',
                            on_click=lambda m=model_id: self._retry_download(m)
                        ).props('flat dense color=primary')
                        ui.button(
                            'Remove',
                            icon='delete',
                            on_click=lambda m=model_id: self._remove_download(m)
                        ).props('flat dense')
                        
    def _pause_download(self, model_id: str):
        """Pause a download"""
        # Pause via download manager
        self.download_manager.pause_download(model_id)
        
        if model_id in self.download_states:
            self.download_states[model_id]['status'] = 'paused'
            ui.notify(f'Paused download: {model_id}', type='info')
            
    async def _resume_download(self, model_id: str):
        """Resume a paused download"""
        if model_id in self.download_states:
            # Resume via download manager
            success = await self.download_manager.resume_download(model_id)
            if success:
                self.download_states[model_id]['status'] = 'downloading'
                ui.notify(f'Resumed download: {model_id}', type='info')
            else:
                ui.notify(f'Failed to resume download: {model_id}', type='negative')
            
    def _retry_download(self, model_id: str):
        """Retry a failed download"""
        if model_id in self.download_states:
            state = self.download_states[model_id]
            self.start_download(model_id, state['model_type'], state['repo_id'])
            
    def _remove_download(self, model_id: str):
        """Remove a download from the list"""
        if model_id in self.download_states:
            del self.download_states[model_id]
            self._update_downloads_list()
        
    def _check_progress(self, model_id: str):
        """Check and update download progress"""
        if model_id in self.downloading_models and model_id in self.progress_cards:
            progress = self.downloading_models[model_id]
            cards = self.progress_cards[model_id]
            
            # Check if we have file-based progress
            task = self.download_manager.get_task(model_id)
            if task and task.total_files > 0:
                # Show both file progress and percentage
                current_file = task.current_file_index if task.current_file_index > 0 else 1
                if task.completed_files < task.total_files:
                    cards['label'].text = f'File {current_file}/{task.total_files} - {progress:.0f}%'
                else:
                    cards['label'].text = f'{task.total_files}/{task.total_files} files - 100%'
            else:
                # Percentage-based progress only
                cards['label'].text = f'{progress:.0f}%'
            
            cards['bar'].value = progress / 100
        else:
            # Download complete or cancelled - stop timer
            if model_id in self.progress_cards and 'timer' in self.progress_cards[model_id]:
                self.progress_cards[model_id]['timer'].active = False
        
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
        
    def _refresh_model_card(self, model_id: str):
        """Refresh a specific model card if it exists"""
        if hasattr(self, 'refreshable_cards') and model_id in self.refreshable_cards:
            try:
                if callable(self.refreshable_cards[model_id]):
                    self.refreshable_cards[model_id].refresh()
                else:
                    logger.warning(f"Refreshable card for {model_id} is not callable")
            except Exception as e:
                logger.warning(f"Failed to refresh model card {model_id}: {e}")