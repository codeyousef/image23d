"""
NeuralForge Studio Desktop Application - Native Only Version
This version ensures the app runs exclusively as a desktop application
"""

import sys
import asyncio
from pathlib import Path
from nicegui import ui, app, native
import warnings

# Suppress browser opening warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.services import ModelManager
from core.config import MODELS_DIR, OUTPUT_DIR
from desktop.ui import apply_theme
from desktop.ui.components import NavigationSidebar
from desktop.ui.pages.create_better import CreatePage
from desktop.ui.pages.library import LibraryPage
from desktop.ui.pages.models import ModelsPage
from desktop.ui.pages.lab import LabPage

class NeuralForgeDesktop:
    """Main desktop application class"""
    
    def __init__(self):
        # Core services
        self.model_manager = ModelManager(MODELS_DIR)
        
        # Pages
        self.pages = {
            'create': CreatePage(OUTPUT_DIR),
            'library': LibraryPage(OUTPUT_DIR),
            'lab': LabPage(OUTPUT_DIR),
            'models': ModelsPage(MODELS_DIR),
            'settings': None
        }
        
        # Current page
        self.current_page = 'create'
        
        # UI components
        self.navigation = NavigationSidebar(on_navigate=self.navigate_to)
        
    async def initialize(self):
        """Initialize the application"""
        # Initialize model manager
        await self.model_manager.initialize()
        
        # Set up UI
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the main UI layout"""
        # Apply theme
        apply_theme()
        
        # Configure native window
        native.window_size = (1400, 900)
        native.window_title = 'NeuralForge Studio'
        
        # Main layout
        with ui.row().classes('w-screen h-screen gap-0'):
            # Navigation sidebar
            self.navigation.render()
            
            # Main content area
            self.content_container = ui.column().classes('flex-1 h-screen overflow-auto bg-background')
            
        # Load initial page
        self.navigate_to(self.current_page)
        
    def navigate_to(self, page_id: str):
        """Navigate to a specific page"""
        if page_id not in self.pages:
            ui.notify(f'Page {page_id} not found', type='warning')
            return
            
        self.current_page = page_id
        
        # Clear content
        self.content_container.clear()
        
        # Render new page
        with self.content_container:
            if page_id == 'create':
                self.pages['create'].render()
            elif page_id == 'library':
                self.pages['library'].render()
            elif page_id == 'lab':
                self.pages['lab'].render()
            elif page_id == 'models':
                self.pages['models'].render()
            elif page_id == 'settings':
                self._render_settings()
                
    def _render_settings(self):
        """Render settings page"""
        with ui.column().classes('w-full h-full p-6'):
            ui.label('Settings').classes('text-2xl font-bold mb-6')
            
            with ui.card().classes('card'):
                ui.label('Application Settings').classes('text-lg font-semibold mb-4')
                
                # GPU settings
                with ui.column().classes('gap-4'):
                    ui.label('GPU Configuration').classes('font-medium')
                    ui.switch('Enable GPU acceleration', value=True)
                    ui.switch('Use FP16 precision', value=True)
                    
                    ui.separator()
                    
                    # API Keys
                    ui.label('API Keys').classes('font-medium')
                    ui.input('RunPod API Key', password=True, placeholder='Enter your RunPod API key')
                    ui.input('Civitai API Key', password=True, placeholder='Enter your Civitai API key')
                    
                    ui.separator()
                    
                    # Paths
                    ui.label('Storage Paths').classes('font-medium')
                    ui.label(f'Models: {MODELS_DIR}').classes('text-sm')
                    ui.label(f'Output: {OUTPUT_DIR}').classes('text-sm')
                    
            # About section
            with ui.card().classes('card mt-4'):
                ui.label('About NeuralForge Studio').classes('text-lg font-semibold mb-4')
                ui.label('Version: 1.0.0').classes('text-sm')
                ui.label('A powerful AI creative suite for generating images, 3D models, and videos').classes('text-sm text-gray-500')

# Global app instance
app_instance = None

@ui.page('/')
async def main_page():
    """Main page setup"""
    global app_instance
    if app_instance is None:
        app_instance = NeuralForgeDesktop()
        await app_instance.initialize()

def main():
    """Main application entry point - Native Desktop Only"""
    import os
    
    # Force native mode environment
    os.environ['NICEGUI_NATIVE_MODE'] = '1'
    
    # Storage path
    storage_path = Path.home() / '.neuralforge'
    storage_path.mkdir(exist_ok=True)
    
    # Native configuration
    native_config = {
        'window': {
            'width': 1400,
            'height': 900,
            'title': 'NeuralForge Studio',
            'resizable': True,
            'frameless': False,
            'easy_drag': False,
            'fullscreen': False,
            'maximized': False,
            'minimized': False,
            'confirm_close': True,
            'background_color': '#1a1a1a',
        },
        'start_args': {
            'debug': False,
            'ssl': False
        }
    }
    
    # Run exclusively as native desktop app
    try:
        ui.run(
            native=True,
            title='NeuralForge Studio',
            window_size=(1400, 900),
            dark=True,
            reload=False,
            port=None,  # No web server
            show=False,  # Don't open browser
            storage_secret=storage_path / '.secret',
            binding_refresh_interval=0.1,
            favicon='🧠',
            on_air=False,  # Disable web access
            **native_config
        )
    except Exception as e:
        # Fallback with minimal config
        print(f"Warning: Some native features unavailable: {e}")
        ui.run(
            native=True,
            title='NeuralForge Studio', 
            window_size=(1400, 900),
            dark=True,
            show=False,
            port=None
        )

if __name__ == '__main__':
    main()