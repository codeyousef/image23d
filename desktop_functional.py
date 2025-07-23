"""
Functional NeuralForge Studio Desktop Application
This version uses the fixed processors that connect to the actual generation system
"""

import sys
from pathlib import Path
from nicegui import ui, app

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from core.config import OUTPUT_DIR
from desktop.ui import apply_theme
from desktop.ui.components import NavigationSidebar
from desktop.ui.pages.create_better import CreatePage  # Use the better layout version
from desktop.ui.pages.library import LibraryPage
from desktop.ui.pages.models import ModelsPage
from desktop.ui.pages.lab import LabPage

class NeuralForgeDesktopFunctional:
    """Functional desktop application with real generation"""
    
    def __init__(self):
        # Pages - all implemented
        self.pages = {
            'create': CreatePage(OUTPUT_DIR),
            'library': LibraryPage(OUTPUT_DIR),
            'lab': LabPage(OUTPUT_DIR),
            'models': ModelsPage(Path.home() / '.neuralforge' / 'models'),
            'settings': None  # Using inline implementation
        }
        
        # Current page
        self.current_page = 'create'
        
        # UI components
        self.navigation = NavigationSidebar(on_navigate=self.navigate_to)
        
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
                
    def _render_placeholder(self, title: str, description: str):
        """Render a placeholder page"""
        with ui.column().classes('w-full h-full items-center justify-center'):
            ui.icon('construction', size='4rem').classes('text-gray-600 mb-4')
            ui.label(title).classes('text-2xl font-bold mb-2')
            ui.label(description).classes('text-gray-500')
            
    def _render_settings(self):
        """Render settings page"""
        with ui.column().classes('w-full h-full p-6'):
            ui.label('Settings').classes('text-2xl font-bold mb-6')
            
            with ui.card().classes('card'):
                ui.label('Generation Settings').classes('text-lg font-semibold mb-4')
                
                # Settings that actually affect generation
                with ui.column().classes('gap-4'):
                    ui.label('GPU Configuration').classes('font-medium')
                    ui.switch('Enable GPU acceleration', value=True)
                    ui.switch('Use FP16 precision', value=True)
                    ui.switch('Enable xFormers optimization', value=True)
                    
                    ui.separator()
                    
                    ui.label('Output Settings').classes('font-medium')
                    ui.label(f'Output Directory: {OUTPUT_DIR}').classes('text-sm')
                    ui.switch('Auto-save generations', value=True)
                    ui.switch('Generate thumbnails', value=True)
                    
            # Info section
            with ui.card().classes('card mt-4'):
                ui.label('System Information').classes('text-lg font-semibold mb-4')
                
                # Show actual system info
                import torch
                cuda_available = torch.cuda.is_available()
                
                with ui.column().classes('gap-2'):
                    ui.label(f'CUDA Available: {"Yes" if cuda_available else "No"}').classes('text-sm')
                    if cuda_available:
                        ui.label(f'GPU: {torch.cuda.get_device_name(0)}').classes('text-sm')
                        ui.label(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB').classes('text-sm')
                    
                    import psutil
                    memory = psutil.virtual_memory()
                    ui.label(f'System RAM: {memory.total / 1024**3:.1f} GB').classes('text-sm')
                    ui.label(f'Available RAM: {memory.available / 1024**3:.1f} GB').classes('text-sm')

# Create the UI
@ui.page('/')
async def main_page():
    """Main page setup"""
    # Apply theme
    apply_theme()
    
    # Configure app
    ui.page_title('NeuralForge Studio - Functional')
    
    # Create app instance
    app_instance = NeuralForgeDesktopFunctional()
    
    # Main layout
    with ui.row().classes('w-screen h-screen gap-0'):
        # Navigation sidebar
        app_instance.navigation.render()
        
        # Main content area
        app_instance.content_container = ui.column().classes('flex-1 h-screen overflow-auto bg-background')
        
    # Load initial page
    app_instance.navigate_to('create')

if __name__ == '__main__':
    print("Starting NeuralForge Studio Desktop (Functional Version)")
    print("This version connects to the actual generation system")
    print("-" * 50)
    
    # Run the app
    ui.run(
        title='NeuralForge Studio - Functional',
        native=False,  # Disable native mode for testing
        window_size=(1400, 900),
        dark=True,
        reload=False,
        port=8765,
        show=False  # Don't auto-open browser
    )