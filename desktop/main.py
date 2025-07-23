"""
NeuralForge Studio Desktop Application
Main entry point for the Nicegui-based desktop interface
"""

import sys
import os
import asyncio
import signal
import atexit
from pathlib import Path
from nicegui import ui, app, native

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
from desktop.ui.pages.settings import SettingsPage

class NeuralForgeDesktop:
    """Main desktop application class"""
    
    def __init__(self):
        # Core services
        self.model_manager = ModelManager(MODELS_DIR)
        
        # Pages - all implemented
        self.pages = {
            'create': CreatePage(OUTPUT_DIR, self.model_manager),
            'library': LibraryPage(OUTPUT_DIR),
            'lab': LabPage(OUTPUT_DIR, self.model_manager),
            'models': ModelsPage(MODELS_DIR, self.model_manager),
            'settings': SettingsPage()
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
        
        # Configure app for native desktop
        ui.page_title('NeuralForge Studio')
        
        # Native window configuration
        if app.native:
            app.native.window_args['resizable'] = True
            app.native.window_args['frameless'] = False
            app.native.window_args['easy_drag'] = False
            app.native.window_args['confirm_close'] = True
            app.native.start_args['debug'] = False
        
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
                self.pages['settings'].render()
                
    def _render_placeholder(self, title: str, description: str):
        """Render a placeholder page"""
        with ui.column().classes('w-full h-full items-center justify-center'):
            ui.icon('construction', size='4rem').classes('text-gray-600 mb-4')
            ui.label(title).classes('text-2xl font-bold mb-2')
            ui.label(description).classes('text-gray-500')
            

@ui.page('/')
async def main_page():
    """Main page setup"""
    app_instance = NeuralForgeDesktop()
    await app_instance.initialize()

def main():
    """Main application entry point"""
    # Set up signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        """Handle shutdown signals gracefully"""
        print("\nShutting down NeuralForge Studio...")
        # Don't raise the signal, just exit cleanly
        os._exit(0)
    
    # Register signal handlers
    try:
        signal.signal(signal.SIGINT, signal_handler)
        if sys.platform != 'win32':
            signal.signal(signal.SIGTERM, signal_handler)
    except:
        pass
    
    # Clean up on exit
    def cleanup():
        """Clean up resources on exit"""
        try:
            # Any cleanup code here
            pass
        except:
            pass
    
    atexit.register(cleanup)
    
    # Run the app with error handling
    try:
        ui.run(
            title='NeuralForge Studio',
            native=True,
            window_size=(1400, 900),
            dark=True,
            reload=False,
            port=8765  # Different port from Gradio app
        )
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError running application: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()