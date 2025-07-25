"""
Simplified NeuralForge Studio Desktop Application
This version works without complex async initialization
"""

import sys
import os
from pathlib import Path
from nicegui import ui, app

def setup_python_path():
    """Setup Python path for imports"""
    # Get current directory and ensure we're working from the right place
    current_dir = Path(__file__).parent.resolve()
    os.chdir(current_dir)
    
    # Add current directory to path
    current_dir_str = str(current_dir)
    if current_dir_str not in sys.path:
        sys.path.insert(0, current_dir_str)
    
    # Add src directory to path for HunYuan3D imports
    src_path = current_dir / "src"
    if src_path.exists():
        src_path_str = str(src_path)
        if src_path_str not in sys.path:
            sys.path.insert(0, src_path_str)

# Setup path before any other imports
setup_python_path()

from desktop.ui import apply_theme
from desktop.ui.components import NavigationSidebar
from desktop.ui.components import EnhancementPanel, ProgressPipeline, PipelineStep

class SimplifiedDesktop:
    """Simplified desktop application"""
    
    def __init__(self):
        self.current_page = 'create'
        self.navigation = NavigationSidebar(on_navigate=self.navigate_to)
        self.content_container = None
        
    def navigate_to(self, page_id: str):
        """Navigate to a page"""
        self.current_page = page_id
        
        if self.content_container:
            self.content_container.clear()
            
            with self.content_container:
                if page_id == 'create':
                    self.render_create_page()
                elif page_id == 'library':
                    self._render_placeholder('Library', 'Your generated assets will appear here')
                elif page_id == 'lab':
                    self._render_placeholder('Lab', 'Advanced tools and experiments')
                elif page_id == 'models':
                    self._render_placeholder('Models', 'Download and manage AI models')
                elif page_id == 'settings':
                    self._render_settings()
                    
    def render_create_page(self):
        """Simplified create page"""
        with ui.column().classes('w-full h-full p-6'):
            # Header
            ui.label('Create').classes('text-2xl font-bold mb-6')
            
            # Main content
            with ui.row().classes('gap-6 w-full'):
                # Left: Input
                with ui.column().classes('flex-1'):
                    with ui.card().classes('card'):
                        # Prompt input
                        ui.textarea(
                            label='Prompt',
                            placeholder='Describe what you want to create...',
                            value=''
                        ).classes('w-full').props('rows=3')
                        
                        # Model selector
                        ui.select(
                            label='Model',
                            options=['flux-1-dev', 'hunyuan3d-2.1', 'stable-video'],
                            value='flux-1-dev'
                        ).classes('w-full mt-4')
                        
                        # Generate button
                        ui.button(
                            'Generate',
                            icon='auto_awesome',
                            on_click=lambda: ui.notify('Generation would start here!', type='info')
                        ).props('unelevated').classes('w-full mt-6')
                        
                # Right: Enhancement
                with ui.column().classes('w-96'):
                    # Enhancement panel
                    enhancement_panel = EnhancementPanel()
                    enhancement_panel.render()
                    
                    ui.space().classes('h-4')
                    
                    # Progress pipeline
                    with ui.card().classes('progress-pipeline'):
                        ui.label('Generation Pipeline').classes('text-lg font-semibold mb-4')
                        ui.label('Progress will appear here during generation').classes('text-sm text-gray-500')
                        
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
                ui.label('Application Settings').classes('text-lg font-semibold mb-4')
                ui.switch('Enable GPU acceleration', value=True)
                ui.switch('Use prompt enhancement', value=True)
                
                ui.separator().classes('my-4')
                
                ui.label('About').classes('font-medium mb-2')
                ui.label('NeuralForge Studio v1.0.0').classes('text-sm text-gray-500')

# Create the UI
@ui.page('/')
def main_page():
    """Main page setup"""
    # Apply theme
    apply_theme()
    
    # Configure app
    ui.page_title('NeuralForge Studio')
    
    # Create app instance
    app_instance = SimplifiedDesktop()
    
    # Main layout
    with ui.row().classes('w-screen h-screen gap-0'):
        # Navigation sidebar
        app_instance.navigation.render()
        
        # Main content area
        app_instance.content_container = ui.column().classes('flex-1 h-screen overflow-auto bg-background')
        
    # Load initial page
    app_instance.navigate_to('create')

if __name__ == '__main__':
    # Run the app
    print("🚀 Starting NeuralForge Studio Desktop App...")
    ui.run(
        title='NeuralForge Studio',
        native=True,  # Desktop mode as originally intended
        window_size=(1400, 900),
        dark=True,
        reload=False,
        port=8765
    )