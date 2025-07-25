#!/usr/bin/env python3
"""
Headless main.py for HunYuan3D - runs without native window dependencies
"""

import sys
import os
from pathlib import Path

# Force headless mode
os.environ['NICEGUI_NATIVE'] = '0'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

def main():
    print("🚀 HunYuan3D Studio - Headless Mode")
    print("=" * 50)
    print()
    
    # Add paths
    project_root = Path(__file__).parent.resolve()
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "src"))
    
    # Import NiceGUI with headless settings
    from nicegui import ui
    from desktop.ui import apply_theme
    from desktop.ui.components import NavigationSidebar
    from desktop.ui.pages.create_page import CreatePage
    from core.config import OUTPUT_DIR
    
    @ui.page('/')
    def index():
        apply_theme()
        ui.page_title('NeuralForge Studio')
        
        with ui.row().classes('w-screen h-screen gap-0'):
            # Navigation
            nav = NavigationSidebar(on_navigate=lambda x: ui.notify(f"Navigate to {x}"))
            nav.render()
            
            # Content
            with ui.column().classes('flex-1 h-screen overflow-auto bg-background'):
                # Create page
                create_page = CreatePage(OUTPUT_DIR)
                create_page.render()
    
    print("📍 Starting server at http://localhost:8765")
    print("⌨️  Press Ctrl+C to stop")
    print()
    
    # Run without native window
    ui.run(
        host='127.0.0.1',
        port=8765,
        title='NeuralForge Studio',
        dark=True,
        reload=False,
        native=False,
        show=False
    )

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
        sys.exit(0)