#!/usr/bin/env python3
"""
NeuralForge Studio Desktop Application Launcher
Ensures the app runs exclusively as a native desktop application
"""

import sys
import os
from pathlib import Path

def check_requirements():
    """Check if all requirements are met for running the desktop app"""
    errors = []
    
    # Check Python version
    if sys.version_info < (3, 10):
        errors.append("Python 3.10 or higher is required")
    
    # Check required modules
    required_modules = ['nicegui']
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            errors.append(f"Missing required module: {module}")
    
    # Check pywebview separately with better error message
    try:
        import pywebview
    except ImportError:
        errors.append("pywebview is required for native desktop mode. Install with: pip install pywebview")
    
    # Check display (for Linux)
    if sys.platform.startswith('linux'):
        if not os.environ.get('DISPLAY'):
            errors.append("No display found. Desktop apps require a display.")
    
    return errors

def main():
    """Launch NeuralForge Studio as a native desktop application"""
    # Check requirements
    errors = check_requirements()
    if errors:
        print("Cannot start NeuralForge Studio:")
        for error in errors:
            print(f"  ❌ {error}")
        print("\nPlease install missing dependencies:")
        print("  pip install nicegui pywebview")
        sys.exit(1)
    
    # Set environment for desktop-only mode
    os.environ['NICEGUI_NATIVE_MODE'] = '1'
    os.environ['NICEGUI_DISABLE_WEB'] = '1'
    
    # Import and run the desktop app
    try:
        from desktop.main import main as run_desktop
        print("🚀 Starting NeuralForge Studio...")
        print("   This is a native desktop application.")
        print("   The browser will NOT open.")
        run_desktop()
    except Exception as e:
        print(f"❌ Failed to start NeuralForge Studio: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()