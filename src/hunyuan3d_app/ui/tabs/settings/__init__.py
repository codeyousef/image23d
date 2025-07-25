"""Settings Tab Module

This module contains all settings-related UI components split into focused submodules.
"""

import gradio as gr
from typing import Any
import logging

logger = logging.getLogger(__name__)


def create_settings_tab(app: Any) -> None:
    """Create the consolidated settings tab with all configuration options"""
    
    gr.Markdown("## ⚙️ Settings & Configuration")
    gr.Markdown("Manage models, credentials, performance settings, and system configuration")
    
    with gr.Tabs() as settings_tabs:
        # Import submodules here to avoid circular imports
        from .model_management import create_model_management_subtab
        from .credentials import create_credentials_subtab
        from .performance import create_performance_subtab
        from .queue_management import create_queue_management_subtab
        from .system_info import create_system_info_subtab
        from .preferences import create_preferences_subtab
        
        # 1. Model Management (from Downloads Manager)
        with gr.Tab("Model Management", elem_id="model_management_tab"):
            create_model_management_subtab(app)
        
        # 2. API Credentials
        with gr.Tab("API Credentials", elem_id="credentials_tab"):
            create_credentials_subtab(app)
        
        # 3. Performance Settings
        with gr.Tab("Performance", elem_id="performance_tab"):
            create_performance_subtab(app)
        
        # 4. Queue Management
        with gr.Tab("Queue Management", elem_id="queue_tab"):
            create_queue_management_subtab(app)
        
        # 5. System Info & Testing
        with gr.Tab("System Info", elem_id="system_tab"):
            create_system_info_subtab(app)
        
        # 6. Preferences
        with gr.Tab("Preferences", elem_id="preferences_tab"):
            create_preferences_subtab(app)


# Re-export for backward compatibility
__all__ = ["create_settings_tab"]