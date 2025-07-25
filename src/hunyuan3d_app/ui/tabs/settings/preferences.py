"""User Preferences UI

Handles user preferences including UI settings, defaults, file handling, and privacy.
"""

import gradio as gr
from typing import Any
import logging

from ...components.common import create_action_button

logger = logging.getLogger(__name__)


def create_preferences_subtab(app: Any) -> None:
    """User preferences and UI settings"""
    
    gr.Markdown("### Preferences")
    
    # UI preferences
    with gr.Group():
        gr.Markdown("#### User Interface")
        
        with gr.Row():
            ui_theme = gr.Dropdown(
                choices=["Light", "Dark", "Auto"],
                value="Auto",
                label="Theme"
            )
            
            ui_scale = gr.Slider(
                0.8, 1.2, 1.0,
                step=0.05,
                label="UI Scale"
            )
        
        show_tips = gr.Checkbox(
            label="Show helpful tips",
            value=True
        )
        
        auto_save = gr.Checkbox(
            label="Auto-save preferences",
            value=True
        )
    
    # Default generation settings
    with gr.Group():
        gr.Markdown("#### Default Generation Settings")
        
        with gr.Row():
            default_image_size = gr.Dropdown(
                choices=["512x512", "768x768", "1024x1024", "1536x1536"],
                value="1024x1024",
                label="Default Image Size"
            )
            
            default_steps = gr.Slider(
                10, 100, 30,
                step=5,
                label="Default Steps"
            )
        
        with gr.Row():
            default_guidance = gr.Slider(
                1, 20, 7.5,
                step=0.5,
                label="Default Guidance Scale"
            )
            
            default_sampler = gr.Dropdown(
                choices=["Euler", "Euler a", "DPM++ 2M", "DPM++ SDE"],
                value="Euler a",
                label="Default Sampler"
            )
    
    # File handling
    with gr.Group():
        gr.Markdown("#### File Handling")
        
        output_format = gr.Dropdown(
            choices=["PNG", "JPEG", "WEBP"],
            value="PNG",
            label="Default Image Format"
        )
        
        with gr.Row():
            auto_download = gr.Checkbox(
                label="Auto-download generated files",
                value=False
            )
            
            organize_by_date = gr.Checkbox(
                label="Organize outputs by date",
                value=True
            )
        
        output_directory = gr.Textbox(
            label="Output Directory",
            value=str(app.output_dir) if hasattr(app, 'output_dir') else "outputs"
        )
    
    # Privacy settings
    with gr.Group():
        gr.Markdown("#### Privacy & Data")
        
        analytics = gr.Checkbox(
            label="Share anonymous usage statistics",
            value=False
        )
        
        history_retention = gr.Slider(
            7, 365, 30,
            step=1,
            label="History retention (days)"
        )
        
        clear_on_exit = gr.Checkbox(
            label="Clear temporary files on exit",
            value=True
        )
    
    # Save preferences
    with gr.Row():
        save_prefs_btn = create_action_button("💾 Save Preferences", variant="primary")
        reset_prefs_btn = create_action_button("🔄 Reset to Defaults", size="sm")
        export_prefs_btn = create_action_button("📤 Export Settings", size="sm")
    
    prefs_status = gr.HTML()