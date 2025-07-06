"""Main UI creation module."""

import gradio as gr
from pathlib import Path
from typing import Any, Optional

from .components.header import create_header
from .components.model_status import create_model_status
from .tabs.quick_generate import create_quick_generate_tab
from .tabs.manual_pipeline import create_manual_pipeline_tab
from .tabs.model_management import create_model_management_tab
from ..config import ALL_IMAGE_MODELS, HUNYUAN3D_MODELS, QUALITY_PRESETS


def load_custom_css() -> str:
    """Load custom CSS for the interface."""
    possible_paths = [
        Path("app_styles_minimal.css"),
        Path("src/app_styles_minimal.css"),
        Path(__file__).parent.parent.parent / "app_styles_minimal.css"
    ]
    
    for css_path in possible_paths:
        if css_path.exists():
            with open(css_path, 'r') as f:
                return f.read()
    return ""


def create_interface(app: Any) -> gr.Blocks:
    """Create the main Gradio interface.
    
    Args:
        app: Hunyuan3DStudio application instance
        
    Returns:
        Gradio Blocks interface
    """
    custom_css = load_custom_css()
    
    with gr.Blocks(
        title="Hunyuan3D Studio - Complete Pipeline",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
            neutral_hue="slate"
        ),
        css=custom_css
    ) as interface:
        # Header
        create_header()
        
        # Model status
        model_status = create_model_status(app)
        
        # Get initial model choices
        initial_image_choices, initial_hunyuan_choices, _, _ = app.get_model_selection_data()
        
        # Create shared dropdown components (used across tabs)
        manual_img_model = gr.Dropdown(
            choices=initial_image_choices,
            label="Model",
            value=initial_image_choices[0] if initial_image_choices else None,
            visible=False,
            interactive=True
        )
        manual_3d_model = gr.Dropdown(
            choices=initial_hunyuan_choices,
            label="Model",
            value=initial_hunyuan_choices[0] if initial_hunyuan_choices else None,
            visible=False,
            interactive=True
        )
        
        # Main tabs
        with gr.Tabs():
            # Quick Generate Tab
            with gr.Tab("🚀 Quick Generate"):
                create_quick_generate_tab(app)
            
            # Manual Pipeline Tab
            with gr.Tab("🔧 Manual Pipeline"):
                create_manual_pipeline_tab(
                    app,
                    manual_img_model,
                    manual_3d_model,
                    model_status
                )
            
            # Model Management Tab
            with gr.Tab("📦 Model Management"):
                create_model_management_tab(
                    app,
                    model_status,
                    manual_img_model,
                    manual_3d_model
                )
        
        # Footer
        gr.HTML("""
        <div style='text-align: center; margin-top: 20px; color: #666;'>
            <p>Powered by Hunyuan3D v2 Models | For research and creative use</p>
        </div>
        """)
    
    return interface