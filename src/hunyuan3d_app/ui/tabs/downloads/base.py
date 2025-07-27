"""Main downloads manager tab creation"""

import gradio as gr
from typing import TYPE_CHECKING

from .status_display import (
    create_status_display_component,
    create_control_buttons
)
from .model_cards import (
    create_model_category_section,
    create_quick_download_button
)
from .pipeline_components import (
    create_texture_warning_banner,
    create_pipeline_components_tab,
    check_texture_components_status
)
from .model_hub_search import create_model_hub_search_tab
from .advanced_features import create_advanced_features_tab

if TYPE_CHECKING:
    from ....app import HunyuanApp


def create_downloads_manager_tab(app: "HunyuanApp"):
    """Create downloads manager tab with concurrent downloads and resume support"""
    
    # Get texture components status
    realesrgan_installed, xatlas_installed, dinov2_installed = check_texture_components_status()
    
    # Show warning banner if texture components missing
    create_texture_warning_banner()
    
    # Add quick download texture components button if needed
    if not realesrgan_installed or not xatlas_installed or not dinov2_installed:
        with gr.Row():
            quick_texture_btn, texture_progress = create_quick_download_button(
                app,
                "🎨 Download Texture Components",
                [("realesrgan", "texture_components"), ("xatlas", "texture_components"), ("dinov2", "texture_components")],
                priority=3  # High priority
            )
    
    # Status display
    status_display = create_status_display_component(app)
    
    # Control buttons
    with gr.Row():
        refresh_btn, stop_all_btn, refresh_status, stop_all_downloads = create_control_buttons(app)
    
    # Set up auto-refresh for status
    refresh_btn.click(refresh_status, outputs=[status_display])
    stop_all_btn.click(stop_all_downloads, outputs=[gr.Textbox(visible=False), status_display])
    
    # Create tabs for different download categories
    with gr.Tabs():
        create_quick_start_tab(app)
        create_3d_models_tab(app)
        create_image_models_tab(app)
        create_video_models_tab(app)
        create_pipeline_components_tab(app)
        create_advanced_features_tab(app)
        create_model_hub_search_tab(app)
    
    # Auto-refresh status every 2 seconds when downloads are active
    status_display.change(
        lambda x: gr.update(),
        inputs=[status_display],
        outputs=[status_display],
        every=2
    )


def create_quick_start_tab(app: "HunyuanApp"):
    """Create quick start tab with preset downloads"""
    with gr.Tab("🚀 Quick Start"):
        gr.Markdown("### 🚀 Quick Start Downloads")
        gr.Markdown("Download recommended model combinations with a single click")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### 💡 Minimal Setup")
                gr.Markdown("For basic 3D generation (8GB VRAM)")
                minimal_btn, minimal_progress = create_quick_download_button(
                    app,
                    "Download Minimal Setup",
                    [
                        ("hunyuan3d-2mini", "3d"),
                        ("sd-turbo", "image"),
                        ("realesrgan", "texture_components"),
                        ("xatlas", "texture_components")
                    ],
                    priority=2
                )
                
            with gr.Column():
                gr.Markdown("#### ⭐ Recommended Setup")
                gr.Markdown("Balanced quality and performance (12GB VRAM)")
                recommended_btn, recommended_progress = create_quick_download_button(
                    app,
                    "Download Recommended Setup",
                    [
                        ("hunyuan3d-20", "3d"),
                        ("sdxl-turbo", "image"),
                        ("ip-adapter-plus_sd15", "ip_adapter"),
                        ("realesrgan", "texture_components"),
                        ("xatlas", "texture_components"),
                        ("dinov2", "texture_components")
                    ],
                    priority=2
                )
                
            with gr.Column():
                gr.Markdown("#### 🏆 Ultimate Setup")
                gr.Markdown("Maximum quality (16GB+ VRAM)")
                ultimate_btn, ultimate_progress = create_quick_download_button(
                    app,
                    "Download Ultimate Setup",
                    [
                        ("hunyuan3d-21", "3d"),
                        ("flux-schnell", "image"),
                        ("ip-adapter-plus_sdxl_vit-h", "ip_adapter"),
                        ("inswapper_128", "face_swap"),
                        ("codeformer", "face_restore"),
                        ("realesrgan", "texture_components"),
                        ("xatlas", "texture_components"),
                        ("dinov2", "texture_components")
                    ],
                    priority=2
                )


def create_3d_models_tab(app: "HunyuanApp"):
    """Create 3D models download tab"""
    with gr.Tab("🎯 3D Models"):
        from ....config import HUNYUAN3D_MODELS
        create_model_category_section(app, "🎯 HunYuan3D Models", HUNYUAN3D_MODELS, "3d")


def create_image_models_tab(app: "HunyuanApp"):
    """Create image models download tab"""
    with gr.Tab("🖼️ Image Models"):
        from ....config import IMAGE_MODELS
        create_model_category_section(app, "🖼️ Text-to-Image Models", IMAGE_MODELS, "image")


def create_video_models_tab(app: "HunyuanApp"):
    """Create video models download tab"""
    with gr.Tab("🎬 Video Models"):
        from ....config import VIDEO_MODELS
        create_model_category_section(app, "🎬 Text-to-Video Models", VIDEO_MODELS, "video")