"""Image Generation Tab Module

This module contains all image generation features split into focused submodules.
"""

import gradio as gr
from typing import Any
import logging

from ...components.media_sidebar import create_media_sidebar

logger = logging.getLogger(__name__)


def create_image_generation_tab(app: Any) -> None:
    """Create the unified image generation tab with all image-related features"""
    
    # Pre-fetch models to ensure they're available
    try:
        downloaded_models = app.model_manager.get_downloaded_models("image")
        logger.info(f"Pre-fetched image models: {downloaded_models}")
    except Exception as e:
        logger.error(f"Error pre-fetching models: {e}")
        downloaded_models = []
    
    # Store models in app for access by subtabs
    app._cached_image_models = downloaded_models
    
    with gr.Row():
        # Main content area
        with gr.Column(scale=4):
            gr.Markdown("## 🎨 Image Generation Studio")
            gr.Markdown("Create, edit, and enhance images with AI-powered tools")
            
            # Feature cards in a grid
            with gr.Tabs() as feature_tabs:
                # Import submodules here to avoid circular imports
                from .generate_image import create_generate_image_subtab
                from .edit_image import create_edit_image_subtab
                from .remove_background import create_remove_background_subtab
                from .upscale_image import create_upscale_image_subtab
                from .style_transfer import create_style_transfer_subtab
                from .image_variations import create_image_variations_subtab
                from .extend_image import create_extend_image_subtab
                from .fix_image import create_fix_image_subtab
                from .face_swap import create_face_swap_subtab
                
                # 1. Generate Image
                with gr.Tab("Generate Image", elem_id="generate_image_tab"):
                    create_generate_image_subtab(app)
                
                # 2. Edit Image
                with gr.Tab("Edit Image", elem_id="edit_image_tab"):
                    create_edit_image_subtab(app)
                
                # 3. Remove Background
                with gr.Tab("Remove Background", elem_id="remove_bg_tab"):
                    create_remove_background_subtab(app)
                
                # 4. Upscale Image
                with gr.Tab("Upscale Image", elem_id="upscale_tab"):
                    create_upscale_image_subtab(app)
                
                # 5. Style Transfer
                with gr.Tab("Style Transfer", elem_id="style_transfer_tab"):
                    create_style_transfer_subtab(app)
                
                # 6. Image Variations
                with gr.Tab("Image Variations", elem_id="variations_tab"):
                    create_image_variations_subtab(app)
                
                # 7. Extend Image
                with gr.Tab("Extend Image", elem_id="extend_tab"):
                    create_extend_image_subtab(app)
                
                # 8. Fix Image
                with gr.Tab("Fix Image", elem_id="fix_image_tab"):
                    create_fix_image_subtab(app)
                
                # 9. Face Swap
                with gr.Tab("Face Swap", elem_id="face_swap_tab"):
                    create_face_swap_subtab(app)
        
        # Sidebar with recent images
        sidebar = create_media_sidebar(
            app,
            media_type="image",
            on_select_callback=None,  # Will implement selection handling
            title="Recent Images"
        )


# Re-export for backward compatibility
__all__ = ["create_image_generation_tab"]