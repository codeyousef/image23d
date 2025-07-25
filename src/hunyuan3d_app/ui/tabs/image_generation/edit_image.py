"""Image Editing UI (img2img)

Handles image-to-image editing with AI guidance.
"""

import gradio as gr
from typing import Any
import logging

from ...components.common import create_action_button

logger = logging.getLogger(__name__)


def create_edit_image_subtab(app: Any) -> None:
    """Create image editing interface (img2img)"""
    
    with gr.Column():
        gr.Markdown("### Edit existing images with AI guidance")
        
        # Input image
        input_image = gr.Image(
            label="Input Image",
            type="pil",
            height=300
        )
        
        # Editing prompt
        edit_prompt = gr.Textbox(
            label="Edit Instructions",
            placeholder="Change the background to a sunset...",
            lines=2
        )
        
        # Strength slider
        edit_strength = gr.Slider(
            0.1, 1.0, 0.5,
            step=0.05,
            label="Edit Strength",
            info="Lower = more similar to original, Higher = more changes"
        )
        
        # Model selection - create dropdown that will be populated dynamically
        model_dropdown = gr.Dropdown(
            label="Model",
            choices=[],  # Will be populated on load
            value=None
        )
        
        # Basic settings
        with gr.Row():
            steps = gr.Slider(10, 100, 30, step=5, label="Steps")
            guidance = gr.Slider(1, 20, 7.5, step=0.5, label="Guidance")
            seed = gr.Number(-1, label="Seed")
        
        # Edit button
        edit_btn = create_action_button("✏️ Edit Image", variant="primary")
        
        # Output
        with gr.Row():
            output_image = gr.Image(label="Edited Image", type="pil")
            edit_info = gr.HTML()
        
        # Placeholder for img2img functionality
        def edit_image(input_img, prompt, strength, model, steps, guidance, seed):
            if not input_img:
                return None, "❌ Please provide an input image"
            if not model:
                return None, "❌ No model selected"
            
            # TODO: Implement img2img pipeline
            return None, "🚧 Image editing feature coming soon!"
        
        edit_btn.click(
            edit_image,
            inputs=[input_image, edit_prompt, edit_strength, model_dropdown, steps, guidance, seed],
            outputs=[output_image, edit_info]
        )