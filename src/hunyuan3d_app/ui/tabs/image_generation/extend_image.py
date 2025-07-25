"""Image Extension UI (Outpainting)

Handles extending images beyond their original borders.
"""

import gradio as gr
from typing import Any
import logging

from ...components.common import create_action_button

logger = logging.getLogger(__name__)


def create_extend_image_subtab(app: Any) -> None:
    """Create image extension/outpainting interface"""
    
    with gr.Column():
        gr.Markdown("### Extend images beyond their borders")
        
        # Input image
        input_image = gr.Image(
            label="Input Image",
            type="pil",
            height=300
        )
        
        # Extension settings
        with gr.Row():
            extend_direction = gr.CheckboxGroup(
                choices=["Top", "Bottom", "Left", "Right"],
                value=["Bottom", "Right"],
                label="Extend Directions"
            )
            
            extend_size = gr.Slider(
                64, 512, 256,
                step=64,
                label="Extension Size (pixels)"
            )
        
        # Extension prompt
        extend_prompt = gr.Textbox(
            label="Extension Description",
            placeholder="Continue the landscape, add more sky...",
            lines=2
        )
        
        # Model selection - create dropdown that will be populated dynamically
        model_dropdown = gr.Dropdown(
            label="Model",
            choices=[],  # Will be populated on load
            value=None
        )
        
        # Extend button
        extend_btn = create_action_button("↔️ Extend Image", variant="primary")
        
        # Output
        with gr.Row():
            output_image = gr.Image(label="Extended Image", type="pil")
            extend_info = gr.HTML()
        
        # Placeholder for outpainting
        def extend_image(input_img, directions, size, prompt, model):
            if not input_img:
                return None, "❌ Please provide an input image"
            if not directions:
                return None, "❌ Please select at least one direction"
            if not model:
                return None, "❌ No model selected"
            
            # TODO: Implement outpainting
            return None, "🚧 Image extension feature coming soon!"
        
        extend_btn.click(
            extend_image,
            inputs=[input_image, extend_direction, extend_size, 
                   extend_prompt, model_dropdown],
            outputs=[output_image, extend_info]
        )