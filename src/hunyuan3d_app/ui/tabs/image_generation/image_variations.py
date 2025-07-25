"""Image Variations UI

Handles generating multiple variations of a reference image.
"""

import gradio as gr
from typing import Any
import logging

from ...components.common import create_action_button

logger = logging.getLogger(__name__)


def create_image_variations_subtab(app: Any) -> None:
    """Create image variations interface"""
    
    with gr.Column():
        gr.Markdown("### Generate variations of an existing image")
        
        # Input image
        input_image = gr.Image(
            label="Reference Image",
            type="pil",
            height=300
        )
        
        # Variation settings
        with gr.Row():
            variation_strength = gr.Slider(
                0.1, 0.9, 0.3,
                step=0.05,
                label="Variation Strength",
                info="Lower = more similar, Higher = more different"
            )
            
            num_variations = gr.Slider(
                1, 4, 4,
                step=1,
                label="Number of Variations"
            )
        
        # Optional prompt for guided variations
        variation_prompt = gr.Textbox(
            label="Variation Guide (Optional)",
            placeholder="Make it more colorful, add flowers...",
            lines=2
        )
        
        # Model selection - create dropdown that will be populated dynamically
        model_dropdown = gr.Dropdown(
            label="Model",
            choices=[],  # Will be populated on load
            value=None
        )
        
        # Generate button
        generate_btn = create_action_button("🎲 Generate Variations", variant="primary")
        
        # Output gallery
        output_gallery = gr.Gallery(
            label="Variations",
            columns=2,
            rows=2,
            height="auto"
        )
        variations_info = gr.HTML()
        
        # Placeholder for variations
        def generate_variations(input_img, strength, num, prompt, model):
            if not input_img:
                return [], "❌ Please provide a reference image"
            if not model:
                return [], "❌ No model selected"
            
            # TODO: Implement image variations
            return [], "🚧 Image variations feature coming soon!"
        
        generate_btn.click(
            generate_variations,
            inputs=[input_image, variation_strength, num_variations, 
                   variation_prompt, model_dropdown],
            outputs=[output_gallery, variations_info]
        )