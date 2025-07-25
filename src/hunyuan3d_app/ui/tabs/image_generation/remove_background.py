"""Background Removal UI

Handles removing backgrounds from images with various output options.
"""

import gradio as gr
from typing import Any
import logging
from PIL import Image

from ...components.common import create_action_button

logger = logging.getLogger(__name__)


def create_remove_background_subtab(app: Any) -> None:
    """Create background removal interface"""
    
    with gr.Column():
        gr.Markdown("### Remove background from any image")
        
        # Input image
        input_image = gr.Image(
            label="Input Image",
            type="pil",
            height=400
        )
        
        # Options
        with gr.Row():
            output_format = gr.Radio(
                choices=["PNG (Transparent)", "White Background", "Custom Color"],
                value="PNG (Transparent)",
                label="Output Format"
            )
            
            bg_color = gr.ColorPicker(
                label="Background Color",
                value="#FFFFFF",
                visible=False
            )
        
        # Show color picker when custom selected
        output_format.change(
            lambda fmt: gr.update(visible=fmt == "Custom Color"),
            inputs=[output_format],
            outputs=[bg_color]
        )
        
        # Remove button
        remove_btn = create_action_button("🎯 Remove Background", variant="primary")
        
        # Output
        with gr.Row():
            output_image = gr.Image(label="Result", type="pil")
            remove_info = gr.HTML()
        
        # Wire up background removal
        def remove_background(input_img, format, color):
            if not input_img:
                return None, "❌ Please provide an input image"
            
            try:
                # Use existing background remover
                result = app.image_generator.remove_background(input_img)
                
                if format == "White Background":
                    # Convert to RGB with white background
                    bg = Image.new('RGB', result.size, (255, 255, 255))
                    bg.paste(result, mask=result.split()[-1])
                    result = bg
                elif format == "Custom Color":
                    # Convert to RGB with custom color
                    rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
                    bg = Image.new('RGB', result.size, rgb)
                    bg.paste(result, mask=result.split()[-1])
                    result = bg
                
                return result, "✅ Background removed successfully!"
                
            except Exception as e:
                logger.error(f"Background removal error: {e}")
                return None, f"❌ Error: {str(e)}"
        
        remove_btn.click(
            remove_background,
            inputs=[input_image, output_format, bg_color],
            outputs=[output_image, remove_info]
        )