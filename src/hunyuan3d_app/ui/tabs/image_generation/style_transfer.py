"""Style Transfer UI

Handles artistic style transfer using LoRA models.
"""

import gradio as gr
from typing import Any
import logging

from ...components.common import create_action_button

logger = logging.getLogger(__name__)


def create_style_transfer_subtab(app: Any) -> None:
    """Create style transfer interface using LoRAs"""
    
    with gr.Column():
        gr.Markdown("### Apply artistic styles using LoRA models")
        
        # Input image (optional for style transfer)
        input_image = gr.Image(
            label="Input Image (Optional)",
            type="pil",
            height=300
        )
        
        # Style prompt
        style_prompt = gr.Textbox(
            label="Style Description",
            placeholder="oil painting, van gogh style, impressionist...",
            lines=2
        )
        
        # LoRA selection
        available_loras = app.get_available_loras()
        
        with gr.Group():
            gr.Markdown("#### Select Style LoRAs")
            
            if available_loras:
                # Allow multiple LoRAs
                lora_1 = gr.Dropdown(
                    choices=[(l["name"], l["name"]) for l in available_loras],
                    label="Style LoRA 1",
                    value=None
                )
                lora_1_weight = gr.Slider(0, 2, 1, step=0.1, label="Weight 1")
                
                lora_2 = gr.Dropdown(
                    choices=[(l["name"], l["name"]) for l in available_loras],
                    label="Style LoRA 2 (Optional)",
                    value=None
                )
                lora_2_weight = gr.Slider(0, 2, 1, step=0.1, label="Weight 2")
            else:
                gr.Markdown("No LoRAs available. Download style LoRAs from Settings → Model Management")
                lora_1 = gr.State(None)
                lora_1_weight = gr.State(1.0)
                lora_2 = gr.State(None)
                lora_2_weight = gr.State(1.0)
        
        # Model selection
        downloaded_models = app.model_manager.get_downloaded_models("image")
        if downloaded_models:
            model_dropdown = gr.Dropdown(
                choices=downloaded_models,
                value=downloaded_models[0],
                label="Base Model"
            )
        else:
            model_dropdown = gr.Dropdown(
                choices=[],
                label="Model (No models downloaded)",
                interactive=False
            )
        
        # Generation settings
        with gr.Row():
            steps = gr.Slider(10, 100, 30, step=5, label="Steps")
            guidance = gr.Slider(1, 20, 7.5, step=0.5, label="Guidance")
            seed = gr.Number(-1, label="Seed")
        
        # Transfer button
        transfer_btn = create_action_button("🎨 Apply Style", variant="primary")
        
        # Output
        with gr.Row():
            output_image = gr.Image(label="Styled Image", type="pil")
            transfer_info = gr.HTML()
        
        # Placeholder for style transfer
        def apply_style_transfer(input_img, prompt, lora1, weight1, lora2, weight2, 
                               model, steps, guidance, seed):
            if not model:
                return None, "❌ No model selected"
            
            # TODO: Implement style transfer with LoRAs
            return None, "🚧 Style transfer feature coming soon!"
        
        transfer_btn.click(
            apply_style_transfer,
            inputs=[input_image, style_prompt, lora_1, lora_1_weight, 
                   lora_2, lora_2_weight, model_dropdown, steps, guidance, seed],
            outputs=[output_image, transfer_info]
        )