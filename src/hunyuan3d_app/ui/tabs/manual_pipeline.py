"""Manual Pipeline tab for step-by-step generation."""

import gradio as gr
from typing import Any

from ..components.common import (
    create_generation_settings,
    create_output_display,
    create_action_button
)


def create_manual_pipeline_tab(app: Any, manual_img_model: gr.Dropdown, manual_3d_model: gr.Dropdown, model_status: gr.HTML):
    """Create the Manual Pipeline tab.
    
    Args:
        app: Application instance
        manual_img_model: Image model dropdown
        manual_3d_model: 3D model dropdown
        model_status: Model status HTML component
    """
    gr.Markdown("""
    ### Step-by-Step Pipeline Control
    Generate images and convert to 3D with full control over each step.
    """)
    
    with gr.Tabs() as pipeline_tabs:
        # Image Generation Tab
        with gr.Tab("1️⃣ Image Generation"):
            with gr.Row():
                with gr.Column(scale=3):
                    # Text prompts
                    with gr.Group():
                        img_prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="Describe what you want to generate...",
                            lines=3
                        )
                        img_negative = gr.Textbox(
                            label="Negative Prompt",
                            value="blurry, low quality, distorted",
                            lines=2
                        )
                    
                    # Model and settings
                    with gr.Row():
                        manual_img_model.visible = True
                        manual_img_model.label = "Image Model"
                    
                    # Generation settings
                    img_seed, img_steps, img_cfg, img_width, img_height = create_generation_settings()
                    
                    # Control buttons
                    with gr.Row():
                        img_generate_btn = create_action_button(
                            label="🎨 Generate Image",
                            variant="primary"
                        )
                        img_stop_btn = create_action_button(
                            label="⏹️ Stop",
                            variant="stop",
                            size="sm"
                        )
                
                with gr.Column(scale=2):
                    manual_img_output = gr.Image(
                        label="Generated Image",
                        type="pil"
                    )
                    manual_img_info = gr.HTML()
        
        # 3D Conversion Tab
        with gr.Tab("2️⃣ 3D Conversion"):
            with gr.Row():
                with gr.Column(scale=3):
                    # Input image
                    with gr.Group():
                        gr.Markdown("### Input Image")
                        input_image = gr.Image(
                            label="Image to Convert",
                            type="pil",
                            height=300
                        )
                        gr.Markdown("💡 **Tip**: Use the generated image from Step 1 or upload your own")
                    
                    # Model and settings
                    with gr.Row():
                        manual_3d_model.visible = True
                        manual_3d_model.label = "Hunyuan3D Model"
                    
                    with gr.Row():
                        remove_bg = gr.Checkbox(
                            label="Remove Background",
                            value=True,
                            info="Automatically remove background for better results"
                        )
                        mesh_format = gr.Dropdown(
                            choices=["glb", "obj", "ply"],
                            value="glb",
                            label="Output Format"
                        )
                    
                    # Convert button
                    convert_btn = create_action_button(
                        label="🔮 Convert to 3D",
                        variant="primary"
                    )
                
                with gr.Column(scale=2):
                    manual_3d_output = gr.Model3D(
                        label="3D Model",
                        height=400
                    )
                    manual_3d_info = gr.HTML()
    
    # Wire up the manual pipeline
    # Image generation
    img_generate_btn.click(
        fn=app.generate_image,
        inputs=[
            img_prompt, img_negative, manual_img_model,
            img_seed, img_steps, img_cfg, img_width, img_height
        ],
        outputs=[manual_img_output, manual_img_info]
    )
    
    img_stop_btn.click(
        fn=app.stop_image_generation,
        outputs=[manual_img_info]
    )
    
    # Copy generated image to 3D input
    manual_img_output.change(
        fn=lambda img: img,
        inputs=[manual_img_output],
        outputs=[input_image]
    )
    
    # 3D conversion
    convert_btn.click(
        fn=app.convert_to_3d,
        inputs=[input_image, manual_3d_model, remove_bg, mesh_format],
        outputs=[manual_3d_output, manual_3d_info]
    )
    
    # Update model status after operations
    for btn in [img_generate_btn, convert_btn]:
        btn.click(
            fn=app.get_model_status,
            outputs=[model_status]
        )