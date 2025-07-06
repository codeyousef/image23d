"""Quick Generate tab for the UI."""

import gradio as gr
from typing import Any

from ..components.common import (
    create_generation_settings,
    create_output_display,
    create_model_selector,
    create_action_button
)
from ...config import ALL_IMAGE_MODELS, HUNYUAN3D_MODELS, QUALITY_PRESETS


def create_quick_generate_tab(app: Any):
    """Create the Quick Generate tab.
    
    Args:
        app: Application instance
    """
    gr.Markdown("""
    ### Complete Pipeline: Text → Image → 3D
    Generate stunning 3D models from text descriptions using the best available models.
    """)
    
    with gr.Row():
        with gr.Column(scale=3):
            # Text prompt
            with gr.Group():
                gr.Markdown("### 📝 Step 1: Describe Your Object")
                prompt = gr.Textbox(
                    label="What would you like to create?",
                    placeholder="A majestic golden crown with intricate details and gemstones...",
                    lines=3
                )
                negative_prompt = gr.Textbox(
                    label="What to avoid",
                    value="blurry, low quality, multiple objects, busy background",
                    lines=2
                )
            
            # Model selection
            with gr.Group():
                gr.Markdown("### 🤖 Step 2: Choose Models")
                
                with gr.Row():
                    # Get model choices
                    img_choices, hunyuan_choices, _, _ = app.get_model_selection_data()
                    
                    image_model = create_model_selector(
                        label="Image Model",
                        choices=img_choices,
                        info="Model for generating the initial image"
                    )
                    
                    hunyuan_model = create_model_selector(
                        label="3D Model",
                        choices=hunyuan_choices,
                        info="Hunyuan3D variant for 3D conversion"
                    )
            
            # Quality settings
            with gr.Group():
                gr.Markdown("### ⚙️ Step 3: Quality Settings")
                
                quality_preset = gr.Radio(
                    choices=list(QUALITY_PRESETS.keys()),
                    value="standard",
                    label="Quality Preset",
                    info="Balance between quality and speed"
                )
                
                # Generation settings
                seed, steps, cfg, width, height = create_generation_settings()
            
            # Generate button
            generate_btn = create_action_button(
                label="🎨 Generate 3D Model",
                variant="primary",
                size="lg"
            )
        
        with gr.Column(scale=2):
            # Output display
            output_image, output_3d, output_info = create_output_display()
    
    # Examples
    with gr.Row():
        gr.Examples(
            examples=[
                ["A detailed ancient Greek marble statue of a philosopher", "modern, contemporary, abstract"],
                ["A ornate Victorian tea set with gold trim and floral patterns", "broken, damaged, incomplete"],
                ["A futuristic robot companion with sleek design", "retro, old-fashioned, rusty"],
                ["A magical crystal orb glowing with inner light", "dull, opaque, ordinary"],
                ["A medieval knight's helmet with intricate engravings", "plain, simple, modern"]
            ],
            inputs=[prompt, negative_prompt],
            label="Example Prompts"
        )
    
    # Wire up the generation
    generate_btn.click(
        fn=app.quick_generate,
        inputs=[
            prompt, negative_prompt, image_model, hunyuan_model,
            quality_preset, seed, steps, cfg, width, height
        ],
        outputs=[output_image, output_3d, output_info]
    )