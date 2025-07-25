"""Basic Image Generation UI

Handles text-to-image generation with model and LoRA support.
"""

import gradio as gr
from pathlib import Path
from typing import Any, Optional
import logging
import uuid

from ...components.common import create_generation_settings, create_action_button

logger = logging.getLogger(__name__)


def create_generate_image_subtab(app: Any) -> None:
    """Create the main image generation interface"""
    
    with gr.Column():
        # Prompt inputs
        prompt = gr.Textbox(
            label="Prompt",
            placeholder="Describe what you want to create...",
            lines=3
        )
        
        negative_prompt = gr.Textbox(
            label="Negative Prompt",
            value="blurry, low quality, distorted",
            lines=2
        )
        
        # Model selection - use cached models
        with gr.Row():
            # Use cached models from parent tab
            cached_models = getattr(app, '_cached_image_models', [])
            
            # Log for debugging
            logger.info(f"Creating dropdown with models: {cached_models}")
            
            model_dropdown = gr.Dropdown(
                label="Model" if cached_models else "Model (No models downloaded - visit Model Management)",
                choices=cached_models,
                value=cached_models[0] if cached_models else None,
                interactive=bool(cached_models)
            )
            
            # Add refresh button
            refresh_btn = gr.Button("🔄 Refresh", scale=0)
            
            # LoRA selection (if available)
            available_loras = app.get_available_loras()
            if available_loras:
                lora_dropdown = gr.Dropdown(
                    choices=[(l["name"], l["name"]) for l in available_loras],
                    label="LoRA (Optional)",
                    value=None
                )
                lora_weight = gr.Slider(0, 2, 1, step=0.1, label="LoRA Weight", visible=False)
                
                # Show weight slider when LoRA selected
                lora_dropdown.change(
                    lambda x: gr.update(visible=x is not None),
                    inputs=[lora_dropdown],
                    outputs=[lora_weight]
                )
            else:
                lora_dropdown = None
                lora_weight = None
        
        # Generation settings
        seed, steps, cfg, width, height = create_generation_settings()
        
        # Generate button
        generate_btn = create_action_button("🎨 Generate Image", variant="primary")
        
        # Output
        with gr.Row():
            output_image = gr.Image(label="Generated Image", type="pil")
            generation_info = gr.HTML()
        
        # Wire up generation
        def generate_image(prompt, negative_prompt, model, lora, lora_weight, 
                         seed, steps, cfg, width, height, progress=gr.Progress()):
            """Generate image with selected settings"""
            
            if not model:
                return None, "❌ No model selected"
            
            try:
                # Load model if needed
                progress(0.1, "Loading model...")
                if app.image_model_name != model:
                    status, loaded_model, model_name = app.model_manager.load_image_model(
                        model, app.image_model, app.image_model_name, "cuda", progress
                    )
                    if "❌" in status:
                        return None, f"Failed to load model: {status}"
                    app.image_model = loaded_model
                    app.image_model_name = model_name
                
                # Apply LoRA if selected
                lora_configs = []
                if lora and lora_weight and hasattr(app, 'lora_manager'):
                    progress(0.2, "Applying LoRA...")
                    from ....features.lora.manager import LoRAInfo
                    lora_info = next((l for l in available_loras if l["name"] == lora), None)
                    if lora_info:
                        lora_obj = LoRAInfo(
                            name=lora_info["name"],
                            path=Path(lora_info["path"]),
                            base_model=lora_info["base_model"],
                            trigger_words=lora_info["trigger_words"]
                        )
                        lora_configs.append((lora_obj, lora_weight))
                        app.lora_manager.apply_multiple_loras(app.image_model, lora_configs)
                
                # Generate image
                progress(0.3, "Generating image...")
                image, info = app.image_generator.generate_image(
                    app.image_model,
                    app.image_model_name,
                    prompt,
                    negative_prompt,
                    width,
                    height,
                    steps,
                    cfg,
                    seed,
                    progress=lambda p, msg: progress(0.3 + p * 0.6, msg)
                )
                
                if image:
                    # Save to history
                    progress(0.9, "Saving to history...")
                    generation_id = str(uuid.uuid4())
                    
                    # Save image
                    image_path = app.output_dir / f"image_{generation_id}.png"
                    image.save(image_path)
                    
                    # Add to history
                    app.history_manager.add_generation(
                        generation_id=generation_id,
                        generation_type="image",
                        model_name=model,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        parameters={
                            "width": width,
                            "height": height,
                            "steps": steps,
                            "guidance_scale": cfg,
                            "seed": seed,
                            "lora": lora if lora else None,
                            "lora_weight": lora_weight if lora else None
                        },
                        output_paths=[str(image_path)],
                        metadata={"feature": "generate_image"}
                    )
                    
                    progress(1.0, "Complete!")
                    return image, info
                else:
                    return None, f"Generation failed: {info}"
                    
            except Exception as e:
                logger.error(f"Image generation error: {e}")
                return None, f"❌ Error: {str(e)}"
        
        # Connect generate button
        inputs = [prompt, negative_prompt, model_dropdown]
        if lora_dropdown:
            inputs.extend([lora_dropdown, lora_weight])
        else:
            inputs.extend([gr.State(None), gr.State(1.0)])
        inputs.extend([seed, steps, cfg, width, height])
        
        generate_btn.click(
            generate_image,
            inputs=inputs,
            outputs=[output_image, generation_info]
        )
        
        # Add refresh functionality
        def refresh_model_choices():
            """Refresh model dropdown with available models"""
            try:
                downloaded_models = app.model_manager.get_downloaded_models("image")
                logger.info(f"Refreshed models: {downloaded_models}")
                if downloaded_models:
                    # Update cached models
                    app._cached_image_models = downloaded_models
                    return gr.update(choices=downloaded_models, value=downloaded_models[0], interactive=True)
                else:
                    return gr.update(choices=[], value=None, label="Model (No models downloaded - visit Settings)", interactive=False)
            except Exception as e:
                logger.error(f"Error refreshing model choices: {e}")
                return gr.update(choices=[], value=None, label="Model (Error loading models)", interactive=False)
        
        # Connect refresh button
        refresh_btn.click(refresh_model_choices, outputs=[model_dropdown])