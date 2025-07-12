"""Image Generation Tab - Consolidated image features"""

import gradio as gr
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging

from ..components.media_sidebar import create_media_sidebar
from ..components.common import create_generation_settings, create_action_button
from ...config import ALL_IMAGE_MODELS

logger = logging.getLogger(__name__)


def create_image_generation_tab(app: Any) -> None:
    """Create the unified image generation tab with all image-related features"""
    
    with gr.Row():
        # Main content area
        with gr.Column(scale=4):
            gr.Markdown("## 🎨 Image Generation Studio")
            gr.Markdown("Create, edit, and enhance images with AI-powered tools")
            
            # Feature cards in a grid
            with gr.Tabs() as feature_tabs:
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


def create_generate_image_subtab(app: Any) -> None:
    """Create the main image generation interface"""
    
    with gr.Column():
        # Get downloaded models
        downloaded_models = app.model_manager.get_downloaded_models("image")
        default_model = downloaded_models[0] if downloaded_models else None
        
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
        
        # Model selection
        with gr.Row():
            if downloaded_models:
                model_dropdown = gr.Dropdown(
                    choices=downloaded_models,
                    value=default_model,
                    label="Model"
                )
            else:
                model_dropdown = gr.Dropdown(
                    choices=[],
                    label="Model (No models downloaded - visit Settings)",
                    interactive=False
                )
            
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
                    from ...features.lora.manager import LoRAInfo
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
                    import uuid
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
        
        # Model selection
        downloaded_models = app.model_manager.get_downloaded_models("image")
        if downloaded_models:
            model_dropdown = gr.Dropdown(
                choices=downloaded_models,
                value=downloaded_models[0],
                label="Model"
            )
        else:
            model_dropdown = gr.Dropdown(
                choices=[],
                label="Model (No models downloaded)",
                interactive=False
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
                    from PIL import Image
                    bg = Image.new('RGB', result.size, (255, 255, 255))
                    bg.paste(result, mask=result.split()[-1])
                    result = bg
                elif format == "Custom Color":
                    # Convert to RGB with custom color
                    from PIL import Image
                    import matplotlib.colors as mcolors
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


def create_upscale_image_subtab(app: Any) -> None:
    """Create image upscaling interface"""
    
    with gr.Column():
        gr.Markdown("### Enhance image resolution with AI")
        
        # Input image
        input_image = gr.Image(
            label="Input Image",
            type="pil",
            height=300
        )
        
        # Upscale settings
        with gr.Row():
            scale_factor = gr.Radio(
                choices=["2x", "4x", "8x"],
                value="2x",
                label="Scale Factor"
            )
            
            model_type = gr.Radio(
                choices=["General", "Anime", "Face"],
                value="General",
                label="Model Type"
            )
        
        # Advanced options
        with gr.Accordion("Advanced Options", open=False):
            denoise_strength = gr.Slider(
                0, 1, 0.5,
                step=0.1,
                label="Denoise Strength"
            )
            
            face_enhance = gr.Checkbox(
                label="Face Enhancement",
                value=False
            )
        
        # Upscale button
        upscale_btn = create_action_button("🔍 Upscale Image", variant="primary")
        
        # Output
        with gr.Row():
            output_image = gr.Image(label="Upscaled Image", type="pil")
            upscale_info = gr.HTML()
        
        # Placeholder for upscaling
        def upscale_image(input_img, scale, model, denoise, face_enh):
            if not input_img:
                return None, "❌ Please provide an input image"
            
            # TODO: Implement Real-ESRGAN or similar upscaling
            return None, "🚧 Upscaling feature coming soon!"
        
        upscale_btn.click(
            upscale_image,
            inputs=[input_image, scale_factor, model_type, denoise_strength, face_enhance],
            outputs=[output_image, upscale_info]
        )


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
        
        # Model selection
        downloaded_models = app.model_manager.get_downloaded_models("image")
        if downloaded_models:
            model_dropdown = gr.Dropdown(
                choices=downloaded_models,
                value=downloaded_models[0],
                label="Model"
            )
        else:
            model_dropdown = gr.Dropdown(
                choices=[],
                label="Model (No models downloaded)",
                interactive=False
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
        
        # Model selection
        downloaded_models = app.model_manager.get_downloaded_models("image")
        if downloaded_models:
            model_dropdown = gr.Dropdown(
                choices=downloaded_models,
                value=downloaded_models[0],
                label="Model"
            )
        else:
            model_dropdown = gr.Dropdown(
                choices=[],
                label="Model (No models downloaded)",
                interactive=False
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


def create_fix_image_subtab(app: Any) -> None:
    """Create image fixing/restoration interface"""
    
    with gr.Column():
        gr.Markdown("### Fix and restore damaged or low-quality images")
        
        # Input image
        input_image = gr.Image(
            label="Input Image",
            type="pil",
            height=300
        )
        
        # Fix options
        with gr.Row():
            fix_type = gr.CheckboxGroup(
                choices=[
                    "Face Restoration",
                    "Denoise",
                    "Deblur",
                    "Color Correction",
                    "Remove Artifacts"
                ],
                value=["Face Restoration", "Denoise"],
                label="Fix Options"
            )
        
        # Advanced settings
        with gr.Accordion("Advanced Settings", open=False):
            restoration_strength = gr.Slider(
                0, 1, 0.5,
                step=0.1,
                label="Restoration Strength"
            )
            
            face_detection_confidence = gr.Slider(
                0.5, 0.99, 0.9,
                step=0.01,
                label="Face Detection Confidence"
            )
        
        # Fix button
        fix_btn = create_action_button("🔧 Fix Image", variant="primary")
        
        # Output
        with gr.Row():
            output_image = gr.Image(label="Fixed Image", type="pil")
            fix_info = gr.HTML()
        
        # Wire up face restoration (partial implementation)
        def fix_image(input_img, fix_types, strength, face_conf):
            if not input_img:
                return None, "❌ Please provide an input image"
            if not fix_types:
                return None, "❌ Please select at least one fix option"
            
            try:
                result = input_img
                applied_fixes = []
                
                # Face restoration using existing face swap manager
                if "Face Restoration" in fix_types and hasattr(app, 'face_swap_manager'):
                    if app.face_swap_manager.facefusion_loaded or app.face_swap_manager.initialize_models()[0]:
                        from ...features.face_swap import FaceSwapParams
                        params = FaceSwapParams(
                            face_restore=True,
                            face_restore_fidelity=strength,
                            background_enhance=False,
                            face_upsample=True
                        )
                        # Use same image as source and target for restoration only
                        result, info = app.face_swap_manager.swap_face(
                            source_image=result,
                            target_image=result,
                            params=params
                        )
                        if result:
                            applied_fixes.append("Face Restoration")
                
                # TODO: Implement other fix types
                for fix in fix_types:
                    if fix not in ["Face Restoration"]:
                        logger.info(f"Fix type '{fix}' not yet implemented")
                
                if applied_fixes:
                    return result, f"✅ Applied: {', '.join(applied_fixes)}"
                else:
                    return None, "🚧 Additional fix features coming soon!"
                    
            except Exception as e:
                logger.error(f"Image fix error: {e}")
                return None, f"❌ Error: {str(e)}"
        
        fix_btn.click(
            fix_image,
            inputs=[input_image, fix_type, restoration_strength, face_detection_confidence],
            outputs=[output_image, fix_info]
        )


def create_face_swap_subtab(app: Any) -> None:
    """Create face swap interface with generate-then-swap option"""
    
    with gr.Column():
        gr.Markdown("### Swap faces in images with AI precision")
        
        # Mode selection
        mode = gr.Radio(
            choices=["Upload Both Images", "Generate Then Swap"],
            value="Upload Both Images",
            label="Mode",
            info="Choose whether to upload both images or generate the target image first"
        )
        
        with gr.Group(visible=True) as upload_mode_group:
            with gr.Row():
                # Source face image
                source_image = gr.Image(
                    label="Source Face",
                    type="pil",
                    height=300
                )
                
                # Target image
                target_image = gr.Image(
                    label="Target Image",
                    type="pil",
                    height=300
                )
        
        with gr.Group(visible=False) as generate_mode_group:
            # Generation prompt
            gen_prompt = gr.Textbox(
                label="Generate Target Image",
                placeholder="A professional business photo with formal attire...",
                lines=2
            )
            
            with gr.Row():
                # Model selection for generation
                downloaded_models = []
                try:
                    downloaded_models = app.model_manager.get_downloaded_models("image")
                except:
                    pass
                
                if downloaded_models:
                    gen_model = gr.Dropdown(
                        choices=downloaded_models,
                        value=downloaded_models[0] if downloaded_models else None,
                        label="Image Model"
                    )
                else:
                    gen_model = gr.Dropdown(
                        choices=[],
                        label="Image Model (No models downloaded)",
                        interactive=False
                    )
                
                generate_target_btn = create_action_button("🎨 Generate Target", size="sm")
            
            # Generated target display
            generated_target = gr.Image(
                label="Generated Target Image",
                type="pil",
                height=300,
                visible=False
            )
            
            # Source face for generate mode
            source_face_gen = gr.Image(
                label="Source Face",
                type="pil",
                height=300
            )
        
        # Swap settings
        with gr.Accordion("Face Swap Settings", open=True):
            with gr.Row():
                face_restore = gr.Checkbox(
                    label="Restore Face Quality",
                    value=True,
                    info="Apply face restoration after swapping"
                )
                
                blend_ratio = gr.Slider(
                    0.0, 1.0, 0.95,
                    step=0.05,
                    label="Blend Ratio",
                    info="How much of the source face to blend (1.0 = full swap)"
                )
            
            with gr.Row():
                face_index = gr.Number(
                    value=0,
                    label="Target Face Index",
                    info="Which face to swap in target image (0 = first/largest)"
                )
                
                similarity_threshold = gr.Slider(
                    0.0, 1.0, 0.6,
                    step=0.1,
                    label="Face Detection Threshold"
                )
        
        # Swap button
        swap_btn = create_action_button("🔄 Swap Faces", variant="primary")
        
        # Output
        with gr.Row():
            output_image = gr.Image(label="Result", type="pil")
            swap_info = gr.HTML()
        
        # Mode switching logic
        def switch_mode(selected_mode):
            if selected_mode == "Upload Both Images":
                return (
                    gr.update(visible=True),   # upload_mode_group
                    gr.update(visible=False),  # generate_mode_group
                    gr.update(visible=False)   # generated_target
                )
            else:
                return (
                    gr.update(visible=False),  # upload_mode_group
                    gr.update(visible=True),   # generate_mode_group
                    gr.update(visible=False)   # generated_target initially hidden
                )
        
        mode.change(
            switch_mode,
            inputs=[mode],
            outputs=[upload_mode_group, generate_mode_group, generated_target]
        )
        
        # Generate target image
        def generate_target_image(prompt, model_name, progress=gr.Progress()):
            """Generate target image for face swap"""
            if not prompt:
                return None, "❌ Please enter a prompt"
            if not model_name:
                return None, "❌ Please select a model"
            
            try:
                progress(0.1, "Generating target image...")
                
                # Submit generation job
                job_id = app.submit_generation_job(
                    job_type="image",
                    params={
                        "prompt": prompt,
                        "model_name": model_name,
                        "width": 512,
                        "height": 512,
                        "num_inference_steps": 30,
                        "guidance_scale": 7.5
                    }
                )
                
                # Wait for completion
                import time
                job = app.queue_manager.get_job(job_id)
                
                while job and job.status.value in ["pending", "running"]:
                    time.sleep(0.5)
                    job = app.queue_manager.get_job(job_id)
                    if job:
                        progress(0.1 + job.progress * 0.8, job.progress_message or "Generating...")
                
                if job and job.status.value == "completed" and job.result:
                    logger.info(f"Job result keys: {list(job.result.keys())}")
                    
                    # Try different possible keys for the image path
                    image_path = job.result.get("path") or job.result.get("image_path")
                    
                    # Also check if image is directly in result
                    image = job.result.get("image")
                    
                    if image_path and Path(image_path).exists():
                        from PIL import Image
                        image = Image.open(image_path)
                        progress(1.0, "Target image generated!")
                        return gr.update(value=image, visible=True), "✅ Target image generated!"
                    elif image:
                        progress(1.0, "Target image generated!")
                        return gr.update(value=image, visible=True), "✅ Target image generated!"
                    else:
                        logger.error(f"No valid image found in result. Path: {image_path}, Image: {image is not None}")
                
                error_msg = job.error if job else "Job not found"
                logger.error(f"Generation failed: {error_msg}")
                return None, f"❌ Generation failed: {error_msg}"
                
            except Exception as e:
                logger.error(f"Target generation error: {e}")
                return None, f"❌ Error: {str(e)}"
        
        generate_target_btn.click(
            generate_target_image,
            inputs=[gen_prompt, gen_model],
            outputs=[generated_target, swap_info]
        )
        
        # Main face swap function
        def swap_faces(mode, source_img, target_img, source_gen, generated,
                      restore, blend, face_idx, threshold, progress=gr.Progress()):
            """Perform face swap"""
            
            # Determine which images to use based on mode
            if mode == "Upload Both Images":
                if not source_img or not target_img:
                    return None, "❌ Please provide both source and target images"
                source_face = source_img
                target_face = target_img
            else:
                if not source_gen:
                    return None, "❌ Please provide source face image"
                if not generated:
                    return None, "❌ Please generate a target image first"
                source_face = source_gen
                target_face = generated
            
            try:
                progress(0.1, "Detecting faces...")
                
                # Submit face swap job
                job_id = app.submit_generation_job(
                    job_type="face_swap",
                    params={
                        "source_image": source_face,
                        "target_image": target_face,
                        "face_restore": restore,
                        "blend_ratio": blend,
                        "face_index": int(face_idx),
                        "similarity_threshold": threshold
                    }
                )
                
                progress(0.2, f"Job {job_id[:8]} submitted...")
                
                # Wait for completion
                import time
                job = app.queue_manager.get_job(job_id)
                
                while job and job.status.value in ["pending", "running"]:
                    time.sleep(0.5)
                    job = app.queue_manager.get_job(job_id)
                    if job:
                        progress(0.2 + job.progress * 0.7, job.progress_message or "Processing...")
                
                if job and job.status.value == "completed" and job.result:
                    logger.info(f"Face swap job completed. Result keys: {list(job.result.keys()) if isinstance(job.result, dict) else 'Not a dict'}")
                    # Face swap returns 'path', not 'output_path'
                    result_path = job.result.get("path") or job.result.get("output_path") or job.result.get("image_path")
                    logger.info(f"Looking for result at path: {result_path}")
                    
                    # Handle Windows paths in WSL environment
                    if result_path:
                        result_path_obj = Path(result_path)
                        if not result_path_obj.exists():
                            # Try to extract just the filename and look in outputs directory
                            filename = result_path_obj.name
                            alt_path = Path("outputs") / filename
                            if alt_path.exists():
                                result_path = str(alt_path)
                                logger.info(f"Using alternative path: {result_path}")
                    
                    if result_path and Path(result_path).exists():
                        from PIL import Image
                        result_image = Image.open(result_path)
                        
                        # Get additional info from job result
                        job_info = job.result.get('info', {})
                        
                        info = f"""
                        <div class="success-box">
                            <h4>✅ Face Swap Complete!</h4>
                            <ul>
                                <li><strong>Mode:</strong> {mode}</li>
                                <li><strong>Face Restoration:</strong> {'Enabled' if restore else 'Disabled'}</li>
                                <li><strong>Blend Ratio:</strong> {blend}</li>
                                <li><strong>Source Faces:</strong> {job_info.get('source_faces', 'N/A')}</li>
                                <li><strong>Target Faces:</strong> {job_info.get('target_faces', 'N/A')}</li>
                                <li><strong>Swapped Faces:</strong> {job_info.get('swapped_faces', 'N/A')}</li>
                                <li><strong>Processing Time:</strong> {job_info.get('processing_time', job.result.get('processing_time', 'N/A')):.2f}s</li>
                            </ul>
                        </div>
                        """
                        
                        progress(1.0, "Face swap complete!")
                        return result_image, info
                
                if job and job.error:
                    error = job.error
                elif job and job.status.value == "failed":
                    error = "Face swap failed but no error message provided"
                elif not job:
                    error = "Job not found"
                else:
                    error = f"Job status: {job.status.value if job else 'N/A'}"
                return None, f"❌ Face swap failed: {error}"
                
            except Exception as e:
                logger.error(f"Face swap error: {e}")
                return None, f"❌ Error: {str(e)}"
        
        swap_btn.click(
            swap_faces,
            inputs=[mode, source_image, target_image, source_face_gen, generated_target,
                   face_restore, blend_ratio, face_index, similarity_threshold],
            outputs=[output_image, swap_info]
        )