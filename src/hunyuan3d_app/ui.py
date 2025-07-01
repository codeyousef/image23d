import gradio as gr
from pathlib import Path

from .hunyuan3d_studio import Hunyuan3DStudio
from .config import ALL_IMAGE_MODELS, HUNYUAN3D_MODELS, QUALITY_PRESETS, IMAGE_MODELS, GATED_IMAGE_MODELS

def load_custom_css():
    css_path = Path("app_styles.css")
    if css_path.exists():
        with open(css_path, 'r') as f:
            return f.read()
    return ""

def create_interface(app: Hunyuan3DStudio):
    custom_css = load_custom_css()

    with gr.Blocks(
            title="Hunyuan3D Studio - Complete Pipeline",
            theme=gr.themes.Base(
                primary_hue="sky",
                secondary_hue="indigo",
                font=gr.themes.GoogleFont("Inter")
            ),
            css=custom_css
    ) as interface:
        gr.HTML("""
        <div class="main-header">
            <h1>🎨 Hunyuan3D Studio</h1>
            <p>Complete Text → Image → 3D Pipeline with Best-in-Class Models</p>
        </div>
        """)

        # Model status at the top
        model_status = gr.HTML(value=app.get_model_status())

        # Declare manual_img_model and manual_3d_model at a higher scope
        initial_image_dropdown_update, initial_hunyuan_dropdown_update, _, _ = app.get_model_selection_data()

        manual_img_model = gr.Dropdown(
            choices=initial_image_dropdown_update['choices'],
            label="Model",
            value=initial_image_dropdown_update['value'],
            visible=False # Initially hidden, will be shown in Manual Pipeline tab
        )
        manual_3d_model = gr.Dropdown(
            choices=initial_hunyuan_dropdown_update['choices'],
            label="Model",
            value=initial_hunyuan_dropdown_update['value'],
            visible=False # Initially hidden, will be shown in Manual Pipeline tab
        )

        with gr.Tabs():
            # Quick Generate Tab
            with gr.Tab("🚀 Quick Generate"):
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
                            with gr.Row():
                                image_model = gr.Dropdown(
                                    choices=initial_image_dropdown_update['choices'],
                                    value=initial_image_dropdown_update['value'],
                                    label="Image Model"
                                )
                                hunyuan_model = gr.Dropdown(
                                    choices=initial_hunyuan_dropdown_update['choices'],
                                    value=initial_hunyuan_dropdown_update['value'],
                                    label="3D Model"
                                )

                            quality_preset = gr.Radio(
                                choices=list(QUALITY_PRESETS.keys()),
                                value="standard",
                                label="Quality Preset"
                            )

                        # Generation mode selection
                        with gr.Row():
                            only_generate_image = gr.Checkbox(
                                label="Generate image only (skip 3D conversion)",
                                value=False,
                                info="Generate only the image without creating a 3D model"
                            )

                        # JavaScript to disable 3D-related options when only_generate_image is checked
                        only_generate_image.change(
                            fn=lambda x: [
                                gr.update(interactive=not x, label="3D Model (disabled)" if x else "3D Model"),
                                gr.update(interactive=not x, label="Quality Preset (disabled)" if x else "Quality Preset")
                            ],
                            inputs=[only_generate_image],
                            outputs=[hunyuan_model, quality_preset]
                        )

                        # Advanced options
                        with gr.Accordion("🎛️ Advanced Options", open=False):
                            with gr.Row():
                                image_width = gr.Slider(512, 2048, 1024, step=64, label="Width")
                                image_height = gr.Slider(512, 2048, 1024, step=64, label="Height")

                            image_seed = gr.Number(value=-1, label="Seed (-1 for random)")
                            remove_background = gr.Checkbox(label="Remove Background", value=True)

                            with gr.Row():
                                keep_image_loaded = gr.Checkbox(
                                    label="Keep image model loaded",
                                    value=False,
                                    info="Uses more VRAM but faster for multiple generations"
                                )
                                save_intermediate = gr.Checkbox(
                                    label="Save intermediate outputs",
                                    value=True
                                )

                        # Generate and Stop buttons
                        with gr.Row():
                            generate_btn = gr.Button(
                                "✨ Generate",
                                variant="primary",
                                elem_classes=["generate-button"]
                            )
                            stop_btn = gr.Button(
                                "🛑 Stop Generation",
                                variant="stop",
                                visible=False,
                                elem_classes=["stop-button"]
                            )

                    # Results column
                    with gr.Column(scale=2):
                        gr.Markdown("### 🎭 Results")

                        with gr.Tabs():
                            with gr.Tab("Generated Image"):
                                generated_image = gr.Image(label="Generated Image")

                            with gr.Tab("3D Preview"):
                                preview_3d = gr.Image(label="3D Preview")

                            with gr.Tab("Download"):
                                mesh_file = gr.File(label="3D Model File")

                        generation_info = gr.HTML()

                # Connect stop button
                stop_btn.click(
                    fn=app.stop_generation,
                    outputs=[generation_info]
                )

                # Connect generate button with button state management
                generate_btn.click(
                    # First update the UI to show we're starting generation
                    fn=lambda: (
                        gr.update(interactive=False),  # Disable generate button
                        gr.update(visible=True)        # Show stop button
                    ),
                    outputs=[generate_btn, stop_btn]
                ).then(
                    # Then run the actual generation
                    fn=app.full_pipeline,
                    inputs=[
                        prompt, negative_prompt, image_model,
                        image_width, image_height, image_seed,
                        quality_preset, hunyuan_model,
                        keep_image_loaded, save_intermediate,
                        only_generate_image
                    ],
                    outputs=[generated_image, preview_3d, mesh_file, generation_info]
                ).then(
                    # After generation completes, update model status
                    fn=app.get_model_status,
                    outputs=[model_status]
                ).then(
                    # Update model selection dropdowns
                    fn=app.model_manager.get_model_selection_data,
                    outputs=[image_model, hunyuan_model, manual_img_model, manual_3d_model]
                ).then(
                    # Finally, restore the UI state
                    fn=lambda: (
                        gr.update(interactive=True),   # Re-enable generate button
                        gr.update(visible=False)       # Hide stop button
                    ),
                    outputs=[generate_btn, stop_btn]
                )

            # Model Manager Tab
            with gr.Tab("📦 Model Manager"):
                gr.Markdown("### Download and manage AI models")

                # Add HF token input at the top
                with gr.Group():
                    gr.Markdown("### 🔑 Hugging Face Authentication (Optional)")
                    gr.Markdown(
                        "Only needed for gated models like FLUX. Get your token from [HF Settings](https://huggingface.co/settings/tokens)")
                    with gr.Row():
                        hf_token_input = gr.Textbox(
                            label="HF Token",
                            type="password",
                            placeholder="hf_...",
                            info="Your Hugging Face access token",
                            value=app.model_manager.hf_token
                        )
                        set_token_btn = gr.Button("Set Token", variant="secondary")
                        token_status = gr.HTML()

                    set_token_btn.click(
                        fn=app.model_manager.set_hf_token,
                        inputs=[hf_token_input],
                        outputs=[token_status]
                    )

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 🎨 Open Image Generation Models")
                        gr.Markdown("These models are freely available and don't require authentication.")

                        for name, config in IMAGE_MODELS.items():
                            with gr.Group():
                                gr.Markdown(f"**{name}**")
                                gr.Markdown(f"{config.description}")
                                gr.Markdown(f"Size: {config.size} | VRAM: {config.vram_required}")

                                with gr.Row():
                                    download_img_btn = gr.Button(f"Download {name}", size="sm", variant="primary")
                                    stop_download_btn = gr.Button("Stop", size="sm", variant="stop")
                                    force_redownload = gr.Checkbox(label="Force re-download", value=False)

                                status_html = gr.HTML()

                                def create_download_fn(model_name):
                                    def download_fn(force, progress=gr.Progress()):
                                        yield from app.model_manager.download_model("image", model_name, use_hf_token=False,
                                                                  force_redownload=force, progress=progress)
                                    return download_fn

                                download_img_btn.click(
                                    fn=create_download_fn(name),
                                    inputs=[force_redownload],
                                    outputs=[status_html, image_model, hunyuan_model, manual_img_model, manual_3d_model]
                                )

                                stop_download_btn.click(
                                    fn=app.model_manager.stop_download,
                                    outputs=[status_html]
                                )

                        # Gated models section
                        gr.Markdown("### 🔒 Gated Models (Require HF Login)")
                        gr.Markdown("These models require authentication. Set your HF token above first.")

                        for name, config in GATED_IMAGE_MODELS.items():
                            with gr.Group():
                                gr.Markdown(f"**{name}**")
                                gr.Markdown(f"⚠️ {config.description}")
                                gr.Markdown(f"Size: {config.size} | VRAM: {config.vram_required}")

                                with gr.Row():
                                    download_gated_btn = gr.Button(f"Download {name}", size="sm", variant="secondary")
                                    stop_gated_btn = gr.Button("Stop", size="sm", variant="stop")
                                    force_redownload_gated = gr.Checkbox(label="Force re-download", value=False)

                                gated_status = gr.HTML()

                                def create_gated_download_fn(model_name):
                                    def download_fn(force, progress=gr.Progress()):
                                        yield from app.model_manager.download_model("image", model_name, use_hf_token=True,
                                                                  force_redownload=force, progress=progress)
                                    return download_fn

                                download_gated_btn.click(
                                    fn=create_gated_download_fn(name),
                                    inputs=[force_redownload_gated],
                                    outputs=[gated_status, image_model, hunyuan_model, manual_img_model, manual_3d_model]
                                )

                                stop_gated_btn.click(
                                    fn=app.model_manager.stop_download,
                                    outputs=[gated_status]
                                )

                    with gr.Column():
                        gr.Markdown("### 🎭 Hunyuan3D Models")
                        gr.Markdown("Open 3D generation models from Tencent.")

                        for name, config in HUNYUAN3D_MODELS.items():
                            with gr.Group():
                                gr.Markdown(f"**{name}**")
                                gr.Markdown(f"{config['description']}")
                                gr.Markdown(f"Size: {config['size']} | VRAM: {config['vram_required']}")

                                with gr.Row():
                                    download_3d_btn = gr.Button(f"Download {name}", size="sm", variant="primary")
                                    stop_3d_btn = gr.Button("Stop", size="sm", variant="stop")
                                    force_redownload_3d = gr.Checkbox(label="Force re-download", value=False)

                                status_3d = gr.HTML()

                                def create_3d_download_fn(model_name):
                                    def download_fn(force, progress=gr.Progress()):
                                        yield from app.model_manager.download_model("3d", model_name, force_redownload=force, progress=progress)
                                    return download_fn

                                download_3d_btn.click(
                                    fn=create_3d_download_fn(name),
                                    inputs=[force_redownload_3d],
                                    outputs=[status_3d, image_model, hunyuan_model, manual_img_model, manual_3d_model]
                                )

                                stop_3d_btn.click(
                                    fn=app.model_manager.stop_download,
                                    outputs=[status_3d]
                                )

                # Memory management
                with gr.Group():
                    gr.Markdown("### 🧹 Memory Management")
                    with gr.Row():
                        unload_all_btn = gr.Button("Unload All Models", variant="secondary")
                        clear_cache_btn = gr.Button("Clear Cache", variant="secondary")
                        refresh_btn = gr.Button("Refresh Status", variant="primary")

                    memory_info = gr.HTML()

                    unload_all_btn.click(
                        fn=lambda: (app.model_manager.unload_models(), "✅ All models unloaded"),
                        outputs=[memory_info]
                    ).then(
                        fn=app.get_model_status,
                        outputs=[model_status]
                    )

                    clear_cache_btn.click(
                        fn=lambda: "✅ Cache cleared",
                        outputs=[memory_info]
                    )

                    refresh_btn.click(
                        fn=app.get_model_status,
                        outputs=[model_status]
                    )

                # Missing Components Download
                with gr.Group():
                    gr.Markdown("### 🧩 Download Missing Components")
                    gr.Markdown("If a model is missing specific components (like VAE), you can download them here without re-downloading the entire model.")

                    with gr.Row():
                        missing_components_model_type = gr.Radio(
                            choices=["image", "3d"],
                            value="image",
                            label="Model Type"
                        )

                        # Create a dropdown for model selection
                        missing_components_model = gr.Dropdown(
                            choices=list(IMAGE_MODELS.keys()) + list(GATED_IMAGE_MODELS.keys()),  # Initial choices for "image" type
                            label="Select Model",
                            interactive=True
                        )

                        check_missing_btn = gr.Button("Check Missing Components", variant="secondary")

                    missing_components_status = gr.HTML()

                    with gr.Row(visible=False) as download_components_row:
                        download_components_btn = gr.Button("Download Missing Components", variant="primary")
                        stop_components_btn = gr.Button("Stop", variant="stop")

                    # Function to update model choices based on type
                    def update_model_choices(model_type):
                        if model_type == "image":
                            choices = list(IMAGE_MODELS.keys()) + list(GATED_IMAGE_MODELS.keys())
                        else:
                            choices = list(HUNYUAN3D_MODELS.keys())
                        return gr.update(choices=choices, value=choices[0] if choices else None)

                    # Connect the radio button to update model choices
                    missing_components_model_type.change(
                        fn=update_model_choices,
                        inputs=[missing_components_model_type],
                        outputs=[missing_components_model]
                    )

                    # Function to check missing components
                    def check_missing_components(model_type, model_name):
                        missing = app.check_missing_components(model_type, model_name)

                        if not missing:
                            return """
<div class="success-box">
    <h4>✅ All Components Present</h4>
    <p>All required components for this model are already downloaded.</p>
</div>
""", gr.update(visible=False)

                        if "complete model" in missing:
                            return f"""
<div class="error-box">
    <h4>❌ Model Not Downloaded</h4>
    <p>The model <strong>{model_name}</strong> is not downloaded at all.</p>
    <p>Please download the complete model first using the buttons above.</p>
</div>
""", gr.update(visible=False)

                        components_list = ", ".join(missing)
                        return f"""
<div class="warning-box" style="background-color: #fff3cd; color: #664d03; border: 1px solid #ffeeba; padding: 1em; border-radius: 5px;">
    <h4>⚠️ Missing Components Detected</h4>
    <p>The model <strong>{model_name}</strong> is missing the following components:</p>
    <ul>
        <li>{components_list}</li>
    </ul>
    <p>Click "Download Missing Components" to download only these components.</p>
</div>
""", gr.update(visible=True)

                    # Connect the check button
                    check_missing_btn.click(
                        fn=check_missing_components,
                        inputs=[missing_components_model_type, missing_components_model],
                        outputs=[missing_components_status, download_components_row]
                    )

                    # Function to download missing components
                    def download_missing_components(model_type, model_name, progress=gr.Progress()):
                        yield from app.model_manager.download_missing_components(model_type, model_name, progress=progress)

                    # Connect the download button
                    download_components_btn.click(
                        fn=download_missing_components,
                        inputs=[missing_components_model_type, missing_components_model],
                        outputs=[missing_components_status, image_model, hunyuan_model, manual_img_model, manual_3d_model]
                    )

                    # Connect the stop button
                    stop_components_btn.click(
                        fn=app.model_manager.stop_download,
                        outputs=[missing_components_status]
                    )

            # Manual Pipeline Tab
            with gr.Tab("🎛️ Manual Pipeline"):
                gr.Markdown("""
                ### Manual Step-by-Step Control
                Run each step of the pipeline individually for maximum control.
                """)

                with gr.Row():
                    with gr.Column():
                        # Step 1: Generate Image
                        with gr.Group():
                            gr.Markdown("### Step 1: Generate Image")
                            manual_prompt = gr.Textbox(label="Prompt", lines=3)
                            manual_neg_prompt = gr.Textbox(label="Negative Prompt", lines=2)

                            with gr.Row():
                                # Re-assign the already declared manual_img_model
                                manual_img_model = gr.Dropdown(
                                    choices=list(ALL_IMAGE_MODELS.keys()),
                                    label="Model",
                                    value="SDXL-Turbo",
                                    visible=True # Make it visible here
                                )
                                manual_img_steps = gr.Slider(10, 100, 35, label="Steps")

                            with gr.Row():
                                manual_img_width = gr.Slider(512, 2048, 1024, step=64, label="Width")
                                manual_img_height = gr.Slider(512, 2048, 1024, step=64, label="Height")

                            with gr.Row():
                                manual_img_guidance = gr.Slider(1, 20, 8, step=0.5, label="Guidance")
                                manual_img_seed = gr.Number(value=-1, label="Seed")

                            with gr.Row():
                                generate_img_btn = gr.Button("Generate Image", variant="primary")
                                stop_img_btn = gr.Button("🛑 Stop", variant="stop", visible=False)

                        # Step 2: Convert to 3D
                        with gr.Group():
                            gr.Markdown("### Step 2: Convert to 3D")
                            input_image = gr.Image(label="Input Image", type="pil")

                            with gr.Row():
                                # Re-assign the already declared manual_3d_model
                                manual_3d_model = gr.Dropdown(
                                    choices=list(HUNYUAN3D_MODELS.keys()),
                                    label="Model",
                                    value="hunyuan3d-2mini",
                                    visible=True # Make it visible here
                                )

                            with gr.Row():
                                manual_num_views = gr.Slider(4, 12, 6, label="Number of Views")
                                manual_mesh_res = gr.Slider(128, 512, 256, label="Mesh Resolution")

                            manual_texture_res = gr.Slider(512, 2048, 1024, label="Texture Resolution")

                            with gr.Row():
                                convert_3d_btn = gr.Button("Convert to 3D", variant="primary")
                                stop_3d_btn = gr.Button("🛑 Stop", variant="stop", visible=False)

                    with gr.Column():
                        manual_outputs = gr.HTML()
                        manual_image_out = gr.Image(label="Generated Image")
                        manual_3d_preview = gr.Image(label="3D Preview")
                        manual_3d_out = gr.File(label="3D Model")

                # Connect manual pipeline buttons
                # Connect stop buttons
                stop_img_btn.click(
                    fn=app.stop_generation,
                    outputs=[manual_outputs]
                )

                stop_3d_btn.click(
                    fn=app.stop_generation,
                    outputs=[manual_outputs]
                )

                # Connect generate image button with button state management
                generate_img_btn.click(
                    # First update the UI to show we're starting generation
                    fn=lambda: (
                        gr.update(interactive=False),  # Disable generate button
                        gr.update(visible=True)        # Show stop button
                    ),
                    outputs=[generate_img_btn, stop_img_btn]
                ).then(
                    # Then run the actual generation
                    fn=lambda *args: app.generate_image(*args, progress=gr.Progress()),
                    inputs=[
                        manual_prompt, manual_neg_prompt, manual_img_model,
                        manual_img_width, manual_img_height, manual_img_steps,
                        manual_img_guidance, manual_img_seed
                    ],
                    outputs=[manual_image_out, manual_outputs]
                ).then(
                    # Copy the generated image to the input image for 3D conversion
                    fn=lambda img: img,
                    inputs=[manual_image_out],
                    outputs=[input_image]
                ).then(
                    # Finally, restore the UI state
                    fn=lambda: (
                        gr.update(interactive=True),   # Re-enable generate button
                        gr.update(visible=False)       # Hide stop button
                    ),
                    outputs=[generate_img_btn, stop_img_btn]
                )

                # Connect convert to 3D button with button state management
                convert_3d_btn.click(
                    # First update the UI to show we're starting conversion
                    fn=lambda: (
                        gr.update(interactive=False),  # Disable convert button
                        gr.update(visible=True)        # Show stop button
                    ),
                    outputs=[convert_3d_btn, stop_3d_btn]
                ).then(
                    # Then run the actual conversion
                    fn=lambda *args: app.convert_to_3d(*args, progress=gr.Progress()),
                    inputs=[
                        input_image, manual_3d_model,
                        manual_num_views, manual_mesh_res, manual_texture_res
                    ],
                    outputs=[manual_3d_out, manual_3d_preview, manual_outputs]
                ).then(
                    # Finally, restore the UI state
                    fn=lambda: (
                        gr.update(interactive=True),   # Re-enable convert button
                        gr.update(visible=False)       # Hide stop button
                    ),
                    outputs=[convert_3d_btn, stop_3d_btn]
                )

            # System Requirements Tab
            with gr.Tab("🔍 System Requirements"):
                gr.Markdown("""
                ### System Requirements Check

                This tab shows whether your system meets the requirements for optimal performance.
                """)

                # System requirements status
                system_requirements_html = gr.HTML(value=app.check_system_requirements())

                # Refresh button
                refresh_sys_req_btn = gr.Button("Refresh System Check", variant="primary")
                refresh_sys_req_btn.click(
                    fn=app.check_system_requirements,
                    outputs=[system_requirements_html]
                )

                # Add some CSS for the system requirements display
                gr.HTML("""
                <style>
                .system-requirements {
                    margin: 20px 0;
                    padding: 15px;
                    border-radius: 10px;
                    background-color: #f8f9fa;
                    border: 1px solid #dee2e6;
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                    font-size: 16px;
                    line-height: 1.5;
                    color: #212529;
                }
                .requirements-summary {
                    margin-bottom: 20px;
                    color: #212529;
                }
                .overall-status {
                    padding: 10px;
                    border-radius: 5px;
                    margin-bottom: 15px;
                    font-weight: bold;
                    font-size: 18px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .overall-status.ok {
                    background-color: #d4edda;
                    color: #155724;
                    border: 1px solid #c3e6cb;
                }
                .overall-status.warning {
                    background-color: #fff3cd;
                    color: #856404;
                    border: 1px solid #ffeeba;
                }
                .overall-status.error {
                    background-color: #f8d7da;
                    color: #721c24;
                    border: 1px solid #f5c6cb;
                }
                .requirements-section {
                    margin-bottom: 20px;
                    padding: 15px;
                    border-radius: 8px;
                    background-color: white;
                    border: 1px solid #dee2e6;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
                    color: #212529;
                }
                .requirements-section h4 {
                    margin-top: 0;
                    margin-bottom: 15px;
                    color: #212529;
                    font-size: 18px;
                    font-weight: 600;
                    border-bottom: 1px solid #eee;
                    padding-bottom: 8px;
                }
                .requirements-section ul {
                    margin: 0;
                    padding-left: 25px;
                    color: #212529;
                }
                .requirements-section li {
                    margin-bottom: 10px;
                    line-height: 1.6;
                    color: #212529;
                }
                .requirements-section strong {
                    font-weight: 600;
                    color: #212529;
                }
                .status-icon {
                    margin-right: 8px;
                    font-size: 18px;
                    display: inline-block;
                    vertical-align: middle;
                }
                .status-text {
                    color: inherit;
                }
                .recommendations {
                    background-color: #e2f0fd;
                    border: 1px solid #b8daff;
                    color: #0c5460;
                }
                .recommendations li {
                    color: #0c5460;
                }
                /* Ensure all text in the system requirements is visible */
                .system-requirements * {
                    color: inherit;
                }
                .system-requirements span, 
                .system-requirements p {
                    color: #212529;
                }
                </style>
                """)

            # Examples Tab
            with gr.Tab("💡 Examples"):
                gr.Markdown("""
                ### Best Practices for Image-to-3D Generation

                1. **Clear, Centered Objects**: The image model should generate objects centered with clean backgrounds
                2. **Consistent Lighting**: Avoid harsh shadows or dramatic lighting  
                3. **Full Visibility**: Ensure the entire object is visible, not cropped
                4. **High Detail**: Higher resolution images produce better 3D models

                ### Optimized Prompts for 3D
                """)

                gr.Examples(
                    examples=[
                        ["A ornate golden crown with red gemstones, centered, white background, studio lighting, highly detailed, 3D model reference",
                         "blurry, multiple objects, cropped, shadows"],
                        ["A ceramic vase with blue and white patterns, centered object, neutral background, even lighting, high quality 3D scan",
                         "glass, transparent, reflective, multiple items"],
                        ["A wooden chess piece knight, centered, plain background, soft lighting, detailed carved texture, 360 view",
                         "metallic, shiny, multiple pieces, harsh shadows"],
                        ["A leather hiking boot, single shoe, centered, clean background, product photography style, all angles visible",
                         "pair of shoes, shadows, artistic angle, cropped"],
                        ["A plush teddy bear toy, centered, white background, soft even lighting, full view, high detail fabric texture",
                         "multiple toys, dark background, harsh lighting"],
                        ["An ancient Greek amphora vase, terracotta material, centered, museum lighting, archaeological detail",
                         "modern, glass, metal, multiple objects"],
                        ["A mechanical pocket watch, open face, centered, macro detail, clean background, brass and gold materials",
                         "digital watch, wristwatch, multiple items, reflections"],
                        ["A medieval iron helmet, knight armor piece, centered, studio lighting, aged metal texture, historical accuracy",
                         "modern helmet, shiny chrome, multiple pieces"]
                    ],
                    inputs=[prompt, negative_prompt]
                )
    return interface
