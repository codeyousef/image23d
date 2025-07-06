"""Flux Kontext UI tab for enhanced context-aware generation"""

import gradio as gr
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ...features.flux_kontext import FluxKontextManager, FluxKontextConfig


def create_flux_kontext_tab(app: Any) -> None:
    """Create the Flux Kontext tab
    
    Args:
        app: The enhanced application instance
    """
    gr.Markdown("""
    ### ✨ Flux Kontext Studio
    Generate images with enhanced context awareness using advanced Flux models.
    """)
    
    with gr.Tabs():
        # Single Image Generation
        with gr.Tab("🎨 Context Generation"):
            with gr.Row():
                with gr.Column(scale=2):
                    # Main prompt
                    prompt = gr.Textbox(
                        label="Primary Prompt",
                        placeholder="A detailed description of what you want to create...",
                        lines=3
                    )
                    
                    negative_prompt = gr.Textbox(
                        label="Negative Prompt",
                        placeholder="What to avoid in the generation...",
                        lines=2
                    )
                    
                    # Reference images for context
                    with gr.Accordion("📸 Reference Context", open=True):
                        reference_images = gr.File(
                            label="Reference Images",
                            file_count="multiple",
                            file_types=["image"]
                        )
                        gr.Markdown("*Upload images to provide visual context*")
                        
                        context_strength = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.8,
                            step=0.1,
                            label="Context Strength"
                        )
                        gr.Markdown("*How strongly to apply reference context*")
                        
                        context_mode = gr.Radio(
                            choices=["focused", "balanced", "expansive"],
                            value="balanced",
                            label="Context Mode"
                        )
                        gr.Markdown("*How to interpret the context*")
                        
                    # Model settings
                    with gr.Accordion("🔧 Model Settings", open=True):
                        model_variant = gr.Dropdown(
                            choices=["flux-dev", "flux-schnell"],
                            value="flux-dev",
                            label="Model Variant"
                        )
                        
                        precision = gr.Radio(
                            choices=["fp16", "fp8", "int8"],
                            value="fp16",
                            label="Model Precision"
                        )
                        
                    # Generation parameters
                    with gr.Accordion("⚙️ Generation Settings", open=True):
                        with gr.Row():
                            width = gr.Slider(
                                minimum=512,
                                maximum=2048,
                                value=1024,
                                step=64,
                                label="Width"
                            )
                            
                            height = gr.Slider(
                                minimum=512,
                                maximum=2048,
                                value=1024,
                                step=64,
                                label="Height"
                            )
                            
                        with gr.Row():
                            guidance_scale = gr.Slider(
                                minimum=1.0,
                                maximum=15.0,
                                value=3.5,
                                step=0.5,
                                label="Guidance Scale"
                            )
                            
                            num_inference_steps = gr.Slider(
                                minimum=4,
                                maximum=50,
                                value=28,
                                step=1,
                                label="Inference Steps"
                            )
                            
                        seed = gr.Number(
                            value=-1,
                            label="Seed (-1 for random)"
                        )
                        
                    # Advanced Kontext settings
                    with gr.Accordion("🧠 Advanced Kontext", open=False):
                        context_injection_step = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=5,
                            step=1,
                            label="Context Injection Step"
                        )
                        gr.Markdown("*When to inject context during generation*")
                        
                        context_preservation = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.7,
                            step=0.1,
                            label="Context Preservation"
                        )
                        gr.Markdown("*How much to preserve original context*")
                        
                        semantic_layers = gr.Textbox(
                            label="Semantic Layers",
                            value="0,6,12"
                        )
                        gr.Markdown("*Transformer layers for semantic injection (comma-separated)*")
                        
                        attention_layers = gr.Textbox(
                            label="Attention Layers", 
                            value="3,9,15"
                        )
                        gr.Markdown("*Transformer layers for attention modification (comma-separated)*")
                        
                    # Generation controls
                    with gr.Row():
                        generate_btn = gr.Button(
                            "✨ Generate with Kontext",
                            variant="primary",
                            size="lg"
                        )
                        
                        stop_btn = gr.Button(
                            "⏹️ Stop",
                            variant="stop"
                        )
                        
                with gr.Column(scale=1):
                    # Output
                    output_image = gr.Image(
                        label="Generated Image",
                        type="pil"
                    )
                    
                    generation_info = gr.HTML()
                    
                    # Model status
                    model_status = gr.HTML()
                    
                    # Reference preview
                    reference_preview = gr.Gallery(
                        label="Reference Context",
                        columns=2,
                        height="auto",
                        visible=False
                    )
                    
        # Context Presets
        with gr.Tab("💾 Context Presets"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Save Current Context")
                    
                    preset_name = gr.Textbox(
                        label="Preset Name",
                        placeholder="My awesome context preset"
                    )
                    
                    preset_description = gr.Textbox(
                        label="Description",
                        placeholder="Description of this context preset...",
                        lines=2
                    )
                    
                    save_preset_btn = gr.Button(
                        "💾 Save Preset",
                        variant="primary"
                    )
                    
                    save_status = gr.HTML()
                    
                with gr.Column():
                    gr.Markdown("### Load Saved Presets")
                    
                    preset_list = gr.Dropdown(
                        label="Available Presets",
                        choices=[],
                        interactive=True
                    )
                    
                    load_preset_btn = gr.Button(
                        "📂 Load Preset"
                    )
                    
                    delete_preset_btn = gr.Button(
                        "🗑️ Delete Preset",
                        variant="stop"
                    )
                    
                    preset_info = gr.HTML()
                    
            refresh_presets_btn = gr.Button(
                "🔄 Refresh Preset List"
            )
            
        # Batch Generation
        with gr.Tab("📦 Batch Kontext"):
            gr.Markdown("""
            Generate multiple images with the same context but different prompts.
            """)
            
            with gr.Row():
                with gr.Column():
                    # Batch settings
                    batch_base_prompt = gr.Textbox(
                        label="Base Prompt",
                        placeholder="Base description for all generations...",
                        lines=2
                    )
                    
                    batch_prompt_variations = gr.Textbox(
                        label="Prompt Variations",
                        placeholder="variation 1\nvariation 2\nvariation 3",
                        lines=5
                    )
                    gr.Markdown("*One variation per line*")
                    
                    batch_reference_images = gr.File(
                        label="Batch Reference Images",
                        file_count="multiple",
                        file_types=["image"]
                    )
                    
                    batch_context_strength = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.8,
                        step=0.1,
                        label="Context Strength"
                    )
                    
                    batch_generate_btn = gr.Button(
                        "🚀 Generate Batch",
                        variant="primary",
                        size="lg"
                    )
                    
                with gr.Column():
                    batch_progress = gr.HTML()
                    
                    batch_results = gr.Gallery(
                        label="Batch Results",
                        columns=3,
                        height="auto"
                    )
                    
                    batch_download = gr.File(
                        label="Download Results",
                        visible=False
                    )
                    
    # Helper functions
    def update_reference_preview(files):
        """Update reference images preview"""
        if not files:
            return gr.update(visible=False)
            
        images = []
        for file in files:
            if hasattr(file, 'name'):
                images.append(file.name)
            else:
                images.append(file)
                
        return gr.update(value=images, visible=True)
        
    reference_images.change(
        update_reference_preview,
        inputs=[reference_images],
        outputs=[reference_preview]
    )
    
    # Update model status
    def get_model_status():
        """Get current model status"""
        if hasattr(app, 'flux_kontext_manager'):
            info = app.flux_kontext_manager.get_model_info()
            
            if info["status"] == "loaded":
                return f"""
                <div style='padding: 10px; background: #e8f5e9; border-radius: 5px;'>
                    <h4>✅ {info['model_type']}</h4>
                    <ul>
                        <li><strong>Device:</strong> {info['device']}</li>
                        <li><strong>Precision:</strong> {info['precision']}</li>
                        <li><strong>Kontext Enabled:</strong> {'Yes' if info['kontext_enabled'] else 'No'}</li>
                        <li><strong>Context Layers:</strong> {info['context_layers']}</li>
                    </ul>
                </div>
                """
            else:
                return """
                <div style='padding: 10px; background: #fff3cd; border-radius: 5px;'>
                    <h4>⚠️ Model Not Loaded</h4>
                    <p>Flux Kontext model needs to be initialized</p>
                </div>
                """
        else:
            return """
            <div style='padding: 10px; background: #ffebee; border-radius: 5px;'>
                <h4>❌ Flux Kontext Not Available</h4>
                <p>Please check model installation</p>
            </div>
            """
            
    # Add refresh button since HTML components don't support .load()
    refresh_status_btn = gr.Button("🔄 Refresh Model Status", variant="secondary")
    refresh_status_btn.click(get_model_status, outputs=[model_status])
    
    # Main generation function
    def generate_with_kontext(
        prompt, negative_prompt, ref_images,
        context_strength, context_mode, model_variant, precision,
        width, height, guidance_scale, steps, seed,
        injection_step, preservation, semantic_layers_str, attention_layers_str
    ):
        """Generate image with Flux Kontext"""
        try:
            # Initialize Kontext manager if needed
            if not hasattr(app, 'flux_kontext_manager'):
                app.flux_kontext_manager = FluxKontextManager()
                
            if not app.flux_kontext_manager.models_loaded:
                yield None, "<p>Initializing Flux Kontext models...</p>"
                success, msg = app.flux_kontext_manager.initialize_models()
                if not success:
                    yield None, f"<p style='color: red;'>Failed to initialize models: {msg}</p>"
                    return
                    
            # Parse layer configurations
            try:
                semantic_layers = [f"transformer.transformer_blocks.{i.strip()}" 
                                 for i in semantic_layers_str.split(",") if i.strip()]
                attention_layers = [f"transformer.transformer_blocks.{i.strip()}" 
                                  for i in attention_layers_str.split(",") if i.strip()]
            except:
                semantic_layers = None
                attention_layers = None
                
            # Create configuration
            config = FluxKontextConfig(
                context_strength=context_strength,
                context_mode=context_mode,
                model_variant=model_variant,
                precision=precision,
                guidance_scale=guidance_scale,
                num_inference_steps=int(steps),
                width=int(width),
                height=int(height),
                seed=int(seed),
                context_injection_step=int(injection_step),
                context_preservation=preservation,
                semantic_layers=semantic_layers,
                attention_layers=attention_layers
            )
            
            # Update manager config
            app.flux_kontext_manager.update_config(**config.__dict__)
            
            # Prepare reference images
            reference_imgs = []
            if ref_images:
                from PIL import Image
                for img_file in ref_images:
                    try:
                        if hasattr(img_file, 'name'):
                            img = Image.open(img_file.name).convert('RGB')
                        else:
                            img = Image.open(img_file).convert('RGB')
                        reference_imgs.append(img)
                    except Exception as e:
                        logger.warning(f"Failed to load reference image: {e}")
                        
            yield None, "<p>Generating with Flux Kontext...</p>"
            
            # Generate image
            result_img, info = app.flux_kontext_manager.generate_with_kontext(
                prompt=prompt,
                negative_prompt=negative_prompt,
                reference_images=reference_imgs,
                config=config
            )
            
            if result_img:
                info_html = f"""
                <div style='padding: 10px; background: #e8f5e9; border-radius: 5px;'>
                    <h4>✅ Generation Successful!</h4>
                    <ul>
                        <li><strong>Model:</strong> {info['model']}</li>
                        <li><strong>Resolution:</strong> {info['config']['resolution']}</li>
                        <li><strong>Steps:</strong> {info['config']['steps']}</li>
                        <li><strong>Guidance:</strong> {info['config']['guidance_scale']}</li>
                        <li><strong>Context Used:</strong> {'Yes' if info['context_used'] else 'No'}</li>
                        <li><strong>Reference Images:</strong> {info['reference_images']}</li>
                        <li><strong>Generation Time:</strong> {info['generation_time']}</li>
                        <li><strong>Seed:</strong> {info['config']['seed']}</li>
                    </ul>
                </div>
                """
                yield result_img, info_html
            else:
                error_msg = info.get("error", "Unknown error")
                yield None, f"<p style='color: red;'>Generation failed: {error_msg}</p>"
                
        except Exception as e:
            import traceback
            error_html = f"""
            <div style='padding: 10px; background: #ffebee; border-radius: 5px;'>
                <h4>❌ Error</h4>
                <p>{str(e)}</p>
                <details>
                    <summary>Traceback</summary>
                    <pre>{traceback.format_exc()}</pre>
                </details>
            </div>
            """
            yield None, error_html
            
    generate_btn.click(
        generate_with_kontext,
        inputs=[
            prompt, negative_prompt, reference_images,
            context_strength, context_mode, model_variant, precision,
            width, height, guidance_scale, num_inference_steps, seed,
            context_injection_step, context_preservation, 
            semantic_layers, attention_layers
        ],
        outputs=[output_image, generation_info]
    )
    
    # Preset management functions
    def save_context_preset(name, description, current_prompt):
        """Save current context as preset"""
        try:
            if not name:
                return "<p style='color: red;'>Please enter a preset name</p>"
                
            if not hasattr(app, 'flux_kontext_manager') or not app.flux_kontext_manager.models_loaded:
                return "<p style='color: red;'>Flux Kontext not initialized</p>"
                
            success = app.flux_kontext_manager.save_context_preset(name, current_prompt)
            
            if success:
                return f"""
                <div style='padding: 10px; background: #e8f5e9; border-radius: 5px;'>
                    <h4>✅ Preset Saved</h4>
                    <p>Context preset "{name}" saved successfully</p>
                </div>
                """
            else:
                return "<p style='color: red;'>Failed to save preset</p>"
                
        except Exception as e:
            return f"<p style='color: red;'>Error: {str(e)}</p>"
            
    save_preset_btn.click(
        save_context_preset,
        inputs=[preset_name, preset_description, prompt],
        outputs=[save_status]
    )
    
    def refresh_preset_list():
        """Refresh the list of available presets"""
        try:
            if hasattr(app, 'flux_kontext_manager'):
                presets = app.flux_kontext_manager.list_context_presets()
                return gr.update(choices=presets)
            else:
                return gr.update(choices=[])
        except:
            return gr.update(choices=[])
            
    refresh_presets_btn.click(
        refresh_preset_list,
        outputs=[preset_list]
    )
    
    def load_context_preset(preset_name):
        """Load a context preset"""
        try:
            if not preset_name:
                return "<p style='color: red;'>Please select a preset</p>"
                
            if not hasattr(app, 'flux_kontext_manager'):
                return "<p style='color: red;'>Flux Kontext not available</p>"
                
            success = app.flux_kontext_manager.load_context_preset(preset_name)
            
            if success:
                return f"""
                <div style='padding: 10px; background: #e8f5e9; border-radius: 5px;'>
                    <h4>✅ Preset Loaded</h4>
                    <p>Context preset "{preset_name}" loaded successfully</p>
                </div>
                """
            else:
                return f"<p style='color: red;'>Failed to load preset: {preset_name}</p>"
                
        except Exception as e:
            return f"<p style='color: red;'>Error: {str(e)}</p>"
            
    load_preset_btn.click(
        load_context_preset,
        inputs=[preset_list],
        outputs=[preset_info]
    )
    
    # Batch generation function
    def generate_batch_kontext(
        base_prompt, variations, ref_images, context_strength
    ):
        """Generate batch with context"""
        try:
            if not base_prompt and not variations:
                return "", [], gr.update()
                
            if not hasattr(app, 'flux_kontext_manager') or not app.flux_kontext_manager.models_loaded:
                return "<p style='color: red;'>Flux Kontext not initialized</p>", [], gr.update()
                
            # Parse variations
            variation_list = [v.strip() for v in variations.split('\n') if v.strip()]
            if not variation_list:
                variation_list = [base_prompt]
                
            # Prepare reference images
            reference_imgs = []
            if ref_images:
                from PIL import Image
                for img_file in ref_images:
                    try:
                        if hasattr(img_file, 'name'):
                            img = Image.open(img_file.name).convert('RGB')
                        else:
                            img = Image.open(img_file).convert('RGB')
                        reference_imgs.append(img)
                    except Exception as e:
                        logger.warning(f"Failed to load reference image: {e}")
                        
            # Generate batch
            results = []
            total = len(variation_list)
            
            for i, variation in enumerate(variation_list):
                full_prompt = f"{base_prompt} {variation}".strip()
                
                progress_html = f"""
                <div style='padding: 10px; background: #e3f2fd; border-radius: 5px;'>
                    <h4>Generating Batch...</h4>
                    <div style='width: 100%; background: #ddd; border-radius: 3px; overflow: hidden;'>
                        <div style='width: {(i+1)/total*100:.1f}%; background: #2196f3; height: 20px;'></div>
                    </div>
                    <p>Processing {i+1}/{total}: {variation[:50]}...</p>
                </div>
                """
                
                # Create config for batch
                config = FluxKontextConfig(context_strength=context_strength)
                
                # Generate image
                result_img, info = app.flux_kontext_manager.generate_with_kontext(
                    prompt=full_prompt,
                    reference_images=reference_imgs,
                    config=config
                )
                
                if result_img:
                    # Save image
                    import uuid
                    output_path = Path(app.output_dir) / f"batch_kontext_{uuid.uuid4()}.png"
                    result_img.save(output_path)
                    results.append(str(output_path))
                    
            if results:
                # Create zip file
                import zipfile
                import uuid
                zip_path = Path(app.output_dir) / f"batch_kontext_{uuid.uuid4()}.zip"
                with zipfile.ZipFile(zip_path, 'w') as zf:
                    for img_path in results:
                        zf.write(img_path, Path(img_path).name)
                        
                final_html = f"""
                <div style='padding: 10px; background: #e8f5e9; border-radius: 5px;'>
                    <h4>✅ Batch Complete!</h4>
                    <p>Generated {len(results)} images with context</p>
                </div>
                """
                
                return final_html, results, gr.update(value=str(zip_path), visible=True)
            else:
                return "<p style='color: red;'>Batch generation failed</p>", [], gr.update()
                
        except Exception as e:
            return f"<p style='color: red;'>Error: {str(e)}</p>", [], gr.update()
            
    batch_generate_btn.click(
        generate_batch_kontext,
        inputs=[
            batch_base_prompt, batch_prompt_variations,
            batch_reference_images, batch_context_strength
        ],
        outputs=[batch_progress, batch_results, batch_download]
    )
    
    # Initialize preset list - set up change event instead of .load()
    # The preset list will be populated when refresh button is clicked