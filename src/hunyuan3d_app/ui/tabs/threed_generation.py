"""3D Generation Tab - Consolidated 3D features"""

import gradio as gr
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging

from ..components.media_sidebar import create_media_sidebar
from ..components.common import create_generation_settings, create_action_button
from ...config import HUNYUAN3D_MODELS

logger = logging.getLogger(__name__)


def create_threed_generation_tab(app: Any) -> None:
    """Create the unified 3D generation tab with all 3D-related features"""
    
    with gr.Row():
        # Main content area
        with gr.Column(scale=4):
            gr.Markdown("## 🎲 3D Generation Studio")
            gr.Markdown("Create and edit 3D models with AI-powered tools")
            
            # Feature tabs
            with gr.Tabs() as feature_tabs:
                # 1. Generate 3D Object
                with gr.Tab("Generate 3D Object", elem_id="generate_3d_tab"):
                    create_generate_3d_subtab(app)
                
                # 2. Image to 3D
                with gr.Tab("Image to 3D", elem_id="image_to_3d_tab"):
                    create_image_to_3d_subtab(app)
                
                # 3. 3D Model Variations
                with gr.Tab("3D Variations", elem_id="3d_variations_tab"):
                    create_3d_variations_subtab(app)
                
                # 4. Basic 3D Textures
                with gr.Tab("3D Textures", elem_id="3d_textures_tab"):
                    create_3d_textures_subtab(app)
        
        # Sidebar with recent 3D models
        sidebar = create_media_sidebar(
            app,
            media_type="3d",
            on_select_callback=None,
            title="Recent 3D Models"
        )


def create_generate_3d_subtab(app: Any) -> None:
    """Create the main text-to-3D generation interface"""
    
    with gr.Column():
        # Get downloaded models - wrap in try-except to avoid blocking
        try:
            downloaded_image_models = app.model_manager.get_downloaded_models("image")
            downloaded_3d_models = app.model_manager.get_downloaded_models("3d")
        except Exception as e:
            logger.error(f"Error getting downloaded models: {e}")
            downloaded_image_models = []
            downloaded_3d_models = []
        
        default_image_model = downloaded_image_models[0] if downloaded_image_models else None
        default_3d_model = downloaded_3d_models[0] if downloaded_3d_models else None
        
        # Prompt inputs
        prompt = gr.Textbox(
            label="3D Object Description",
            placeholder="A detailed medieval sword with ornate handle...",
            lines=3
        )
        
        negative_prompt = gr.Textbox(
            label="Negative Prompt",
            value="low quality, distorted, multiple objects",
            lines=2
        )
        
        # Model selection
        with gr.Row():
            if downloaded_image_models:
                image_model = gr.Dropdown(
                    choices=downloaded_image_models,
                    value=default_image_model,
                    label="Image Model"
                )
            else:
                image_model = gr.Dropdown(
                    choices=[],
                    label="Image Model (No models downloaded)",
                    interactive=False
                )
            
            if downloaded_3d_models:
                hunyuan_model = gr.Dropdown(
                    choices=downloaded_3d_models,
                    value=default_3d_model,
                    label="3D Model"
                )
            else:
                hunyuan_model = gr.Dropdown(
                    choices=[],
                    label="3D Model (No models downloaded)",
                    interactive=False
                )
        
        # 3D generation settings
        with gr.Accordion("3D Settings", open=True):
            with gr.Row():
                num_views = gr.Slider(
                    4, 16, 8,
                    step=2,
                    label="Number of Views",
                    info="More views = better quality but slower"
                )
                
                mesh_resolution = gr.Slider(
                    256, 1024, 512,
                    step=128,
                    label="Mesh Resolution"
                )
            
            with gr.Row():
                texture_resolution = gr.Slider(
                    512, 2048, 1024,
                    step=256,
                    label="Texture Resolution"
                )
                
                output_format = gr.Dropdown(
                    choices=["glb", "obj", "ply", "stl"],
                    value="glb",
                    label="Output Format"
                )
        
        # Image generation settings
        with gr.Accordion("Image Generation Settings", open=False):
            seed, steps, cfg, width, height = create_generation_settings()
        
        # Model status indicator
        with gr.Row():
            model_status = gr.HTML(value="""
            <div style='padding: 10px; background: #e3f2fd; border-radius: 5px; margin-bottom: 10px;'>
                <strong>ℹ️ Model Status:</strong> Select a 3D model to see its status
            </div>
            """)
        
        # Generate button
        generate_btn = create_action_button("🎲 Generate 3D Model", variant="primary")
        
        # Output
        with gr.Row():
            with gr.Column():
                output_image = gr.Image(label="Generated Image", type="pil")
                output_3d = gr.Model3D(label="3D Model")
            
            generation_info = gr.HTML()
        
        # Model status check function
        def check_model_status(model_name):
            """Check and display the status of the selected 3D model"""
            if not model_name or "(not downloaded)" in model_name:
                return """
                <div style='padding: 10px; background: #ffebee; border-radius: 5px;'>
                    <strong>❌ Model Not Available:</strong> Please download the model first
                </div>
                """
            
            # Try to check if model can load properly
            try:
                # This is a simplified check - in reality we'd check if the model loads
                hunyuan3d_path = Path("Hunyuan3D")
                if not hunyuan3d_path.exists():
                    return """
                    <div style='padding: 10px; background: #fff3e0; border-radius: 5px;'>
                        <strong>⚠️ Warning:</strong> Hunyuan3D directory not found. The model may fall back to demo mode.
                        <br>Install with: <code>pip install -e ./Hunyuan3D</code>
                    </div>
                    """
                
                # Check if hy3dshape can be imported
                try:
                    import sys
                    hy3dshape_path = hunyuan3d_path / "hy3dshape"
                    if str(hy3dshape_path) not in sys.path:
                        sys.path.insert(0, str(hy3dshape_path))
                    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
                    return f"""
                    <div style='padding: 10px; background: #e8f5e9; border-radius: 5px;'>
                        <strong>✅ Model Ready:</strong> {model_name} is available and should work properly
                    </div>
                    """
                except ImportError:
                    return f"""
                    <div style='padding: 10px; background: #fff3e0; border-radius: 5px;'>
                        <strong>⚠️ Demo Mode:</strong> {model_name} will use simplified demo generation
                        <br><small>Hunyuan3D modules not installed. Run: <code>python install_hunyuan3d.py</code></small>
                    </div>
                    """
            except Exception as e:
                return f"""
                <div style='padding: 10px; background: #ffebee; border-radius: 5px;'>
                    <strong>❌ Error:</strong> {str(e)}
                </div>
                """
        
        # Wire up model selection to status check
        hunyuan_model.change(
            fn=check_model_status,
            inputs=[hunyuan_model],
            outputs=[model_status]
        )
        
        # Wire up generation
        def generate_3d_object(prompt, negative_prompt, img_model, model_3d,
                             num_views, mesh_res, tex_res, format,
                             seed, steps, cfg, width, height,
                             progress=gr.Progress()):
            """Generate 3D object from text"""
            
            if not img_model or not model_3d:
                return None, None, "❌ Please select both image and 3D models"
            
            try:
                # Use direct generation for real-time progress updates
                result = app.generate_3d_direct(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image_model_name=img_model,
                    hunyuan3d_model_name=model_3d,
                    width=width,
                    height=height,
                    steps=steps,
                    guidance_scale=cfg,
                    seed=seed,
                    num_views=num_views,
                    mesh_resolution=mesh_res,
                    texture_resolution=tex_res,
                    output_format=format,
                    progress_callback=progress
                )
                
                if result["success"]:
                    return result["image"], result["mesh_path"], f"""
                    <div class="success-box">
                        <h4>✅ 3D Model Generated!</h4>
                        <ul>
                            <li><strong>Views:</strong> {num_views}</li>
                            <li><strong>Mesh Resolution:</strong> {mesh_res}</li>
                            <li><strong>Texture:</strong> {tex_res}x{tex_res}</li>
                            <li><strong>Format:</strong> {format.upper()}</li>
                        </ul>
                    </div>
                    """
                else:
                    return None, None, f"❌ Generation failed: {result['error']}"
                    
            except Exception as e:
                logger.error(f"3D generation error: {e}")
                return None, None, f"❌ Error: {str(e)}"
        
        generate_btn.click(
            generate_3d_object,
            inputs=[prompt, negative_prompt, image_model, hunyuan_model,
                   num_views, mesh_resolution, texture_resolution, output_format,
                   seed, steps, cfg, width, height],
            outputs=[output_image, output_3d, generation_info]
        )


def create_image_to_3d_subtab(app: Any) -> None:
    """Create image-to-3D conversion interface"""
    
    with gr.Column():
        gr.Markdown("### Convert any image into a 3D model")
        
        # Input image
        input_image = gr.Image(
            label="Input Image",
            type="pil",
            height=400
        )
        
        # Model selection
        try:
            downloaded_3d_models = app.model_manager.get_downloaded_models("3d")
        except Exception as e:
            logger.error(f"Error getting downloaded models: {e}")
            downloaded_3d_models = []
        if downloaded_3d_models:
            model_3d = gr.Dropdown(
                choices=downloaded_3d_models,
                value=downloaded_3d_models[0],
                label="3D Model"
            )
        else:
            model_3d = gr.Dropdown(
                choices=[],
                label="3D Model (No models downloaded)",
                interactive=False
            )
        
        # Conversion settings
        with gr.Row():
            remove_bg = gr.Checkbox(
                label="Remove Background",
                value=True,
                info="Automatically remove background for better results"
            )
            
            auto_center = gr.Checkbox(
                label="Auto Center Object",
                value=True
            )
        
        # 3D settings
        with gr.Row():
            mesh_resolution = gr.Slider(
                256, 1024, 512,
                step=128,
                label="Mesh Resolution"
            )
            
            texture_resolution = gr.Slider(
                512, 2048, 1024,
                step=256,
                label="Texture Resolution"
            )
        
        with gr.Row():
            output_format = gr.Dropdown(
                choices=["glb", "obj", "ply", "stl"],
                value="glb",
                label="Output Format"
            )
            
            quality_preset = gr.Radio(
                choices=["Fast", "Balanced", "Quality"],
                value="Balanced",
                label="Quality Preset"
            )
        
        # Convert button
        convert_btn = create_action_button("🔄 Convert to 3D", variant="primary")
        
        # Output
        with gr.Row():
            output_3d = gr.Model3D(label="3D Model")
            conversion_info = gr.HTML()
        
        # Wire up conversion
        def convert_image_to_3d(input_img, model, remove_background, center,
                              mesh_res, tex_res, format, quality,
                              progress=gr.Progress()):
            """Convert image to 3D model"""
            
            if not input_img:
                return None, "❌ Please provide an input image"
            if not model:
                return None, "❌ Please select a 3D model"
            
            try:
                progress(0.1, "Preparing image...")
                
                # Remove background if requested
                processed_img = input_img
                if remove_background:
                    progress(0.2, "Removing background...")
                    processed_img = app.image_generator.remove_background(input_img)
                
                # Submit 3D conversion job
                progress(0.3, "Submitting conversion job...")
                
                job_id = app.submit_generation_job(
                    job_type="3d",
                    params={
                        "image": processed_img,
                        "model_name": model,
                        "num_views": 8 if quality == "Balanced" else (4 if quality == "Fast" else 16),
                        "mesh_resolution": mesh_res,
                        "texture_resolution": tex_res,
                        "output_format": format
                    }
                )
                
                # Poll for completion
                import time
                job = app.queue_manager.get_job(job_id)
                last_progress = -1  # Track last progress to avoid duplicate logs
                
                while job and job.status.value in ["pending", "running"]:
                    time.sleep(0.5)
                    job = app.queue_manager.get_job(job_id)
                    
                    if job:
                        # Ensure job.progress is a valid number
                        job_progress = job.progress if job.progress is not None else 0.0
                        current_progress = 0.3 + job_progress * 0.65
                        msg = job.progress_message or f"Converting... ({current_progress*100:.0f}%)"
                        progress(current_progress, msg)
                        
                        # Log progress for debugging only when it changes
                        if job_progress != last_progress:
                            logger.info(f"UI Poll (Image2D) - Job {job_id}: status={job.status.value}, progress={job_progress:.3f}, message={msg}")
                            last_progress = job_progress
                
                if job and job.status.value == "completed" and job.result:
                    mesh_path = job.result.get("mesh_path")
                    
                    progress(1.0, "Conversion complete!")
                    
                    return mesh_path, f"""
                    <div class="success-box">
                        <h4>✅ 3D Model Created!</h4>
                        <ul>
                            <li><strong>Quality:</strong> {quality}</li>
                            <li><strong>Mesh Resolution:</strong> {mesh_res}</li>
                            <li><strong>Texture:</strong> {tex_res}x{tex_res}</li>
                            <li><strong>Format:</strong> {format.upper()}</li>
                        </ul>
                    </div>
                    """
                else:
                    error = job.error if job else "Job not found"
                    return None, f"❌ Conversion failed: {error}"
                    
            except Exception as e:
                logger.error(f"Image to 3D error: {e}")
                return None, f"❌ Error: {str(e)}"
        
        convert_btn.click(
            convert_image_to_3d,
            inputs=[input_image, model_3d, remove_bg, auto_center,
                   mesh_resolution, texture_resolution, output_format, quality_preset],
            outputs=[output_3d, conversion_info]
        )


def create_3d_variations_subtab(app: Any) -> None:
    """Create 3D model variations interface"""
    
    with gr.Column():
        gr.Markdown("### Generate variations of existing 3D models")
        
        # Input 3D model
        input_model = gr.Model3D(
            label="Reference 3D Model",
            height=400
        )
        
        # Variation settings
        with gr.Row():
            variation_type = gr.Radio(
                choices=["Geometry", "Texture", "Both"],
                value="Both",
                label="Variation Type"
            )
            
            variation_strength = gr.Slider(
                0.1, 0.9, 0.3,
                step=0.05,
                label="Variation Strength"
            )
        
        # Variation guide
        variation_prompt = gr.Textbox(
            label="Variation Guide",
            placeholder="Make it more ornate, add battle damage...",
            lines=2
        )
        
        with gr.Row():
            num_variations = gr.Slider(
                1, 4, 2,
                step=1,
                label="Number of Variations"
            )
            
            preserve_topology = gr.Checkbox(
                label="Preserve Topology",
                value=True,
                info="Keep the same mesh structure"
            )
        
        # Generate button
        generate_btn = create_action_button("🎲 Generate Variations", variant="primary")
        
        # Output gallery
        output_gallery = gr.Gallery(
            label="3D Variations",
            columns=2,
            rows=2,
            height="auto"
        )
        variations_info = gr.HTML()
        
        # Placeholder for 3D variations
        def generate_3d_variations(input_3d, var_type, strength, prompt, num, preserve):
            if not input_3d:
                return [], "❌ Please provide a reference 3D model"
            
            # TODO: Implement 3D variations
            return [], "🚧 3D variations feature coming soon!"
        
        generate_btn.click(
            generate_3d_variations,
            inputs=[input_model, variation_type, variation_strength,
                   variation_prompt, num_variations, preserve_topology],
            outputs=[output_gallery, variations_info]
        )


def create_3d_textures_subtab(app: Any) -> None:
    """Create 3D texture editing interface"""
    
    with gr.Column():
        gr.Markdown("### Edit and generate textures for 3D models")
        
        # Input 3D model
        input_model = gr.Model3D(
            label="Input 3D Model",
            height=400
        )
        
        # Texture options
        texture_mode = gr.Radio(
            choices=["Generate New", "Edit Existing", "Apply Style"],
            value="Generate New",
            label="Texture Mode"
        )
        
        # Texture description
        texture_prompt = gr.Textbox(
            label="Texture Description",
            placeholder="Weathered bronze with verdigris patina...",
            lines=2
        )
        
        # Texture settings
        with gr.Row():
            texture_resolution = gr.Dropdown(
                choices=["512", "1024", "2048", "4096"],
                value="1024",
                label="Texture Resolution"
            )
            
            texture_type = gr.Dropdown(
                choices=["Diffuse", "PBR", "Stylized"],
                value="PBR",
                label="Texture Type"
            )
        
        # Advanced options
        with gr.Accordion("Advanced Texture Options", open=False):
            with gr.Row():
                roughness = gr.Slider(
                    0, 1, 0.5,
                    step=0.1,
                    label="Roughness"
                )
                
                metallic = gr.Slider(
                    0, 1, 0.0,
                    step=0.1,
                    label="Metallic"
                )
            
            with gr.Row():
                normal_strength = gr.Slider(
                    0, 2, 1.0,
                    step=0.1,
                    label="Normal Map Strength"
                )
                
                ao_strength = gr.Slider(
                    0, 1, 0.8,
                    step=0.1,
                    label="Ambient Occlusion"
                )
        
        # Apply button
        apply_btn = create_action_button("🎨 Apply Texture", variant="primary")
        
        # Output
        with gr.Row():
            output_3d = gr.Model3D(label="Textured Model")
            texture_info = gr.HTML()
        
        # Texture preview
        texture_preview = gr.Image(
            label="Texture Preview",
            type="pil",
            visible=False
        )
        
        # Placeholder for texture generation
        def apply_texture(input_3d, mode, prompt, resolution, tex_type,
                        rough, metal, normal, ao):
            if not input_3d:
                return None, "", None, "❌ Please provide a 3D model"
            
            # TODO: Implement texture generation/editing
            return None, None, None, "🚧 Texture editing feature coming soon!"
        
        apply_btn.click(
            apply_texture,
            inputs=[input_model, texture_mode, texture_prompt, texture_resolution,
                   texture_type, roughness, metallic, normal_strength, ao_strength],
            outputs=[output_3d, texture_preview, texture_info]
        )