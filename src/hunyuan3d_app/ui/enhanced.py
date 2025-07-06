"""Enhanced UI with all new features integrated"""

import gradio as gr
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .main import create_interface as create_base_interface
from .modern import ModernUI, load_modern_css
from ..config import ALL_IMAGE_MODELS, HUNYUAN3D_MODELS, QUALITY_PRESETS

logger = logging.getLogger(__name__)


def create_enhanced_interface(app):
    """Create enhanced interface with all new features"""
    
    # Load custom CSS
    custom_css = load_modern_css()
    
    with gr.Blocks(
        title="Hunyuan3D Studio Enhanced",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
            neutral_hue="slate"
        ),
        css=custom_css
    ) as interface:
        
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>🎨 Hunyuan3D Studio Enhanced</h1>
            <p>Professional Text → Image → 3D Pipeline with Advanced Features</p>
        </div>
        """)
        
        # Dashboard stats
        with gr.Row():
            stats = app.get_system_stats()
            with gr.Column(scale=1):
                ModernUI.create_stat_card(
                    str(stats.get('total_generations', 0)),
                    "Total Generations",
                    "🎨"
                )
            with gr.Column(scale=1):
                ModernUI.create_stat_card(
                    str(stats.get('models_loaded', 0)),
                    "Models Loaded",
                    "🤖"
                )
            with gr.Column(scale=1):
                ModernUI.create_stat_card(
                    f"{stats.get('vram_used', 0):.1f}GB",
                    "VRAM Used",
                    "💾"
                )
            with gr.Column(scale=1):
                ModernUI.create_stat_card(
                    str(stats.get('queue_pending', 0)),
                    "Queue Pending",
                    "⏳"
                )
        
        # Main tabs
        with gr.Tabs() as main_tabs:
            # Quick Generate Tab (existing functionality)
            with gr.Tab("🚀 Quick Generate"):
                create_quick_generate_tab(app)
                
            # Advanced Pipeline Tab
            with gr.Tab("🔧 Advanced Pipeline"):
                create_advanced_pipeline_tab(app)
                
            # Downloads Manager Tab
            with gr.Tab("📦 Downloads Manager"):
                from .tabs.downloads import create_downloads_manager_tab
                create_downloads_manager_tab(app)
                
            # Media Gallery Tab
            with gr.Tab("🖼️ Media Gallery", elem_id="media_gallery_tab") as media_gallery_tab:
                from .tabs.media_gallery import create_media_gallery_tab
                create_media_gallery_tab(app)
                
            # Queue & History Tab
            with gr.Tab("📋 Queue & History", elem_id="queue_history_tab") as queue_history_tab:
                create_queue_history_tab(app)
                
            # Video Generation Tab
            with gr.Tab("🎬 Video Generation"):
                from .tabs.video import create_video_generation_tab
                create_video_generation_tab(app)
                
            # Character Studio Tab
            with gr.Tab("👤 Character Studio"):
                from .tabs.character import create_character_studio_tab
                create_character_studio_tab(app)
                
            # Face Swap Tab
            with gr.Tab("🔄 Face Swap"):
                from .tabs.face_swap import create_face_swap_tab
                create_face_swap_tab(app)
                
            # Flux Kontext Tab
            with gr.Tab("✨ Flux Kontext"):
                from .tabs.flux_kontext import create_flux_kontext_tab
                create_flux_kontext_tab(app)
                
            # LoRA Studio Tab
            with gr.Tab("🎨 LoRA Studio"):
                from .tabs.lora import create_lora_tab
                create_lora_tab(app)
                
            # Model Comparison Tab
            with gr.Tab("📊 Benchmarks"):
                app.model_comparison.create_ui_component()
                
            # Settings Tab
            with gr.Tab("⚙️ Settings"):
                create_settings_tab(app)
                
        # Add real-time progress display
        from .components.progress import create_progress_component
        progress_display = create_progress_component()
        
        # Badge update system
        def update_tab_badges():
            """Update badges for tabs with counts"""
            # Get queue status
            queue_status = app.queue_manager.get_queue_status()
            queue_count = queue_status.get("pending", 0) + queue_status.get("active", 0)
            
            # Get unviewed media count
            unviewed_count = app.history_manager.get_unviewed_count()
            
            # Create badge CSS/JS with auto-refresh
            badge_script = f"""
            <div id="badge-update-container" style="display: none;">
                <span id="queue-count">{queue_count}</span>
                <span id="unviewed-count">{unviewed_count}</span>
            </div>
            <script>
            (function() {{
                // Function to update badges
                function updateBadges() {{
                    var queueCount = document.getElementById('queue-count') ? parseInt(document.getElementById('queue-count').textContent) : 0;
                    var unviewedCount = document.getElementById('unviewed-count') ? parseInt(document.getElementById('unviewed-count').textContent) : 0;
                    
                    // Update queue badge
                    var queueTab = document.querySelector('#queue_history_tab button');
                    if (queueTab) {{
                        var queueBadge = queueTab.querySelector('.tab-badge');
                        if (!queueBadge) {{
                            queueBadge = document.createElement('span');
                            queueBadge.className = 'tab-badge';
                            queueBadge.style.cssText = 'background: #ff4444; color: white; border-radius: 10px; padding: 2px 6px; font-size: 11px; margin-left: 5px; position: relative; top: -1px; display: none;';
                            queueTab.appendChild(queueBadge);
                        }}
                        if (queueCount > 0) {{
                            queueBadge.textContent = queueCount;
                            queueBadge.style.display = 'inline';
                        }} else {{
                            queueBadge.style.display = 'none';
                        }}
                    }}
                    
                    // Update media gallery badge
                    var mediaTab = document.querySelector('#media_gallery_tab button');
                    if (mediaTab) {{
                        var mediaBadge = mediaTab.querySelector('.tab-badge');
                        if (!mediaBadge) {{
                            mediaBadge = document.createElement('span');
                            mediaBadge.className = 'tab-badge';
                            mediaBadge.style.cssText = 'background: #4CAF50; color: white; border-radius: 10px; padding: 2px 6px; font-size: 11px; margin-left: 5px; position: relative; top: -1px; display: none;';
                            mediaTab.appendChild(mediaBadge);
                        }}
                        if (unviewedCount > 0) {{
                            mediaBadge.textContent = unviewedCount;
                            mediaBadge.style.display = 'inline';
                        }} else {{
                            mediaBadge.style.display = 'none';
                        }}
                    }}
                }}
                
                // Initial update
                updateBadges();
                
                // Set up periodic updates
                setInterval(updateBadges, 2000);
                
                // Also update when counts change
                var observer = new MutationObserver(updateBadges);
                var container = document.getElementById('badge-update-container');
                if (container) {{
                    observer.observe(container, {{ childList: true, subtree: true }});
                }}
            }})();
            </script>
            """
            
            return badge_script
        
        # Add badge updater component that refreshes periodically
        with gr.Row(visible=False):
            badge_updater = gr.HTML(value=update_tab_badges())
            
        # Refresh badges every 2 seconds using Gradio's built-in refresh
        def refresh_badges():
            return gr.update(value=update_tab_badges())
            
        # Set up a timer to refresh badges if the interface supports it
        try:
            interface.load(
                refresh_badges,
                outputs=[badge_updater],
                every=2  # Refresh every 2 seconds
            )
        except Exception as e:
            logger.warning(f"Could not set up badge auto-refresh: {e}")
        
        # Auto-refresh for stats
        def refresh_stats():
            stats = app.get_system_stats()
            # Would need to return updated HTML for stat cards
            pass
            
    return interface


def create_quick_generate_tab(app):
    """Create the quick generate tab with existing functionality"""
    
    # Get downloaded models using the improved detection
    downloaded_image_models = app.model_manager.get_downloaded_models("image")
    downloaded_3d_models = app.model_manager.get_downloaded_models("3d")
    
    # Auto-select single models
    default_image_model = downloaded_image_models[0] if len(downloaded_image_models) == 1 else (downloaded_image_models[0] if downloaded_image_models else None)
    default_3d_model = downloaded_3d_models[0] if len(downloaded_3d_models) == 1 else (downloaded_3d_models[0] if downloaded_3d_models else None)
    
    # Add mode selection
    with gr.Row():
        image_only_mode = gr.Checkbox(
            label="🖼️ Generate Image Only (skip 3D generation)",
            value=False
        )
    
    gr.Markdown("### Generation Pipeline")
    
    with gr.Row():
        with gr.Column(scale=3):
            # Text prompt
            prompt = gr.Textbox(
                label="What would you like to create?",
                placeholder="A majestic golden crown with intricate details...",
                lines=3
            )
            negative_prompt = gr.Textbox(
                label="What to avoid",
                value="blurry, low quality, multiple objects",
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
                        label="Image Model (No models downloaded - visit Downloads Manager)",
                        interactive=False
                    )
                
                if downloaded_3d_models:
                    hunyuan_model = gr.Dropdown(
                        choices=downloaded_3d_models,
                        value=default_3d_model,
                        label="3D Model",
                        visible=True
                    )
                else:
                    hunyuan_model = gr.Dropdown(
                        choices=[],
                        label="3D Model (No models downloaded - visit Downloads Manager)",
                        interactive=False,
                        visible=True
                    )
                
            # Generation settings
            with gr.Accordion("Generation Settings", open=False):
                with gr.Row():
                    width = gr.Slider(512, 2048, 1024, step=64, label="Width")
                    height = gr.Slider(512, 2048, 1024, step=64, label="Height")
                with gr.Row():
                    steps = gr.Slider(10, 100, 30, step=5, label="Steps")
                    guidance = gr.Slider(1, 20, 7.5, step=0.5, label="Guidance")
                    seed = gr.Number(-1, label="Seed (-1 for random)")
                    
            # Generate button
            generate_btn = gr.Button("🎨 Generate 3D Asset", variant="primary", size="lg")
            
        with gr.Column(scale=2):
            # Progress display
            with gr.Group():
                generation_progress = gr.Progress()
                progress_text = gr.HTML(value="<p style='text-align: center; color: #666;'>Ready to generate</p>")
                progress_status = gr.HTML(value="", visible=True)
            
            # Results
            output_image = gr.Image(label="Generated Image", type="pil")
            output_3d = gr.Model3D(label="3D Model", visible=True)
            output_info = gr.HTML()
    
    # Update UI based on mode
    def update_ui_mode(image_only):
        if image_only:
            return {
                hunyuan_model: gr.update(visible=False),
                output_3d: gr.update(visible=False),
                generate_btn: gr.update(value="🎨 Generate Image")
            }
        else:
            return {
                hunyuan_model: gr.update(visible=True),
                output_3d: gr.update(visible=True),
                generate_btn: gr.update(value="🎨 Generate 3D Asset")
            }
    
    image_only_mode.change(
        update_ui_mode,
        inputs=[image_only_mode],
        outputs=[hunyuan_model, output_3d, generate_btn]
    )
            
    # Wire up the generation
    def generate_pipeline(image_only, prompt, negative_prompt, image_model, hunyuan_model, 
                         width, height, steps, guidance, seed, progress=gr.Progress()):
        # Check if models are available
        if not image_model:
            return None, None, "❌ No image model selected. Please download a model from the Downloads Manager tab.", "<p style='color: red;'>No model selected</p>"
        
        if not image_only and not hunyuan_model:
            return None, None, "❌ No 3D model selected. Please download a model from the Downloads Manager tab or enable 'Generate Image Only' mode.", "<p style='color: red;'>No 3D model selected</p>"
        
        try:
            progress(0.05, "Submitting generation job...")
            
            # For image-only mode, we can run directly for better UX
            if image_only:
                progress(0.1, "Loading image model...")
                
                # Load model if needed
                if app.image_model_name != image_model:
                    status, model, model_name = app.model_manager.load_image_model(
                        image_model, app.image_model, app.image_model_name, "cuda", progress
                    )
                    if "❌" in status:
                        return None, None, f"❌ Failed to load model: {status}", f"<p style='color: red;'>Model loading failed</p>"
                    app.image_model = model
                    app.image_model_name = model_name
                
                progress(0.3, "Generating image...")
                
                # Generate image directly
                image, info = app.image_generator.generate_image(
                    app.image_model,
                    app.image_model_name,
                    prompt,
                    negative_prompt,
                    width,
                    height,
                    steps,
                    guidance,
                    seed,
                    progress=lambda p, msg: progress(0.3 + p * 0.6, msg)
                )
                
                if image:
                    progress(1.0, "Generation complete!")
                    return image, None, "✅ Image generated successfully!", f"<p style='color: green;'>Generation complete!</p>"
                else:
                    return None, None, f"❌ Generation failed: {info}", f"<p style='color: red;'>Generation failed</p>"
            else:
                # For full pipeline, submit job and wait for results
                progress(0.1, "Submitting full pipeline job...")
                
                job_id = app.submit_generation_job(
                    job_type="full_pipeline",
                    params={
                        "prompt": prompt,
                        "negative_prompt": negative_prompt,
                        "model_name": image_model,
                        "hunyuan3d_model_name": hunyuan_model,
                        "width": width,
                        "height": height,
                        "steps": steps,
                        "guidance_scale": guidance,
                        "seed": seed
                    }
                )
                
                progress(0.15, f"Job {job_id[:8]} submitted, waiting for completion...")
                
                # Poll for job completion with progress updates
                import time
                job = app.queue_manager.get_job(job_id)
                last_progress = 0.15
                poll_count = 0
                
                while job and job.status.value in ["pending", "running"]:
                    time.sleep(0.5)  # Poll every 500ms
                    job = app.queue_manager.get_job(job_id)
                    poll_count += 1
                    
                    if job:
                        # Update progress based on job progress
                        job_progress = job.progress
                        current_progress = 0.15 + job_progress * 0.8
                        
                        # Show detailed progress message
                        if job.progress_message:
                            progress_msg = job.progress_message
                        elif job.status.value == "pending":
                            progress_msg = f"Waiting in queue... (position: ~{poll_count//2})"
                        else:
                            progress_msg = f"Processing... ({current_progress*100:.0f}%)"
                        
                        progress(current_progress, progress_msg)
                        last_progress = current_progress
                    
                    # Add timeout to prevent infinite loops
                    if poll_count > 600:  # 5 minutes timeout (full pipeline can take 2-3 minutes)
                        logger.warning(f"Job {job_id} timed out after {poll_count/2} seconds")
                        break
                
                # Check final job status
                if not job:
                    return None, None, "❌ Job disappeared from queue", "<p style='color: red;'>Job error</p>"
                
                logger.info(f"Final job status for {job_id}: {job.status.value}")
                logger.info(f"Job result present: {job.result is not None}")
                if job.result:
                    logger.info(f"Job result type: {type(job.result)}")
                
                if job.status.value == "failed":
                    return None, None, f"❌ Generation failed: {job.error}", f"<p style='color: red;'>Error: {job.error}</p>"
                
                if job.status.value == "completed":
                    progress(1.0, "Generation complete!")
                    
                    # Check if job has results
                    if not job.result:
                        logger.error(f"Job {job_id} completed but result is None/empty")
                        logger.error(f"Job attributes: status={job.status}, id={job.id}, type={job.type}")
                        
                        # Try to find the result from job history as a fallback
                        try:
                            from pathlib import Path
                            import json
                            job_history_paths = [
                                Path("outputs/job_history") / f"{job_id}.json",
                                app.output_dir / "job_history" / f"{job_id}.json" if hasattr(app, 'output_dir') else None
                            ]
                            
                            for history_path in job_history_paths:
                                if history_path and history_path.exists():
                                    logger.info(f"Attempting to load job result from history: {history_path}")
                                    with open(history_path, 'r') as f:
                                        job_data = json.load(f)
                                        if job_data.get('result'):
                                            job.result = job_data['result']
                                            logger.info(f"Loaded job result from history with keys: {list(job.result.keys())}")
                                            break
                        except Exception as e:
                            logger.error(f"Failed to load job from history: {e}")
                        
                        # If still no result, return error
                        if not job.result:
                            return None, None, "❌ Job completed but result data is missing", "<p style='color: red;'>No result data</p>"
                    
                    result = job.result
                    logger.info(f"Job {job_id} result keys: {list(result.keys())}")
                    
                    # Extract results - handle both direct image and serialized image_path
                    image = result.get("image")
                    image_path = result.get("image_path")
                    
                    # Try to load image from path if image object not present
                    if not image and image_path:
                        try:
                            from PIL import Image as PILImage
                            if Path(image_path).exists():
                                image = PILImage.open(image_path)
                                logger.info(f"Loaded image from path: {image_path}")
                            else:
                                logger.warning(f"Image path does not exist: {image_path}")
                        except Exception as e:
                            logger.warning(f"Failed to load image from path: {e}")
                    
                    # Also check for the original path field (not the serialized one)
                    original_image_path = result.get("path")
                    
                    mesh_path = result.get("mesh_path")
                    image_info = result.get("image_info", "")
                    mesh_info = result.get("mesh_info", "")
                    
                    # Check if we have any valid results
                    has_results = (image is not None or image_path is not None or 
                                 mesh_path is not None or original_image_path is not None)
                    
                    # Combine info
                    if result.get("success"):
                        combined_info = f"""
                        <div style="background: #e8f5e9; padding: 15px; border-radius: 8px; margin: 10px 0;">
                            <h4>✅ Full Pipeline Complete!</h4>
                            <div style="margin: 10px 0;">
                                <strong>Image Generation:</strong> ✅ Success<br>
                                <strong>3D Conversion:</strong> ✅ Success<br>
                                <strong>Output:</strong> {mesh_path.split('/')[-1] if mesh_path else 'N/A'}
                            </div>
                        </div>
                        {mesh_info}
                        """
                    else:
                        combined_info = f"""
                        <div style="background: #fff3e0; padding: 15px; border-radius: 8px; margin: 10px 0;">
                            <h4>⚠️ Partial Success</h4>
                            <div style="margin: 10px 0;">
                                <strong>Image Generation:</strong> ✅ Success<br>
                                <strong>3D Conversion:</strong> ❌ {result.get('error', 'Failed')}<br>
                            </div>
                        </div>
                        """
                    
                    if has_results:
                        return image, mesh_path, combined_info, "<p style='color: green;'>Pipeline complete!</p>"
                    else:
                        # Log the actual result for debugging
                        logger.warning(f"Job completed but no displayable results found. Result keys: {list(result.keys())}")
                        return None, None, "❌ Job completed but no displayable results found", "<p style='color: red;'>No results</p>"
                elif poll_count > 600:
                    # Timeout reached
                    return None, None, "⏱️ Generation timed out. The job may still be running in the background. Check the output folder.", "<p style='color: orange;'>Timeout - check outputs folder</p>"
                else:
                    # Job is not completed yet
                    if job.status.value == "running":
                        return None, None, "⏳ Job is still running... Please wait for completion.", "<p style='color: orange;'>Still processing...</p>"
                    else:
                        return None, None, f"❌ Job status: {job.status.value}", f"<p style='color: red;'>Job not completed</p>"
                
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return None, None, f"❌ Generation failed: {str(e)}", f"<p style='color: red;'>Error: {str(e)}</p>"
        
    generate_btn.click(
        generate_pipeline,
        inputs=[image_only_mode, prompt, negative_prompt, image_model, hunyuan_model,
                width, height, steps, guidance, seed],
        outputs=[output_image, output_3d, output_info, progress_text]
    )


def create_advanced_pipeline_tab(app):
    """Create advanced pipeline with LoRA support"""
    gr.Markdown("### Advanced Generation with LoRA & Custom Settings")
    
    with gr.Row():
        with gr.Column():
            # Model and LoRA selection
            with gr.Group():
                gr.Markdown("#### Model Configuration")
                
                base_model = gr.Dropdown(
                    choices=[m for m in ALL_IMAGE_MODELS.keys()],
                    label="Base Model"
                )
                
                # LoRA selection
                available_loras = app.get_available_loras()
                lora_choices = [(l["name"], l["name"]) for l in available_loras]
                
                with gr.Group():
                    gr.Markdown("#### LoRA Configuration")
                    lora_1 = gr.Dropdown(choices=lora_choices, label="LoRA 1", value=None)
                    lora_1_weight = gr.Slider(0, 2, 1, step=0.1, label="LoRA 1 Weight")
                    
                    lora_2 = gr.Dropdown(choices=lora_choices, label="LoRA 2", value=None)
                    lora_2_weight = gr.Slider(0, 2, 1, step=0.1, label="LoRA 2 Weight")
                    
            # Prompt with auto-suggestion
            prompt = gr.Textbox(label="Prompt", lines=3)
            negative_prompt = gr.Textbox(label="Negative Prompt", lines=2)
            
            # LoRA auto-suggestion
            with gr.Group():
                auto_suggest_loras = gr.Checkbox(
                    label="🤖 Auto-suggest LoRAs based on prompt",
                    value=True
                )
                
                suggestion_status = gr.HTML(visible=False)
                
                with gr.Row(visible=False) as suggestion_row:
                    suggested_loras = gr.CheckboxGroup(
                        label="Suggested LoRAs",
                        choices=[],
                        value=[]
                    )
                    
                    download_suggested_btn = gr.Button(
                        "📥 Download Selected",
                        size="sm"
                    )
            
        with gr.Column():
            # Output
            output = gr.Image(label="Result", type="pil")
            info = gr.HTML()
            
    # Generation button
    generate_btn = gr.Button("Generate with LoRAs", variant="primary")
    
    def generate_with_loras(base_model, lora_1, lora_1_weight, lora_2, lora_2_weight,
                           prompt, negative_prompt):
        # Build LoRA configs
        lora_configs = []
        if lora_1:
            lora_info = next((l for l in available_loras if l["name"] == lora_1), None)
            if lora_info:
                # Need to create LoRAInfo object
                from ..features.lora.manager import LoRAInfo
                lora_obj = LoRAInfo(
                    name=lora_info["name"],
                    path=Path(lora_info["path"]),
                    base_model=lora_info["base_model"],
                    trigger_words=lora_info["trigger_words"]
                )
                lora_configs.append((lora_obj, lora_1_weight))
                
        if lora_2:
            lora_info = next((l for l in available_loras if l["name"] == lora_2), None)
            if lora_info:
                from ..features.lora.manager import LoRAInfo
                lora_obj = LoRAInfo(
                    name=lora_info["name"],
                    path=Path(lora_info["path"]),
                    base_model=lora_info["base_model"],
                    trigger_words=lora_info["trigger_words"]
                )
                lora_configs.append((lora_obj, lora_2_weight))
                
        # Submit job
        job_id = app.submit_generation_job(
            job_type="image",
            params={
                "model_name": base_model,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "lora_configs": lora_configs,
                "width": 1024,
                "height": 1024,
                "steps": 30,
                "guidance_scale": 7.5,
                "seed": -1
            }
        )
        
        return None, f"Job submitted: {job_id[:8]}..."
        
    generate_btn.click(
        generate_with_loras,
        inputs=[base_model, lora_1, lora_1_weight, lora_2, lora_2_weight,
                prompt, negative_prompt],
        outputs=[output, info]
    )


def create_model_hub_tab(app):
    """Create model hub with Civitai integration"""
    with gr.Tabs():
        # Civitai search
        with gr.Tab("🔍 Search Civitai"):
            with gr.Row():
                search_query = gr.Textbox(
                    label="Search Query",
                    placeholder="anime, realistic, fantasy..."
                )
                model_type = gr.Dropdown(
                    choices=["LORA", "Checkpoint", "TextualInversion"],
                    value="LORA",
                    label="Model Type"
                )
                base_model = gr.Dropdown(
                    choices=["FLUX.1", "SDXL", "SD 1.5"],
                    value="FLUX.1",
                    label="Base Model"
                )
                search_btn = gr.Button("Search", variant="primary")
                
            # Results
            search_results = gr.HTML()
            
            def search_civitai(query, model_type, base_model):
                results = app.search_civitai_models(query, model_type, base_model)
                
                html = "<div class='bento-container bento-grid-2'>"
                for model in results[:10]:
                    html += f"""
                    <div class='bento-card'>
                        <h4>{model['name']}</h4>
                        <p>{model['description'][:100]}...</p>
                        <div style='margin-top: 0.5rem;'>
                            <span>📥 {model['downloads']:,}</span>
                            <span style='margin-left: 1rem;'>❤️ {model['likes']:,}</span>
                        </div>
                        <button class='quick-action-btn' onclick='downloadModel({model['id']})'>
                            Download
                        </button>
                    </div>
                    """
                html += "</div>"
                
                return html
                
            search_btn.click(
                search_civitai,
                inputs=[search_query, model_type, base_model],
                outputs=[search_results]
            )
            
        # Downloaded models
        with gr.Tab("📦 Downloaded Models"):
            gr.Markdown("### Your Downloaded Models")
            
            # LoRA list
            lora_gallery = gr.Gallery(
                label="LoRAs",
                columns=4,
                height="auto"
            )
            
            def load_lora_gallery():
                loras = app.get_available_loras()
                # Would need to create thumbnails/previews
                return []
                
            # Refresh on tab load
            # Add refresh button since HTML components don't support .load()
            refresh_lora_btn = gr.Button("🔄 Refresh LoRAs", variant="secondary")
            refresh_lora_btn.click(load_lora_gallery, outputs=[lora_gallery])


def create_queue_history_tab(app):
    """Create queue and history management tab"""
    with gr.Tabs():
        # Queue management
        with gr.Tab("⏳ Queue"):
            app.queue_manager.create_ui_component()
            
        # Generation history
        with gr.Tab("📚 History"):
            app.history_manager.create_ui_component()


def create_settings_tab(app):
    """Create settings tab with credential management"""
    with gr.Tabs():
        # API Credentials
        with gr.Tab("🔐 API Credentials"):
            app.credential_manager.create_ui_component()
            
        # Performance Settings
        with gr.Tab("⚡ Performance"):
            gr.Markdown("### Performance Settings")
            
            with gr.Group():
                gr.Markdown("#### Memory Optimization")
                
                enable_xformers = gr.Checkbox(
                    label="Enable xFormers (memory efficient attention)",
                    value=True
                )
                
                enable_cpu_offload = gr.Checkbox(
                    label="Enable CPU Offload (for low VRAM)",
                    value=False
                )
                
                batch_size = gr.Slider(
                    1, 8, 1, step=1,
                    label="Batch Size"
                )
                
            with gr.Group():
                gr.Markdown("#### Queue Settings")
                
                max_workers = gr.Slider(
                    1, 4, 2, step=1,
                    label="Max Concurrent Jobs"
                )
                
                auto_cleanup_days = gr.Slider(
                    7, 90, 30, step=1,
                    label="Auto-cleanup after (days)"
                )
                
            save_settings_btn = gr.Button("💾 Save Settings", variant="primary")
            
        # System Info
        with gr.Tab("💻 System Info"):
            system_info = gr.HTML(app.check_system_requirements())
            
            refresh_btn = gr.Button("🔄 Refresh System Info")
            refresh_btn.click(
                app.check_system_requirements,
                outputs=[system_info]
            )