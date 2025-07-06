"""Comprehensive Downloads Manager UI for all model types"""

import gradio as gr
from pathlib import Path
from typing import Dict, List, Tuple

from ...config import (
    ALL_IMAGE_MODELS, HUNYUAN3D_MODELS, VIDEO_MODELS,
    IP_ADAPTER_MODELS, FACE_SWAP_MODELS, FACE_RESTORE_MODELS
)


def create_downloads_manager_tab(app):
    """Create comprehensive Downloads Manager UI"""
    
    with gr.Tabs():
        # Image & 3D Models Tab
        with gr.Tab("🎨 Image & 3D Models"):
            create_image_3d_downloads_tab(app)
            
        # Video Models Tab
        with gr.Tab("🎬 Video Models"):
            create_video_downloads_tab(app)
            
        # Character & Face Models Tab
        with gr.Tab("👤 Character & Face Models"):
            create_character_face_downloads_tab(app)
            
        # Model Hub Search Tab
        with gr.Tab("🔍 Model Hub Search"):
            create_model_hub_search_tab(app)
            

def create_image_3d_downloads_tab(app):
    """Create UI for downloading image and 3D models"""
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🖼️ Image Generation Models")
            
            # Group models by type
            with gr.Accordion("Standard Models", open=True):
                for model_name, model_info in ALL_IMAGE_MODELS.items():
                    # Check if it's GGUF (either has is_gguf attribute or GGUF in name)
                    is_gguf = (model_info.get("is_gguf", False) if isinstance(model_info, dict) else getattr(model_info, "is_gguf", False))
                    # Check if it's gated
                    is_gated = (model_info.get("gated", False) if isinstance(model_info, dict) else getattr(model_info, "gated", False))
                    # Only show non-GGUF, non-gated models here
                    if not is_gguf and not is_gated and model_name not in ["FLUX.1-schnell", "FLUX.1-dev"]:
                        create_model_download_row(
                            app, model_name, model_info, "image"
                        )
                        
            with gr.Accordion("GGUF Quantized Models", open=False):
                for model_name, model_info in ALL_IMAGE_MODELS.items():
                    # Check if it's GGUF (either has is_gguf attribute or GGUF in name)
                    is_gguf = (model_info.get("is_gguf", False) if isinstance(model_info, dict) else getattr(model_info, "is_gguf", False))
                    if is_gguf:
                        create_model_download_row(
                            app, model_name, model_info, "image"
                        )
                        
            with gr.Accordion("Gated Models (Requires HF Token)", open=False):
                for model_name, model_info in ALL_IMAGE_MODELS.items():
                    is_gated = (model_info.get('gated', False) if isinstance(model_info, dict) else getattr(model_info, 'gated', False))
                    # Show gated models and the original FLUX models that require login
                    if is_gated or model_name in ["FLUX.1-schnell", "FLUX.1-dev"]:
                        create_model_download_row(
                            app, model_name, model_info, "image"
                        )
                        
        with gr.Column(scale=1):
            gr.Markdown("### 🎭 3D Generation Models")
            
            for model_name, model_info in HUNYUAN3D_MODELS.items():
                create_model_download_row(
                    app, model_name, model_info, "3d"
                )


def create_video_downloads_tab(app):
    """Create UI for downloading video models"""
    
    gr.Markdown("### 🎬 Video Generation Models")
    gr.Markdown("Download state-of-the-art video generation models for creating stunning animations.")
    
    with gr.Row():
        for model_name, model_info in VIDEO_MODELS.items():
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown(f"#### {model_info['name']}")
                    gr.Markdown(f"**Size:** {model_info['size']} | **VRAM:** {model_info['vram_required']}")
                    gr.Markdown(f"*{model_info['description']}*")
                    
                    # Check if downloaded
                    downloaded = app.model_manager.get_downloaded_models("video")
                    is_downloaded = model_name in downloaded
                    
                    status = gr.HTML(
                        value=f"<div style='color: {'green' if is_downloaded else 'gray'};'>{'✅ Downloaded' if is_downloaded else '⬇️ Not Downloaded'}</div>"
                    )
                    
                    download_btn = gr.Button(
                        f"{'Re-download' if is_downloaded else 'Download'} {model_info['name']}",
                        variant="secondary" if is_downloaded else "primary",
                        size="sm"
                    )
                    
                    progress_info = gr.HTML(visible=False)
                    
                    def download_video_model(model_name=model_name):
                        for update in app.model_manager.download_video_model(model_name):
                            # Extract only the HTML content, ignore the other values
                            if isinstance(update, tuple):
                                html_content = update[0]
                            else:
                                html_content = update
                            yield gr.update(visible=True, value=html_content)
                            
                    download_btn.click(
                        download_video_model,
                        outputs=[progress_info]
                    )


def create_character_face_downloads_tab(app):
    """Create UI for character consistency and face models"""
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🎭 IP-Adapter Models")
            gr.Markdown("For character consistency across generations")
            
            for model_name, model_info in IP_ADAPTER_MODELS.items():
                with gr.Group():
                    gr.Markdown(f"**{model_info['name']}**")
                    gr.Markdown(f"Size: {model_info['size']} | Base: {model_info['base_model']}")
                    gr.Markdown(f"*{model_info['description']}*")
                    
                    downloaded = app.model_manager.get_downloaded_models("ip_adapter")
                    is_downloaded = model_name in downloaded
                    
                    with gr.Row():
                        status = gr.HTML(
                            value=f"{'✅ Downloaded' if is_downloaded else '⬇️ Not Downloaded'}"
                        )
                        download_btn = gr.Button(
                            "Download" if not is_downloaded else "Re-download",
                            size="sm",
                            variant="primary" if not is_downloaded else "secondary"
                        )
                        
                    progress = gr.HTML(visible=False)
                    
                    def download_ip_adapter(model_name=model_name):
                        for update in app.model_manager.download_ip_adapter_model(model_name):
                            # Extract only the HTML content, ignore the other values
                            if isinstance(update, tuple):
                                html_content = update[0]
                            else:
                                html_content = update
                            yield gr.update(visible=True, value=html_content)
                            
                    download_btn.click(
                        download_ip_adapter,
                        outputs=[progress]
                    )
                    
        with gr.Column(scale=1):
            gr.Markdown("### 🔄 Face Swap Models")
            gr.Markdown("InsightFace models for face swapping")
            
            for model_name, model_info in FACE_SWAP_MODELS.items():
                with gr.Group():
                    gr.Markdown(f"**{model_info['name']}**")
                    gr.Markdown(f"Size: {model_info['size']}")
                    gr.Markdown(f"*{model_info['description']}*")
                    
                    downloaded = app.model_manager.get_downloaded_models("face_swap")
                    is_downloaded = model_name in downloaded
                    
                    with gr.Row():
                        status = gr.HTML(
                            value=f"{'✅ Downloaded' if is_downloaded else '⬇️ Not Downloaded'}"
                        )
                        download_btn = gr.Button(
                            "Download" if not is_downloaded else "Re-download",
                            size="sm",
                            variant="primary" if not is_downloaded else "secondary"
                        )
                        
                    progress = gr.HTML(visible=False)
                    
                    def download_face_swap(model_name=model_name):
                        for update in app.model_manager.download_face_swap_model(model_name):
                            # Extract only the HTML content, ignore the other values
                            if isinstance(update, tuple):
                                html_content = update[0]
                            else:
                                html_content = update
                            yield gr.update(visible=True, value=html_content)
                            
                    download_btn.click(
                        download_face_swap,
                        outputs=[progress]
                    )
                    
            gr.Markdown("### 🔧 Face Restoration Models")
            gr.Markdown("Enhance face quality after swapping")
            
            for model_name, model_info in FACE_RESTORE_MODELS.items():
                with gr.Group():
                    gr.Markdown(f"**{model_info['name']}**")
                    gr.Markdown(f"Size: {model_info['size']}")
                    gr.Markdown(f"*{model_info['description']}*")
                    
                    downloaded = app.model_manager.get_downloaded_models("face_restore")
                    is_downloaded = model_name in downloaded
                    
                    with gr.Row():
                        status = gr.HTML(
                            value=f"{'✅ Downloaded' if is_downloaded else '⬇️ Not Downloaded'}"
                        )
                        download_btn = gr.Button(
                            "Download" if not is_downloaded else "Re-download",
                            size="sm",
                            variant="primary" if not is_downloaded else "secondary"
                        )
                        
                    progress = gr.HTML(visible=False)
                    
                    def download_face_restore(model_name=model_name):
                        for update in app.model_manager.download_face_restore_model(model_name):
                            # Extract only the HTML content, ignore the other values
                            if isinstance(update, tuple):
                                html_content = update[0]
                            else:
                                html_content = update
                            yield gr.update(visible=True, value=html_content)
                            
                    download_btn.click(
                        download_face_restore,
                        outputs=[progress]
                    )


def create_model_download_row(app, model_name, model_info, model_type):
    """Create a download row for a model"""
    
    with gr.Group():
        with gr.Row():
            with gr.Column(scale=3):
                # Handle both dict and object types
                if isinstance(model_info, dict):
                    name = model_info.get('name', model_name)
                    size = model_info.get('size', 'Unknown')
                    vram = model_info.get('vram_required', 'Unknown')
                else:
                    name = getattr(model_info, 'name', model_name)
                    size = getattr(model_info, 'size', 'Unknown')
                    vram = getattr(model_info, 'vram_required', 'Unknown')
                
                gr.Markdown(f"**{name}**")
                gr.Markdown(f"Size: {size} | VRAM: {vram}")
                
            with gr.Column(scale=1):
                # Check if downloaded
                if model_type == "image":
                    model_path = app.model_manager.models_dir / "image" / model_name
                    is_complete = app.model_manager.check_model_complete(model_path, "image", model_name)
                else:
                    model_path = app.model_manager.models_dir / "3d" / model_name
                    is_complete = app.model_manager.check_model_complete(model_path, "3d", model_name)
                
                # Check if partially downloaded
                is_partial = model_path.exists() and not is_complete
                
                if is_complete:
                    status_text = "✅ Complete"
                    status_color = "green"
                elif is_partial:
                    status_text = "⏸️ Partial (Resume Available)"
                    status_color = "orange"
                else:
                    status_text = "⬇️ Not Downloaded"
                    status_color = "gray"
                    
                status = gr.HTML(
                    value=f"<div style='color: {status_color};'>{status_text}</div>"
                )
                
                if is_partial:
                    btn_text = "Resume Download"
                    btn_variant = "primary"
                elif is_complete:
                    btn_text = "Re-download"
                    btn_variant = "secondary"
                else:
                    btn_text = "Download"
                    btn_variant = "primary"
                    
                download_btn = gr.Button(
                    btn_text,
                    size="sm",
                    variant=btn_variant
                )
                
        # Add force re-download checkbox
        with gr.Row():
            force_redownload = gr.Checkbox(
                label="Force re-download (delete existing files)",
                value=False,
                visible=is_partial or is_complete
            )
                
        progress = gr.HTML(visible=False)
        
        def download_model(force_redownload_value, model_name=model_name, model_type=model_type):
            # Clear any existing message first
            yield gr.update(visible=True, value="<div style='color: #059669;'>Starting download...</div>")
            
            for update in app.model_manager.download_model(
                model_type, model_name, 
                use_hf_token=True, 
                force_redownload=force_redownload_value, 
                progress=None
            ):
                # Extract only the HTML content, ignore the other values
                if isinstance(update, tuple):
                    html_content = update[0]
                else:
                    html_content = update
                yield gr.update(visible=True, value=html_content)
                
        download_btn.click(
            download_model,
            inputs=[force_redownload],
            outputs=[progress]
        )


def create_model_hub_search_tab(app):
    """Create model hub search interface"""
    
    gr.Markdown("### 🔍 Search Model Hubs")
    gr.Markdown("Search and download models from Hugging Face and Civitai")
    
    with gr.Tabs():
        # Hugging Face Search
        with gr.Tab("🤗 Hugging Face"):
            with gr.Row():
                search_query = gr.Textbox(
                    label="Search Query",
                    placeholder="stable-diffusion, flux, controlnet..."
                )
                search_type = gr.Dropdown(
                    choices=["text-to-image", "image-to-image", "controlnet", "lora"],
                    value="text-to-image",
                    label="Model Type"
                )
                search_btn = gr.Button("Search", variant="primary")
                
            search_results = gr.HTML()
            
            def search_huggingface(query, model_type):
                # This would search HuggingFace models
                # For now, return placeholder
                return f"""
                <div class="info-box">
                    <h4>🔍 Search Results</h4>
                    <p>Searching for "{query}" in {model_type} models...</p>
                    <p>HuggingFace search integration coming soon!</p>
                </div>
                """
                
            search_btn.click(
                search_huggingface,
                inputs=[search_query, search_type],
                outputs=[search_results]
            )
            
        # Civitai Search  
        with gr.Tab("🎨 Civitai"):
            with gr.Row():
                civitai_query = gr.Textbox(
                    label="Search Query",
                    placeholder="anime, realistic, fantasy..."
                )
                civitai_type = gr.Dropdown(
                    choices=["Checkpoint", "LORA", "TextualInversion", "Hypernetwork"],
                    value="LORA",
                    label="Model Type"
                )
                base_model = gr.Dropdown(
                    choices=["SDXL", "SD 1.5", "FLUX.1"],
                    value="SDXL",
                    label="Base Model"
                )
                civitai_search_btn = gr.Button("Search Civitai", variant="primary")
                
            civitai_results = gr.HTML()
            
            def search_civitai(query, model_type, base):
                # This would search Civitai
                # For now, return placeholder
                return f"""
                <div class="info-box">
                    <h4>🎨 Civitai Search</h4>
                    <p>Searching for "{query}" {model_type} models for {base}...</p>
                    <p>Civitai integration with automatic downloads coming soon!</p>
                </div>
                """
                
            civitai_search_btn.click(
                search_civitai,
                inputs=[civitai_query, civitai_type, base_model],
                outputs=[civitai_results]
            )
            
    # Download queue status
    gr.Markdown("### 📊 Download Status")
    with gr.Row():
        with gr.Column():
            queue_status = gr.HTML(
                value="""
                <div class="stat-card">
                    <h4>Download Queue</h4>
                    <p>No active downloads</p>
                </div>
                """
            )
            
        with gr.Column():
            storage_status = gr.HTML(
                value=app.model_manager.get_storage_status()
            )
            
    # Refresh button
    refresh_btn = gr.Button("🔄 Refresh Status", size="sm")
    
    def refresh_status():
        return app.model_manager.get_storage_status()
        
    refresh_btn.click(
        refresh_status,
        outputs=[storage_status]
    )