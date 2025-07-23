"""Comprehensive Downloads Manager UI for all model types"""

import gradio as gr
from pathlib import Path
from typing import Dict, List, Tuple

from ...config import (
    ALL_IMAGE_MODELS, HUNYUAN3D_MODELS, VIDEO_MODELS,
    IP_ADAPTER_MODELS, FACE_SWAP_MODELS, FACE_RESTORE_MODELS,
    TEXTURE_PIPELINE_COMPONENTS
)


def get_download_status_html(app) -> str:
    """Generate HTML for download status panel."""
    try:
        status = app.model_manager.get_download_status()
        active_downloads = status.get('active', {})
        queued_downloads = status.get('queued', [])
        max_concurrent = status.get('max_concurrent', 3)
        
        html = '<div class="download-status-panel" style="padding: 10px; border: 1px solid #ddd; border-radius: 5px; margin-bottom: 10px;">'
        html += f'<h4 style="margin-top: 0;">📥 Download Status (Max {max_concurrent} concurrent)</h4>'
        
        if not active_downloads and not queued_downloads:
            html += '<p style="color: #666;">No active downloads</p>'
        else:
            # Active downloads
            if active_downloads:
                html += '<div class="active-downloads">'
                html += f'<h5>🔄 Active Downloads ({len(active_downloads)})</h5>'
                for download_id, info in active_downloads.items():
                    model_name = info.get('model_name', 'Unknown')
                    progress = info.get('progress', {})
                    percentage = getattr(progress, 'percentage', 0)
                    current_file = getattr(progress, 'current_file', '')
                    speed = getattr(progress, 'speed', 0) / (1024**2) if hasattr(progress, 'speed') else 0
                    
                    html += f'<div style="margin-bottom: 10px;">'
                    html += f'<strong>{model_name}</strong><br>'
                    html += f'<div style="background: #f0f0f0; height: 20px; border-radius: 3px; overflow: hidden;">'
                    html += f'<div style="background: #4CAF50; height: 100%; width: {percentage}%; transition: width 0.3s;"></div>'
                    html += f'</div>'
                    html += f'<small>{percentage}% - {speed:.1f} MB/s'
                    if current_file:
                        html += f' - {current_file}'
                    html += '</small></div>'
                html += '</div>'
            
            # Queued downloads
            if queued_downloads:
                html += '<div class="queued-downloads" style="margin-top: 15px;">'
                html += f'<h5>⏳ Queued Downloads ({len(queued_downloads)})</h5>'
                html += '<ol style="margin: 5px 0; padding-left: 20px;">'
                for idx, info in enumerate(queued_downloads):
                    model_name = info.get('model_name', 'Unknown')
                    html += f'<li>{model_name}</li>'
                html += '</ol>'
                html += '</div>'
        
        html += '</div>'
        return html
        
    except Exception as e:
        return f'<div class="error">Error getting download status: {str(e)}</div>'


def create_downloads_manager_tab(app):
    """Create comprehensive Downloads Manager UI"""
    
    # Check if texture components are installed
    from ...config import TEXTURE_PIPELINE_COMPONENTS
    from pathlib import Path
    
    realesrgan_installed = (Path("Hunyuan3D") / "hy3dpaint" / "ckpt" / "RealESRGAN_x4plus.pth").exists()
    try:
        import xatlas
        xatlas_installed = True
    except ImportError:
        xatlas_installed = False
    
    # Show warning if texture components are missing
    if not realesrgan_installed or not xatlas_installed:
        with gr.Row():
            gr.Markdown("""
            <div style="background: #fef3c7; border: 2px solid #f59e0b; border-radius: 8px; padding: 16px; margin-bottom: 16px;">
                <h4 style="margin-top: 0; color: #d97706;">⚠️ Texture Components Missing!</h4>
                <p style="margin-bottom: 8px;">Essential texture pipeline components are not installed. Your 3D models will generate without textures.</p>
                <p style="margin-bottom: 0;"><strong>Quick Fix:</strong> Click the <strong>"🎨 Download Texture Components"</strong> button below, or go to the <strong>"🎨 Texture Pipeline Components"</strong> tab.</p>
            </div>
            """)
    
    # Add download status panel at the top
    with gr.Row():
        download_status = gr.HTML(
            value=get_download_status_html(app),
            elem_id="download-status-panel"
        )
    
    # Control buttons
    with gr.Row():
        refresh_status_btn = gr.Button("🔄 Refresh Status", size="sm")
        download_all_btn = gr.Button("⬇️ Download All Essential Models", size="sm", variant="primary")
        download_gguf_btn = gr.Button("💾 Download Best GGUF Models", size="sm", variant="secondary")
        download_video_btn = gr.Button("🎬 Download All Video Models", size="sm", variant="secondary")
        download_texture_btn = gr.Button("🎨 Download Texture Components", size="sm", variant="secondary")
    
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
            
        # Texture Pipeline Components Tab
        with gr.Tab("🎨 Texture Pipeline Components"):
            create_pipeline_components_tab(app)
            
        # Model Hub Search Tab
        with gr.Tab("🔍 Model Hub Search"):
            create_model_hub_search_tab(app)
    
    # Auto-refresh download status
    def refresh_download_status():
        return get_download_status_html(app)
    
    refresh_status_btn.click(
        refresh_download_status,
        outputs=[download_status]
    )
    
    # Set up periodic refresh (every 2 seconds when downloads are active)
    download_status.change(
        lambda: gr.update(),
        outputs=[download_status],
        every=2
    )
    
    # Batch download handlers
    def download_all_essential():
        """Download all essential models including texture components"""
        from ...config import TEXTURE_PIPELINE_COMPONENTS
        
        essential_models = [
            ("SDXL-Turbo", "image", 3),
            ("hunyuan3d-2mini", "3d", 2),
            ("hunyuan3d-21", "3d", 1)
        ]
        
        results = []
        # Download models
        for model_name, model_type, priority in essential_models:
            success, message, download_id = app.model_manager.download_model_concurrent(
                model_name=model_name,
                model_type=model_type,
                priority=priority
            )
            results.append(f"{model_name}: {message}")
        
        # Also download essential texture components
        essential_texture_components = ["realesrgan", "xatlas"]
        priority = 4
        
        for comp_name in essential_texture_components:
            if comp_name in TEXTURE_PIPELINE_COMPONENTS:
                comp_info = TEXTURE_PIPELINE_COMPONENTS[comp_name]
                success, message, download_id = app.model_manager.download_texture_component(
                    component_name=comp_name,
                    component_info=comp_info,
                    priority=priority
                )
                results.append(f"{comp_name}: {message}")
                priority += 1
        
        return get_download_status_html(app)
    
    def download_best_gguf():
        """Download best GGUF models for different VRAM levels"""
        gguf_models = [
            ("FLUX.1-dev-Q8", "image", 1),  # 14GB+ VRAM
            ("FLUX.1-dev-Q4", "image", 2),  # 8GB+ VRAM
            ("FLUX.1-schnell-Q4", "image", 3)  # Fast option
        ]
        
        results = []
        for model_name, model_type, priority in gguf_models:
            success, message, download_id = app.model_manager.download_model_concurrent(
                model_name=model_name,
                model_type=model_type,
                priority=priority
            )
            results.append(f"{model_name}: {message}")
        
        return get_download_status_html(app)
    
    def download_all_video():
        """Download all video models"""
        video_models = [
            ("hunyuanvideo", "video", 1),
            ("mochi-1", "video", 2),
            ("wan-2.1", "video", 3),
            ("ltxvideo", "video", 4)
        ]
        
        results = []
        for model_name, model_type, priority in video_models:
            success, message, download_id = app.model_manager.download_model_concurrent(
                model_name=model_name,
                model_type=model_type,
                priority=priority
            )
            results.append(f"{model_name}: {message}")
        
        return get_download_status_html(app)
    
    # Connect batch download buttons
    download_all_btn.click(
        download_all_essential,
        outputs=[download_status]
    )
    
    download_gguf_btn.click(
        download_best_gguf,
        outputs=[download_status]
    )
    
    download_video_btn.click(
        download_all_video,
        outputs=[download_status]
    )
    
    def download_all_texture_components():
        """Download all texture pipeline components"""
        from ...config import TEXTURE_PIPELINE_COMPONENTS
        
        results = []
        priority = 1
        
        for comp_name, comp_info in TEXTURE_PIPELINE_COMPONENTS.items():
            if comp_info.get('type') == 'model':
                # Download models
                success, message, download_id = app.model_manager.download_texture_component(
                    component_name=comp_name,
                    component_info=comp_info,
                    priority=priority
                )
                results.append(f"{comp_name}: {message}")
                priority += 1
            elif comp_info.get('type') == 'dependency':
                # Install pip dependencies
                success, message, download_id = app.model_manager.download_texture_component(
                    component_name=comp_name,
                    component_info=comp_info,
                    priority=10  # Lower priority for pip installs
                )
                results.append(f"{comp_name}: {message}")
        
        # Special handling for RealESRGAN - move to correct location after download
        realesrgan_source = app.model_manager.models_dir / "texture_components" / "realesrgan" / "RealESRGAN_x4plus.pth"
        realesrgan_target = Path("Hunyuan3D") / "hy3dpaint" / "ckpt" / "RealESRGAN_x4plus.pth"
        
        if realesrgan_source.exists() and not realesrgan_target.exists():
            realesrgan_target.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(realesrgan_source, realesrgan_target)
            results.append("RealESRGAN: Moved to HunYuan3D directory")
        
        return get_download_status_html(app)
    
    download_texture_btn.click(
        download_all_texture_components,
        outputs=[download_status]
    )
            

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
            
            # Add texture components notice
            gr.Markdown("""
            <div style="background: #f0f9ff; border: 1px solid #0ea5e9; border-radius: 6px; padding: 12px; margin-bottom: 12px;">
                <p style="margin: 0; font-size: 14px;"><strong>💡 Tip:</strong> For textures on your 3D models, make sure to also download the 
                <a href="#" onclick="document.querySelector('[id*=tab-][id*=texture]').click(); return false;" style="color: #0369a1; text-decoration: underline;">Texture Pipeline Components</a>.</p>
            </div>
            """)
            
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
                        # Use concurrent download
                        success, message, download_id = app.model_manager.download_model_concurrent(
                            model_name=model_name,
                            model_type="video",
                            priority=1
                        )
                        
                        if success:
                            yield gr.update(visible=True, value=f"<div style='color: #059669;'>{message}</div>")
                        else:
                            yield gr.update(visible=True, value=f"<div style='color: #dc3545;'>{message}</div>")
                            
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
                        # Use concurrent download
                        success, message, download_id = app.model_manager.download_model_concurrent(
                            model_name=model_name,
                            model_type="ip_adapter",
                            priority=1
                        )
                        
                        if success:
                            yield gr.update(visible=True, value=f"<div style='color: #059669;'>{message}</div>")
                        else:
                            yield gr.update(visible=True, value=f"<div style='color: #dc3545;'>{message}</div>")
                            
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
                        # Use concurrent download
                        success, message, download_id = app.model_manager.download_model_concurrent(
                            model_name=model_name,
                            model_type="face_swap",
                            priority=1
                        )
                        
                        if success:
                            yield gr.update(visible=True, value=f"<div style='color: #059669;'>{message}</div>")
                        else:
                            yield gr.update(visible=True, value=f"<div style='color: #dc3545;'>{message}</div>")
                            
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
                        # Use concurrent download
                        success, message, download_id = app.model_manager.download_model_concurrent(
                            model_name=model_name,
                            model_type="face_restore",
                            priority=1
                        )
                        
                        if success:
                            yield gr.update(visible=True, value=f"<div style='color: #059669;'>{message}</div>")
                        else:
                            yield gr.update(visible=True, value=f"<div style='color: #dc3545;'>{message}</div>")
                            
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
            # Use concurrent download
            success, message, download_id = app.model_manager.download_model_concurrent(
                model_name=model_name,
                model_type=model_type,
                priority=1  # Normal priority
            )
            
            if success:
                yield gr.update(visible=True, value=f"<div style='color: #059669;'>{message}</div>")
            else:
                yield gr.update(visible=True, value=f"<div style='color: #dc3545;'>{message}</div>")
                
        download_btn.click(
            download_model,
            inputs=[force_redownload],
            outputs=[progress]
        )


def create_pipeline_components_tab(app):
    """Create UI for texture pipeline components"""
    
    gr.Markdown("### 🎨 Texture Pipeline Components")
    gr.Markdown("**Essential components required for high-quality texture generation in HunYuan3D models**")
    
    # Add a prominent notice
    with gr.Row():
        gr.Markdown("""
        <div style="background: #f0f9ff; border: 2px solid #0ea5e9; border-radius: 8px; padding: 16px; margin: 10px 0;">
            <h4 style="margin-top: 0; color: #0369a1;">⚠️ Important: Download These Components for Full Texture Support</h4>
            <p style="margin-bottom: 0;">Without these components, HunYuan3D will generate 3D models without textures or with low-quality textures. 
            Click the <strong>"🎨 Download Texture Components"</strong> button above to download all components at once.</p>
        </div>
        """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("#### 🎨 Texture Enhancement")
            
            for comp_name, comp_info in TEXTURE_PIPELINE_COMPONENTS.items():
                if comp_info.get("type") == "model" and "texture" in comp_info.get("required_for", "").lower():
                    with gr.Group():
                        gr.Markdown(f"**{comp_info['name']}**")
                        gr.Markdown(f"Size: {comp_info['size']}")
                        gr.Markdown(f"*{comp_info['description']}*")
                        gr.Markdown(f"Required for: {comp_info['required_for']}")
                        
                        # Check if downloaded
                        if comp_name == "realesrgan":
                            install_path = Path(comp_info.get("install_path", ""))
                            is_downloaded = (Path("Hunyuan3D") / "hy3dpaint" / "ckpt" / "RealESRGAN_x4plus.pth").exists()
                        else:
                            is_downloaded = False
                            
                        status = gr.HTML(
                            value=f"<div style='color: {'green' if is_downloaded else 'gray'};'>{'✅ Downloaded' if is_downloaded else '⬇️ Not Downloaded'}</div>"
                        )
                        
                        download_btn = gr.Button(
                            f"{'Re-download' if is_downloaded else 'Download'} {comp_info['name']}",
                            variant="secondary" if is_downloaded else "primary",
                            size="sm"
                        )
                        
                        progress_info = gr.HTML(visible=False)
                        
                        def download_component(comp_name=comp_name, comp_info=comp_info):
                            # Use the new concurrent download system
                            success, message, download_id = app.model_manager.download_texture_component(
                                component_name=comp_name,
                                component_info=comp_info,
                                priority=1
                            )
                            
                            # Special handling for RealESRGAN to move it to the correct location
                            if success and comp_name == "realesrgan":
                                # Move the downloaded file to the expected location
                                source_path = app.model_manager.models_dir / "texture_components" / comp_name / "RealESRGAN_x4plus.pth"
                                target_path = Path("Hunyuan3D") / "hy3dpaint" / "ckpt" / "RealESRGAN_x4plus.pth"
                                
                                if source_path.exists() and not target_path.exists():
                                    target_path.parent.mkdir(parents=True, exist_ok=True)
                                    import shutil
                                    shutil.copy2(source_path, target_path)
                                    yield gr.update(visible=True, value=f"<div style='color: green;'>✅ {message} and moved to HunYuan3D directory</div>")
                                else:
                                    yield gr.update(visible=True, value=f"<div style='color: green;'>✅ {message}</div>")
                            else:
                                if success:
                                    yield gr.update(visible=True, value=f"<div style='color: #059669;'>{message}</div>")
                                else:
                                    yield gr.update(visible=True, value=f"<div style='color: #dc3545;'>{message}</div>")
                                
                        download_btn.click(
                            download_component,
                            outputs=[progress_info]
                        )
                        
        with gr.Column(scale=1):
            gr.Markdown("#### 📦 Dependencies")
            
            for comp_name, comp_info in TEXTURE_PIPELINE_COMPONENTS.items():
                if comp_info.get("type") == "dependency":
                    with gr.Group():
                        gr.Markdown(f"**{comp_info['name']}**")
                        gr.Markdown(f"Package: `{comp_info.get('pip_package', 'N/A')}`")
                        gr.Markdown(f"Size: {comp_info['size']}")
                        gr.Markdown(f"*{comp_info['description']}*")
                        
                        # Check if installed
                        try:
                            if comp_name == "xatlas":
                                import xatlas
                                is_installed = True
                        except ImportError:
                            is_installed = False
                            
                        status = gr.HTML(
                            value=f"<div style='color: {'green' if is_installed else 'gray'};'>{'✅ Installed' if is_installed else '⬇️ Not Installed'}</div>"
                        )
                        
                        install_btn = gr.Button(
                            f"{'Reinstall' if is_installed else 'Install'} {comp_info['name']}",
                            variant="secondary" if is_installed else "primary",
                            size="sm"
                        )
                        
                        install_progress = gr.HTML(visible=False)
                        
                        def install_dependency(comp_info=comp_info):
                            yield gr.update(visible=True, value="<div style='color: #059669;'>Installing...</div>")
                            
                            import subprocess
                            pip_package = comp_info.get('pip_package', '')
                            if pip_package:
                                result = subprocess.run([
                                    "pip", "install", pip_package
                                ], capture_output=True, text=True)
                                if result.returncode == 0:
                                    yield gr.update(visible=True, value=f"<div style='color: green;'>✅ {comp_info['name']} installed successfully!</div>")
                                else:
                                    yield gr.update(visible=True, value=f"<div style='color: red;'>❌ Installation failed: {result.stderr}</div>")
                            else:
                                yield gr.update(visible=True, value="<div style='color: orange;'>⚠️ No pip package specified</div>")
                                
                        install_btn.click(
                            install_dependency,
                            outputs=[install_progress]
                        )
                        
            gr.Markdown("#### 🔧 Other Components")
            
            for comp_name, comp_info in TEXTURE_PIPELINE_COMPONENTS.items():
                if comp_info.get("type") == "model" and "texture" not in comp_info.get("required_for", "").lower():
                    with gr.Group():
                        gr.Markdown(f"**{comp_info['name']}**")
                        gr.Markdown(f"Size: {comp_info['size']}")
                        gr.Markdown(f"*{comp_info['description']}*")
                        gr.Markdown(f"Required for: {comp_info.get('required_for', 'Various features')}")
                        
                        # Check if downloaded
                        is_downloaded = False
                        if comp_name == "dinov2":
                            check_path = app.model_manager.models_dir / "texture_components" / comp_name
                            is_downloaded = check_path.exists() and any(check_path.glob("*.bin"))
                        elif comp_name == "background_remover":
                            check_path = app.model_manager.models_dir / "texture_components" / comp_name
                            is_downloaded = check_path.exists() and any(check_path.glob("*.pth"))
                        
                        status = gr.HTML(
                            value=f"<div style='color: {'green' if is_downloaded else 'gray'};'>{'✅ Downloaded' if is_downloaded else '⬇️ Not Downloaded'}</div>"
                        )
                        
                        download_btn = gr.Button(
                            f"{'Re-download' if is_downloaded else 'Download'} {comp_info['name']}",
                            variant="secondary" if is_downloaded else "primary",
                            size="sm"
                        )
                        
                        progress = gr.HTML(visible=False)
                        
                        def download_other_component(comp_name=comp_name, comp_info=comp_info):
                            # Use the new concurrent download system
                            success, message, download_id = app.model_manager.download_texture_component(
                                component_name=comp_name,
                                component_info=comp_info,
                                priority=2
                            )
                            
                            if success:
                                yield gr.update(visible=True, value=f"<div style='color: #059669;'>{message}</div>")
                            else:
                                yield gr.update(visible=True, value=f"<div style='color: #dc3545;'>{message}</div>")
                        
                        download_btn.click(
                            download_other_component,
                            outputs=[progress]
                        )
    
    # Instructions section
    gr.Markdown("---")
    gr.Markdown("### 📚 Instructions")
    gr.Markdown("""
    **Texture Pipeline Components:**
    - **RealESRGAN**: Enhances texture resolution by 4x for better quality
    - **xatlas**: Required for proper UV unwrapping in texture generation
    - **DINOv2**: Feature extraction for advanced texture synthesis
    - **Background Remover**: Preprocesses images for cleaner 3D generation
    
    **Installation Order:**
    1. Install xatlas dependency first
    2. Download RealESRGAN model
    3. Other components are optional but recommended
    
    **Note:** These components enhance the quality of 3D texture generation but the basic pipeline works without them.
    """)


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