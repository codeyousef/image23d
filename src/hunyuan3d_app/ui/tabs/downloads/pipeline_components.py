"""Pipeline components download management"""

import gradio as gr
from pathlib import Path
import shutil
from typing import Dict, Any

from ....config import TEXTURE_PIPELINE_COMPONENTS


def check_texture_components_status() -> tuple:
    """Check if texture components are installed
    
    Returns:
        Tuple of (realesrgan_installed, xatlas_installed, dinov2_installed)
    """
    realesrgan_installed = (Path("Hunyuan3D") / "hy3dpaint" / "ckpt" / "RealESRGAN_x4plus.pth").exists()
    
    try:
        import xatlas
        xatlas_installed = True
    except ImportError:
        xatlas_installed = False
    
    # Check if DINOv2 is in cache
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    dinov2_installed = any(
        d.name.startswith("models--facebook--dinov2-giant") 
        for d in cache_dir.iterdir() 
        if d.is_dir()
    ) if cache_dir.exists() else False
    
    return realesrgan_installed, xatlas_installed, dinov2_installed


def create_texture_warning_banner() -> gr.Markdown:
    """Create warning banner for missing texture components"""
    realesrgan_installed, xatlas_installed, dinov2_installed = check_texture_components_status()
    
    if not realesrgan_installed or not xatlas_installed or not dinov2_installed:
        missing_components = []
        if not realesrgan_installed:
            missing_components.append("RealESRGAN")
        if not xatlas_installed:
            missing_components.append("xatlas")
        if not dinov2_installed:
            missing_components.append("DINOv2")
            
        return gr.Markdown(f"""
        <div style="background: #fef3c7; border: 2px solid #f59e0b; border-radius: 8px; padding: 16px; margin-bottom: 16px;">
            <h4 style="margin-top: 0; color: #d97706;">⚠️ Texture Components Missing!</h4>
            <p style="margin-bottom: 8px;">Essential texture pipeline components are not installed: {', '.join(missing_components)}</p>
            <p style="margin-bottom: 8px;">Your 3D models will generate with reduced texture quality.</p>
            <p style="margin-bottom: 0;"><strong>Quick Fix:</strong> Click the <strong>"🎨 Download Texture Components"</strong> button below, or go to the <strong>"🎨 Texture Pipeline Components"</strong> tab.</p>
        </div>
        """)
    else:
        return gr.Markdown("")  # Empty if all components installed


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
            create_texture_enhancement_section(app)
                        
        with gr.Column(scale=1):
            create_dependencies_section(app)


def create_texture_enhancement_section(app):
    """Create texture enhancement models section"""
    gr.Markdown("#### 🎨 Texture Enhancement")
    
    for comp_name, comp_info in TEXTURE_PIPELINE_COMPONENTS.items():
        if comp_info.get("type") == "model" and "texture" in comp_info.get("required_for", "").lower():
            create_texture_component_card(app, comp_name, comp_info)


def create_texture_component_card(app, comp_name: str, comp_info: Dict[str, Any]):
    """Create a download card for a texture component"""
    with gr.Group():
        gr.Markdown(f"**{comp_info['name']}**")
        gr.Markdown(f"Size: {comp_info['size']}")
        gr.Markdown(f"*{comp_info['description']}*")
        gr.Markdown(f"Required for: {comp_info['required_for']}")
        
        # Check if downloaded
        if comp_name == "realesrgan":
            is_downloaded = (Path("Hunyuan3D") / "hy3dpaint" / "ckpt" / "RealESRGAN_x4plus.pth").exists()
        elif comp_name == "dinov2":
            # Check if DINOv2 model is in HuggingFace cache
            from pathlib import Path
            import os
            cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
            # Look for facebook--dinov2-giant in cache
            is_downloaded = any(
                d.name.startswith("models--facebook--dinov2-giant") 
                for d in cache_dir.iterdir() 
                if d.is_dir()
            ) if cache_dir.exists() else False
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


def create_dependencies_section(app):
    """Create dependencies section"""
    gr.Markdown("#### 📦 Dependencies")
    
    for comp_name, comp_info in TEXTURE_PIPELINE_COMPONENTS.items():
        if comp_info.get("type") == "dependency":
            create_dependency_card(app, comp_name, comp_info)


def create_dependency_card(app, comp_name: str, comp_info: Dict[str, Any]):
    """Create a card for a dependency"""
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
            else:
                is_installed = False
        except ImportError:
            is_installed = False
            
        status = gr.HTML(
            value=f"<div style='color: {'green' if is_installed else 'gray'};'>{'✅ Installed' if is_installed else '⬇️ Not Installed'}</div>"
        )
        
        install_btn = gr.Button(
            f"{'Re-install' if is_installed else 'Install'} {comp_info['name']}",
            variant="secondary" if is_installed else "primary",
            size="sm"
        )
        
        progress_info = gr.HTML(visible=False)
        
        def install_dependency(comp_name=comp_name, comp_info=comp_info):
            # Use the texture component download system
            success, message, download_id = app.model_manager.download_texture_component(
                component_name=comp_name,
                component_info=comp_info,
                priority=1
            )
            
            if success:
                yield gr.update(visible=True, value=f"<div style='color: #059669;'>{message}</div>")
            else:
                yield gr.update(visible=True, value=f"<div style='color: #dc3545;'>{message}</div>")
                
        install_btn.click(
            install_dependency,
            outputs=[progress_info]
        )