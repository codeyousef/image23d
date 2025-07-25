"""Texture Components Management UI

Handles installation and management of texture pipeline components.
"""

import gradio as gr
from pathlib import Path
from typing import Any, Tuple
import logging
import shutil

from ...components.common import create_action_button

logger = logging.getLogger(__name__)


def check_texture_components_status() -> Tuple[bool, bool]:
    """Check if texture components are installed"""
    realesrgan_installed = (Path("Hunyuan3D") / "hy3dpaint" / "ckpt" / "RealESRGAN_x4plus.pth").exists()
    try:
        import xatlas
        xatlas_installed = True
    except ImportError:
        xatlas_installed = False
    
    return realesrgan_installed, xatlas_installed


def create_texture_components_tab(app: Any) -> None:
    """Create UI for texture pipeline components"""
    from ....config import TEXTURE_PIPELINE_COMPONENTS
    
    gr.Markdown("### 🎨 Essential Texture Pipeline Components")
    gr.Markdown("""
    These components are **required** for generating textures on your 3D models. 
    Without them, HunYuan3D will only generate untextured (gray) 3D models.
    """)
    
    # Quick install all button
    with gr.Row():
        install_all_btn = create_action_button(
            "🚀 Install All Texture Components",
            variant="primary",
            size="lg"
        )
        status_all = gr.HTML()
    
    def install_all_components(progress=gr.Progress()):
        """Install all texture components"""
        results = []
        components = list(TEXTURE_PIPELINE_COMPONENTS.items())
        total = len(components)
        
        progress(0, desc="Starting texture component installation...")
        
        # Install each component
        for idx, (comp_name, comp_info) in enumerate(components):
            progress((idx / total), desc=f"Installing {comp_info.get('name', comp_name)}...")
            
            success, message, _ = app.model_manager.download_texture_component(
                component_name=comp_name,
                component_info=comp_info,
                priority=1
            )
            
            status_icon = "✅" if success else "❌"
            results.append(f"{status_icon} {comp_name}: {message}")
            
            # Update progress
            progress((idx + 1) / total, desc=f"Completed {comp_info.get('name', comp_name)}")
        
        # Special handling for RealESRGAN
        progress(0.9, desc="Moving RealESRGAN to HunYuan3D directory...")
        source_path = app.model_manager.models_dir / "texture_components" / "realesrgan" / "RealESRGAN_x4plus.pth"
        target_path = Path("Hunyuan3D") / "hy3dpaint" / "ckpt" / "RealESRGAN_x4plus.pth"
        
        if source_path.exists() and not target_path.exists():
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)
            results.append("✅ RealESRGAN: Moved to HunYuan3D directory")
        
        progress(1.0, desc="Installation complete!")
        
        return "<div style='padding: 10px;'>" + "<br>".join(results) + "</div>"
    
    install_all_btn.click(
        install_all_components,
        outputs=[status_all]
    )
    
    # Individual components
    with gr.Row():
        with gr.Column():
            create_realesrgan_component(app)
            
        with gr.Column():
            create_xatlas_component(app)
    
    # Optional components
    gr.Markdown("### 🔧 Optional Components")
    gr.Markdown("These components enhance texture quality but are not strictly required.")
    
    with gr.Row():
        with gr.Column():
            create_dinov2_component(app)
            
        with gr.Column():
            create_rembg_component(app)


def create_realesrgan_component(app: Any) -> None:
    """Create RealESRGAN installation UI"""
    from ....config import TEXTURE_PIPELINE_COMPONENTS
    
    gr.Markdown("#### 🖼️ RealESRGAN (Texture Enhancement)")
    gr.Markdown("**Required for:** High-quality texture generation")
    gr.Markdown("**Size:** ~64 MB")
    
    # Check if installed
    realesrgan_path = Path("Hunyuan3D") / "hy3dpaint" / "ckpt" / "RealESRGAN_x4plus.pth"
    realesrgan_status_display = gr.HTML(
        value=f"<div style='color: {'green' if realesrgan_path.exists() else 'red'};'>{'✅ Installed' if realesrgan_path.exists() else '❌ Not Installed'}</div>"
    )
        
    with gr.Row():
        install_realesrgan_btn = create_action_button("Install RealESRGAN", variant="primary")
        realesrgan_status = gr.HTML()
        
    def install_realesrgan(progress=gr.Progress()):
        progress(0, desc="Starting RealESRGAN download...")
        
        comp_info = TEXTURE_PIPELINE_COMPONENTS.get("realesrgan", {})
        success, message, _ = app.model_manager.download_texture_component(
            component_name="realesrgan",
            component_info=comp_info,
            priority=1
        )
        
        progress(0.5, desc="Download complete, moving to HunYuan3D directory...")
        
        # Move to correct location
        if success:
            source = app.model_manager.models_dir / "texture_components" / "realesrgan" / "RealESRGAN_x4plus.pth"
            target = Path("Hunyuan3D") / "hy3dpaint" / "ckpt" / "RealESRGAN_x4plus.pth"
            if source.exists() and not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                progress(1.0, desc="Installation complete!")
                result = f"<div style='color: green;'>✅ {message} and moved to HunYuan3D</div>"
            else:
                progress(1.0, desc="Installation complete!")
                result = f"<div style='color: green;'>✅ {message}</div>"
        else:
            progress(1.0, desc="Installation failed")
            result = f"<div style='color: red;'>❌ {message}</div>"
        
        # Return both the result and updated status
        return result, "<div style='color: green;'>✅ Installed</div>" if success else "<div style='color: red;'>❌ Not Installed</div>"
    
    install_realesrgan_btn.click(
        install_realesrgan,
        outputs=[realesrgan_status, realesrgan_status_display]
    )


def create_xatlas_component(app: Any) -> None:
    """Create xatlas installation UI"""
    from ....config import TEXTURE_PIPELINE_COMPONENTS
    
    gr.Markdown("#### 📐 xatlas (UV Mapping)")
    gr.Markdown("**Required for:** Proper texture mapping on 3D models")
    gr.Markdown("**Size:** ~2 MB")
    
    # Check if installed
    try:
        import xatlas
        xatlas_installed = True
    except ImportError:
        xatlas_installed = False
        
    xatlas_status_display = gr.HTML(
        value=f"<div style='color: {'green' if xatlas_installed else 'red'};'>{'✅ Installed' if xatlas_installed else '❌ Not Installed'}</div>"
    )
        
    with gr.Row():
        install_xatlas_btn = create_action_button("Install xatlas", variant="primary")
        xatlas_status = gr.HTML()
        
    def install_xatlas(progress=gr.Progress()):
        progress(0, desc="Installing xatlas package...")
        
        comp_info = TEXTURE_PIPELINE_COMPONENTS.get("xatlas", {})
        success, message, _ = app.model_manager.download_texture_component(
            component_name="xatlas",
            component_info=comp_info,
            priority=1
        )
        
        progress(1.0, desc="Installation complete!" if success else "Installation failed")
        
        result = f"<div style='color: {'green' if success else 'red'};'>{'✅' if success else '❌'} {message}</div>"
        status = "<div style='color: green;'>✅ Installed</div>" if success else "<div style='color: red;'>❌ Not Installed</div>"
        
        return result, status
    
    install_xatlas_btn.click(
        install_xatlas,
        outputs=[xatlas_status, xatlas_status_display]
    )


def create_dinov2_component(app: Any) -> None:
    """Create DINOv2 installation UI"""
    from ....config import TEXTURE_PIPELINE_COMPONENTS
    
    gr.Markdown("#### 🔍 DINOv2 (Feature Extraction)")
    gr.Markdown("**Enhances:** Texture detail and consistency")
    gr.Markdown("**Size:** ~4.5 GB")
    
    # Check if installed
    dinov2_path = app.model_manager.models_dir / "texture_components" / "dinov2"
    dinov2_installed = dinov2_path.exists() and any(dinov2_path.iterdir()) if dinov2_path.exists() else False
    
    dinov2_status_display = gr.HTML(
        value=f"<div style='color: {'green' if dinov2_installed else 'red'};'>{'✅ Installed' if dinov2_installed else '❌ Not Installed'}</div>"
    )
    
    with gr.Row():
        install_dino_btn = create_action_button("Install DINOv2", variant="secondary", size="sm")
        dino_status = gr.HTML()
    
    def install_dinov2(progress=gr.Progress()):
        progress(0, desc="Starting DINOv2 download...")
        
        comp_info = TEXTURE_PIPELINE_COMPONENTS.get("dinov2", {})
        success, message, _ = app.model_manager.download_texture_component(
            component_name="dinov2",
            component_info=comp_info,
            priority=1
        )
        
        progress(1.0, desc="Installation complete!" if success else "Installation failed")
        
        result = f"<div style='color: {'green' if success else 'red'};'>{'✅' if success else '❌'} {message}</div>"
        status = "<div style='color: green;'>✅ Installed</div>" if success else "<div style='color: red;'>❌ Not Installed</div>"
        
        return result, status
    
    install_dino_btn.click(
        install_dinov2,
        outputs=[dino_status, dinov2_status_display]
    )


def create_rembg_component(app: Any) -> None:
    """Create Background Remover installation UI"""
    from ....config import TEXTURE_PIPELINE_COMPONENTS
    
    gr.Markdown("#### 🎨 Background Remover")
    gr.Markdown("**Enhances:** Clean 3D generation from images")
    gr.Markdown("**Size:** ~176 MB")
    
    # Check if installed
    rembg_path = app.model_manager.models_dir / "texture_components" / "background_remover"
    rembg_installed = rembg_path.exists() and any(rembg_path.iterdir()) if rembg_path.exists() else False
    
    rembg_status_display = gr.HTML(
        value=f"<div style='color: {'green' if rembg_installed else 'red'};'>{'✅ Installed' if rembg_installed else '❌ Not Installed'}</div>"
    )
    
    with gr.Row():
        install_rembg_btn = create_action_button("Install RemBG", variant="secondary", size="sm")
        rembg_status = gr.HTML()
    
    def install_rembg(progress=gr.Progress()):
        progress(0, desc="Starting RemBG download...")
        
        comp_info = TEXTURE_PIPELINE_COMPONENTS.get("background_remover", {})
        success, message, _ = app.model_manager.download_texture_component(
            component_name="background_remover",
            component_info=comp_info,
            priority=1
        )
        
        progress(1.0, desc="Installation complete!" if success else "Installation failed")
        
        result = f"<div style='color: {'green' if success else 'red'};'>{'✅' if success else '❌'} {message}</div>"
        status = "<div style='color: green;'>✅ Installed</div>" if success else "<div style='color: red;'>❌ Not Installed</div>"
        
        return result, status
    
    install_rembg_btn.click(
        install_rembg,
        outputs=[rembg_status, rembg_status_display]
    )