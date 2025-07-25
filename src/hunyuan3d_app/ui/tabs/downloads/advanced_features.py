"""Advanced features model downloads"""

import gradio as gr
from typing import TYPE_CHECKING

from ....config import (
    IP_ADAPTER_MODELS, 
    FACE_SWAP_MODELS, 
    FACE_RESTORE_MODELS
)

if TYPE_CHECKING:
    from ....app import HunyuanApp


def create_advanced_features_tab(app: "HunyuanApp"):
    """Create advanced features download tab"""
    with gr.Tab("🔧 Advanced Features"):
        gr.Markdown("### 🔧 Advanced Feature Models")
        gr.Markdown("Models for IP-Adapter, Face Swap, and other advanced features")
        
        with gr.Row():
            create_ip_adapter_section(app)
            create_face_swap_section(app)


def create_ip_adapter_section(app: "HunyuanApp"):
    """Create IP-Adapter models section"""
    with gr.Column(scale=1):
        gr.Markdown("### 🎯 IP-Adapter Models")
        gr.Markdown("Image prompt adapters for style/content guidance")
        
        for model_name, model_info in IP_ADAPTER_MODELS.items():
            create_feature_model_card(app, model_name, model_info, "ip_adapter")


def create_face_swap_section(app: "HunyuanApp"):
    """Create face swap models section"""
    with gr.Column(scale=1):
        gr.Markdown("### 🔄 Face Swap Models")
        gr.Markdown("InsightFace models for face swapping")
        
        for model_name, model_info in FACE_SWAP_MODELS.items():
            create_feature_model_card(app, model_name, model_info, "face_swap")
            
        gr.Markdown("### 🔧 Face Restoration Models")
        gr.Markdown("Enhance face quality after swapping")
        
        for model_name, model_info in FACE_RESTORE_MODELS.items():
            create_feature_model_card(app, model_name, model_info, "face_restore")


def create_feature_model_card(app: "HunyuanApp", model_name: str, model_info: dict, model_type: str):
    """Create a download card for a feature model"""
    with gr.Group():
        gr.Markdown(f"**{model_info['name']}**")
        gr.Markdown(f"Size: {model_info['size']}")
        gr.Markdown(f"*{model_info['description']}*")
        
        downloaded = app.model_manager.get_downloaded_models(model_type)
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
        
        def download_feature_model(model_name=model_name, model_type=model_type):
            # Use concurrent download
            success, message, download_id = app.model_manager.download_model_concurrent(
                model_name=model_name,
                model_type=model_type,
                priority=1
            )
            
            if success:
                yield gr.update(visible=True, value=f"<div style='color: #059669;'>{message}</div>")
            else:
                yield gr.update(visible=True, value=f"<div style='color: #dc3545;'>{message}</div>")
                
        download_btn.click(
            download_feature_model,
            outputs=[progress]
        )