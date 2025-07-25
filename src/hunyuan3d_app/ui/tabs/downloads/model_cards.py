"""Model download card creation"""

import gradio as gr
from pathlib import Path
from typing import Dict, Any, Union


def create_model_download_row(app, model_name: str, model_info: Union[Dict, Any], model_type: str):
    """Create a download row for a model
    
    Args:
        app: Application instance
        model_name: Name of the model
        model_info: Model information (dict or object)
        model_type: Type of model (image, 3d, etc.)
    """
    
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
                    model_path = app.model_manager.models_dir / model_type / model_name
                    is_complete = app.model_manager.check_model_complete(model_path, model_type, model_name)
                
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


def create_model_category_section(app, category_title: str, models_dict: Dict[str, Any], model_type: str):
    """Create a section for a category of models
    
    Args:
        app: Application instance
        category_title: Title for the category
        models_dict: Dictionary of models
        model_type: Type of models in this category
    """
    gr.Markdown(f"### {category_title}")
    
    for model_name, model_info in models_dict.items():
        create_model_download_row(app, model_name, model_info, model_type)


def create_quick_download_button(app, button_text: str, models_to_download: list, priority: int = 2):
    """Create a quick download button for multiple models
    
    Args:
        app: Application instance
        button_text: Text for the button
        models_to_download: List of (model_name, model_type) tuples
        priority: Download priority
    """
    button = gr.Button(button_text, variant="primary")
    progress = gr.HTML(visible=False)
    
    def download_all():
        results = []
        for model_name, model_type in models_to_download:
            success, message, download_id = app.model_manager.download_model_concurrent(
                model_name=model_name,
                model_type=model_type,
                priority=priority
            )
            results.append((model_name, success, message))
        
        # Format results
        html = "<div style='margin-top: 10px;'>"
        html += "<h4>Download Results:</h4>"
        html += "<ul>"
        for model_name, success, message in results:
            color = "#059669" if success else "#dc3545"
            html += f"<li style='color: {color};'>{model_name}: {message}</li>"
        html += "</ul>"
        html += "</div>"
        
        return gr.update(visible=True, value=html)
    
    button.click(download_all, outputs=[progress])
    
    return button, progress