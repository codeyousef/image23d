"""Model Management tab for downloading and managing models."""

import gradio as gr
from typing import Any, List

from ..components.common import update_model_dropdowns_helper, create_action_button
from ...config import IMAGE_MODELS, GATED_IMAGE_MODELS, GGUF_IMAGE_MODELS, HUNYUAN3D_MODELS, VIDEO_MODELS, TEXTURE_PIPELINE_COMPONENTS


def create_model_management_tab(app: Any, model_status: gr.HTML, manual_img_model: gr.Dropdown, manual_3d_model: gr.Dropdown):
    """Create the Model Management tab.
    
    Args:
        app: Application instance
        model_status: Model status HTML component
        manual_img_model: Manual pipeline image model dropdown
        manual_3d_model: Manual pipeline 3D model dropdown
    """
    
    # Helper function to extract size in GB from size string
    def get_size_gb(size_str):
        """Extract size in GB from size string like '~10 GB' or '10GB'"""
        if isinstance(size_str, (int, float)):
            return size_str
        import re
        match = re.search(r'(\d+(?:\.\d+)?)', str(size_str))
        return float(match.group(1)) if match else 8.0
    
    # Helper function to extract VRAM in GB
    def get_vram_gb(vram_str):
        """Extract VRAM requirement from string like '12GB+' """
        if isinstance(vram_str, (int, float)):
            return vram_str
        import re
        match = re.search(r'(\d+)', str(vram_str))
        return int(match.group(1)) if match else 8
    gr.Markdown("""
    ### Download and Manage AI Models
    Download models for offline use and manage your model library.
    """)
    
    # HuggingFace Token Section
    with gr.Group():
        gr.Markdown("""
        #### 🔐 HuggingFace Token (Optional)
        Some models require authentication. Get your token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
        """)
        
        with gr.Row():
            hf_token = gr.Textbox(
                label="HuggingFace Token",
                type="password",
                placeholder="hf_...",
                scale=4
            )
            set_token_btn = create_action_button(
                label="Set Token",
                variant="secondary",
                size="sm"
            )
        
        token_status = gr.HTML()
    
    with gr.Tabs():
        # Image Models Tab
        with gr.Tab("🎨 Image Models"):
            # Standard Models
            gr.Markdown("#### Standard Models (Recommended)")
            standard_models = gr.Dataset(
                components=[gr.Textbox(visible=False)],
                headers=["Model", "Size", "VRAM", "Description", "Status"],
                samples=[
                    [name, info.get('size', 'Unknown'), info.get('vram_required', '8GB+'), 
                     info.get('description', ''), app.check_model_downloaded("image", name)]
                    for name, info in IMAGE_MODELS.items()
                ],
                elem_id="standard-models-table"
            )
            
            # Gated Models
            gr.Markdown("#### Gated Models (Require HF Token)")
            gated_models = gr.Dataset(
                components=[gr.Textbox(visible=False)],
                headers=["Model", "Size", "VRAM", "Description", "Status"],
                samples=[
                    [name, info.get('size', 'Unknown'), info.get('vram_required', '8GB+'),
                     info.get('description', ''), app.check_model_downloaded("image", name)]
                    for name, info in GATED_IMAGE_MODELS.items()
                ],
                elem_id="gated-models-table"
            )
            
            # GGUF Models
            gr.Markdown("#### GGUF Quantized Models (Lower VRAM)")
            gguf_models = gr.Dataset(
                components=[gr.Textbox(visible=False)],
                headers=["Model", "Size", "VRAM", "Description", "Status"],
                samples=[
                    [name, info.get('size', 'Unknown'), info.get('vram_required', '6GB+'),
                     info.get('description', ''), app.check_model_downloaded("image", name)]
                    for name, info in GGUF_IMAGE_MODELS.items()
                ],
                elem_id="gguf-models-table"
            )
            
            # Model actions
            with gr.Row():
                selected_img_model = gr.Textbox(
                    label="Selected Model",
                    interactive=False
                )
                img_download_btn = create_action_button(
                    label="📥 Download",
                    variant="primary",
                    size="sm"
                )
                img_delete_btn = create_action_button(
                    label="🗑️ Delete",
                    variant="stop",
                    size="sm"
                )
        
        # 3D Models Tab
        with gr.Tab("🔮 3D Models"):
            gr.Markdown("#### Hunyuan3D Models")
            hunyuan_models = gr.Dataset(
                components=[gr.Textbox(visible=False)],
                headers=["Model", "Size", "VRAM", "Description", "Status"],
                samples=[
                    [name, info.get('size', 'Unknown'), info.get('vram_required', '12GB+'),
                     info.get('description', ''), app.check_model_downloaded("3d", name)]
                    for name, info in HUNYUAN3D_MODELS.items()
                ],
                elem_id="hunyuan-models-table"
            )
            
            # Model actions
            with gr.Row():
                selected_3d_model = gr.Textbox(
                    label="Selected Model",
                    interactive=False
                )
                hunyuan_download_btn = create_action_button(
                    label="📥 Download",
                    variant="primary",
                    size="sm"
                )
                hunyuan_delete_btn = create_action_button(
                    label="🗑️ Delete",
                    variant="stop",
                    size="sm"
                )
        
        # Video Models Tab
        with gr.Tab("🎬 Video Models"):
            gr.Markdown("#### Text-to-Video Models")
            video_models = gr.Dataset(
                components=[gr.Textbox(visible=False)],
                headers=["Model", "Size", "VRAM", "Description", "Status"],
                samples=[
                    [name, info.get('size', 'Unknown'), info.get('vram_required', '12GB+'),
                     info.get('description', ''), app.check_model_downloaded("video", name)]
                    for name, info in VIDEO_MODELS.items()
                ],
                elem_id="video-models-table"
            )
            
            # Model actions
            with gr.Row():
                selected_video_model = gr.Textbox(
                    label="Selected Model",
                    interactive=False
                )
                video_download_btn = create_action_button(
                    label="📥 Download",
                    variant="primary",
                    size="sm"
                )
                video_delete_btn = create_action_button(
                    label="🗑️ Delete",
                    variant="stop",
                    size="sm"
                )
        
        # Texture Components Tab
        with gr.Tab("🎨 Texture Components"):
            from .settings.texture_components import create_texture_components_tab
            create_texture_components_tab(app)
    
    # Download progress
    with gr.Group():
        gr.Markdown("#### Download Progress")
        download_progress = gr.HTML()
        cancel_download_btn = create_action_button(
            label="❌ Cancel Download",
            variant="stop",
            size="sm"
        )
    
    # Wire up token setting
    set_token_btn.click(
        fn=app.set_hf_token,
        inputs=[hf_token],
        outputs=[token_status]
    )
    
    # Model selection handlers
    def select_model(evt: gr.SelectData, model_type: str):
        if evt.index is not None:
            return evt.value[0]  # Return model name
        return ""
    
    standard_models.select(
        fn=lambda evt: select_model(evt, "standard"),
        outputs=[selected_img_model]
    )
    
    gated_models.select(
        fn=lambda evt: select_model(evt, "gated"),
        outputs=[selected_img_model]
    )
    
    gguf_models.select(
        fn=lambda evt: select_model(evt, "gguf"),
        outputs=[selected_img_model]
    )
    
    hunyuan_models.select(
        fn=lambda evt: select_model(evt, "3d"),
        outputs=[selected_3d_model]
    )
    
    video_models.select(
        fn=lambda evt: select_model(evt, "video"),
        outputs=[selected_video_model]
    )
    
    # Download handlers
    def download_and_update(model_name: str, model_type: str):
        result = app.download_model(model_name, model_type)
        
        # After download, update the model dropdowns
        choices = app.get_model_selection_data()
        updates = update_model_dropdowns_helper(choices)
        
        return result, updates[0], updates[1]
    
    img_download_btn.click(
        fn=lambda name: download_and_update(name, "image"),
        inputs=[selected_img_model],
        outputs=[download_progress, manual_img_model, manual_3d_model]
    )
    
    hunyuan_download_btn.click(
        fn=lambda name: download_and_update(name, "3d"),
        inputs=[selected_3d_model],
        outputs=[download_progress, manual_img_model, manual_3d_model]
    )
    
    video_download_btn.click(
        fn=lambda name: download_and_update(name, "video"),
        inputs=[selected_video_model],
        outputs=[download_progress, manual_img_model, manual_3d_model]
    )
    
    # Delete handlers
    def delete_and_update(model_name: str, model_type: str):
        result = app.delete_model(model_name, model_type)
        
        # After deletion, update the model dropdowns
        choices = app.get_model_selection_data()
        updates = update_model_dropdowns_helper(choices)
        
        return result, app.get_model_status(), updates[0], updates[1]
    
    img_delete_btn.click(
        fn=lambda name: delete_and_update(name, "image"),
        inputs=[selected_img_model],
        outputs=[download_progress, model_status, manual_img_model, manual_3d_model]
    )
    
    hunyuan_delete_btn.click(
        fn=lambda name: delete_and_update(name, "3d"),
        inputs=[selected_3d_model],
        outputs=[download_progress, model_status, manual_img_model, manual_3d_model]
    )
    
    video_delete_btn.click(
        fn=lambda name: delete_and_update(name, "video"),
        inputs=[selected_video_model],
        outputs=[download_progress, model_status, manual_img_model, manual_3d_model]
    )
    
    # Cancel download
    cancel_download_btn.click(
        fn=app.cancel_download,
        outputs=[download_progress]
    )
    
    # Auto-refresh download progress
    download_progress.change(
        fn=app.get_download_status,
        outputs=[download_progress],
        every=1  # Update every second
    )