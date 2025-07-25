"""API Credentials Management UI

Handles saving and managing API credentials for various services.
"""

import gradio as gr
from typing import Any
import logging

from ...components.common import create_action_button

logger = logging.getLogger(__name__)


def create_credentials_subtab(app: Any) -> None:
    """API credentials management"""
    
    if hasattr(app, 'credential_manager'):
        app.credential_manager.create_ui_component()
    else:
        gr.Markdown("### API Credentials")
        
        # Manual credential inputs
        with gr.Group():
            gr.Markdown("#### Hugging Face")
            hf_token = gr.Textbox(
                label="HF Token",
                type="password",
                placeholder="hf_..."
            )
            
        with gr.Group():
            gr.Markdown("#### Civitai")
            civitai_token = gr.Textbox(
                label="API Key",
                type="password",
                placeholder="Your Civitai API key"
            )
        
        save_creds_btn = create_action_button("💾 Save Credentials", variant="primary")
        creds_status = gr.HTML()