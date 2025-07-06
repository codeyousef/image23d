"""Model status display component."""

import gradio as gr
from typing import Any


def create_model_status(app: Any) -> gr.HTML:
    """Create model status display component.
    
    Args:
        app: Application instance
        
    Returns:
        HTML component for model status
    """
    return gr.HTML(value=app.get_model_status())