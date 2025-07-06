"""Header component for the UI."""

import gradio as gr


def create_header():
    """Create the main header component."""
    gr.HTML("""
    <div class="main-header">
        <h1>🎨 Hunyuan3D Studio</h1>
        <p>Complete Text → Image → 3D Pipeline with Best-in-Class Models</p>
    </div>
    """)