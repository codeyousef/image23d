"""Performance Settings UI

Handles performance optimization settings including memory, generation, and cache configuration.
"""

import gradio as gr
from typing import Any
import logging

from ...components.common import create_action_button

logger = logging.getLogger(__name__)


def create_performance_subtab(app: Any) -> None:
    """Performance and optimization settings"""
    
    gr.Markdown("### Performance Settings")
    
    # Memory optimization
    with gr.Group():
        gr.Markdown("#### Memory Optimization")
        
        enable_xformers = gr.Checkbox(
            label="Enable xFormers (memory efficient attention)",
            value=True,
            info="Reduces VRAM usage with minimal quality impact"
        )
        
        enable_cpu_offload = gr.Checkbox(
            label="Enable CPU Offload",
            value=False,
            info="Move model parts to CPU when not in use (slower but saves VRAM)"
        )
        
        enable_sequential_cpu_offload = gr.Checkbox(
            label="Enable Sequential CPU Offload",
            value=False,
            info="More aggressive CPU offloading (very slow but minimal VRAM)"
        )
        
        attention_slicing = gr.Dropdown(
            choices=["Disabled", "Auto", "Max"],
            value="Auto",
            label="Attention Slicing",
            info="Process attention in chunks to save memory"
        )
    
    # Generation settings
    with gr.Group():
        gr.Markdown("#### Generation Settings")
        
        with gr.Row():
            default_batch_size = gr.Slider(
                1, 8, 1,
                step=1,
                label="Default Batch Size"
            )
            
            vae_tiling = gr.Checkbox(
                label="Enable VAE Tiling",
                value=False,
                info="Process large images in tiles"
            )
        
        with gr.Row():
            torch_compile = gr.Checkbox(
                label="Enable Torch Compile",
                value=False,
                info="Compile models for faster inference (initial slowdown)"
            )
            
            channels_last = gr.Checkbox(
                label="Channels Last Memory Format",
                value=True,
                info="Optimize memory layout for better performance"
            )
    
    # Cache settings
    with gr.Group():
        gr.Markdown("#### Cache Settings")
        
        with gr.Row():
            model_cache_size = gr.Slider(
                1, 10, 3,
                step=1,
                label="Max Models in Memory",
                info="Number of models to keep loaded"
            )
            
            clear_cache_btn = create_action_button("🧹 Clear Cache", size="sm")
        
        cache_info = gr.HTML()
    
    # Apply settings
    apply_perf_btn = create_action_button("💾 Apply Settings", variant="primary")
    perf_status = gr.HTML()