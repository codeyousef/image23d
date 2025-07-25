"""Queue Management UI

Handles job queue monitoring and configuration.
"""

import gradio as gr
from typing import Any
import logging

from ...components.common import create_action_button

logger = logging.getLogger(__name__)


def create_queue_management_subtab(app: Any) -> None:
    """Queue and job management"""
    
    gr.Markdown("### Queue Management")
    
    # Queue status
    with gr.Row():
        queue_status = gr.HTML()
        refresh_queue_btn = create_action_button("🔄 Refresh", size="sm")
    
    # Active jobs
    with gr.Group():
        gr.Markdown("#### Active Jobs")
        
        active_jobs = gr.DataFrame(
            headers=["ID", "Type", "Status", "Progress", "Started", "Actions"],
            datatype=["str", "str", "str", "number", "str", "str"]
        )
        
        with gr.Row():
            pause_queue_btn = create_action_button("⏸️ Pause Queue", size="sm")
            resume_queue_btn = create_action_button("▶️ Resume Queue", size="sm")
            clear_completed_btn = create_action_button("🧹 Clear Completed", size="sm")
    
    # Queue settings
    with gr.Group():
        gr.Markdown("#### Queue Settings")
        
        with gr.Row():
            max_workers = gr.Slider(
                1, 8, 2,
                step=1,
                label="Max Concurrent Jobs",
                info="Number of jobs to process simultaneously"
            )
            
            job_timeout = gr.Slider(
                60, 1800, 600,
                step=60,
                label="Job Timeout (seconds)",
                info="Maximum time for a single job"
            )
        
        auto_cleanup = gr.Checkbox(
            label="Auto-cleanup completed jobs",
            value=True,
            info="Remove completed jobs after 24 hours"
        )
    
    # Job history
    with gr.Group():
        gr.Markdown("#### Job History")
        
        job_history = gr.DataFrame(
            headers=["ID", "Type", "Status", "Duration", "Completed", "Size"],
            datatype=["str", "str", "str", "str", "str", "str"]
        )
        
        with gr.Row():
            export_history_btn = create_action_button("📤 Export History", size="sm")
            clear_history_btn = create_action_button("🗑️ Clear History", variant="stop", size="sm")
    
    # Wire up queue display
    if hasattr(app, 'queue_manager'):
        app.queue_manager.create_ui_component()