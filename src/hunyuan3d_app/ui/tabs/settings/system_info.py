"""System Information and Diagnostics UI

Provides system checks, performance metrics, and diagnostic tools.
"""

import gradio as gr
from typing import Any
import logging

from ...components.common import create_action_button

logger = logging.getLogger(__name__)


def create_system_info_subtab(app: Any) -> None:
    """System information and diagnostics"""
    
    gr.Markdown("### System Information")
    
    # System requirements check
    with gr.Group():
        gr.Markdown("#### System Requirements")
        
        def get_system_requirements_display():
            """Get formatted system requirements display"""
            if hasattr(app, 'check_system_requirements'):
                try:
                    results = app.check_system_requirements()
                    if isinstance(results, dict):
                        html = "<h4>System Status</h4>"
                        
                        # Overall status
                        status = results.get("overall_status", "unknown")
                        status_color = {"ok": "green", "warning": "orange", "error": "red"}.get(status, "gray")
                        html += f"<p><b>Status:</b> <span style='color:{status_color}'>{status.upper()}</span></p>"
                        
                        # Warnings
                        if results.get("warnings"):
                            html += "<p><b>Warnings:</b></p><ul>"
                            for warning in results["warnings"]:
                                html += f"<li>{warning}</li>"
                            html += "</ul>"
                        
                        # Errors
                        if results.get("errors"):
                            html += "<p><b>Errors:</b></p><ul>"
                            for error in results["errors"]:
                                html += f"<li style='color:red'>{error}</li>"
                            html += "</ul>"
                        
                        return html
                    else:
                        return f"<p>System check returned: {str(results)}</p>"
                except Exception as e:
                    return f"<p>Error checking system requirements: {str(e)}</p>"
            else:
                return "<p>System requirements check not available</p>"
        
        system_check = gr.HTML(
            value=get_system_requirements_display()
        )
        
        check_system_btn = create_action_button("🔍 Check System", variant="primary")
    
    # Performance metrics
    with gr.Group():
        gr.Markdown("#### Performance Metrics")
        
        perf_metrics = gr.HTML()
        
        with gr.Row():
            benchmark_btn = create_action_button("📊 Run Benchmark", variant="primary")
            export_metrics_btn = create_action_button("📤 Export Metrics", size="sm")
    
    # GPU information
    with gr.Group():
        gr.Markdown("#### GPU Information")
        
        gpu_info = gr.HTML()
        monitor_gpu = gr.Checkbox(
            label="Enable GPU Monitoring",
            value=True
        )
    
    # Diagnostics
    with gr.Group():
        gr.Markdown("#### Diagnostics")
        
        with gr.Row():
            test_image_gen = create_action_button("🖼️ Test Image Generation", size="sm")
            test_3d_gen = create_action_button("🎲 Test 3D Generation", size="sm")
            test_video_gen = create_action_button("🎬 Test Video Generation", size="sm")
        
        diagnostic_output = gr.Textbox(
            label="Diagnostic Output",
            lines=10,
            max_lines=20
        )
    
    # Logs
    with gr.Group():
        gr.Markdown("#### Application Logs")
        
        log_level = gr.Dropdown(
            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
            value="INFO",
            label="Log Level"
        )
        
        logs_output = gr.Textbox(
            label="Recent Logs",
            lines=15,
            max_lines=30
        )
        
        with gr.Row():
            refresh_logs_btn = create_action_button("🔄 Refresh Logs", size="sm")
            export_logs_btn = create_action_button("📤 Export Logs", size="sm")
            clear_logs_btn = create_action_button("🗑️ Clear Logs", size="sm")
    
    # Wire up system check
    if hasattr(app, 'check_system_requirements'):
        check_system_btn.click(
            lambda: app.check_system_requirements()["html"],
            outputs=[system_check]
        )