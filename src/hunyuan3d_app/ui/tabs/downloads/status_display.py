"""Download status display functionality"""

import gradio as gr
from typing import Dict, Any


def get_download_status_html(app) -> str:
    """Generate HTML for download status panel."""
    try:
        status = app.model_manager.get_download_status()
        active_downloads = status.get('active', {})
        queued_downloads = status.get('queued', [])
        max_concurrent = status.get('max_concurrent', 3)
        
        html = '<div class="download-status-panel" style="padding: 10px; border: 1px solid #ddd; border-radius: 5px; margin-bottom: 10px;">'
        html += f'<h4 style="margin-top: 0;">📥 Download Status (Max {max_concurrent} concurrent)</h4>'
        
        if not active_downloads and not queued_downloads:
            html += '<p style="color: #666;">No active downloads</p>'
        else:
            # Active downloads
            if active_downloads:
                html += '<div class="active-downloads">'
                html += f'<h5>🔄 Active Downloads ({len(active_downloads)})</h5>'
                for download_id, info in active_downloads.items():
                    model_name = info.get('model_name', 'Unknown')
                    progress = info.get('progress', {})
                    percentage = getattr(progress, 'percentage', 0)
                    current_file = getattr(progress, 'current_file', '')
                    speed = getattr(progress, 'speed', 0) / (1024**2) if hasattr(progress, 'speed') else 0
                    
                    html += f'<div style="margin-bottom: 10px;">'
                    html += f'<strong>{model_name}</strong><br>'
                    html += f'<div style="background: #f0f0f0; height: 20px; border-radius: 3px; overflow: hidden;">'
                    html += f'<div style="background: #4CAF50; height: 100%; width: {percentage}%; transition: width 0.3s;"></div>'
                    html += f'</div>'
                    html += f'<small>{percentage}% - {speed:.1f} MB/s'
                    if current_file:
                        html += f' - {current_file}'
                    html += '</small></div>'
                html += '</div>'
            
            # Queued downloads
            if queued_downloads:
                html += '<div class="queued-downloads" style="margin-top: 15px;">'
                html += f'<h5>⏳ Queued Downloads ({len(queued_downloads)})</h5>'
                html += '<ol style="margin: 5px 0; padding-left: 20px;">'
                for idx, info in enumerate(queued_downloads):
                    model_name = info.get('model_name', 'Unknown')
                    html += f'<li>{model_name}</li>'
                html += '</ol>'
                html += '</div>'
        
        html += '</div>'
        return html
        
    except Exception as e:
        return f'<div class="error">Error getting download status: {str(e)}</div>'


def create_status_display_component(app) -> gr.HTML:
    """Create the download status display component."""
    return gr.HTML(
        value=get_download_status_html(app),
        elem_id="download-status-panel"
    )


def create_control_buttons(app) -> tuple:
    """Create control buttons for download management."""
    refresh_btn = gr.Button("🔄 Refresh Status", size="sm")
    stop_all_btn = gr.Button("⏹️ Stop All Downloads", size="sm", variant="stop")
    
    def refresh_status():
        return get_download_status_html(app)
    
    def stop_all_downloads():
        # TODO: Implement stop all functionality
        app.model_manager.stop_download()
        return "All downloads stopped", get_download_status_html(app)
    
    return refresh_btn, stop_all_btn, refresh_status, stop_all_downloads