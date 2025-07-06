"""Enhanced UI with 5-tab layout"""

import gradio as gr
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .modern import ModernUI, load_modern_css
from ..config import ALL_IMAGE_MODELS, HUNYUAN3D_MODELS, QUALITY_PRESETS

logger = logging.getLogger(__name__)


def create_enhanced_interface(app):
    """Create enhanced interface with new 5-tab layout"""
    
    # Load custom CSS with additional styles for the new layout
    custom_css = load_modern_css() + """
    /* Additional styles for 5-tab layout */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 12px;
        margin-bottom: 2rem;
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        margin: 0;
        font-weight: 700;
    }
    
    .main-header p {
        font-size: 1.1rem;
        margin-top: 0.5rem;
        opacity: 0.9;
    }
    
    /* Tab styling */
    .tabs > .tab-nav {
        border-bottom: 2px solid #e5e7eb;
        margin-bottom: 1.5rem;
    }
    
    .tabs > .tab-nav > button {
        font-size: 1.1rem;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        border-bottom: 3px solid transparent;
        transition: all 0.3s ease;
    }
    
    .tabs > .tab-nav > button.selected {
        color: #667eea;
        border-bottom-color: #667eea;
    }
    
    /* Feature card styling */
    .feature-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .feature-card:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        transform: translateY(-2px);
    }
    
    .feature-card h4 {
        margin: 0 0 0.5rem 0;
        color: #1f2937;
        font-size: 1.2rem;
    }
    
    .feature-card p {
        margin: 0;
        color: #6b7280;
        font-size: 0.95rem;
    }
    
    /* Sidebar styling */
    .sidebar-gallery {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 0.5rem;
    }
    
    .sidebar-gallery .thumbnail-item {
        cursor: pointer;
        transition: transform 0.2s ease;
    }
    
    .sidebar-gallery .thumbnail-item:hover {
        transform: scale(1.05);
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        
        .tabs > .tab-nav > button {
            font-size: 1rem;
            padding: 0.5rem 1rem;
        }
    }
    """
    
    with gr.Blocks(
        title="Hunyuan3D Studio - AI Creative Suite",
        theme=gr.themes.Soft(
            primary_hue="purple",
            secondary_hue="blue",
            neutral_hue="slate"
        ),
        css=custom_css,
        analytics_enabled=False
    ) as interface:
        
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>🎨 Hunyuan3D Studio</h1>
            <p>AI-Powered Creative Suite for Images, Videos, and 3D Models</p>
        </div>
        """)
        
        # Dashboard stats (compact version) - Comment out to prevent blocking
        # TODO: Load stats asynchronously
        with gr.Row():
            # stats = app.get_system_stats()
            with gr.Column(scale=1):
                ModernUI.create_stat_card(
                    "0",  # str(stats.get('total_generations', 0)),
                    "Total Creations",
                    "🎨"
                )
            with gr.Column(scale=1):
                ModernUI.create_stat_card(
                    "0",  # str(stats.get('models_loaded', 0)),
                    "Active Models",
                    "🤖"
                )
            with gr.Column(scale=1):
                ModernUI.create_stat_card(
                    "0.0GB",  # f"{stats.get('vram_used', 0):.1f}GB",
                    "VRAM Used",
                    "💾"
                )
            with gr.Column(scale=1):
                ModernUI.create_stat_card(
                    "0",  # str(stats.get('queue_pending', 0)),
                    "Queue",
                    "⏳"
                )
        
        # Main 5-tab layout
        with gr.Tabs(elem_classes=["main-tabs"]) as main_tabs:
            # 1. Image Generation Tab
            with gr.Tab("🎨 Image Generation", elem_id="image_gen_tab"):
                from .tabs.image_generation import create_image_generation_tab
                create_image_generation_tab(app)
            
            # 2. Video Generation Tab
            with gr.Tab("🎬 Video Generation", elem_id="video_gen_tab"):
                from .tabs.video_generation import create_video_generation_tab
                create_video_generation_tab(app)
            
            # 3. 3D Generation Tab
            with gr.Tab("🎲 3D Generation", elem_id="threed_gen_tab"):
                from .tabs.threed_generation import create_threed_generation_tab
                create_threed_generation_tab(app)
            
            # 4. Generated Media Tab
            with gr.Tab("🖼️ Generated Media", elem_id="media_gallery_tab"):
                from .tabs.media_gallery import create_media_gallery_tab
                create_media_gallery_tab(app)
            
            # 5. Settings Tab
            with gr.Tab("⚙️ Settings", elem_id="settings_tab"):
                from .tabs.settings import create_settings_tab
                create_settings_tab(app)
        
        # Add real-time progress display (hidden, updates automatically)
        from .components.progress import create_progress_component
        progress_display = create_progress_component()
        
        # Badge update system for media gallery
        def update_tab_badges():
            """Update badges for tabs with counts"""
            try:
                # Get unviewed media count
                unviewed_count = app.history_manager.get_unviewed_count() if hasattr(app.history_manager, 'get_unviewed_count') else 0
                
                # Get queue status
                queue_status = app.queue_manager.get_queue_status()
                queue_count = queue_status.get("pending", 0) + queue_status.get("active", 0)
                
                # Create badge update script
                badge_script = f"""
                <div id="badge-update-container" style="display: none;">
                    <span id="media-count">{unviewed_count}</span>
                    <span id="queue-count">{queue_count}</span>
                </div>
                <script>
                (function() {{
                    // Function to update badges
                    function updateBadges() {{
                        var mediaCount = parseInt(document.getElementById('media-count')?.textContent || '0');
                        var queueCount = parseInt(document.getElementById('queue-count')?.textContent || '0');
                        
                        // Update media gallery badge
                        var mediaTab = document.querySelector('#media_gallery_tab button');
                        if (mediaTab) {{
                            var mediaBadge = mediaTab.querySelector('.tab-badge');
                            if (!mediaBadge) {{
                                mediaBadge = document.createElement('span');
                                mediaBadge.className = 'tab-badge';
                                mediaBadge.style.cssText = 'background: #4CAF50; color: white; border-radius: 10px; padding: 2px 6px; font-size: 11px; margin-left: 5px; position: relative; top: -1px; display: none;';
                                mediaTab.appendChild(mediaBadge);
                            }}
                            if (mediaCount > 0) {{
                                mediaBadge.textContent = mediaCount;
                                mediaBadge.style.display = 'inline';
                            }} else {{
                                mediaBadge.style.display = 'none';
                            }}
                        }}
                    }}
                    
                    // Initial update
                    updateBadges();
                    
                    // Set up periodic updates
                    setInterval(updateBadges, 3000);
                }})();
                </script>
                """
                
                return badge_script
            except Exception as e:
                logger.error(f"Error updating badges: {e}")
                return ""
        
        # Add badge updater component
        with gr.Row(visible=False):
            badge_updater = gr.HTML(value="")  # Don't call update_tab_badges() synchronously
        
        # Auto-refresh badges
        def refresh_badges():
            return gr.update(value=update_tab_badges())
        
        # Set up periodic refresh if supported
        try:
            interface.load(
                refresh_badges,
                outputs=[badge_updater],
                every=3  # Refresh every 3 seconds
            )
        except Exception as e:
            logger.debug(f"Badge auto-refresh not available: {e}")
        
        # Auto-refresh stats display
        def refresh_stats_display():
            """Refresh the stats cards"""
            stats = app.get_system_stats()
            # This would need to return updated HTML for all stat cards
            # For now, stats are static until page refresh
            pass
        
        # Footer with version info
        gr.HTML("""
        <div style="text-align: center; margin-top: 3rem; padding: 1rem; color: #6b7280; font-size: 0.9rem;">
            <p>Hunyuan3D Studio v2.0 | Powered by AI | 
            <a href="https://github.com/yourusername/hunyuan3d-app" target="_blank" style="color: #667eea;">GitHub</a>
            </p>
        </div>
        """)
    
    return interface