"""Media Sidebar Component for showing recent generations"""

import gradio as gr
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def create_media_sidebar(
    app: Any,
    media_type: str,
    on_select_callback: Optional[Callable] = None,
    title: str = "Recent Generations"
) -> gr.Column:
    """Create a sidebar showing recent media filtered by type
    
    Args:
        app: Application instance
        media_type: Type to filter ('image', 'video', '3d')
        on_select_callback: Function to call when item is selected
        title: Sidebar title
        
    Returns:
        gr.Column containing the sidebar
    """
    
    with gr.Column(scale=1, min_width=200) as sidebar:
        gr.Markdown(f"### {title}")
        
        # Quick filters
        with gr.Row():
            time_filter = gr.Radio(
                choices=["1h", "24h", "7d", "All"],
                value="24h",
                label="Time Range",
                scale=3
            )
            refresh_btn = gr.Button("🔄", scale=1, size="sm")
        
        # Gallery
        sidebar_gallery = gr.Gallery(
            label=None,
            columns=2,
            rows=6,
            height="auto",
            object_fit="cover",
            show_label=False,
            elem_classes=["sidebar-gallery"]
        )
        
        # Hidden state for tracking selection
        selected_item = gr.State(None)
        
        def load_recent_media(time_range: str):
            """Load recent media based on time filter"""
            try:
                # Calculate date range
                end_date = datetime.now()
                if time_range == "1h":
                    start_date = end_date - timedelta(hours=1)
                elif time_range == "24h":
                    start_date = end_date - timedelta(days=1)
                elif time_range == "7d":
                    start_date = end_date - timedelta(days=7)
                else:
                    start_date = None
                
                # Get items from history
                records = app.history_manager.get_history(
                    limit=12,
                    generation_type=media_type,
                    start_date=start_date,
                    end_date=end_date
                )
                
                # Build gallery items
                gallery_items = []
                import tempfile
                import shutil
                
                temp_dir = Path(tempfile.gettempdir()) / "hunyuan3d_sidebar_temp"
                temp_dir.mkdir(exist_ok=True)
                
                for record in records:
                    try:
                        # Get thumbnail or first output
                        preview_path = None
                        
                        if record.thumbnails and len(record.thumbnails) > 0:
                            preview_path = record.thumbnails[0]
                        elif record.output_paths and len(record.output_paths) > 0:
                            output_path = Path(record.output_paths[0])
                            if output_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp']:
                                preview_path = str(output_path)
                        
                        if preview_path and Path(preview_path).exists():
                            # Copy to temp for Gradio access
                            src_path = Path(preview_path)
                            temp_file = temp_dir / f"sidebar_{record.id}{src_path.suffix}"
                            
                            if not temp_file.exists() or temp_file.stat().st_mtime < src_path.stat().st_mtime:
                                shutil.copy2(src_path, temp_file)
                            
                            # Add with truncated prompt as label
                            label = f"{record.prompt[:30]}..." if record.prompt else "Untitled"
                            gallery_items.append((str(temp_file), label))
                            
                    except Exception as e:
                        logger.error(f"Error processing sidebar item {record.id}: {e}")
                        continue
                
                return gallery_items
                
            except Exception as e:
                logger.error(f"Error loading sidebar media: {e}")
                return []
        
        def on_gallery_select(evt: gr.SelectData, time_range: str):
            """Handle gallery selection"""
            try:
                # Get the selected record
                end_date = datetime.now()
                if time_range == "1h":
                    start_date = end_date - timedelta(hours=1)
                elif time_range == "24h":
                    start_date = end_date - timedelta(days=1)
                elif time_range == "7d":
                    start_date = end_date - timedelta(days=7)
                else:
                    start_date = None
                
                records = app.history_manager.get_history(
                    limit=12,
                    generation_type=media_type,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if evt.index < len(records):
                    record = records[evt.index]
                    
                    # Get the actual file path
                    if record.output_paths and len(record.output_paths) > 0:
                        file_path = record.output_paths[0]
                        
                        # Call the callback if provided
                        if on_select_callback:
                            return on_select_callback(file_path, record)
                        else:
                            return file_path
                            
            except Exception as e:
                logger.error(f"Error handling sidebar selection: {e}")
                
            return None
        
        # Wire up events
        refresh_btn.click(
            load_recent_media,
            inputs=[time_filter],
            outputs=[sidebar_gallery]
        )
        
        time_filter.change(
            load_recent_media,
            inputs=[time_filter],
            outputs=[sidebar_gallery]
        )
        
        if on_select_callback:
            sidebar_gallery.select(
                on_gallery_select,
                inputs=[time_filter],
                outputs=[selected_item]
            )
        
        # Don't set initial value synchronously - let it load asynchronously
        # sidebar_gallery.value = load_recent_media("24h")
        
    return sidebar