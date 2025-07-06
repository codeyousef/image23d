"""Media Gallery Tab UI Component

A comprehensive gallery view for all generated media with filtering,
preview, and management capabilities.
"""

import gradio as gr
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import logging
import time

logger = logging.getLogger(__name__)


def create_media_gallery_tab(app) -> None:
    """Create media gallery tab with badge support
    
    Args:
        app: The main application instance
    """
    
    # Track unviewed count
    unviewed_count = gr.State(0)
    current_selection = gr.State(None)
    
    gr.Markdown("### 🖼️ Generated Media Gallery")
    gr.Markdown("Click **Refresh Gallery** to load your media files.", elem_id="gallery_init_message")
    
    # Auto-load state
    has_loaded = gr.State(False)
    
    # Import button for existing files
    with gr.Row():
        import_btn = gr.Button("📥 Import Existing Files from Outputs", variant="secondary")
        refresh_btn = gr.Button("🔄 Refresh Gallery", variant="secondary")
        import_status = gr.Textbox(label="Import Status", visible=False)
    
    # Filters
    with gr.Row():
        type_filter = gr.Dropdown(
            choices=["All", "Images", "3D Models", "Videos"],
            value="All",
            label="Type",
            scale=1
        )
        date_filter = gr.Dropdown(
            choices=["Today", "This Week", "This Month", "All Time"],
            value="All Time",
            label="Date Range",
            scale=1
        )
        model_filter = gr.Dropdown(
            choices=["All Models"],
            value="All Models",
            label="Model",
            scale=2
        )
        unviewed_only = gr.Checkbox(
            label="Show new only",
            value=False,
            scale=1
        )
        
    # Gallery grid
    gallery = gr.Gallery(
        label="Generated Media",
        columns=4,
        rows=3,
        height="auto",
        show_label=False,
        elem_id="media_gallery_grid",
        object_fit="cover"
    )
    
    # Selected item viewer
    with gr.Row():
        with gr.Column(scale=2):
            # Large preview
            image_viewer = gr.Image(
                label="Preview",
                visible=True,
                elem_id="media_preview"
            )
            model_viewer = gr.Model3D(
                label="3D Preview",
                visible=False
            )
            
        with gr.Column(scale=1):
            # Item details
            item_info = gr.JSON(
                label="Generation Details",
                value={}
            )
            
            # Actions
            with gr.Row():
                download_btn = gr.Button("💾 Download", size="sm")
                favorite_btn = gr.Button("⭐ Favorite", size="sm")
                delete_btn = gr.Button("🗑️ Delete", size="sm", variant="stop")
                
            with gr.Row():
                regenerate_btn = gr.Button("🔄 Regenerate", size="sm")
                copy_prompt_btn = gr.Button("📋 Copy Prompt", size="sm")
                
    # Pagination
    with gr.Row():
        prev_btn = gr.Button("◀ Previous", size="sm")
        page_info = gr.Textbox(
            value="Page 1 of 1",
            interactive=False,
            show_label=False,
            scale=2
        )
        next_btn = gr.Button("Next ▶", size="sm")
        
    # Hidden states
    current_page = gr.State(0)
    total_pages = gr.State(1)
    
    # Load gallery function
    def load_gallery(type_filter, date_filter, model_filter, unviewed_only, page):
        """Load gallery items based on filters"""
        try:
            import shutil
            from pathlib import Path
            import tempfile
            
            # Calculate date range
            end_date = datetime.now()
            start_date = None
            
            if date_filter == "Today":
                start_date = end_date - timedelta(days=1)
            elif date_filter == "This Week":
                start_date = end_date - timedelta(weeks=1)
            elif date_filter == "This Month":
                start_date = end_date - timedelta(days=30)
            
            # Map type filter
            generation_type = None
            if type_filter == "Images":
                generation_type = "image"
            elif type_filter == "3D Models":
                generation_type = "3d"
            elif type_filter == "Videos":
                generation_type = "video"
            
            # Get items from history
            limit = 12
            offset = page * limit
            
            records = app.history_manager.get_history(
                limit=limit,
                offset=offset,
                generation_type=generation_type,
                model_name=model_filter if model_filter != "All Models" else None,
                start_date=start_date,
                end_date=end_date,
                favorites_only=False
            )
            
            # If unviewed only, filter further
            if unviewed_only:
                records = [r for r in records if not getattr(r, 'viewed', False)]
            
            # Build gallery items
            gallery_items = []
            
            # Get temp directory for Gradio-safe file access
            temp_dir = Path(tempfile.gettempdir()) / "hunyuan3d_gallery_temp"
            temp_dir.mkdir(exist_ok=True)
            
            for record in records:
                try:
                    # Get thumbnail or first output
                    preview_path = None
                    
                    if record.thumbnails and len(record.thumbnails) > 0:
                        preview_path = record.thumbnails[0]
                    elif record.output_paths and len(record.output_paths) > 0:
                        # Use first output if it's an image
                        output_path = Path(record.output_paths[0])
                        if output_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp']:
                            preview_path = str(output_path)
                    
                    if preview_path and Path(preview_path).exists():
                        # Copy file to temp directory for Gradio access
                        src_path = Path(preview_path)
                        temp_file = temp_dir / f"gallery_{record.id}{src_path.suffix}"
                        
                        # Only copy if not already in temp
                        if not temp_file.exists() or temp_file.stat().st_mtime < src_path.stat().st_mtime:
                            shutil.copy2(src_path, temp_file)
                        
                        label = f"{record.prompt[:50]}..." if record.prompt else "Untitled"
                        gallery_items.append((str(temp_file), label))
                except Exception as e:
                    logger.error(f"Error processing record {record.id}: {e}")
                    continue
            
            # Calculate total pages
            total_count = app.history_manager.get_history_count(
                generation_type=generation_type,
                model_name=model_filter if model_filter != "All Models" else None,
                start_date=start_date,
                end_date=end_date
            )
            total_pages = max(1, (total_count + limit - 1) // limit)
            
            page_text = f"Page {page + 1} of {total_pages}"
            
            # Mark items as viewed
            for record in records:
                if not getattr(record, 'viewed', False):
                    try:
                        app.history_manager.mark_as_viewed(record.id)
                    except Exception as e:
                        logger.error(f"Error marking record as viewed: {e}")
            
            # Clean up old temp files (older than 1 hour)
            try:
                import time
                current_time = time.time()
                for old_file in temp_dir.glob("gallery_*"):
                    if current_time - old_file.stat().st_mtime > 3600:  # 1 hour
                        old_file.unlink()
            except Exception as e:
                logger.debug(f"Error cleaning temp files: {e}")
            
            return gallery_items, page_text, total_pages
            
        except Exception as e:
            logger.error(f"Error loading gallery: {e}")
            return [], "Page 1 of 1", 1
    
    # Handle gallery selection
    def on_gallery_select(evt: gr.SelectData, type_filter, date_filter, model_filter, unviewed_only, page):
        """Handle gallery item selection"""
        
        # Get the selected record
        limit = 12
        offset = page * limit
        
        # Calculate date range
        end_date = datetime.now()
        start_date = None
        
        if date_filter == "Today":
            start_date = end_date - timedelta(days=1)
        elif date_filter == "This Week":
            start_date = end_date - timedelta(weeks=1)
        elif date_filter == "This Month":
            start_date = end_date - timedelta(days=30)
        
        # Map type filter
        generation_type = None
        if type_filter == "Images":
            generation_type = "image"
        elif type_filter == "3D Models":
            generation_type = "3d"
        elif type_filter == "Videos":
            generation_type = "video"
        
        records = app.history_manager.get_history(
            limit=limit,
            offset=offset,
            generation_type=generation_type,
            model_name=model_filter if model_filter != "All Models" else None,
            start_date=start_date,
            end_date=end_date
        )
        
        if evt.index < len(records):
            record = records[evt.index]
            
            # Prepare info display
            info = {
                "id": record.id,
                "timestamp": datetime.fromtimestamp(record.timestamp).strftime("%Y-%m-%d %H:%M:%S"),
                "type": record.generation_type,
                "model": record.model_name,
                "prompt": record.prompt,
                "negative_prompt": record.negative_prompt,
                "parameters": record.parameters,
                "tags": record.tags,
                "favorite": record.favorite
            }
            
            # Determine preview type and visibility
            image_update = gr.update(visible=False)
            model_update = gr.update(visible=False)
            
            if record.output_paths and len(record.output_paths) > 0:
                output_path = Path(record.output_paths[0])
                
                # Copy to temp directory for Gradio access
                import tempfile
                import shutil
                temp_dir = Path(tempfile.gettempdir()) / "hunyuan3d_gallery_temp"
                temp_dir.mkdir(exist_ok=True)
                
                if output_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp']:
                    # Copy image to temp
                    temp_file = temp_dir / f"preview_{record.id}{output_path.suffix}"
                    if output_path.exists():
                        shutil.copy2(output_path, temp_file)
                        image_update = gr.update(visible=True, value=str(temp_file))
                elif output_path.suffix.lower() in ['.glb', '.obj', '.ply', '.stl']:
                    # Copy 3D model to temp
                    temp_file = temp_dir / f"model_{record.id}{output_path.suffix}"
                    if output_path.exists():
                        shutil.copy2(output_path, temp_file)
                        model_update = gr.update(visible=True, value=str(temp_file))
            
            return info, image_update, model_update, record.id
        
        return {}, gr.update(visible=False), gr.update(visible=False), None
    
    # Pagination handlers
    def next_page(current, total):
        """Go to next page"""
        new_page = min(current + 1, total - 1)
        return new_page
    
    def prev_page(current):
        """Go to previous page"""
        new_page = max(0, current - 1)
        return new_page
    
    # Action handlers
    def download_item(selection_id):
        """Download selected item"""
        if not selection_id:
            return gr.update(value="No item selected")
        
        record = app.history_manager.get_record(selection_id)
        if record and record.output_paths:
            # In a real implementation, this would trigger a download
            return gr.update(value=f"Download ready: {record.output_paths[0]}")
        
        return gr.update(value="No file to download")
    
    def toggle_favorite(selection_id):
        """Toggle favorite status"""
        if not selection_id:
            return gr.update(value="No item selected"), {}
        
        record = app.history_manager.get_record(selection_id)
        if record:
            app.history_manager.toggle_favorite(selection_id)
            # Reload info
            info = {
                "id": record.id,
                "timestamp": datetime.fromtimestamp(record.timestamp).strftime("%Y-%m-%d %H:%M:%S"),
                "type": record.generation_type,
                "model": record.model_name,
                "prompt": record.prompt,
                "negative_prompt": record.negative_prompt,
                "parameters": record.parameters,
                "tags": record.tags,
                "favorite": not record.favorite  # Toggle
            }
            return gr.update(value="Favorite toggled!"), info
        
        return gr.update(value="Failed to toggle favorite"), {}
    
    def delete_item(selection_id):
        """Delete selected item"""
        if not selection_id:
            return gr.update(value="No item selected")
        
        success = app.history_manager.delete_generation(selection_id)
        if success:
            return gr.update(value="Item deleted successfully")
        
        return gr.update(value="Failed to delete item")
    
    def copy_prompt(item_info):
        """Copy prompt to clipboard"""
        if item_info and 'prompt' in item_info:
            # In Gradio, we can't directly copy to clipboard,
            # but we can show the prompt for manual copying
            return gr.update(value=f"Prompt: {item_info['prompt']}")
        
        return gr.update(value="No prompt to copy")
    
    def import_existing_files():
        """Import existing files from outputs directory"""
        try:
            # Get outputs directory from app config
            outputs_dir = app.output_dir if hasattr(app, 'output_dir') else Path("outputs")
            
            # Scan and import files
            imported_count = app.history_manager.scan_outputs_directory(outputs_dir)
            
            if imported_count > 0:
                return gr.update(
                    value=f"✅ Successfully imported {imported_count} files!", 
                    visible=True
                ), load_gallery("All", "All Time", "All Models", False, 0)[0]
            else:
                return gr.update(
                    value="ℹ️ No new files found to import.", 
                    visible=True
                ), gr.update()
                
        except Exception as e:
            logger.error(f"Failed to import files: {e}")
            return gr.update(
                value=f"❌ Import failed: {str(e)}", 
                visible=True
            ), gr.update()
    
    # Wire up events
    
    # Import button
    import_btn.click(
        import_existing_files,
        outputs=[import_status, gallery]
    )
    
    # Refresh button with auto-load on first click
    def refresh_gallery(has_loaded_state, t, d, m, u, p):
        """Refresh gallery and handle first load"""
        gallery_items, page_text, total_p = load_gallery(t, d, m, u, p)
        
        # If this is the first load and no items found, try importing
        if not has_loaded_state and len(gallery_items) == 0:
            try:
                outputs_dir = app.output_dir if hasattr(app, 'output_dir') else Path("outputs")
                imported_count = app.history_manager.scan_outputs_directory(outputs_dir)
                if imported_count > 0:
                    # Re-load after import
                    gallery_items, page_text, total_p = load_gallery(t, d, m, u, p)
            except Exception as e:
                logger.error(f"Auto-import failed: {e}")
        
        return gallery_items, page_text, total_p, True
    
    refresh_btn.click(
        refresh_gallery,
        inputs=[has_loaded, type_filter, date_filter, model_filter, unviewed_only, current_page],
        outputs=[gallery, page_info, total_pages, has_loaded]
    )
    
    # Remove gallery.change to avoid circular updates
    
    # Filter changes
    for filter_component in [type_filter, date_filter, model_filter, unviewed_only]:
        filter_component.change(
            lambda t, d, m, u: (0, *load_gallery(t, d, m, u, 0)),
            inputs=[type_filter, date_filter, model_filter, unviewed_only],
            outputs=[current_page, gallery, page_info, total_pages]
        )
    
    # Gallery selection
    gallery.select(
        on_gallery_select,
        inputs=[type_filter, date_filter, model_filter, unviewed_only, current_page],
        outputs=[item_info, image_viewer, model_viewer, current_selection]
    )
    
    # Pagination
    next_btn.click(
        lambda c, t: (next_page(c, t), *load_gallery(
            type_filter.value, date_filter.value, model_filter.value, 
            unviewed_only.value, next_page(c, t)
        )[:2]),
        inputs=[current_page, total_pages],
        outputs=[current_page, gallery, page_info]
    )
    
    prev_btn.click(
        lambda c: (prev_page(c), *load_gallery(
            type_filter.value, date_filter.value, model_filter.value,
            unviewed_only.value, prev_page(c)
        )[:2]),
        inputs=[current_page],
        outputs=[current_page, gallery, page_info]
    )
    
    # Actions
    download_btn.click(download_item, inputs=[current_selection], outputs=[gr.Textbox(visible=False)])
    favorite_btn.click(toggle_favorite, inputs=[current_selection], outputs=[gr.Textbox(visible=False), item_info])
    delete_btn.click(delete_item, inputs=[current_selection], outputs=[gr.Textbox(visible=False)])
    copy_prompt_btn.click(copy_prompt, inputs=[item_info], outputs=[gr.Textbox(visible=False)])
    
    # Initial load happens through event handlers, not direct value setting
    # Auto-import will happen on first filter change or manual import
    
    # Add tab select event to auto-load gallery when tab is clicked
    def on_tab_select():
        """Load gallery when tab is selected"""
        return load_gallery("All", "All Time", "All Models", False, 0)
    
    # This will be connected when the tab is created in the parent component