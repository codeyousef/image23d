"""Gradio UI component for history viewing"""

import logging
from pathlib import Path
from typing import Optional, Callable, Any

import gradio as gr

logger = logging.getLogger(__name__)


class HistoryUIComponent:
    """Creates Gradio UI component for history viewing"""
    
    def __init__(self, history_manager):
        self.history_manager = history_manager
    
    def create_ui_component(self) -> gr.Group:
        """Create the history viewer UI component"""
        with gr.Group() as history_group:
            gr.Markdown("## 📚 Generation History")
            
            # Filters and search
            with gr.Row():
                search_box = gr.Textbox(
                    label="Search",
                    placeholder="Search prompts, tags...",
                    scale=3
                )
                type_filter = gr.Dropdown(
                    label="Type",
                    choices=["All", "image", "3d", "full_pipeline"],
                    value="All",
                    scale=1
                )
                favorites_only = gr.Checkbox(
                    label="Favorites Only",
                    value=False
                )
                
            # Gallery and details
            with gr.Row():
                with gr.Column(scale=2):
                    gallery = gr.Gallery(
                        label="History",
                        show_label=False,
                        elem_id="history_gallery",
                        columns=4,
                        rows=3,
                        height=600,
                        object_fit="contain"
                    )
                    
                with gr.Column(scale=1):
                    # Selected item details
                    selected_info = gr.JSON(
                        label="Generation Details",
                        value={}
                    )
                    
                with gr.Column(scale=1):
                    # Actions
                    favorite_btn = gr.Button("⭐ Toggle Favorite", size="sm")
                    regenerate_btn = gr.Button("🔄 Regenerate", size="sm")
                    delete_btn = gr.Button("🗑️ Delete", size="sm", variant="stop")
                    
            # Pagination
            with gr.Row():
                prev_btn = gr.Button("◀ Previous", size="sm")
                page_info = gr.Textbox(
                    value="Page 1",
                    interactive=False,
                    show_label=False
                )
                next_btn = gr.Button("Next ▶", size="sm")
                
            # Hidden state
            current_page = gr.State(0)
            selected_id = gr.State(None)
            
            # Wire up events
            self._setup_events(
                search_box, type_filter, favorites_only,
                gallery, selected_info, selected_id,
                favorite_btn, regenerate_btn, delete_btn,
                prev_btn, page_info, next_btn,
                current_page
            )
            
        return history_group
    
    def _setup_events(
        self,
        search_box, type_filter, favorites_only,
        gallery, selected_info, selected_id,
        favorite_btn, regenerate_btn, delete_btn,
        prev_btn, page_info, next_btn,
        current_page
    ):
        """Set up event handlers for UI components"""
        
        def load_gallery(search, type_filter, favorites, page):
            """Load gallery images"""
            limit = 12
            offset = page * limit
            
            records = self.history_manager.get_history(
                limit=limit,
                offset=offset,
                generation_type=type_filter if type_filter != "All" else None,
                search_query=search if search else None,
                favorites_only=favorites
            )
            
            images = []
            for record in records:
                if record.thumbnails:
                    # Use first thumbnail
                    thumb_path = record.thumbnails[0]
                    if Path(thumb_path).exists():
                        images.append((thumb_path, record.prompt[:50] + "..."))
                        
            return images, f"Page {page + 1}"
        
        def select_item(evt: gr.SelectData, search, type_filter, favorites, page):
            """Handle gallery selection"""
            limit = 12
            offset = page * limit
            
            records = self.history_manager.get_history(
                limit=limit,
                offset=offset,
                generation_type=type_filter if type_filter != "All" else None,
                search_query=search if search else None,
                favorites_only=favorites
            )
            
            if evt.index < len(records):
                record = records[evt.index]
                # Mark as viewed
                self.history_manager.mark_as_viewed(record.id)
                return record.to_dict(), record.id
                
            return {}, None
        
        def toggle_favorite(selected_id):
            """Toggle favorite status"""
            if selected_id:
                record = self.history_manager.get_record(selected_id)
                if record:
                    self.history_manager.update_record(
                        selected_id, 
                        favorite=not record.favorite
                    )
        
        def delete_item(selected_id):
            """Delete selected item"""
            if selected_id:
                self.history_manager.delete_record(selected_id, delete_files=False)
        
        # Search and filter changes
        for component in [search_box, type_filter, favorites_only]:
            component.change(
                lambda s, t, f: load_gallery(s, t, f, 0),
                inputs=[search_box, type_filter, favorites_only],
                outputs=[gallery, page_info]
            ).then(
                lambda: 0,
                outputs=[current_page]
            )
        
        # Gallery selection
        gallery.select(
            select_item,
            inputs=[search_box, type_filter, favorites_only, current_page],
            outputs=[selected_info, selected_id]
        )
        
        # Action buttons
        favorite_btn.click(
            toggle_favorite,
            inputs=[selected_id]
        ).then(
            lambda s, t, f, p: load_gallery(s, t, f, p),
            inputs=[search_box, type_filter, favorites_only, current_page],
            outputs=[gallery, page_info]
        )
        
        delete_btn.click(
            delete_item,
            inputs=[selected_id]
        ).then(
            lambda s, t, f, p: load_gallery(s, t, f, p),
            inputs=[search_box, type_filter, favorites_only, current_page],
            outputs=[gallery, page_info]
        )
        
        # Pagination
        prev_btn.click(
            lambda p: max(0, p - 1),
            inputs=[current_page],
            outputs=[current_page]
        ).then(
            lambda s, t, f, p: load_gallery(s, t, f, p),
            inputs=[search_box, type_filter, favorites_only, current_page],
            outputs=[gallery, page_info]
        )
        
        next_btn.click(
            lambda p: p + 1,
            inputs=[current_page],
            outputs=[current_page]
        ).then(
            lambda s, t, f, p: load_gallery(s, t, f, p),
            inputs=[search_box, type_filter, favorites_only, current_page],
            outputs=[gallery, page_info]
        )