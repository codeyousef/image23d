"""
Library page - Browse and manage generated assets
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from nicegui import ui
import shutil

class LibraryPage:
    """Library page for browsing generated assets"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.current_filter = "all"
        self.selected_items: List[Path] = []
        self.items_per_page = 20
        self.current_page = 0
        
    def render(self):
        """Render the library page"""
        with ui.column().classes('w-full h-full'):
            # Header
            with ui.card().classes('w-full mb-4').style('background-color: #1F1F1F; border: 1px solid #333333'):
                with ui.row().classes('items-center justify-between'):
                    ui.label('Library').classes('text-2xl font-bold')
                    
                    # Stats
                    stats = self._get_library_stats()
                    with ui.row().classes('gap-4'):
                        ui.label(f"{stats['total']} items").classes('text-sm text-gray-400')
                        ui.label(f"{stats['size']:.1f} GB").classes('text-sm text-gray-400')
                        
            # Filters and actions
            with ui.card().classes('w-full mb-4').style('background-color: #1F1F1F; border: 1px solid #333333'):
                with ui.row().classes('items-center justify-between'):
                    # Filter tabs
                    with ui.row().classes('gap-2'):
                        self.filter_tabs = ui.toggle(
                            ['all', 'images', '3d', 'videos'],
                            value='all'
                        ).on('change', self._on_filter_change)
                        
                    # Search
                    self.search_input = ui.input(
                        placeholder='Search...',
                        on_change=self._refresh_library
                    ).props('dense outlined').classes('w-64')
                    
                    # Actions
                    with ui.row().classes('gap-2'):
                        ui.button(
                            'Export Selected',
                            icon='download',
                            on_click=self._export_selected
                        ).props('flat').bind_enabled_from(self, 'selected_items')
                        
                        ui.button(
                            'Delete Selected',
                            icon='delete',
                            on_click=self._delete_selected
                        ).props('flat color=negative').bind_enabled_from(self, 'selected_items')
                        
                        ui.button(
                            icon='refresh',
                            on_click=self._refresh_library
                        ).props('flat round')
                        
            # Main content area
            with ui.card().classes('w-full flex-grow').style('background-color: #1F1F1F; border: 1px solid #333333'):
                # Grid/List view toggle
                with ui.row().classes('items-center justify-between mb-4'):
                    ui.label('Generated Assets').classes('text-lg font-semibold')
                    
                    self.view_toggle = ui.toggle(
                        ['grid', 'list'],
                        value='grid'
                    ).props('dense').on('change', self._refresh_library)
                    
                # Content container
                self.content_container = ui.column().classes('w-full')
                
                # Initial load
                self._refresh_library()
                
    def _get_library_stats(self) -> Dict[str, Any]:
        """Get library statistics"""
        total_items = 0
        total_size = 0
        
        # Count files in output directories
        for subdir in ['images', '3d', 'videos']:
            dir_path = self.output_dir / subdir
            if dir_path.exists():
                for file in dir_path.rglob('*'):
                    if file.is_file():
                        total_items += 1
                        total_size += file.stat().st_size
                        
        return {
            'total': total_items,
            'size': total_size / (1024**3)  # Convert to GB
        }
        
    def _on_filter_change(self, e):
        """Handle filter change"""
        self.current_filter = e.value
        self.current_page = 0
        self._refresh_library()
        
    def _refresh_library(self):
        """Refresh the library view"""
        self.content_container.clear()
        self.selected_items.clear()
        
        # Get filtered items
        items = self._get_filtered_items()
        
        if not items:
            with self.content_container:
                with ui.column().classes('w-full h-64 items-center justify-center'):
                    ui.icon('folder_open', size='4rem').classes('text-gray-600 mb-4')
                    ui.label('No items found').classes('text-gray-500')
            return
            
        # Render based on view type
        with self.content_container:
            if self.view_toggle.value == 'grid':
                self._render_grid_view(items)
            else:
                self._render_list_view(items)
                
            # Pagination
            if len(items) > self.items_per_page:
                self._render_pagination(items)
                
    def _get_filtered_items(self) -> List[Dict[str, Any]]:
        """Get filtered library items"""
        items = []
        
        # Determine which directories to search
        if self.current_filter == 'all':
            search_dirs = ['images', '3d', 'videos']
        elif self.current_filter == 'images':
            search_dirs = ['images']
        elif self.current_filter == '3d':
            search_dirs = ['3d']
        elif self.current_filter == 'videos':
            search_dirs = ['videos']
        else:
            search_dirs = []
            
        # Search for files
        search_term = self.search_input.value.lower() if hasattr(self, 'search_input') else ""
        
        for subdir in search_dirs:
            dir_path = self.output_dir / subdir
            if not dir_path.exists():
                continue
                
            # Look for actual generated files
            for file_path in dir_path.rglob('*'):
                if not file_path.is_file():
                    continue
                    
                # Skip temp files and non-media files
                if file_path.name.startswith('.') or file_path.suffix.lower() not in ['.png', '.jpg', '.jpeg', '.glb', '.obj', '.ply', '.stl', '.mp4', '.gif']:
                    continue
                    
                # Apply search filter
                if search_term and search_term not in file_path.name.lower():
                    continue
                    
                # Get file info
                stat = file_path.stat()
                items.append({
                    'path': file_path,
                    'name': file_path.name,
                    'type': subdir,
                    'size': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime),
                    'extension': file_path.suffix.lower()
                })
                
        # Sort by modified date (newest first)
        items.sort(key=lambda x: x['modified'], reverse=True)
        
        return items
        
    def _render_grid_view(self, items: List[Dict[str, Any]]):
        """Render items in grid view"""
        # Paginate items
        start_idx = self.current_page * self.items_per_page
        end_idx = start_idx + self.items_per_page
        page_items = items[start_idx:end_idx]
        
        with ui.grid(columns='repeat(auto-fill, minmax(200px, 1fr))').classes('gap-4 w-full'):
            for item in page_items:
                self._render_grid_item(item)
                
    def _render_grid_item(self, item: Dict[str, Any]):
        """Render a single grid item"""
        with ui.card().classes('cursor-pointer hover:border-purple-500 transition-all').style(
            'background-color: #0A0A0A; border: 1px solid #333333; overflow: hidden'
        ) as card:
            # Checkbox for selection
            with ui.row().classes('absolute top-2 left-2 z-10'):
                checkbox = ui.checkbox().props('dense dark')
                checkbox.on('change', lambda e, path=item['path']: self._toggle_selection(path, e.value))
                
            # Preview area
            with ui.column().classes('w-full h-48 items-center justify-center bg-gray-900'):
                if item['type'] == 'images':
                    # Show actual image
                    ui.image(str(item['path'])).classes('max-w-full max-h-full object-contain')
                elif item['type'] == '3d':
                    # Show 3D icon for now (could show preview render)
                    ui.icon('view_in_ar', size='4rem').classes('text-purple-500')
                    ui.label(item['extension'].upper()).classes('text-sm text-gray-400 mt-2')
                else:
                    # Video icon
                    ui.icon('movie', size='4rem').classes('text-blue-500')
                    
            # Info section
            with ui.column().classes('p-3'):
                ui.label(item['name']).classes('text-sm font-medium truncate')
                with ui.row().classes('gap-2 mt-1'):
                    ui.label(f"{item['size'] / 1024:.1f} KB").classes('text-xs text-gray-400')
                    ui.label(item['modified'].strftime('%m/%d %H:%M')).classes('text-xs text-gray-400')
                    
            # Click handler
            card.on('click', lambda i=item: self._show_item_details(i))
            
    def _render_list_view(self, items: List[Dict[str, Any]]):
        """Render items in list view"""
        # Paginate items
        start_idx = self.current_page * self.items_per_page
        end_idx = start_idx + self.items_per_page
        page_items = items[start_idx:end_idx]
        
        with ui.column().classes('w-full gap-2'):
            # Header
            with ui.row().classes('w-full px-4 py-2 text-sm text-gray-400'):
                ui.label('Name').classes('flex-grow')
                ui.label('Type').classes('w-20')
                ui.label('Size').classes('w-24')
                ui.label('Modified').classes('w-32')
                
            # Items
            for item in page_items:
                self._render_list_item(item)
                
    def _render_list_item(self, item: Dict[str, Any]):
        """Render a single list item"""
        with ui.row().classes('w-full px-4 py-2 items-center hover:bg-gray-800 cursor-pointer rounded').style(
            'background-color: #0A0A0A'
        ) as row:
            # Checkbox
            checkbox = ui.checkbox().props('dense')
            checkbox.on('change', lambda e, path=item['path']: self._toggle_selection(path, e.value))
            
            # Icon based on type
            if item['type'] == 'images':
                ui.icon('image', size='sm').classes('text-green-500')
            elif item['type'] == '3d':
                ui.icon('view_in_ar', size='sm').classes('text-purple-500')
            else:
                ui.icon('movie', size='sm').classes('text-blue-500')
                
            # Name
            ui.label(item['name']).classes('flex-grow text-sm truncate px-2')
            
            # Type
            ui.label(item['extension'][1:].upper()).classes('w-20 text-sm text-gray-400')
            
            # Size
            size_str = f"{item['size'] / 1024:.1f} KB" if item['size'] < 1024*1024 else f"{item['size'] / 1024 / 1024:.1f} MB"
            ui.label(size_str).classes('w-24 text-sm text-gray-400')
            
            # Modified
            ui.label(item['modified'].strftime('%Y-%m-%d %H:%M')).classes('w-32 text-sm text-gray-400')
            
            # Click handler
            row.on('click', lambda i=item: self._show_item_details(i))
            
    def _render_pagination(self, items: List[Dict[str, Any]]):
        """Render pagination controls"""
        total_pages = (len(items) + self.items_per_page - 1) // self.items_per_page
        
        with ui.row().classes('w-full justify-center mt-4 gap-2'):
            # Previous button
            ui.button(
                icon='chevron_left',
                on_click=lambda: self._change_page(-1)
            ).props('flat round').bind_enabled_from(self, 'current_page', lambda x: x > 0)
            
            # Page numbers
            for i in range(max(0, self.current_page - 2), min(total_pages, self.current_page + 3)):
                ui.button(
                    str(i + 1),
                    on_click=lambda p=i: self._go_to_page(p)
                ).props('flat' if i != self.current_page else 'unelevated').classes(
                    '' if i != self.current_page else 'bg-purple-600'
                )
                
            # Next button
            ui.button(
                icon='chevron_right',
                on_click=lambda: self._change_page(1)
            ).props('flat round').bind_enabled_from(
                self, 'current_page', 
                lambda x, t=total_pages: x < t - 1
            )
            
    def _toggle_selection(self, path: Path, selected: bool):
        """Toggle item selection"""
        if selected and path not in self.selected_items:
            self.selected_items.append(path)
        elif not selected and path in self.selected_items:
            self.selected_items.remove(path)
            
    def _change_page(self, delta: int):
        """Change page by delta"""
        self.current_page += delta
        self._refresh_library()
        
    def _go_to_page(self, page: int):
        """Go to specific page"""
        self.current_page = page
        self._refresh_library()
        
    def _show_item_details(self, item: Dict[str, Any]):
        """Show detailed view of an item"""
        with ui.dialog() as dialog, ui.card().classes('w-[800px] max-w-[90vw]'):
            with ui.row().classes('items-center justify-between mb-4'):
                ui.label(item['name']).classes('text-xl font-semibold')
                ui.button(icon='close', on_click=dialog.close).props('flat round')
                
            # Preview
            with ui.column().classes('w-full items-center mb-4'):
                if item['type'] == 'images':
                    ui.image(str(item['path'])).classes('max-w-full').style('max-height: 500px')
                elif item['type'] == '3d':
                    # For 3D, show info and download button
                    ui.icon('view_in_ar', size='8rem').classes('text-purple-500 mb-4')
                    ui.label(f'3D Model ({item["extension"][1:].upper()})').classes('text-lg')
                    
            # Metadata
            with ui.column().classes('w-full gap-2'):
                ui.label('File Information').classes('font-semibold mb-2')
                with ui.row().classes('gap-4'):
                    ui.label(f'Size: {item["size"] / 1024 / 1024:.2f} MB').classes('text-sm')
                    ui.label(f'Modified: {item["modified"].strftime("%Y-%m-%d %H:%M:%S")}').classes('text-sm')
                    ui.label(f'Path: {item["path"].relative_to(self.output_dir)}').classes('text-sm')
                    
            # Actions
            with ui.row().classes('gap-2 mt-4'):
                ui.button('Export', icon='download', on_click=lambda: self._export_item(item['path'])).props('unelevated')
                ui.button('Delete', icon='delete', on_click=lambda: self._delete_item(item['path'], dialog)).props('flat color=negative')
                
        dialog.open()
        
    def _export_selected(self):
        """Export selected items"""
        if not self.selected_items:
            return
            
        # In a real app, this would open a file dialog
        ui.notify(f'Exporting {len(self.selected_items)} items...', type='info')
        
    def _delete_selected(self):
        """Delete selected items with confirmation"""
        if not self.selected_items:
            return
            
        with ui.dialog() as dialog, ui.card():
            ui.label(f'Delete {len(self.selected_items)} items?').classes('text-lg mb-4')
            ui.label('This action cannot be undone.').classes('text-sm text-gray-400')
            
            with ui.row().classes('gap-2 mt-4'):
                ui.button('Cancel', on_click=dialog.close).props('flat')
                ui.button(
                    'Delete',
                    on_click=lambda: self._confirm_delete_selected(dialog)
                ).props('unelevated color=negative')
                
        dialog.open()
        
    def _confirm_delete_selected(self, dialog):
        """Confirm deletion of selected items"""
        for path in self.selected_items:
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
            except Exception as e:
                ui.notify(f'Failed to delete {path.name}: {e}', type='negative')
                
        ui.notify(f'Deleted {len(self.selected_items)} items', type='positive')
        self.selected_items.clear()
        dialog.close()
        self._refresh_library()
        
    def _export_item(self, path: Path):
        """Export a single item"""
        ui.notify(f'Exported: {path.name}', type='positive')
        
    def _delete_item(self, path: Path, dialog):
        """Delete a single item"""
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            ui.notify(f'Deleted: {path.name}', type='positive')
            dialog.close()
            self._refresh_library()
        except Exception as e:
            ui.notify(f'Failed to delete: {e}', type='negative')