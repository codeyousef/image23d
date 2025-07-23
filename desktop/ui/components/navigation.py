"""
Navigation sidebar component
"""

from typing import Callable, Optional, Dict
from nicegui import ui
from dataclasses import dataclass
from ...localization import get_translator

@dataclass
class NavItem:
    """Navigation item configuration"""
    id: str
    label: str
    icon: str
    description: str

NAV_ITEMS = [
    NavItem("create", "Create", "add_circle", "Generate images, 3D models, and videos"),
    NavItem("library", "Library", "collections", "Browse and manage your creations"),
    NavItem("lab", "Lab", "science", "Advanced tools and experiments"),
    NavItem("models", "Models", "inventory_2", "Download and manage AI models"),
]

class NavigationSidebar:
    """Left navigation sidebar"""
    
    def __init__(self, on_navigate: Callable[[str], None]):
        self.on_navigate = on_navigate
        self.current_page = "create"
        self.nav_items: Dict[str, ui.element] = {}
        self.translator = get_translator()
        
    def render(self):
        """Render the navigation sidebar"""
        t = self.translator.t
        rtl_classes = 'dir-rtl' if self.translator.is_rtl() else ''
        font_style = f'font-family: {self.translator.get_font_family()}'
        
        with ui.column().classes(f'navigation-sidebar w-64 h-screen {rtl_classes}').style(font_style):
            # Logo and title
            with ui.row().classes('items-center gap-3 mb-8'):
                ui.icon('auto_awesome', size='2rem').style('color: #7C3AED')
                with ui.column().classes('gap-0'):
                    ui.label(t('app.title')).classes('text-xl font-bold')
                    ui.label('Studio').classes('text-sm text-gray-500')
            
            # Navigation items
            for item in NAV_ITEMS:
                self.nav_items[item.id] = self._create_nav_item(item)
                
            # Spacer
            ui.space().classes('flex-grow')
            
            # Settings at bottom
            with ui.row().classes('nav-item').on('click', lambda: self.on_navigate('settings')):
                ui.icon('settings', size='1.25rem')
                ui.label(t('navigation.settings'))
                
    def _create_nav_item(self, item: NavItem) -> ui.element:
        """Create a navigation item"""
        t = self.translator.t
        
        with ui.row().classes('nav-item').on('click', lambda i=item: self._handle_navigation(i.id)) as nav_element:
            ui.icon(item.icon, size='1.25rem')
            with ui.column().classes('gap-0'):
                # Use translation keys for navigation items
                label_key = f'navigation.{item.id}'
                ui.label(t(label_key)).classes('text-sm font-medium')
                # For now, keep descriptions in English
                ui.label(item.description).classes('text-xs opacity-60')
                
        # Set initial active state
        if item.id == self.current_page:
            nav_element.classes('active', remove='')
            
        return nav_element
        
    def _handle_navigation(self, page_id: str):
        """Handle navigation click"""
        if page_id == self.current_page:
            return
            
        # Update active states
        if self.current_page in self.nav_items:
            self.nav_items[self.current_page].classes(remove='active')
        if page_id in self.nav_items:
            self.nav_items[page_id].classes('active')
            
        self.current_page = page_id
        self.on_navigate(page_id)
        
    def set_active_page(self, page_id: str):
        """Set the active page programmatically"""
        self._handle_navigation(page_id)