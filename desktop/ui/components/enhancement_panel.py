"""
Prompt enhancement panel with field-based UI
"""

from typing import Dict, Any, Optional, Callable, List
from nicegui import ui
from core.models.enhancement import ModelType
from core.config import FLUX_ENHANCEMENT_FIELDS, HUNYUAN_ENHANCEMENT_FIELDS

class EnhancementPanel:
    """Panel for prompt enhancement fields"""
    
    def __init__(self, on_change: Optional[Callable[[Dict[str, Any]], None]] = None):
        self.on_change = on_change
        self.current_model_type = ModelType.FLUX_1_DEV
        self.field_values: Dict[str, Any] = {}
        self.field_elements: Dict[str, ui.element] = {}
        
    def render(self, container: Optional[ui.element] = None):
        """Render the enhancement panel"""
        with container or ui.column():
            with ui.card().classes('enhancement-panel'):
                # Header
                with ui.row().classes('items-center justify-between mb-4'):
                    ui.label('Prompt Enhancement').classes('text-lg font-semibold')
                    self.llm_toggle = ui.switch('Use LLM', value=True).classes('ml-auto')
                    
                # Fields container
                self.fields_container = ui.column().classes('gap-3')
                self._render_fields()
                
    def set_model_type(self, model_type: ModelType):
        """Update fields based on model type"""
        if model_type == self.current_model_type:
            return
            
        self.current_model_type = model_type
        self.field_values.clear()
        self._render_fields()
        
    def get_values(self) -> Dict[str, Any]:
        """Get current field values"""
        return {
            "use_llm": self.llm_toggle.value,
            "fields": self.field_values.copy()
        }
        
    def _render_fields(self):
        """Render fields based on current model type"""
        self.fields_container.clear()
        self.field_elements.clear()
        
        # Get fields for current model type
        if self.current_model_type in [ModelType.FLUX_1_DEV, ModelType.FLUX_1_SCHNELL]:
            fields = FLUX_ENHANCEMENT_FIELDS
        else:
            fields = HUNYUAN_ENHANCEMENT_FIELDS
            
        with self.fields_container:
            for field_id, field_config in fields.items():
                self._render_field(field_id, field_config)
                
    def _render_field(self, field_id: str, config: Dict[str, Any]):
        """Render a single enhancement field"""
        field_type = config.get('type', 'dropdown')
        
        with ui.column().classes('enhancement-field'):
            # Label
            ui.label(config['label']).classes('text-sm font-medium mb-2')
            
            if field_type == 'multi_checkbox':
                # Multi-checkbox field
                options = config.get('options', {})
                checkboxes = {}
                
                with ui.column().classes('gap-2'):
                    for option_id, option_label in options.items():
                        checkbox = ui.checkbox(option_id).classes('text-sm')
                        checkbox.on('change', lambda e, fid=field_id: self._handle_multi_checkbox_change(fid))
                        checkboxes[option_id] = checkbox
                        
                self.field_elements[field_id] = checkboxes
                self.field_values[field_id] = []
                
            else:
                # Dropdown field
                options = config.get('options', {})
                if options:
                    select = ui.select(
                        options=list(options.keys()),
                        label=None,
                        value=None
                    ).classes('w-full')
                    
                    select.on('change', lambda e, fid=field_id: self._handle_field_change(fid, e.value))
                    self.field_elements[field_id] = select
                    
    def _handle_field_change(self, field_id: str, value: Any):
        """Handle field value change"""
        self.field_values[field_id] = value
        if self.on_change:
            self.on_change(self.get_values())
            
    def _handle_multi_checkbox_change(self, field_id: str):
        """Handle multi-checkbox field change"""
        checkboxes = self.field_elements.get(field_id, {})
        selected = [cb_id for cb_id, cb in checkboxes.items() if cb.value]
        self.field_values[field_id] = selected
        if self.on_change:
            self.on_change(self.get_values())
            
    def reset(self):
        """Reset all field values"""
        self.field_values.clear()
        
        for field_id, element in self.field_elements.items():
            if isinstance(element, dict):  # Multi-checkbox
                for checkbox in element.values():
                    checkbox.value = False
            else:  # Dropdown
                element.value = None
                
        if self.on_change:
            self.on_change(self.get_values())