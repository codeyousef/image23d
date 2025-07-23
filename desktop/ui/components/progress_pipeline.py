"""
Visual pipeline progress component
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
from nicegui import ui
import asyncio

class StepStatus(Enum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class PipelineStep:
    """Individual pipeline step"""
    id: str
    label: str
    icon: str
    duration_estimate: Optional[int] = None  # seconds

class ProgressPipeline:
    """Visual representation of generation pipeline progress"""
    
    def __init__(self):
        self.steps: List[PipelineStep] = []
        self.current_step_index = -1
        self.step_elements: Dict[str, ui.element] = {}
        self.progress_bar: Optional[ui.linear_progress] = None
        self.status_label: Optional[ui.label] = None
        self.time_label: Optional[ui.label] = None
        self.start_time: Optional[float] = None
        
    def render(self, container: Optional[ui.element] = None):
        """Render the progress pipeline"""
        # Store the container reference for updates from background tasks
        self.container = container or ui.column()
        
        with self.container:
            with ui.card().classes('progress-pipeline').style('padding: 12px') as self.card:
                # Header - more compact
                with ui.row().classes('items-center justify-between mb-2'):
                    ui.label('Generation Pipeline').classes('text-base font-semibold')
                    self.time_label = ui.label('').classes('text-xs text-gray-500')
                    
                # Progress bar - more visible
                self.progress_bar = ui.linear_progress(value=0).classes('mb-3 w-full').style('height: 10px; width: 100%;')
                
                # Status - smaller text
                self.status_label = ui.label('Ready to generate').classes('text-xs mb-2')
                
                # Steps container - compact
                self.steps_container = ui.column().classes('gap-1')
                
    def set_steps(self, steps: List[PipelineStep]):
        """Set pipeline steps"""
        self.steps = steps
        self.current_step_index = -1
        self._render_steps()
        
    def _render_steps(self):
        """Render all pipeline steps"""
        self.steps_container.clear()
        self.step_elements.clear()
        
        with self.steps_container:
            for i, step in enumerate(self.steps):
                self.step_elements[step.id] = self._render_step(step, i)
                
    def _render_step(self, step: PipelineStep, index: int) -> ui.element:
        """Render a single pipeline step"""
        with ui.row().classes('progress-step items-center').style('padding: 4px 0') as step_element:
            # Step number or icon - smaller
            with ui.element('div').classes('w-6 h-6 rounded-full bg-gray-700 flex items-center justify-center'):
                if not hasattr(self, 'step_icons'):
                    self.step_icons = {}
                self.step_icons[step.id] = ui.icon(step.icon, size='0.75rem').classes('text-gray-400')
                
            # Step info
            with ui.column().classes('flex-grow gap-0 ml-3'):
                ui.label(step.label).classes('text-sm font-medium')
                if step.duration_estimate:
                    ui.label(f'~{step.duration_estimate}s').classes('text-xs text-gray-500')
                    
            # Status indicator
            if not hasattr(self, 'step_status_icons'):
                self.step_status_icons = {}
            self.step_status_icons[step.id] = ui.icon('', size='1.25rem').classes('hidden ml-2')
            
        return step_element
        
    def start(self):
        """Start the pipeline progress"""
        self.current_step_index = -1
        self.start_time = asyncio.get_event_loop().time()
        self.progress_bar.value = 0
        self.status_label.text = 'Starting generation...'
        
        # Reset all steps
        for step_id, element in self.step_elements.items():
            element.classes(remove='active completed')
            self.step_status_icons[step_id].classes('hidden')
            
    def advance_step(self, message: Optional[str] = None):
        """Advance to the next step"""
        # Mark current step as completed
        if self.current_step_index >= 0:
            current_step = self.steps[self.current_step_index]
            self.step_elements[current_step.id].classes('completed', remove='active')
            self.step_status_icons[current_step.id].props('name=check_circle').classes(remove='hidden').style('color: #10B981')
        
        # Move to next step
        self.current_step_index += 1
        if self.current_step_index < len(self.steps):
            next_step = self.steps[self.current_step_index]
            self.step_elements[next_step.id].classes('active')
            self.status_label.text = message or f'{next_step.label}...'
            
            # Update progress
            progress = (self.current_step_index + 1) / len(self.steps)
            self.progress_bar.value = progress
        
        # Update time
        if self.start_time:
            elapsed = int(asyncio.get_event_loop().time() - self.start_time)
            self.time_label.text = f'{elapsed}s'
            
    def complete(self, message: str = 'Generation complete!'):
        """Mark pipeline as complete"""
        # Mark last step as completed
        if self.current_step_index >= 0 and self.current_step_index < len(self.steps):
            current_step = self.steps[self.current_step_index]
            self.step_elements[current_step.id].classes('completed', remove='active')
            self.step_status_icons[current_step.id].props('name=check_circle').classes(remove='hidden').style('color: #10B981')
            
        self.progress_bar.value = 1.0
        self.status_label.text = message
        self.status_label.style('color: #10B981')
        
    def fail(self, message: str = 'Generation failed'):
        """Mark pipeline as failed"""
        if self.current_step_index >= 0 and self.current_step_index < len(self.steps):
            current_step = self.steps[self.current_step_index]
            self.step_elements[current_step.id].classes(remove='active').style('border-color: #EF4444')
            self.step_status_icons[current_step.id].props('name=error').classes(remove='hidden').style('color: #EF4444')
            
        self.status_label.text = message
        self.status_label.style('color: #EF4444')
        
    def reset(self):
        """Reset the pipeline"""
        self.current_step_index = -1
        self.start_time = None
        self.progress_bar.value = 0
        self.status_label.text = 'Ready to generate'
        self.status_label.style('')
        self.time_label.text = ''
        
        # Reset all steps
        for step_id, element in self.step_elements.items():
            element.classes(remove='active completed')
            element.style('')
            self.step_status_icons[step_id].classes('hidden')