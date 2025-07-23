"""
Visual pipeline progress component with full-width fixes
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
    """Visual representation of generation pipeline progress with full-width components"""
    
    def __init__(self):
        self.steps: List[PipelineStep] = []
        self.current_step_index = -1
        self.step_elements: Dict[str, ui.element] = {}
        self.progress_bar: Optional[ui.linear_progress] = None
        self.status_label: Optional[ui.label] = None
        self.time_label: Optional[ui.label] = None
        self.progress_label: Optional[ui.label] = None
        self.start_time: Optional[float] = None
        
    def render(self, container: Optional[ui.element] = None):
        """Render the progress pipeline with full-width components"""
        # Store the container reference
        self.container = container or ui.column()
        
        with self.container:
            # Main card with full width - remove padding from card and force full width
            with ui.card().classes('w-full').style('padding: 0; width: 100% !important; display: block !important;') as self.card:
                # Inner container with padding
                with ui.column().classes('w-full p-4').style('width: 100% !important;'):
                    # Header - full width
                    with ui.row().classes('items-center justify-between mb-3 w-full').style('width: 100% !important;'):
                        ui.label('Generation Pipeline').classes('text-lg font-semibold')
                        self.time_label = ui.label('').classes('text-sm text-gray-500')
                        
                    # Progress bar - thick and full width with percentage
                    with ui.column().classes('w-full mb-3').style('width: 100% !important;'):
                        self.progress_bar = ui.linear_progress(value=0, show_value=False).classes('w-full')
                        # Force full width and thick height
                        self.progress_bar.style('height: 20px !important; width: 100% !important; border-radius: 10px;')
                        # Add percentage label
                        self.progress_label = ui.label('0%').classes('text-sm text-gray-400 mt-1')
                    
                    # Status
                    self.status_label = ui.label('Ready to generate').classes('text-base mb-3 text-gray-300')
                    
                    # Steps container - full width
                    self.steps_container = ui.column().classes('gap-2 w-full')
                    self.steps_container.style('width: 100% !important;')
                
    def set_steps(self, steps: List[PipelineStep]):
        """Set pipeline steps"""
        self.steps = steps
        self.current_step_index = -1
        self._render_steps()
        
    def _render_steps(self):
        """Render all pipeline steps with full width"""
        self.steps_container.clear()
        self.step_elements.clear()
        
        with self.steps_container:
            for i, step in enumerate(self.steps):
                self.step_elements[step.id] = self._render_step(step, i)
                
    def _render_step(self, step: PipelineStep, index: int) -> ui.element:
        """Render a single pipeline step with full width"""
        with ui.row().classes('w-full items-center').style('padding: 8px 0; width: 100% !important;') as step_element:
            # Step icon container
            with ui.element('div').classes('w-10 h-10 rounded-full bg-gray-700 flex items-center justify-center flex-shrink-0'):
                if not hasattr(self, 'step_icons'):
                    self.step_icons = {}
                self.step_icons[step.id] = ui.icon(step.icon).classes('text-gray-400')
                self.step_icons[step.id].style('font-size: 1.25rem;')
                
            # Step info - takes remaining width
            with ui.column().classes('flex-grow ml-4'):
                ui.label(step.label).classes('text-base font-medium text-gray-200')
                if step.duration_estimate:
                    ui.label(f'~{step.duration_estimate}s').classes('text-sm text-gray-500')
                    
            # Status indicator
            if not hasattr(self, 'step_status_icons'):
                self.step_status_icons = {}
            self.step_status_icons[step.id] = ui.icon('').classes('hidden ml-auto')
            self.step_status_icons[step.id].style('font-size: 1.5rem;')
            
        return step_element
        
    def start(self):
        """Start the pipeline progress"""
        self.current_step_index = -1
        self.start_time = asyncio.get_event_loop().time()
        self.progress_bar.value = 0
        if hasattr(self, 'progress_label'):
            self.progress_label.text = '0%'
        self.status_label.text = 'Starting generation...'
        
        # Reset all steps
        for step_id, element in self.step_elements.items():
            element.classes(remove='active completed')
            element.style('background-color: transparent;')
            self.step_status_icons[step_id].classes('hidden')
            self.step_icons[step_id].classes('text-gray-400')
            
    def advance_step(self, message: Optional[str] = None):
        """Advance to the next step"""
        # Mark current step as completed
        if self.current_step_index >= 0 and self.current_step_index < len(self.steps):
            current_step = self.steps[self.current_step_index]
            self.step_elements[current_step.id].style('background-color: rgba(16, 185, 129, 0.1);')
            self.step_icons[current_step.id].classes('text-green-500', remove='text-gray-400')
            self.step_status_icons[current_step.id].props('name=check_circle')
            self.step_status_icons[current_step.id].classes('text-green-500', remove='hidden')
        
        # Move to next step
        self.current_step_index += 1
        if self.current_step_index < len(self.steps):
            next_step = self.steps[self.current_step_index]
            self.step_elements[next_step.id].style('background-color: rgba(59, 130, 246, 0.1);')
            self.step_icons[next_step.id].classes('text-blue-500', remove='text-gray-400')
            self.status_label.text = message or f'{next_step.label}...'
            
            # Update progress with percentage
            progress = (self.current_step_index + 1) / len(self.steps)
            self.progress_bar.value = progress
            self.progress_label.text = f'{int(progress * 100)}%'
        
        # Update time
        if self.start_time:
            elapsed = int(asyncio.get_event_loop().time() - self.start_time)
            self.time_label.text = f'{elapsed}s'
            
    def complete(self, message: str = 'Generation complete!'):
        """Mark pipeline as complete"""
        # Mark last step as completed
        if self.current_step_index >= 0 and self.current_step_index < len(self.steps):
            current_step = self.steps[self.current_step_index]
            self.step_elements[current_step.id].style('background-color: rgba(16, 185, 129, 0.1);')
            self.step_icons[current_step.id].classes('text-green-500', remove='text-gray-400 text-blue-500')
            self.step_status_icons[current_step.id].props('name=check_circle')
            self.step_status_icons[current_step.id].classes('text-green-500', remove='hidden')
            
        self.progress_bar.value = 1.0
        self.progress_label.text = '100%'
        self.status_label.text = message
        self.status_label.classes('text-green-500')
        
    def fail(self, message: str = 'Generation failed'):
        """Mark pipeline as failed"""
        if self.current_step_index >= 0 and self.current_step_index < len(self.steps):
            current_step = self.steps[self.current_step_index]
            self.step_elements[current_step.id].style('background-color: rgba(239, 68, 68, 0.1);')
            self.step_icons[current_step.id].classes('text-red-500', remove='text-gray-400 text-blue-500')
            self.step_status_icons[current_step.id].props('name=error')
            self.step_status_icons[current_step.id].classes('text-red-500', remove='hidden')
            
        self.status_label.text = message
        self.status_label.classes('text-red-500')
        
    def reset(self):
        """Reset the pipeline"""
        self.current_step_index = -1
        self.start_time = None
        self.progress_bar.value = 0
        self.progress_label.text = '0%'
        self.status_label.text = 'Ready to generate'
        self.status_label.classes(remove='text-green-500 text-red-500')
        self.time_label.text = ''
        
        # Reset all steps
        for step_id, element in self.step_elements.items():
            element.style('background-color: transparent;')
            self.step_icons[step_id].classes('text-gray-400', remove='text-green-500 text-blue-500 text-red-500')
            self.step_status_icons[step_id].classes('hidden')