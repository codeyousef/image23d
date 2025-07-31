"""
Progress pipeline that renders directly into a card container
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from nicegui import ui
import asyncio

@dataclass
class PipelineStep:
    id: str
    label: str
    icon: str
    duration_estimate: Optional[int] = None
    weight: float = 1.0  # Weight for overall progress calculation

class ProgressPipeline:
    def __init__(self):
        self.steps: List[PipelineStep] = []
        self.current_step_index = -1
        self.current_step_progress = 0.0  # Progress within current step (0.0 to 1.0)
        self.step_elements: Dict[str, Any] = {}
        self.progress_bar = None
        self.status_label = None
        self.time_label = None
        self.progress_label = None
        self.start_time = None
        self.current_sub_message = ""  # For sub-step messages like "Diffusion Sampling"
        
    def render(self, container: Optional[ui.element] = None):
        """Render directly into container (assumes it's inside a card)"""
        self.container = container or ui.column()
        
        with self.container:
            # Header
            with ui.row().classes('items-center justify-between mb-4 w-full'):
                ui.label('Generation Pipeline').classes('text-lg font-semibold')
                self.time_label = ui.label('').classes('text-sm text-gray-500')
            
            # Progress bar section
            ui.label('Progress').classes('text-sm font-medium mb-2 text-gray-400')
            
            # Progress bar
            self.progress_bar = ui.linear_progress(value=0, show_value=False).classes('w-full mb-2')
            self.progress_bar.style('height: 20px; border-radius: 10px;')
            
            # Progress percentage
            self.progress_label = ui.label('0%').classes('text-sm text-gray-400 mb-4')
            
            # Status
            self.status_label = ui.label('Ready to generate').classes('text-base mb-4')
            
            # Steps section
            ui.label('Steps').classes('text-sm font-medium mb-3 text-gray-400')
            
            # Steps container
            self.steps_container = ui.column().classes('w-full gap-2')
                
    def set_steps(self, steps: List[PipelineStep]):
        self.steps = steps
        self.current_step_index = -1
        self._render_steps()
        
    def _render_steps(self):
        self.steps_container.clear()
        self.step_elements.clear()
        
        with self.steps_container:
            for i, step in enumerate(self.steps):
                # Step row
                with ui.row().classes('w-full items-center p-3 rounded-lg bg-gray-800') as step_element:
                    # Icon container
                    with ui.element('div').classes('w-10 h-10 rounded-full bg-gray-700 flex items-center justify-center flex-shrink-0'):
                        icon = ui.icon(step.icon).classes('text-gray-400')
                    
                    # Step info
                    with ui.column().classes('flex-grow ml-3'):
                        label = ui.label(step.label).classes('text-sm font-medium')
                        if step.duration_estimate:
                            ui.label(f'~{step.duration_estimate}s').classes('text-xs text-gray-500')
                    
                    # Status icon
                    status = ui.icon('').classes('hidden')
                    
                    # Store references
                    self.step_elements[step.id] = {
                        'element': step_element,
                        'icon': icon,
                        'label': label,
                        'status': status
                    }
        
    def start(self):
        self.current_step_index = -1
        self.start_time = asyncio.get_event_loop().time()
        self.progress_bar.value = 0
        self.progress_label.text = '0%'
        self.status_label.text = 'Starting generation...'
        
        # Reset all steps
        for step_id, refs in self.step_elements.items():
            refs['element'].classes('bg-gray-800', remove='bg-green-900 bg-blue-900')
            refs['icon'].classes('text-gray-400', remove='text-green-500 text-blue-500')
            refs['status'].classes('hidden')
            
    def advance_step(self, message: Optional[str] = None):
        # Complete current step
        if self.current_step_index >= 0 and self.current_step_index < len(self.steps):
            current = self.steps[self.current_step_index]
            refs = self.step_elements[current.id]
            refs['element'].classes('bg-green-900', remove='bg-gray-800 bg-blue-900')
            refs['icon'].classes('text-green-500', remove='text-gray-400')
            refs['status'].props('name=check_circle').classes('text-green-500', remove='hidden')
        
        # Move to next step
        self.current_step_index += 1
        if self.current_step_index < len(self.steps):
            next_step = self.steps[self.current_step_index]
            refs = self.step_elements[next_step.id]
            refs['element'].classes('bg-blue-900', remove='bg-gray-800')
            refs['icon'].classes('text-blue-500', remove='text-gray-400')
            self.status_label.text = message or f'{next_step.label}...'
            
            # Update progress
            progress = (self.current_step_index + 1) / len(self.steps)
            self.progress_bar.value = progress
            self.progress_label.text = f'{int(progress * 100)}%'
        
        # Update time
        if self.start_time:
            elapsed = int(asyncio.get_event_loop().time() - self.start_time)
            self.time_label.text = f'{elapsed}s'
    
    def update_step_progress(self, progress: float, sub_message: str = ""):
        """Update progress within the current step with clean integer rounding"""
        if self.current_step_index < 0 or self.current_step_index >= len(self.steps):
            return
            
        # Store the progress and sub-message
        self.current_step_progress = max(0.0, min(1.0, progress))  # Clamp to 0-1
        self.current_sub_message = sub_message
        
        # Calculate overall progress using weighted steps
        total_weight = sum(step.weight for step in self.steps)
        completed_weight = sum(step.weight for step in self.steps[:self.current_step_index])
        current_step_weight = self.steps[self.current_step_index].weight
        
        # Overall progress = (completed steps + current step progress) / total
        overall_progress = (completed_weight + (current_step_weight * self.current_step_progress)) / total_weight
        
        # Apply clean rounding logic (same as downloads)
        raw_progress_percent = overall_progress * 100
        if raw_progress_percent >= 99.5 and overall_progress < 1.0:
            # Cap at 99% if not actually completed to avoid false 100%
            display_progress = 99
        else:
            display_progress = round(raw_progress_percent)
        
        # Update progress bar (uses 0-1 scale)
        self.progress_bar.value = overall_progress
        self.progress_label.text = f'{display_progress}%'
        
        # Update status with step and sub-step info
        step = self.steps[self.current_step_index]
        step_num = self.current_step_index + 1
        total_steps = len(self.steps)
        
        if sub_message:
            # Show both step and sub-step: "Step 3/5 - Diffusion Sampling: 85%"
            step_progress_percent = round(self.current_step_progress * 100)
            if step_progress_percent >= 99.5 and self.current_step_progress < 1.0:
                step_progress_percent = 99
            self.status_label.text = f'Step {step_num}/{total_steps} - {sub_message}: {step_progress_percent}%'
        else:
            # Show step progress: "Step 3/5 - Generate Views: 85%"
            step_progress_percent = round(self.current_step_progress * 100)
            if step_progress_percent >= 99.5 and self.current_step_progress < 1.0:
                step_progress_percent = 99
            self.status_label.text = f'Step {step_num}/{total_steps} - {step.label}: {step_progress_percent}%'
        
        # Update time
        if self.start_time:
            elapsed = int(asyncio.get_event_loop().time() - self.start_time)
            self.time_label.text = f'{elapsed}s'
            
    def complete(self, message: str = 'Generation complete!'):
        # Complete last step
        if self.current_step_index >= 0 and self.current_step_index < len(self.steps):
            current = self.steps[self.current_step_index]
            refs = self.step_elements[current.id]
            refs['element'].classes('bg-green-900', remove='bg-gray-800 bg-blue-900')
            refs['icon'].classes('text-green-500', remove='text-gray-400 text-blue-500')
            refs['status'].props('name=check_circle').classes('text-green-500', remove='hidden')
            
        self.progress_bar.value = 1.0
        self.progress_label.text = '100%'
        self.status_label.text = message
        self.status_label.classes('text-green-500')
        
    def fail(self, message: str = 'Generation failed'):
        if self.current_step_index >= 0 and self.current_step_index < len(self.steps):
            current = self.steps[self.current_step_index]
            refs = self.step_elements[current.id]
            refs['element'].classes('bg-red-900', remove='bg-gray-800 bg-blue-900')
            refs['icon'].classes('text-red-500', remove='text-gray-400 text-blue-500')
            refs['status'].props('name=error').classes('text-red-500', remove='hidden')
            
        self.status_label.text = message
        self.status_label.classes('text-red-500')
        
    def reset(self):
        self.current_step_index = -1
        self.start_time = None
        self.progress_bar.value = 0
        self.progress_label.text = '0%'
        self.status_label.text = 'Ready to generate'
        self.status_label.classes(remove='text-green-500 text-red-500')
        self.time_label.text = ''
        
        for step_id, refs in self.step_elements.items():
            refs['element'].classes('bg-gray-800', remove='bg-green-900 bg-blue-900 bg-red-900')
            refs['icon'].classes('text-gray-400', remove='text-green-500 text-blue-500 text-red-500')
            refs['status'].classes('hidden')