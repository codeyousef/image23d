"""Desktop UI components"""

from .navigation import NavigationSidebar, NavItem
from .enhancement_panel import EnhancementPanel
from .progress_pipeline import ProgressPipeline, PipelineStep, StepStatus

__all__ = [
    "NavigationSidebar",
    "NavItem",
    "EnhancementPanel", 
    "ProgressPipeline",
    "PipelineStep",
    "StepStatus"
]