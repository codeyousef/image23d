"""UI-related type definitions."""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any


class TabType(Enum):
    """Types of UI tabs."""
    IMAGE_GENERATION = "image_generation"
    THREED_GENERATION = "3d_generation"
    VIDEO_GENERATION = "video_generation"
    DOWNLOADS = "downloads"
    SETTINGS = "settings"
    LIBRARY = "library"
    LAB = "lab"


class NotificationType(Enum):
    """Types of UI notifications."""
    INFO = "info"
    SUCCESS = "positive"
    WARNING = "warning"
    ERROR = "negative"


class ComponentVisibility(Enum):
    """Visibility states for UI components."""
    VISIBLE = "visible"
    HIDDEN = "hidden"
    DISABLED = "disabled"


@dataclass
class UIState:
    """State of the UI."""
    active_tab: TabType
    is_generating: bool = False
    is_downloading: bool = False
    show_advanced: bool = False
    theme: str = "dark"
    language: str = "en"
    metadata: Optional[Dict[str, Any]] = None