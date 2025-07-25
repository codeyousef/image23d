"""Generation History Module

Manages generation history with SQLite backend.
"""

from .models import GenerationRecord
from .base import HistoryManager

__all__ = ["GenerationRecord", "HistoryManager"]