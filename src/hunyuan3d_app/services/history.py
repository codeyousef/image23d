"""Generation history and metadata tracking

This module now serves as a backward compatibility wrapper that imports
the refactored components from the history package.
"""

# Import the refactored components
from .history import GenerationRecord, HistoryManager

# Export for backward compatibility
__all__ = ["GenerationRecord", "HistoryManager"]