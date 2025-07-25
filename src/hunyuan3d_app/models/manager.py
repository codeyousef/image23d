"""Main model management module.

This module now serves as a backward compatibility wrapper that imports
the refactored ModelManager from the manager package.
"""

# Import the refactored ModelManager
from .manager import ModelManager

# Export it for backward compatibility
__all__ = ["ModelManager"]