"""
Better Create page with improved layout

This module now serves as a backward compatibility wrapper that imports
the refactored CreatePage from the create_page package.
"""

# Import the refactored CreatePage
from .create_page import CreatePage

# Export it for backward compatibility
__all__ = ["CreatePage"]