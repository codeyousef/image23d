"""Model downloading functionality.

This module now serves as a backward compatibility wrapper that imports
the refactored ModelDownloader from the download package.
"""

# Import the refactored ModelDownloader
from .download import ModelDownloader

# Export for backward compatibility
__all__ = ["ModelDownloader"]