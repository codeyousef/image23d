"""Settings Tab - Backward Compatibility Wrapper

This file maintains backward compatibility by re-exporting the refactored
settings components.
"""

# Re-export the main function from the refactored module
from .settings import create_settings_tab

__all__ = ["create_settings_tab"]