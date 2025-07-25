"""Downloads manager tab - backward compatibility wrapper

This module maintains backward compatibility by re-exporting
functionality from the refactored downloads package.
"""

# Re-export main function for backward compatibility
from .downloads.base import create_downloads_manager_tab

# Re-export any additional functions that might be used elsewhere
from .downloads.status_display import (
    get_download_status_html,
    create_status_display_component,
    create_control_buttons
)
from .downloads.model_cards import (
    create_model_download_row,
    create_model_category_section,
    create_quick_download_button
)
from .downloads.pipeline_components import (
    check_texture_components_status,
    create_texture_warning_banner,
    create_texture_enhancement_section,
    create_dependencies_section
)
from .downloads.model_hub_search import (
    create_model_hub_search_tab,
    format_search_results
)

__all__ = [
    'create_downloads_manager_tab',
    'get_download_status_html',
    'create_status_display_component',
    'create_control_buttons',
    'create_model_download_row',
    'create_model_category_section',
    'create_quick_download_button',
    'check_texture_components_status',
    'create_texture_warning_banner',
    'create_texture_enhancement_section',
    'create_dependencies_section',
    'create_model_hub_search_tab',
    'format_search_results'
]