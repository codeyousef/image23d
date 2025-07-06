"""UI tab modules for the 5-tab layout."""

# Main tabs for the new 5-tab structure
from .image_generation import create_image_generation_tab
from .video_generation import create_video_generation_tab
from .threed_generation import create_threed_generation_tab
from .media_gallery import create_media_gallery_tab
from .settings import create_settings_tab

# Legacy tabs (kept for backward compatibility)
from .quick_generate import create_quick_generate_tab
from .manual_pipeline import create_manual_pipeline_tab
from .model_management import create_model_management_tab
from .downloads import create_downloads_manager_tab
from .lora import create_lora_tab
from .face_swap import create_face_swap_tab
from .flux_kontext import create_flux_kontext_tab
from .character import create_character_studio_tab

__all__ = [
    # New 5-tab structure
    "create_image_generation_tab",
    "create_video_generation_tab",
    "create_threed_generation_tab",
    "create_media_gallery_tab",
    "create_settings_tab",
    # Legacy tabs
    "create_quick_generate_tab",
    "create_manual_pipeline_tab",
    "create_model_management_tab",
    "create_downloads_manager_tab",
    "create_lora_tab",
    "create_face_swap_tab",
    "create_flux_kontext_tab",
    "create_character_studio_tab"
]