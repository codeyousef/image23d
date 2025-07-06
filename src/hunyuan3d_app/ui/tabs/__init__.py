"""UI tab modules."""

from .quick_generate import create_quick_generate_tab
from .manual_pipeline import create_manual_pipeline_tab
from .model_management import create_model_management_tab
from .downloads import create_downloads_manager_tab
from .media_gallery import create_media_gallery_tab
from .lora import create_lora_tab
from .video import create_video_generation_tab
from .face_swap import create_face_swap_tab
from .flux_kontext import create_flux_kontext_tab
from .character import create_character_studio_tab

__all__ = [
    "create_quick_generate_tab",
    "create_manual_pipeline_tab",
    "create_model_management_tab",
    "create_downloads_manager_tab",
    "create_media_gallery_tab",
    "create_lora_tab",
    "create_video_generation_tab",
    "create_face_swap_tab",
    "create_flux_kontext_tab",
    "create_character_studio_tab"
]