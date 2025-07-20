import logging
import json
import torch

logger = logging.getLogger(__name__)

# Import auth functions from utils package for backward compatibility
from .utils.auth import save_hf_token, load_hf_token

# --- Helper Functions ---


class TensorEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        return super().default(obj)


def load_custom_css():
    css_path = Path("app_styles.css")
    if css_path.exists():
        with open(css_path, 'r') as f:
            return f.read()
    return ""

def format_bytes(bytes):
    """Format bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.2f} TB"
