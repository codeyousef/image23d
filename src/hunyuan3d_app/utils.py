import base64
import logging
import os
from pathlib import Path
from typing import Optional, Any
import json
import torch

logger = logging.getLogger(__name__)

# --- App Configuration ---
SECRETS_DIR = Path.cwd() / ".secrets"
HF_TOKEN_FILE = SECRETS_DIR / "hf_token.txt"


# --- Helper Functions ---

def save_hf_token(token: str):
    """Base64 encode and save the Hugging Face token."""
    if not token:
        return
    SECRETS_DIR.mkdir(exist_ok=True)
    encoded_token = base64.b64encode(token.encode('utf-8'))
    HF_TOKEN_FILE.write_bytes(encoded_token)


def load_hf_token() -> Optional[str]:
    """Load and decode the Hugging Face token."""
    if not HF_TOKEN_FILE.exists():
        return None
    try:
        encoded_token = HF_TOKEN_FILE.read_bytes()
        return base64.b64decode(encoded_token).decode('utf-8')
    except Exception as e:
        logger.error(f"Could not load HF token: {e}")
        return None


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
