"""Authentication utilities for Hugging Face models."""

import os
import base64
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Token storage location
SECRETS_DIR = Path.cwd() / ".secrets"
HF_TOKEN_FILE = SECRETS_DIR / "hf_token.txt"


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


def get_hf_token_from_all_sources() -> Optional[str]:
    """Get HF token from all possible sources in priority order."""
    # 1. Check environment variables
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        logger.info("Found HF token in environment variables")
        return token
    
    # 2. Check our app's saved token
    token = load_hf_token()
    if token:
        logger.info("Found HF token in app storage")
        return token
    
    # 3. Check HuggingFace's default token location
    hf_token_path = Path.home() / ".cache" / "huggingface" / "token"
    if hf_token_path.exists():
        try:
            token = hf_token_path.read_text().strip()
            if token:
                logger.info("Found HF token in HuggingFace cache")
                return token
        except Exception as e:
            logger.warning(f"Could not read HF token from cache: {e}")
    
    logger.warning("No HF token found in any location")
    return None


def validate_hf_token(token: str) -> bool:
    """Validate if an HF token is properly formatted."""
    if not token:
        return False
    
    # HF tokens typically start with 'hf_' and have a specific length
    if token.startswith("hf_") and len(token) > 10:
        return True
    
    # Also accept tokens that don't start with hf_ but are long enough
    if len(token) > 20:
        return True
    
    return False