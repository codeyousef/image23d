"""
Services module for desktop app
"""

import sys
import os
from pathlib import Path

# Add parent directory to path to import from main app
desktop_dir = Path(__file__).parent.parent
project_root = desktop_dir.parent
sys.path.insert(0, str(project_root))

# Import ModelManager from main app
try:
    from src.hunyuan3d_app.models.manager import ModelManager
except ImportError:
    # Fallback - create a simple ModelManager class
    class ModelManager:
        """Simple model manager for desktop app"""
        
        def __init__(self, models_dir: Path):
            self.models_dir = Path(models_dir)
            
        def is_model_available(self, model_id: str) -> bool:
            """Check if a model is downloaded"""
            # Check common model locations
            possible_paths = [
                self.models_dir / 'image' / model_id,
                self.models_dir / '3d' / model_id,
                self.models_dir / 'video' / model_id,
                self.models_dir / 'texture' / model_id,
            ]
            
            for path in possible_paths:
                if path.exists() and any(path.iterdir()):
                    return True
            return False