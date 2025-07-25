"""Feature Initialization Mixin

Handles initialization of all enhanced features and managers.
"""

import logging
from pathlib import Path
from typing import Any

from ...services.credentials import CredentialManager
from ...services.civitai import CivitaiManager
from ...features.lora.manager import LoRAManager
from ...features.lora.suggestion import LoRASuggestionEngine
from ...services.queue import QueueManager
from ...services.history import HistoryManager
from ...models.comparison import ModelComparison
from ...generation.video import VideoGenerator
from ...features.character.consistency import CharacterConsistencyManager
from ...features.face_swap.manager import FaceSwapManager
from ...config import MODELS_DIR, OUTPUT_DIR, CACHE_DIR

logger = logging.getLogger(__name__)


class FeatureInitializationMixin:
    """Mixin for initializing enhanced features"""
    
    def _initialize_features(self):
        """Initialize all enhanced features and managers"""
        # Get progress manager first
        from ...services.websocket import get_progress_manager
        self.progress_manager = get_progress_manager()
        
        # Re-create ModelManager with WebSocket server
        from ...models.manager import ModelManager
        self.model_manager = ModelManager(
            MODELS_DIR, 
            Path(__file__).parent.parent.parent / "models",
            websocket_server=self.progress_manager
        )
        
        # Initialize new managers
        self.credential_manager = CredentialManager()
        self.civitai_manager = CivitaiManager(
            cache_dir=CACHE_DIR / "civitai",
            api_key=self.credential_manager.get_credential("civitai")
        )
        
        # LoRA system with suggestion engine
        self.lora_suggestion_engine = LoRASuggestionEngine(
            civitai_manager=self.civitai_manager,
            cache_dir=CACHE_DIR / "lora_suggestions"
        )
        self.lora_manager = LoRAManager(
            lora_dir=MODELS_DIR / "loras",
            suggestion_engine=self.lora_suggestion_engine
        )
        
        # Queue and history
        self.queue_manager = QueueManager(
            max_workers=2,
            job_history_dir=OUTPUT_DIR / "job_history"
        )
        self.history_manager = HistoryManager(
            db_path=CACHE_DIR / "generation_history.db",
            thumbnails_dir=CACHE_DIR / "thumbnails"
        )
        self.model_comparison = ModelComparison(
            output_dir=OUTPUT_DIR / "benchmarks",
            cache_dir=CACHE_DIR / "benchmarks"
        )
        
        # New advanced features
        self.video_generator = VideoGenerator(
            cache_dir=CACHE_DIR / "video"
        )
        self.character_consistency_manager = CharacterConsistencyManager(
            profiles_dir=CACHE_DIR / "characters" / "profiles",
            embeddings_dir=CACHE_DIR / "characters" / "embeddings",
            cache_dir=CACHE_DIR / "characters"
        )
        self.face_swap_manager = FaceSwapManager(
            model_dir=MODELS_DIR / "insightface",
            cache_dir=CACHE_DIR / "faceswap"
        )
        
        # Additional setup
        self.output_dir = OUTPUT_DIR
        
        # Load LoRAs on startup
        self.available_loras = self.lora_manager.scan_lora_directory()
    
    def _initialize_advanced_models(self):
        """Initialize advanced model features"""
        # Initialize face swap models in background
        import threading
        
        def init_face_swap():
            try:
                logger.info("Initializing FaceFusion models in background...")
                initialized, msg = self.face_swap_manager.initialize_models()
                if initialized:
                    logger.info("FaceFusion models ready")
                else:
                    logger.warning(f"FaceFusion initialization: {msg}")
            except Exception as e:
                logger.error(f"Failed to initialize FaceFusion: {e}")
        
        # Start in background thread
        face_thread = threading.Thread(target=init_face_swap, daemon=True)
        face_thread.start()
        
        # Initialize video models if needed
        if hasattr(self, 'video_generator'):
            def init_video():
                try:
                    logger.info("Checking video generation models...")
                    # Video models will be loaded on demand
                except Exception as e:
                    logger.error(f"Failed to check video models: {e}")
            
            video_thread = threading.Thread(target=init_video, daemon=True)
            video_thread.start()
    
    def _register_job_handlers(self):
        """Register job handlers for the queue"""
        # Image generation
        self.queue_manager.register_handler("image", self._process_image_job)
        
        # 3D generation
        self.queue_manager.register_handler("3d", self._process_3d_job)
        
        # Full pipeline (image + 3D)
        self.queue_manager.register_handler("full_pipeline", self._process_full_pipeline_job)
        
        # Video generation
        self.queue_manager.register_handler("video", self._process_video_job)
        
        # Face swap
        self.queue_manager.register_handler("face_swap", self._process_face_swap_job)