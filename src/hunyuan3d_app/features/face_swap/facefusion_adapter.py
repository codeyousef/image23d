"""FaceFusion 3.2.0 adapter for advanced face swapping functionality."""

import sys
import os
import logging
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class FaceFusionModel(Enum):
    """Available FaceFusion face swapper models"""
    INSWAPPER_128 = "inswapper_128"
    HYPERSWAP_1A_256 = "hyperswap_1a_256"
    HYPERSWAP_1B_256 = "hyperswap_1b_256" 
    HYPERSWAP_1C_256 = "hyperswap_1c_256"
    GHOST_1_256 = "ghost_1_256"
    GHOST_2_256 = "ghost_2_256"
    GHOST_3_256 = "ghost_3_256"
    SIMSWAP_256 = "simswap_256"
    SIMSWAP_UNOFFICIAL_512 = "simswap_unofficial_512"
    BLENDSWAP_256 = "blendswap_256"
    UNIFACE_256 = "uniface_256"
    HIFIFACE_UNOFFICIAL_256 = "hififace_unofficial_256"


class FaceDetectorModel(Enum):
    """Available face detector models"""
    YOLO_FACE = "yolo_face"
    RETINAFACE = "retinaface" 
    SCRFD = "scrfd"


class FaceSelectorMode(Enum):
    """Face selection modes"""
    MANY = "many"  # Process all detected faces
    ONE = "one"    # Process first/best face
    REFERENCE = "reference"  # Use reference face for matching


@dataclass
class FaceFusionConfig:
    """Configuration for FaceFusion face swapping"""
    # Model settings
    face_swapper_model: FaceFusionModel = FaceFusionModel.INSWAPPER_128
    face_detector_model: FaceDetectorModel = FaceDetectorModel.YOLO_FACE
    
    # Detection settings
    face_detector_score: float = 0.5
    face_selector_mode: FaceSelectorMode = FaceSelectorMode.ONE
    
    # Masking settings
    face_mask_types: List[str] = None
    face_mask_blur: float = 0.3
    face_mask_padding: Tuple[int, int, int, int] = (0, 0, 0, 0)
    
    # Enhancement settings
    pixel_boost: str = "256x256"  # 2025 feature
    live_portrait: bool = False   # 2025 feature
    
    # Execution settings
    execution_providers: List[str] = None
    execution_thread_count: int = 1
    
    # Output settings
    output_image_quality: int = 95
    keep_temp: bool = False
    
    def __post_init__(self):
        if self.face_mask_types is None:
            self.face_mask_types = ["box"]
        if self.execution_providers is None:
            self.execution_providers = ["cuda"] if self._has_cuda() else ["cpu"]
    
    def _has_cuda(self) -> bool:
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False


class FaceFusionAdapter:
    """Adapter for FaceFusion 3.2.0 face swapping functionality"""
    
    def __init__(self, 
                 facefusion_path: Optional[Path] = None,
                 config: Optional[FaceFusionConfig] = None):
        """Initialize FaceFusion adapter
        
        Args:
            facefusion_path: Path to FaceFusion installation
            config: FaceFusion configuration
        """
        self.facefusion_path = facefusion_path or Path("./models/facefusion")
        self.config = config or FaceFusionConfig()
        self.initialized = False
        self._facefusion_imported = False
        
        # Temp directory for processing
        self.temp_dir = Path(tempfile.gettempdir()) / "facefusion_adapter"
        self.temp_dir.mkdir(exist_ok=True)
        
    def _import_facefusion(self) -> bool:
        """Import FaceFusion modules"""
        if self._facefusion_imported:
            return True
            
        try:
            # Add FaceFusion to Python path
            facefusion_root = str(self.facefusion_path)
            if facefusion_root not in sys.path:
                sys.path.insert(0, facefusion_root)
            
            # Also add the inner facefusion directory
            facefusion_inner = str(self.facefusion_path / "facefusion")
            if facefusion_inner not in sys.path:
                sys.path.insert(0, facefusion_inner)
            
            # Import required modules
            from facefusion import state_manager
            from facefusion import core  
            from facefusion.args import apply_args
            from facefusion.processors.modules import face_swapper
            
            self.state_manager = state_manager
            self.core = core
            self.apply_args = apply_args
            self.face_swapper = face_swapper
            
            self._facefusion_imported = True
            logger.info("FaceFusion modules imported successfully")
            return True
            
        except ImportError as e:
            logger.error(f"Failed to import FaceFusion: {e}")
            logger.error(f"Make sure FaceFusion is installed at {self.facefusion_path}")
            logger.error(f"Python path: {sys.path[:3]}...")
            return False
    
    def initialize(self) -> Tuple[bool, str]:
        """Initialize FaceFusion with configuration
        
        Returns:
            Tuple of (success, message)
        """
        try:
            # Import FaceFusion modules
            if not self._import_facefusion():
                return False, "Failed to import FaceFusion modules"
            
            # Initialize FaceFusion core system
            if not self._initialize_facefusion_core():
                return False, "Failed to initialize FaceFusion core"
            
            self.initialized = True
            logger.info(f"FaceFusion initialized with model: {self.config.face_swapper_model.value}")
            return True, "FaceFusion initialized successfully"
            
        except Exception as e:
            logger.error(f"Error initializing FaceFusion: {e}")
            return False, f"Initialization failed: {str(e)}"
    
    def _initialize_facefusion_core(self) -> bool:
        """Initialize FaceFusion core system"""
        try:
            # Set up minimal required environment
            import os
            os.environ['OMP_NUM_THREADS'] = '1'
            
            # Initialize state manager with minimal args
            default_args = {
                'processors': ['face_swapper'],
                'face_swapper_model': self.config.face_swapper_model.value,
                'execution_providers': self.config.execution_providers,
                'execution_thread_count': 1,
                'temp_path': str(self.temp_dir),
                'command': 'headless-run'  # Set a command to avoid CLI mode
            }
            
            # Apply minimal configuration
            self.apply_args(default_args, self.state_manager.init_item)
            
            # Check if models need downloading/validation
            # For now, just return True - actual model check will happen during swap
            logger.info("FaceFusion core initialized")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing FaceFusion core: {e}")
            return False
    
    def swap_face(self,
                  source_image: Union[Image.Image, np.ndarray, str, Path],
                  target_image: Union[Image.Image, np.ndarray, str, Path],
                  **kwargs) -> Tuple[Optional[Image.Image], Dict[str, Any]]:
        """Perform face swap using FaceFusion
        
        Args:
            source_image: Source face image
            target_image: Target image
            **kwargs: Additional configuration options
            
        Returns:
            Tuple of (result image, info dict)
        """
        if not self.initialized:
            success, msg = self.initialize()
            if not success:
                return None, {"error": f"Initialization failed: {msg}"}
        
        try:
            # Save input images to temp files
            source_path = self._save_temp_image(source_image, "source")
            target_path = self._save_temp_image(target_image, "target")
            output_path = self.temp_dir / f"output_{os.getpid()}_{int(time.time() * 1000)}.png"
            
            # Update configuration with kwargs
            if kwargs:
                self.apply_args(kwargs, self.state_manager.set_item)
            
            # Set current operation parameters
            current_args = {
                'source_paths': [str(source_path)],
                'target_path': str(target_path),
                'output_path': str(output_path),
            }
            self.apply_args(current_args, self.state_manager.set_item)
            
            # Perform face swap
            import time
            start_time = time.time()
            
            self.face_swapper.process_image(
                [str(source_path)], 
                str(target_path), 
                str(output_path)
            )
            
            processing_time = time.time() - start_time
            
            # Check if output was created
            if not output_path.exists():
                return None, {"error": "Face swap failed - no output generated"}
            
            # Load result image
            result_image = Image.open(output_path)
            
            # Clean up temp files
            if not self.config.keep_temp:
                try:
                    source_path.unlink(missing_ok=True)
                    target_path.unlink(missing_ok=True)
                    output_path.unlink(missing_ok=True)
                except Exception:
                    pass  # Ignore cleanup errors
            
            # Prepare info dict
            info = {
                "processing_time": processing_time,
                "model": self.config.face_swapper_model.value,
                "pixel_boost": self.config.pixel_boost,
                "detector": self.config.face_detector_model.value,
                "success": True
            }
            
            logger.info(f"FaceFusion face swap completed in {processing_time:.2f}s")
            return result_image, info
            
        except Exception as e:
            logger.error(f"Error during FaceFusion face swap: {e}")
            return None, {"error": f"Face swap failed: {str(e)}"}
    
    def batch_swap_faces(self,
                        source_image: Union[Image.Image, np.ndarray, str, Path],
                        target_images: List[Union[Image.Image, np.ndarray, str, Path]],
                        progress_callback: Optional[callable] = None) -> List[Tuple[Optional[Image.Image], Dict[str, Any]]]:
        """Perform batch face swapping
        
        Args:
            source_image: Source face image
            target_images: List of target images
            progress_callback: Optional progress callback
            
        Returns:
            List of (result image, info dict) tuples
        """
        results = []
        
        for idx, target_image in enumerate(target_images):
            if progress_callback:
                progress_callback(idx / len(target_images), f"Processing image {idx + 1}/{len(target_images)}")
            
            result, info = self.swap_face(source_image, target_image)
            results.append((result, info))
        
        if progress_callback:
            progress_callback(1.0, "Batch processing complete")
        
        return results
    
    def configure(self, **options):
        """Update configuration options
        
        Args:
            **options: Configuration options to update
        """
        if self.initialized:
            self.apply_args(options, self.state_manager.set_item)
        
        # Update local config for future initialization
        for key, value in options.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    def set_model(self, model: Union[FaceFusionModel, str]):
        """Change the face swapper model
        
        Args:
            model: New model to use
        """
        if isinstance(model, str):
            model = FaceFusionModel(model)
        
        self.config.face_swapper_model = model
        
        if self.initialized:
            self.state_manager.set_item('face_swapper_model', model.value)
            # Re-check model after change
            self.face_swapper.pre_check()
    
    def get_available_models(self) -> List[str]:
        """Get list of available FaceFusion models
        
        Returns:
            List of model names
        """
        return [model.value for model in FaceFusionModel]
    
    def _save_temp_image(self, 
                        image: Union[Image.Image, np.ndarray, str, Path],
                        prefix: str) -> Path:
        """Save image to temporary file
        
        Args:
            image: Input image
            prefix: Filename prefix
            
        Returns:
            Path to saved temporary file
        """
        import time
        
        temp_path = self.temp_dir / f"{prefix}_{os.getpid()}_{int(time.time() * 1000)}.png"
        
        if isinstance(image, (str, Path)):
            # Copy existing file
            import shutil
            shutil.copy2(image, temp_path)
        elif isinstance(image, Image.Image):
            # Save PIL Image
            image.save(temp_path, "PNG")
        elif isinstance(image, np.ndarray):
            # Convert numpy array to PIL and save
            if image.shape[2] == 3:
                # BGR to RGB conversion
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            pil_image = Image.fromarray(image_rgb)
            pil_image.save(temp_path, "PNG")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        return temp_path
    
    def cleanup(self):
        """Clean up temporary files and resources"""
        try:
            import shutil
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception:
            pass  # Ignore cleanup errors
        
        if self.initialized and hasattr(self, 'face_swapper'):
            try:
                # Clear FaceFusion state/memory
                self.face_swapper.post_process()
            except Exception:
                pass
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.cleanup()


# Convenience function for simple face swapping
def swap_faces_with_facefusion(source_image: Union[Image.Image, str, Path],
                              target_image: Union[Image.Image, str, Path],
                              model: Union[FaceFusionModel, str] = FaceFusionModel.INSWAPPER_128,
                              **kwargs) -> Tuple[Optional[Image.Image], Dict[str, Any]]:
    """Simple function to perform face swap using FaceFusion
    
    Args:
        source_image: Source face image
        target_image: Target image
        model: Face swapper model to use
        **kwargs: Additional configuration options
        
    Returns:
        Tuple of (result image, info dict)
    """
    config = FaceFusionConfig(face_swapper_model=model)
    adapter = FaceFusionAdapter(config=config)
    
    try:
        return adapter.swap_face(source_image, target_image, **kwargs)
    finally:
        adapter.cleanup()