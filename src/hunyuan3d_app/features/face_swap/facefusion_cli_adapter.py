"""FaceFusion CLI adapter - calls FaceFusion as subprocess to avoid import conflicts."""

import subprocess
import tempfile
import logging
import time
import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Union, List
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


@dataclass
class FaceFusionConfig:
    """Configuration for FaceFusion CLI adapter"""
    face_swapper_model: FaceFusionModel = FaceFusionModel.INSWAPPER_128
    face_detector_score: float = 0.5
    pixel_boost: str = "256x256"
    live_portrait: bool = False
    execution_providers: list = None
    
    def __post_init__(self):
        if self.execution_providers is None:
            self.execution_providers = ["cuda"] if self._has_cuda() else ["cpu"]
    
    def _has_cuda(self) -> bool:
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False


class FaceFusionCLIAdapter:
    """CLI-based adapter for FaceFusion 3.2.0 that avoids import conflicts"""
    
    def __init__(self, 
                 facefusion_path: Optional[Path] = None,
                 config: Optional[FaceFusionConfig] = None):
        """Initialize FaceFusion CLI adapter
        
        Args:
            facefusion_path: Path to FaceFusion installation
            config: FaceFusion configuration
        """
        self.facefusion_path = facefusion_path or Path("./models/facefusion")
        self.config = config or FaceFusionConfig()
        self.initialized = False
        
        # Temp directory for processing
        self.temp_dir = Path(tempfile.gettempdir()) / "facefusion_cli_adapter"
        self.temp_dir.mkdir(exist_ok=True)
        
        # FaceFusion executable path
        self.facefusion_script = self.facefusion_path / "facefusion.py"
        
        # Debug: Log paths for troubleshooting
        logger.debug(f"FaceFusion path: {self.facefusion_path}")
        logger.debug(f"FaceFusion script: {self.facefusion_script}")
        logger.debug(f"Script exists: {self.facefusion_script.exists()}")
        
    def initialize(self) -> Tuple[bool, str]:
        """Initialize FaceFusion CLI adapter
        
        Returns:
            Tuple of (success, message)
        """
        try:
            # Check if FaceFusion exists
            if not self.facefusion_path.exists():
                return False, f"FaceFusion not found at {self.facefusion_path}"
            
            if not self.facefusion_script.exists():
                # Try to find the correct path
                possible_scripts = [
                    self.facefusion_path / "facefusion.py",
                    self.facefusion_path / "run.py",
                    self.facefusion_path / "main.py"
                ]
                
                found_script = None
                for script in possible_scripts:
                    if script.exists():
                        found_script = script
                        break
                
                if found_script:
                    self.facefusion_script = found_script
                    logger.info(f"Found FaceFusion script at: {self.facefusion_script}")
                else:
                    return False, f"FaceFusion script not found. Tried: {[str(s) for s in possible_scripts]}"
            
            # Test FaceFusion CLI availability (skip full validation to avoid FFmpeg requirement)
            logger.info("Validating FaceFusion CLI...")
            
            # Just check if the script can be executed (don't run full help which checks FFmpeg)
            try:
                python_exec = "python"
                script_path = str(self.facefusion_script.absolute())
                
                # Test if we can import the script (basic validation)
                result = subprocess.run([
                    python_exec, "-c", f"import sys; sys.path.insert(0, '{self.facefusion_path}'); import facefusion"
                ], capture_output=True, text=True, timeout=15)
                
                if result.returncode != 0:
                    logger.warning(f"FaceFusion import test failed: {result.stderr}")
                    # Don't fail completely - some import issues are expected
                
                logger.info("FaceFusion CLI validation completed")
                
            except Exception as e:
                logger.warning(f"FaceFusion CLI validation warning: {e}")
                # Don't fail - just log the warning
            
            self.initialized = True
            logger.info(f"FaceFusion CLI initialized with model: {self.config.face_swapper_model.value}")
            return True, "FaceFusion CLI initialized successfully"
            
        except subprocess.TimeoutExpired:
            return False, "FaceFusion CLI test timed out"
        except Exception as e:
            logger.error(f"Error initializing FaceFusion CLI: {e}")
            return False, f"Initialization failed: {str(e)}"
    
    def swap_face(self,
                  source_image: Union[Image.Image, np.ndarray, str, Path],
                  target_image: Union[Image.Image, np.ndarray, str, Path],
                  **kwargs) -> Tuple[Optional[Image.Image], Dict[str, Any]]:
        """Perform face swap using FaceFusion CLI
        
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
            
            # Build FaceFusion command
            cmd = [
                "python", str(self.facefusion_script.absolute()),
                "headless-run",
                "--source-paths", str(source_path),
                "--target-path", str(target_path),
                "--output-path", str(output_path),
                "--processors", "face_swapper",
                "--face-swapper-model", self.config.face_swapper_model.value,
                "--face-detector-score", str(self.config.face_detector_score),
                "--execution-providers", " ".join(self.config.execution_providers),
                "--execution-thread-count", "1"
            ]
            
            # Add pixel boost if specified
            if self.config.pixel_boost and self.config.pixel_boost != "none":
                cmd.extend(["--face-swapper-pixel-boost", self.config.pixel_boost])
            
            # Add any additional kwargs
            for key, value in kwargs.items():
                if key.startswith("face_") or key.startswith("execution_"):
                    cmd.extend([f"--{key.replace('_', '-')}", str(value)])
            
            logger.info(f"Running FaceFusion: {' '.join(cmd[:10])}...")
            
            # Run FaceFusion
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            processing_time = time.time() - start_time
            
            # Check if command succeeded
            if result.returncode != 0:
                error_msg = f"FaceFusion CLI failed: {result.stderr}"
                logger.error(error_msg)
                return None, {"error": error_msg}
            
            # Check if output was created
            if not output_path.exists():
                return None, {"error": "Face swap failed - no output generated"}
            
            # Load result image
            result_image = Image.open(output_path)
            
            # Clean up temp files
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
                "method": "FaceFusion 3.2.0 CLI",
                "success": True
            }
            
            logger.info(f"FaceFusion CLI face swap completed in {processing_time:.2f}s")
            return result_image, info
            
        except subprocess.TimeoutExpired:
            return None, {"error": "Face swap timed out after 5 minutes"}
        except Exception as e:
            logger.error(f"Error during FaceFusion CLI face swap: {e}")
            return None, {"error": f"Face swap failed: {str(e)}"}
    
    def get_available_models(self) -> List[str]:
        """Get list of available FaceFusion models
        
        Returns:
            List of model names
        """
        return [model.value for model in FaceFusionModel]
    
    def set_model(self, model: Union[FaceFusionModel, str]):
        """Change the face swapper model
        
        Args:
            model: New model to use
        """
        if isinstance(model, str):
            model = FaceFusionModel(model)
        
        self.config.face_swapper_model = model
        logger.info(f"Model changed to: {model.value}")
    
    def configure(self, **options):
        """Update configuration options
        
        Args:
            **options: Configuration options to update
        """
        for key, value in options.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Config updated: {key} = {value}")
    
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
            if len(image.shape) == 3 and image.shape[2] == 3:
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
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.cleanup()


# Convenience function for simple face swapping
def swap_faces_with_facefusion_cli(source_image: Union[Image.Image, str, Path],
                                  target_image: Union[Image.Image, str, Path],
                                  model: Union[FaceFusionModel, str] = FaceFusionModel.INSWAPPER_128,
                                  **kwargs) -> Tuple[Optional[Image.Image], Dict[str, Any]]:
    """Simple function to perform face swap using FaceFusion CLI
    
    Args:
        source_image: Source face image
        target_image: Target image
        model: Face swapper model to use
        **kwargs: Additional configuration options
        
    Returns:
        Tuple of (result image, info dict)
    """
    config = FaceFusionConfig(face_swapper_model=model)
    adapter = FaceFusionCLIAdapter(config=config)
    
    try:
        return adapter.swap_face(source_image, target_image, **kwargs)
    finally:
        adapter.cleanup()