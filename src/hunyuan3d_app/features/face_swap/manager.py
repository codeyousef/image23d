"""Face swap management and processing."""

import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np
from PIL import Image

try:
    import torch
except ImportError:
    torch = None

from .facefusion_adapter import FaceFusionAdapter, FaceFusionConfig, FaceFusionModel

logger = logging.getLogger(__name__)


class FaceRestoreModel(Enum):
    """Available face restoration models"""
    CODEFORMER = "CodeFormer"
    GFPGAN = "GFPGAN"
    RESTOREFORMER = "RestoreFormer"
    NONE = "None"


class BlendMode(Enum):
    """Face blending modes"""
    SEAMLESS = "seamless"
    HARD = "hard"
    SOFT = "soft"
    POISSON = "poisson"


@dataclass
class FaceSwapParams:
    """Parameters for face swapping"""
    source_face_index: int = 0  # Which face from source to use
    target_face_index: int = -1  # Which face in target to replace (-1 = all)
    similarity_threshold: float = 0.0  # Face similarity threshold (0 = swap all faces, 1 = only identical faces)
    blend_mode: BlendMode = BlendMode.SEAMLESS
    
    # FaceFusion options (2025 features)
    use_facefusion: bool = True  # Use FaceFusion instead of legacy inswapper
    facefusion_model: FaceFusionModel = FaceFusionModel.INSWAPPER_128
    pixel_boost: str = "256x256"  # 2025 pixel boost feature
    live_portrait: bool = False   # 2025 live portrait feature
    face_detector_score: float = 0.5
    
    # Enhancement options (legacy)
    face_restore: bool = True
    face_restore_model: FaceRestoreModel = FaceRestoreModel.CODEFORMER
    face_restore_fidelity: float = 0.5
    background_enhance: bool = False
    face_upsample: bool = True
    upscale_factor: int = 2
    
    # Advanced options
    preserve_expression: bool = False
    expression_weight: float = 0.3
    preserve_lighting: bool = True
    lighting_weight: float = 0.5
    preserve_age: bool = True
    age_weight: float = 0.7
    
    # Video options
    temporal_smoothing: bool = True
    smoothing_window: int = 5


@dataclass
class FaceInfo:
    """Information about a detected face"""
    bbox: np.ndarray  # Bounding box [x1, y1, x2, y2]
    kps: np.ndarray  # Keypoints
    det_score: float  # Detection confidence
    embedding: np.ndarray  # Face embedding
    age: Optional[int] = None
    gender: Optional[str] = None


class FaceSwapManager:
    """Manages face swapping pipeline."""
    
    def __init__(
        self,
        model_dir: Optional[Path] = None,
        cache_dir: Optional[Path] = None
    ):
        self.model_dir = model_dir or Path("./models/insightface")
        self.cache_dir = cache_dir or Path("./cache/faceswap")
        
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize FaceFusion adapter (only method)
        facefusion_path = Path("./models/facefusion")
        if not facefusion_path.exists():
            # Try absolute path from project root
            facefusion_path = Path(__file__).parent.parent.parent.parent.parent / "models" / "facefusion"
        self.facefusion_adapter = FaceFusionAdapter(facefusion_path=facefusion_path)
        
        # FaceFusion loaded flag
        self.facefusion_loaded = False
        
    def initialize_models(
        self,
        device: str = "cuda",
        download_if_missing: bool = True
    ) -> Tuple[bool, str]:
        """Initialize FaceFusion 3.2.0 models.
        
        Args:
            device: Device to use (cuda/cpu)
            download_if_missing: Whether to download missing models
            
        Returns:
            Tuple of (success, message)
        """
        try:
            logger.info("Initializing FaceFusion 3.2.0...")
            ff_success, ff_msg = self.facefusion_adapter.initialize()
            if ff_success:
                self.facefusion_loaded = True
                logger.info("FaceFusion 3.2.0 initialized successfully")
                return True, "FaceFusion 3.2.0 initialized successfully"
            else:
                logger.error(f"FaceFusion initialization failed: {ff_msg}")
                return False, f"FaceFusion initialization failed: {ff_msg}"
            
        except Exception as e:
            logger.error(f"Error initializing FaceFusion: {e}")
            return False, f"Error initializing FaceFusion: {str(e)}"
    
    def swap_face(
        self,
        source_image: Union[Image.Image, np.ndarray, str, Path],
        target_image: Union[Image.Image, np.ndarray, str, Path],
        params: Optional[FaceSwapParams] = None
    ) -> Tuple[Optional[Image.Image], Dict[str, Any]]:
        """Perform face swap between source and target images.
        
        Args:
            source_image: Source face image
            target_image: Target image to swap face into
            params: Swap parameters
            
        Returns:
            Tuple of (result image, info dict)
        """
        if not self.facefusion_loaded:
            return None, {"error": "FaceFusion not initialized. Call initialize_models() first."}
        
        if params is None:
            params = FaceSwapParams()
        
        # Use FaceFusion 3.2.0 (only method)
        return self._swap_face_facefusion(source_image, target_image, params)
    
    def _swap_face_facefusion(self,
                             source_image: Union[Image.Image, np.ndarray, str, Path],
                             target_image: Union[Image.Image, np.ndarray, str, Path],
                             params: FaceSwapParams) -> Tuple[Optional[Image.Image], Dict[str, Any]]:
        """Perform face swap using FaceFusion 3.2.0"""
        try:
            start_time = time.time()
            
            # Configure FaceFusion with parameters
            facefusion_config = {
                'face_swapper_model': params.facefusion_model.value,
                'face_detector_score': params.face_detector_score,
                'face_swapper_pixel_boost': params.pixel_boost,
                'execution_providers': ['cuda'] if torch and torch.cuda.is_available() else ['cpu']
            }
            
            # Apply live portrait if enabled (2025 feature)
            if params.live_portrait:
                facefusion_config['live_portrait'] = True
            
            # Perform face swap
            result_image, info = self.facefusion_adapter.swap_face(
                source_image, 
                target_image,
                **facefusion_config
            )
            
            if result_image is None:
                error_msg = info.get("error", "FaceFusion face swap failed")
                return None, {"error": f"Face swap failed: {error_msg}"}
            
            # Add FaceFusion-specific info
            info.update({
                "method": "FaceFusion 3.2.0",
                "model": params.facefusion_model.value,
                "pixel_boost": params.pixel_boost,
                "live_portrait": params.live_portrait,
                "processing_time": time.time() - start_time
            })
            
            logger.info(f"FaceFusion face swap completed successfully in {info['processing_time']:.2f}s")
            return result_image, info
            
        except Exception as e:
            logger.error(f"Error during FaceFusion face swap: {e}")
            return None, {"error": f"FaceFusion error: {str(e)}"}
    
    
    
    def batch_swap_faces(
        self,
        source_image: Union[Image.Image, np.ndarray, str, Path],
        target_images: List[Union[Image.Image, np.ndarray, str, Path]],
        params: Optional[FaceSwapParams] = None,
        progress_callback: Optional[callable] = None
    ) -> List[Tuple[Optional[Image.Image], Dict[str, Any]]]:
        """Perform face swap on multiple target images.
        
        Args:
            source_image: Source face image
            target_images: List of target images
            params: Swap parameters
            progress_callback: Optional progress callback
            
        Returns:
            List of (result image, info dict) tuples
        """
        results = []
        
        for idx, target_image in enumerate(target_images):
            if progress_callback:
                progress_callback(idx / len(target_images), f"Processing image {idx + 1}/{len(target_images)}")
            
            result, info = self.swap_face(source_image, target_image, params)
            results.append((result, info))
        
        if progress_callback:
            progress_callback(1.0, "Batch processing complete")
        
        return results
    
    def process_video(
        self,
        source_image: Union[Image.Image, np.ndarray, str, Path],
        target_video: Union[str, Path],
        output_path: Union[str, Path],
        params: Optional[FaceSwapParams] = None,
        progress_callback: Optional[callable] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """Process a video file for face swapping.
        
        Args:
            source_image: Source face image
            target_video: Target video path
            output_path: Output video path
            params: Swap parameters
            progress_callback: Optional progress callback
            
        Returns:
            Tuple of (success, info dict)
        """
        try:
            import cv2
            
            # Open video
            cap = cv2.VideoCapture(str(target_video))
            if not cap.isOpened():
                return False, {"error": "Failed to open video file"}
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Create output video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
            # Process frames
            processed_frames = 0
            frame_idx = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert frame to PIL Image
                frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                # Swap face
                result_img, info = self.swap_face(source_image, frame_pil, params)
                
                if result_img:
                    # Convert back to OpenCV format
                    result_cv = cv2.cvtColor(np.array(result_img), cv2.COLOR_RGB2BGR)
                    out.write(result_cv)
                    processed_frames += 1
                else:
                    # Write original frame if swap failed
                    out.write(frame)
                
                frame_idx += 1
                
                # Progress callback
                if progress_callback and frame_idx % 10 == 0:
                    progress = frame_idx / total_frames
                    progress_callback(progress, f"Processing frame {frame_idx}/{total_frames}")
            
            # Clean up
            cap.release()
            out.release()
            
            if progress_callback:
                progress_callback(1.0, "Video processing complete")
            
            return True, {
                "total_frames": total_frames,
                "processed_frames": processed_frames,
                "fps": fps,
                "resolution": f"{width}x{height}"
            }
            
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            return False, {"error": str(e)}
    
    def batch_process(
        self,
        source_images: List[Union[Image.Image, np.ndarray, str, Path]],
        target_images: List[Union[Image.Image, np.ndarray, str, Path]],
        output_dir: Union[str, Path],
        params: Optional[FaceSwapParams] = None,
        progress_callback: Optional[callable] = None
    ) -> List[Tuple[bool, Union[str, Path]]]:
        """Process multiple images in batch.
        
        Args:
            source_images: List of source face images
            target_images: List of target images
            output_dir: Output directory
            params: Swap parameters
            progress_callback: Optional progress callback
            
        Returns:
            List of (success, output_path) tuples
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        for idx, (source, target) in enumerate(zip(source_images, target_images)):
            if progress_callback:
                progress = idx / len(target_images)
                progress_callback(progress, f"Processing image {idx + 1}/{len(target_images)}")
            
            # Process image
            result_img, info = self.swap_face(source, target, params)
            
            if result_img:
                # Save result
                output_path = output_dir / f"swapped_{idx:04d}.png"
                result_img.save(output_path)
                results.append((True, output_path))
            else:
                results.append((False, info.get("error", "Unknown error")))
        
        if progress_callback:
            progress_callback(1.0, "Batch processing complete")
        
        return results