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

from .models import FaceDetector, FaceSwapper, FaceRestorer, FaceEnhancer

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
    similarity_threshold: float = 0.6  # Face similarity threshold
    blend_mode: BlendMode = BlendMode.SEAMLESS
    
    # Enhancement options
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
        
        # Initialize components
        self.face_detector = FaceDetector(model_dir=self.model_dir)
        self.face_swapper = FaceSwapper(model_dir=self.model_dir)
        self.face_restorer = FaceRestorer(model_dir=self.model_dir)
        self.face_enhancer = FaceEnhancer(model_dir=self.model_dir)
        
        # Models loaded flag
        self.models_loaded = False
        
    def initialize_models(
        self,
        device: str = "cuda",
        download_if_missing: bool = True
    ) -> Tuple[bool, str]:
        """Initialize all required models.
        
        Args:
            device: Device to use (cuda/cpu)
            download_if_missing: Whether to download missing models
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Initialize each component
            components = [
                ("Face Detector", self.face_detector),
                ("Face Swapper", self.face_swapper),
                ("Face Restorer", self.face_restorer),
                ("Face Enhancer", self.face_enhancer)
            ]
            
            for name, component in components:
                logger.info(f"Initializing {name}...")
                success, msg = component.initialize(device, download_if_missing)
                if not success:
                    return False, f"Failed to initialize {name}: {msg}"
            
            self.models_loaded = True
            return True, "All models initialized successfully"
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            return False, f"Error initializing models: {str(e)}"
    
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
        if not self.models_loaded:
            return None, {"error": "Models not initialized. Call initialize_models() first."}
        
        if params is None:
            params = FaceSwapParams()
        
        try:
            start_time = time.time()
            
            # Convert inputs to numpy arrays
            source_img = self._load_image(source_image)
            target_img = self._load_image(target_image)
            
            # Detect faces in both images
            source_faces = self.face_detector.detect_faces(source_img)
            target_faces = self.face_detector.detect_faces(target_img)
            
            if not source_faces:
                return None, {"error": "No faces detected in source image"}
            
            if not target_faces:
                return None, {"error": "No faces detected in target image"}
            
            # Select source face
            if params.source_face_index >= len(source_faces):
                return None, {"error": f"Source face index {params.source_face_index} out of range"}
            
            source_face = source_faces[params.source_face_index]
            
            # Perform swapping
            result = target_img.copy()
            swapped_count = 0
            
            for idx, target_face in enumerate(target_faces):
                # Check if we should swap this face
                if params.target_face_index != -1 and idx != params.target_face_index:
                    continue
                
                # Check similarity if needed
                if params.similarity_threshold > 0:
                    similarity = self._calculate_face_similarity(
                        source_face.embedding,
                        target_face.embedding
                    )
                    if similarity < params.similarity_threshold:
                        continue
                
                # Swap the face
                result = self.face_swapper.swap_face(
                    result,
                    source_face,
                    target_face,
                    blend_mode=params.blend_mode
                )
                swapped_count += 1
            
            if swapped_count == 0:
                return None, {"error": "No faces were swapped (similarity threshold too high?)"}
            
            # Apply enhancements if requested
            if params.face_restore:
                result = self.face_restorer.restore_faces(
                    result,
                    model=params.face_restore_model,
                    fidelity=params.face_restore_fidelity
                )
            
            if params.face_upsample:
                result = self.face_enhancer.enhance_faces(
                    result,
                    upscale_factor=params.upscale_factor
                )
            
            if params.background_enhance:
                result = self.face_enhancer.enhance_background(result)
            
            # Convert back to PIL Image
            result_img = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            
            # Prepare info
            info = {
                "source_faces": len(source_faces),
                "target_faces": len(target_faces),
                "swapped_faces": swapped_count,
                "processing_time": time.time() - start_time,
                "parameters": {
                    "blend_mode": params.blend_mode.value,
                    "face_restore": params.face_restore,
                    "face_upsample": params.face_upsample,
                    "background_enhance": params.background_enhance
                }
            }
            
            return result_img, info
            
        except Exception as e:
            logger.error(f"Error during face swap: {e}")
            return None, {"error": f"Face swap failed: {str(e)}"}
    
    def _load_image(self, image: Union[Image.Image, np.ndarray, str, Path]) -> np.ndarray:
        """Load image from various input types.
        
        Args:
            image: Input image
            
        Returns:
            OpenCV image array (BGR)
        """
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"Could not load image from {image}")
            return img
        elif isinstance(image, Image.Image):
            return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        elif isinstance(image, np.ndarray):
            if image.shape[2] == 3:
                return image
            elif image.shape[2] == 4:
                return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
    
    def _calculate_face_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between face embeddings.
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            
        Returns:
            Similarity score (0-1)
        """
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return max(0.0, min(1.0, similarity))
    
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