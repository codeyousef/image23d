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

from .models import FaceDetector, FaceSwapper, FaceRestorer, FaceEnhancer
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
        
        # Initialize FaceFusion adapter (2025 primary method)
        facefusion_path = Path("./models/facefusion")
        self.facefusion_adapter = FaceFusionAdapter(facefusion_path=facefusion_path)
        
        # Initialize legacy components (fallback)
        self.face_detector = FaceDetector(model_dir=self.model_dir)
        self.face_swapper = FaceSwapper(model_dir=self.model_dir)
        self.face_restorer = FaceRestorer(model_dir=self.model_dir)
        self.face_enhancer = FaceEnhancer(model_dir=self.model_dir)
        
        # Models loaded flags
        self.models_loaded = False
        self.facefusion_loaded = False
        
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
            # First try to initialize FaceFusion (2025 primary method)
            logger.info("Initializing FaceFusion 3.2.0...")
            ff_success, ff_msg = self.facefusion_adapter.initialize()
            if ff_success:
                self.facefusion_loaded = True
                logger.info("FaceFusion 3.2.0 initialized successfully")
                # Also initialize legacy components for fallback/compatibility
                self._initialize_legacy_components(device, download_if_missing)
                return True, "FaceFusion 3.2.0 initialized successfully"
            else:
                logger.warning(f"FaceFusion initialization failed: {ff_msg}")
                logger.info("Falling back to legacy face swap components...")
            
            # Fallback: Initialize legacy components
            components = [
                ("Face Detector", self.face_detector),
                ("Face Swapper", self.face_swapper),
                ("Face Restorer", self.face_restorer),
                ("Face Enhancer", self.face_enhancer)
            ]
            
            initialized_components = []
            failed_components = []
            
            for name, component in components:
                logger.info(f"Initializing {name}...")
                success, msg = component.initialize(device, download_if_missing)
                if success:
                    initialized_components.append(name)
                else:
                    failed_components.append((name, msg))
                    # Only fail completely if Face Detector fails (it's essential)
                    if name == "Face Detector":
                        return False, f"Failed to initialize {name}: {msg}"
            
            # Mark as loaded if at least face detector is working
            if "Face Detector" in initialized_components:
                self.models_loaded = True
                
                if failed_components:
                    warning_msg = "Legacy models partially initialized. Failed components:\n"
                    for comp, err in failed_components:
                        warning_msg += f"- {comp}: {err}\n"
                    logger.warning(warning_msg)
                    return True, warning_msg
                else:
                    return True, "Legacy models initialized successfully"
            else:
                return False, "Failed to initialize essential components"
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            return False, f"Error initializing models: {str(e)}"
    
    def _initialize_legacy_components(self, device: str, download_if_missing: bool):
        """Initialize legacy components for compatibility"""
        try:
            components = [
                ("Face Detector", self.face_detector),
                ("Face Restorer", self.face_restorer), 
                ("Face Enhancer", self.face_enhancer)
            ]
            
            for name, component in components:
                try:
                    success, msg = component.initialize(device, download_if_missing)
                    if success:
                        logger.info(f"Legacy {name} initialized")
                    else:
                        logger.warning(f"Legacy {name} failed: {msg}")
                except Exception as e:
                    logger.warning(f"Legacy {name} error: {e}")
            
            self.models_loaded = True
            
        except Exception as e:
            logger.warning(f"Error initializing legacy components: {e}")
    
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
        if not self.models_loaded and not self.facefusion_loaded:
            return None, {"error": "Models not initialized. Call initialize_models() first."}
        
        if params is None:
            params = FaceSwapParams()
        
        # Use FaceFusion if available and enabled (2025 primary method)
        if params.use_facefusion and self.facefusion_loaded:
            return self._swap_face_facefusion(source_image, target_image, params)
        
        # Fallback to legacy method
        logger.info("Using legacy face swap method")
        return self._swap_face_legacy(source_image, target_image, params)
    
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
                logger.warning("FaceFusion swap failed, falling back to legacy method")
                return self._swap_face_legacy(source_image, target_image, params)
            
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
            logger.info("Falling back to legacy face swap method")
            return self._swap_face_legacy(source_image, target_image, params)
    
    def _swap_face_legacy(self,
                         source_image: Union[Image.Image, np.ndarray, str, Path],
                         target_image: Union[Image.Image, np.ndarray, str, Path],
                         params: FaceSwapParams) -> Tuple[Optional[Image.Image], Dict[str, Any]]:
        """Legacy face swap method using InsightFace components"""
        logger.warning("Using legacy face swap method - may have limited functionality")
        
        # Use simple face swap as the most reliable legacy method
        try:
            from .simple_swap import SimpleFaceSwap
            simple_swapper = SimpleFaceSwap()
            
            # Convert images to PIL format
            if isinstance(source_image, Image.Image):
                source_pil = source_image
            else:
                source_img = self._load_image(source_image)
                source_pil = Image.fromarray(cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB))
            
            if isinstance(target_image, Image.Image):
                target_pil = target_image
            else:
                target_img = self._load_image(target_image)
                target_pil = Image.fromarray(cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB))
            
            # Use simple face swap method
            result_img, info = simple_swapper.swap_faces(
                source_pil,
                target_pil,
                source_bbox=None,  # Auto-detect
                target_bbox=None,  # Auto-detect
                blend_mode=params.blend_mode.value
            )
            
            if result_img is None:
                return None, {"error": "Legacy face swap failed"}
            
            # Add legacy method info
            info.update({
                "method": "Legacy Simple Swap",
                "note": "Consider upgrading to FaceFusion for better results"
            })
            
            logger.info("Legacy face swap completed using simple method")
            return result_img, info
            
        except Exception as e:
            logger.error(f"Error during legacy face swap: {e}")
            return None, {"error": f"Legacy face swap failed: {str(e)}"}
    
    def detect_faces(self, image: Union[Image.Image, np.ndarray, str, Path]) -> List[FaceInfo]:
        """Detect faces in an image.
        
        Args:
            image: Input image
            
        Returns:
            List of detected faces
        """
        if not self.models_loaded:
            raise RuntimeError("Models not initialized. Call initialize_models() first.")
        
        img = self._load_image(image)
        return self.face_detector.detect_faces(img)
    
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
        # Ensure embeddings are 1D
        emb1 = embedding1.flatten()
        emb2 = embedding2.flatten()
        
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        logger.debug(f"Embedding shapes: {emb1.shape}, {emb2.shape}")
        logger.debug(f"Embedding norms: {norm1:.3f}, {norm2:.3f}")
        logger.debug(f"Dot product: {dot_product:.3f}")
        
        if norm1 == 0 or norm2 == 0:
            logger.warning("One or both embeddings have zero norm!")
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        logger.debug(f"Raw similarity: {similarity:.3f}")
        
        # Note: High similarity between different faces suggests the embeddings
        # are not discriminative enough. This is a limitation of the face detector.
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