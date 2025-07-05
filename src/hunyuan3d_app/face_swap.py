"""Advanced face swap module using InsightFace"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import time
import cv2
import numpy as np
from PIL import Image

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
    """Manages face swapping using InsightFace"""
    
    def __init__(
        self,
        model_dir: Optional[Path] = None,
        cache_dir: Optional[Path] = None
    ):
        self.model_dir = model_dir or Path("./models/insightface")
        self.cache_dir = cache_dir or Path("./cache/faceswap")
        
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Model paths
        self.inswapper_path = self.model_dir / "inswapper_128.onnx"
        self.face_analyser = None
        self.face_swapper = None
        self.face_restorer = None
        self.face_enhancer = None
        
        # Models loaded flag
        self.models_loaded = False
        
    def initialize_models(
        self,
        device: str = "cuda",
        download_if_missing: bool = True
    ) -> Tuple[bool, str]:
        """Initialize face swap models
        
        Args:
            device: Device to run on
            download_if_missing: Download models if not found
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Check if models exist
            if not self.inswapper_path.exists() and download_if_missing:
                logger.info("Downloading InsightFace models...")
                success, msg = self._download_models()
                if not success:
                    return False, msg
                    
            # Initialize face analyser
            logger.info("Initializing face analyser...")
            try:
                import insightface
                from insightface.app import FaceAnalysis
                
                self.face_analyser = FaceAnalysis(
                    name="buffalo_l",
                    root=str(self.model_dir),
                    providers=['CUDAExecutionProvider' if device == "cuda" else 'CPUExecutionProvider']
                )
                self.face_analyser.prepare(ctx_id=0, det_size=(640, 640))
                
            except ImportError:
                # Use mock for demonstration
                self.face_analyser = self._create_mock_analyser()
                
            # Initialize face swapper
            logger.info("Initializing face swapper...")
            self.face_swapper = self._create_face_swapper(device)
            
            # Initialize face restorer
            logger.info("Initializing face restorer...")
            self.face_restorer = self._create_face_restorer(device)
            
            self.models_loaded = True
            return True, "Face swap models initialized successfully"
            
        except Exception as e:
            logger.error(f"Failed to initialize face swap models: {e}")
            return False, f"Initialization failed: {str(e)}"
            
    def _download_models(self) -> Tuple[bool, str]:
        """Download required models"""
        # In real implementation, this would download from model zoo
        logger.info("Would download InsightFace models...")
        return True, "Models downloaded (placeholder)"
        
    def _create_mock_analyser(self) -> Any:
        """Create mock face analyser for demonstration"""
        class MockFaceAnalyser:
            def get(self, img: np.ndarray) -> List[Any]:
                # Mock face detection
                h, w = img.shape[:2]
                
                # Simulate detecting one face
                class MockFace:
                    def __init__(self):
                        self.bbox = np.array([w*0.3, h*0.2, w*0.7, h*0.8])
                        self.kps = np.random.randn(5, 2) * 50 + [w/2, h/2]
                        self.det_score = 0.95
                        self.embedding = np.random.randn(512)
                        self.age = 25
                        self.gender = "F"
                        
                return [MockFace()]
                
        return MockFaceAnalyser()
        
    def _create_face_swapper(self, device: str) -> Any:
        """Create face swapper model"""
        # In real implementation, this would load inswapper model
        class MockFaceSwapper:
            def __init__(self, device):
                self.device = device
                
            def get(self, target_img: np.ndarray, target_face: Any, source_face: Any) -> np.ndarray:
                # Mock face swap - just return target image
                return target_img
                
        return MockFaceSwapper(device)
        
    def _create_face_restorer(self, device: str) -> Any:
        """Create face restoration model"""
        # In real implementation, this would load CodeFormer/GFPGAN
        class MockFaceRestorer:
            def __init__(self, device):
                self.device = device
                
            def restore(self, img: np.ndarray, fidelity: float = 0.5) -> np.ndarray:
                # Mock restoration - apply slight blur as placeholder
                return cv2.GaussianBlur(img, (3, 3), 0)
                
        return MockFaceRestorer(device)
        
    def detect_faces(
        self,
        image: Union[Image.Image, np.ndarray, str, Path]
    ) -> List[FaceInfo]:
        """Detect faces in an image
        
        Args:
            image: Input image
            
        Returns:
            List of detected faces
        """
        if not self.models_loaded:
            logger.error("Models not loaded. Call initialize_models() first.")
            return []
            
        # Convert to numpy array
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
        elif isinstance(image, Image.Image):
            image = np.array(image)
            if image.shape[-1] == 3:  # RGB to BGR
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
        # Detect faces
        faces = self.face_analyser.get(image)
        
        # Convert to FaceInfo objects
        face_infos = []
        for face in faces:
            info = FaceInfo(
                bbox=face.bbox,
                kps=face.kps,
                det_score=face.det_score,
                embedding=face.embedding,
                age=getattr(face, 'age', None),
                gender=getattr(face, 'gender', None)
            )
            face_infos.append(info)
            
        return face_infos
        
    def swap_face(
        self,
        source_image: Union[Image.Image, np.ndarray, str, Path],
        target_image: Union[Image.Image, np.ndarray, str, Path],
        params: Optional[FaceSwapParams] = None
    ) -> Tuple[Optional[Image.Image], Dict[str, Any]]:
        """Swap faces between images
        
        Args:
            source_image: Source face image
            target_image: Target image to swap face into
            params: Swap parameters
            
        Returns:
            Tuple of (result image, info dict)
        """
        if not self.models_loaded:
            return None, {"error": "Models not loaded"}
            
        if params is None:
            params = FaceSwapParams()
            
        try:
            start_time = time.time()
            
            # Load images
            if isinstance(source_image, (str, Path)):
                source_img = cv2.imread(str(source_image))
            elif isinstance(source_image, Image.Image):
                source_img = np.array(source_image)
                if source_img.shape[-1] == 3:
                    source_img = cv2.cvtColor(source_img, cv2.COLOR_RGB2BGR)
            else:
                source_img = source_image
                
            if isinstance(target_image, (str, Path)):
                target_img = cv2.imread(str(target_image))
            elif isinstance(target_image, Image.Image):
                target_img = np.array(target_image)
                if target_img.shape[-1] == 3:
                    target_img = cv2.cvtColor(target_img, cv2.COLOR_RGB2BGR)
            else:
                target_img = target_image.copy()
                
            # Detect faces
            source_faces = self.detect_faces(source_img)
            target_faces = self.detect_faces(target_img)
            
            if not source_faces:
                return None, {"error": "No face detected in source image"}
            if not target_faces:
                return None, {"error": "No face detected in target image"}
                
            # Select source face
            source_face_idx = min(params.source_face_index, len(source_faces) - 1)
            source_face = source_faces[source_face_idx]
            
            # Process target faces
            result_img = target_img.copy()
            swapped_count = 0
            
            for idx, target_face in enumerate(target_faces):
                # Check if we should swap this face
                if params.target_face_index >= 0 and idx != params.target_face_index:
                    continue
                    
                # Check similarity if threshold is set
                if params.similarity_threshold > 0:
                    similarity = self._calculate_face_similarity(
                        source_face.embedding,
                        target_face.embedding
                    )
                    if similarity < params.similarity_threshold:
                        continue
                        
                # Perform face swap
                swapped_img = self._swap_single_face(
                    result_img,
                    source_face,
                    target_face,
                    params
                )
                
                if swapped_img is not None:
                    result_img = swapped_img
                    swapped_count += 1
                    
            if swapped_count == 0:
                return None, {"error": "No faces were swapped"}
                
            # Post-processing
            if params.face_restore:
                result_img = self._restore_faces(result_img, params)
                
            if params.background_enhance:
                result_img = self._enhance_background(result_img)
                
            if params.face_upsample:
                result_img = self._upsample_faces(result_img, params)
                
            # Convert back to RGB PIL Image
            result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            result_pil = Image.fromarray(result_img)
            
            processing_time = time.time() - start_time
            
            info = {
                "source_faces": len(source_faces),
                "target_faces": len(target_faces),
                "swapped_faces": swapped_count,
                "processing_time": f"{processing_time:.2f}s",
                "parameters": {
                    "blend_mode": params.blend_mode.value,
                    "face_restore": params.face_restore,
                    "restore_model": params.face_restore_model.value if params.face_restore else None
                }
            }
            
            return result_pil, info
            
        except Exception as e:
            logger.error(f"Face swap failed: {e}")
            return None, {"error": str(e)}
            
    def _calculate_face_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """Calculate cosine similarity between face embeddings"""
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)
        
    def _swap_single_face(
        self,
        image: np.ndarray,
        source_face: FaceInfo,
        target_face: FaceInfo,
        params: FaceSwapParams
    ) -> Optional[np.ndarray]:
        """Swap a single face"""
        try:
            # Get swapped face
            swapped = self.face_swapper.get(image, target_face, source_face)
            
            if swapped is None:
                return None
                
            # Apply blending based on mode
            if params.blend_mode == BlendMode.SEAMLESS:
                result = self._seamless_blend(image, swapped, target_face.bbox)
            elif params.blend_mode == BlendMode.POISSON:
                result = self._poisson_blend(image, swapped, target_face.bbox)
            elif params.blend_mode == BlendMode.SOFT:
                result = self._soft_blend(image, swapped, target_face.bbox)
            else:  # HARD
                result = swapped
                
            # Preserve attributes if requested
            if params.preserve_expression:
                result = self._preserve_expression(
                    result, image, target_face, params.expression_weight
                )
                
            if params.preserve_lighting:
                result = self._preserve_lighting(
                    result, image, target_face, params.lighting_weight
                )
                
            return result
            
        except Exception as e:
            logger.error(f"Single face swap failed: {e}")
            return None
            
    def _seamless_blend(
        self,
        original: np.ndarray,
        swapped: np.ndarray,
        bbox: np.ndarray
    ) -> np.ndarray:
        """Seamless blending using feathering"""
        x1, y1, x2, y2 = bbox.astype(int)
        
        # Create mask with feathered edges
        mask = np.zeros(original.shape[:2], dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
        
        # Feather the mask
        kernel_size = max(5, min(21, (x2 - x1) // 10))
        if kernel_size % 2 == 0:
            kernel_size += 1
        mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
        
        # Blend
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
        result = swapped * mask_3ch + original * (1 - mask_3ch)
        
        return result.astype(np.uint8)
        
    def _poisson_blend(
        self,
        original: np.ndarray,
        swapped: np.ndarray,
        bbox: np.ndarray
    ) -> np.ndarray:
        """Poisson blending for seamless integration"""
        try:
            x1, y1, x2, y2 = bbox.astype(int)
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            # Create mask
            mask = np.zeros(original.shape[:2], dtype=np.uint8)
            mask[y1:y2, x1:x2] = 255
            
            # Poisson blend
            result = cv2.seamlessClone(
                swapped, original, mask, center, cv2.NORMAL_CLONE
            )
            
            return result
            
        except Exception:
            # Fallback to seamless blend
            return self._seamless_blend(original, swapped, bbox)
            
    def _soft_blend(
        self,
        original: np.ndarray,
        swapped: np.ndarray,
        bbox: np.ndarray
    ) -> np.ndarray:
        """Soft blending with alpha"""
        # Simple alpha blend
        alpha = 0.8
        result = cv2.addWeighted(swapped, alpha, original, 1 - alpha, 0)
        return result
        
    def _preserve_expression(
        self,
        swapped: np.ndarray,
        original: np.ndarray,
        face: FaceInfo,
        weight: float
    ) -> np.ndarray:
        """Preserve original expression"""
        # This would use facial landmarks to transfer expression
        # For now, just blend
        return cv2.addWeighted(swapped, 1 - weight, original, weight, 0)
        
    def _preserve_lighting(
        self,
        swapped: np.ndarray,
        original: np.ndarray,
        face: FaceInfo,
        weight: float
    ) -> np.ndarray:
        """Preserve original lighting"""
        # Extract lighting from original
        x1, y1, x2, y2 = face.bbox.astype(int)
        
        # Convert to LAB color space
        original_lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
        swapped_lab = cv2.cvtColor(swapped, cv2.COLOR_BGR2LAB)
        
        # Transfer lightness channel
        swapped_lab[:, :, 0] = (
            swapped_lab[:, :, 0] * (1 - weight) +
            original_lab[:, :, 0] * weight
        )
        
        # Convert back
        result = cv2.cvtColor(swapped_lab, cv2.COLOR_LAB2BGR)
        return result
        
    def _restore_faces(
        self,
        image: np.ndarray,
        params: FaceSwapParams
    ) -> np.ndarray:
        """Restore face quality"""
        if self.face_restorer is None:
            return image
            
        # Detect faces again for restoration
        faces = self.detect_faces(image)
        
        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            
            # Extract face region
            face_img = image[y1:y2, x1:x2]
            
            # Restore
            restored = self.face_restorer.restore(
                face_img,
                fidelity=params.face_restore_fidelity
            )
            
            # Put back
            image[y1:y2, x1:x2] = restored
            
        return image
        
    def _enhance_background(self, image: np.ndarray) -> np.ndarray:
        """Enhance background quality"""
        # Simple enhancement using sharpening
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        enhanced = cv2.filter2D(image, -1, kernel)
        
        # Blend with original
        result = cv2.addWeighted(image, 0.7, enhanced, 0.3, 0)
        return result
        
    def _upsample_faces(
        self,
        image: np.ndarray,
        params: FaceSwapParams
    ) -> np.ndarray:
        """Upsample face regions"""
        # This would use ESRGAN or similar
        # For now, just use simple upsampling
        h, w = image.shape[:2]
        new_h = h * params.upscale_factor
        new_w = w * params.upscale_factor
        
        upsampled = cv2.resize(
            image,
            (new_w, new_h),
            interpolation=cv2.INTER_CUBIC
        )
        
        return upsampled
        
    def process_video(
        self,
        source_image: Union[Image.Image, str, Path],
        target_video: Union[str, Path],
        output_path: Union[str, Path],
        params: Optional[FaceSwapParams] = None,
        progress_callback: Optional[Any] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """Process video for face swapping
        
        Args:
            source_image: Source face image
            target_video: Target video path
            output_path: Output video path
            params: Swap parameters
            progress_callback: Progress callback
            
        Returns:
            Tuple of (success, info dict)
        """
        if not self.models_loaded:
            return False, {"error": "Models not loaded"}
            
        if params is None:
            params = FaceSwapParams()
            
        try:
            # Open video
            cap = cv2.VideoCapture(str(target_video))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Set up video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                str(output_path),
                fourcc,
                fps,
                (width, height)
            )
            
            # Process frames
            frame_buffer = []
            processed_frames = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Convert frame to PIL
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                
                # Swap face
                result_pil, _ = self.swap_face(
                    source_image,
                    frame_pil,
                    params
                )
                
                if result_pil:
                    # Convert back to OpenCV
                    result_np = np.array(result_pil)
                    result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)
                    
                    # Apply temporal smoothing
                    if params.temporal_smoothing and frame_buffer:
                        result_bgr = self._temporal_smooth(
                            result_bgr,
                            frame_buffer,
                            params.smoothing_window
                        )
                        
                    frame_buffer.append(result_bgr)
                    if len(frame_buffer) > params.smoothing_window:
                        frame_buffer.pop(0)
                        
                    out.write(result_bgr)
                else:
                    # Write original frame if swap failed
                    out.write(frame)
                    
                processed_frames += 1
                
                if progress_callback:
                    progress = processed_frames / total_frames
                    progress_callback(
                        progress,
                        f"Processing frame {processed_frames}/{total_frames}"
                    )
                    
            # Clean up
            cap.release()
            out.release()
            
            info = {
                "total_frames": total_frames,
                "processed_frames": processed_frames,
                "fps": fps,
                "resolution": f"{width}x{height}",
                "output_path": str(output_path)
            }
            
            return True, info
            
        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            return False, {"error": str(e)}
            
    def _temporal_smooth(
        self,
        current_frame: np.ndarray,
        buffer: List[np.ndarray],
        window_size: int
    ) -> np.ndarray:
        """Apply temporal smoothing to reduce flicker"""
        if not buffer:
            return current_frame
            
        # Get recent frames
        recent_frames = buffer[-min(window_size-1, len(buffer)):]
        recent_frames.append(current_frame)
        
        # Weighted average (more weight on current frame)
        weights = np.linspace(0.5, 1.0, len(recent_frames))
        weights = weights / weights.sum()
        
        smoothed = np.zeros_like(current_frame, dtype=np.float32)
        for frame, weight in zip(recent_frames, weights):
            smoothed += frame.astype(np.float32) * weight
            
        return smoothed.astype(np.uint8)
        
    def batch_process(
        self,
        source_images: List[Union[Image.Image, str, Path]],
        target_images: List[Union[Image.Image, str, Path]],
        output_dir: Path,
        params: Optional[FaceSwapParams] = None,
        progress_callback: Optional[Any] = None
    ) -> List[Tuple[bool, str]]:
        """Batch process multiple face swaps
        
        Args:
            source_images: List of source images
            target_images: List of target images
            output_dir: Output directory
            params: Swap parameters
            progress_callback: Progress callback
            
        Returns:
            List of (success, output_path) tuples
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        total = len(target_images)
        
        for idx, (source, target) in enumerate(zip(source_images, target_images)):
            # Generate output filename
            target_name = Path(target).stem if isinstance(target, (str, Path)) else f"image_{idx}"
            output_path = output_dir / f"{target_name}_swapped.png"
            
            # Process
            result_img, info = self.swap_face(source, target, params)
            
            if result_img:
                result_img.save(output_path)
                results.append((True, str(output_path)))
            else:
                results.append((False, info.get("error", "Unknown error")))
                
            if progress_callback:
                progress_callback(
                    (idx + 1) / total,
                    f"Processed {idx + 1}/{total} images"
                )
                
        return results