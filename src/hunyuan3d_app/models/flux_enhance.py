"""Post-processing and enhancement pipeline for FLUX outputs.

This module implements upscaling, detail enhancement, face restoration,
and other post-generation optimizations for FLUX images.
"""

import torch
import numpy as np
import cv2
import logging
from typing import Optional, Dict, Any, List, Union, Tuple
from PIL import Image, ImageEnhance, ImageFilter
from pathlib import Path
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class EnhancementConfig:
    """Configuration for post-processing enhancements."""
    upscale_factor: float = 2.0
    face_restoration: bool = True
    detail_enhancement: bool = True
    color_correction: bool = True
    sharpening: float = 0.5  # 0-1
    contrast: float = 1.1    # 1.0 = no change
    saturation: float = 1.1  # 1.0 = no change
    denoise_strength: float = 0.3  # 0-1


class PostProcessingPipeline:
    """Comprehensive post-processing pipeline for FLUX outputs."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.upscaler = None
        self.face_restorer = None
        self.detail_enhancer = None
        self._setup_processors()
    
    def _setup_processors(self):
        """Setup post-processing models."""
        # Setup Real-ESRGAN for upscaling
        self._setup_upscaler()
        
        # Setup face restoration
        self._setup_face_restorer()
        
        # Setup detail enhancement
        self._setup_detail_enhancer()
    
    def _setup_upscaler(self):
        """Setup Real-ESRGAN upscaler."""
        try:
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet
            
            # RealESRGAN x4plus model
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            
            self.upscaler = RealESRGANer(
                scale=4,
                model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
                model=model,
                tile=0,  # 0 = no tile, reduces memory for small images
                tile_pad=10,
                pre_pad=0,
                half=True if self.device == "cuda" else False,
                device=self.device
            )
            
            logger.info("Real-ESRGAN upscaler loaded")
            
        except ImportError:
            logger.warning("Real-ESRGAN not available, using fallback upscaling")
            self.upscaler = self._fallback_upscale
        except Exception as e:
            logger.warning(f"Could not load Real-ESRGAN: {e}")
            self.upscaler = self._fallback_upscale
    
    def _setup_face_restorer(self):
        """Setup GFPGAN face restoration."""
        try:
            from gfpgan import GFPGANer
            
            self.face_restorer = GFPGANer(
                model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth',
                upscale=1,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=None,
                device=self.device
            )
            
            logger.info("GFPGAN face restorer loaded")
            
        except ImportError:
            logger.warning("GFPGAN not available for face restoration")
            self.face_restorer = None
        except Exception as e:
            logger.warning(f"Could not load GFPGAN: {e}")
            self.face_restorer = None
    
    def _setup_detail_enhancer(self):
        """Setup detail enhancement models."""
        # For now, using CV2-based enhancement
        # Could integrate more sophisticated models here
        self.detail_enhancer = self._cv2_detail_enhance
    
    def _fallback_upscale(self, img: np.ndarray, outscale: float = 2.0) -> np.ndarray:
        """Fallback upscaling using CV2."""
        height, width = img.shape[:2]
        new_height = int(height * outscale)
        new_width = int(width * outscale)
        
        # Use Lanczos for quality
        upscaled = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        
        # Apply slight sharpening
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]]) / 10
        upscaled = cv2.filter2D(upscaled, -1, kernel)
        
        return upscaled
    
    def _cv2_detail_enhance(self, img: np.ndarray, strength: float = 0.5) -> np.ndarray:
        """Enhance details using CV2 operations."""
        # Create detail layer
        blurred = cv2.GaussianBlur(img, (0, 0), 3.0)
        detail_layer = cv2.subtract(img, blurred)
        
        # Enhance details
        enhanced = cv2.add(img, detail_layer * strength)
        
        # Ensure valid range
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        return enhanced
    
    def enhance_image(self,
                     image: Union[Image.Image, np.ndarray],
                     config: Optional[EnhancementConfig] = None) -> Image.Image:
        """Apply full enhancement pipeline to image."""
        
        if config is None:
            config = EnhancementConfig()
        
        # Convert to numpy if needed
        if isinstance(image, Image.Image):
            img_np = np.array(image)
        else:
            img_np = image
        
        original_size = img_np.shape[:2]
        
        logger.info("Starting post-processing pipeline...")
        start_time = time.time()
        
        # Step 1: Denoise if needed
        if config.denoise_strength > 0:
            img_np = self._denoise(img_np, config.denoise_strength)
        
        # Step 2: Face restoration (before upscaling for speed)
        if config.face_restoration and self.face_restorer:
            img_np = self._restore_faces(img_np)
        
        # Step 3: Upscaling
        if config.upscale_factor > 1.0:
            img_np = self._upscale(img_np, config.upscale_factor)
        
        # Step 4: Detail enhancement
        if config.detail_enhancement and self.detail_enhancer:
            img_np = self.detail_enhancer(img_np, strength=config.sharpening)
        
        # Step 5: Color correction
        if config.color_correction:
            img_pil = Image.fromarray(img_np)
            img_pil = self._color_correct(img_pil, config)
            img_np = np.array(img_pil)
        
        # Convert back to PIL
        result = Image.fromarray(img_np)
        
        processing_time = time.time() - start_time
        logger.info(f"Post-processing completed in {processing_time:.2f}s")
        logger.info(f"Size: {original_size} -> {result.size}")
        
        return result
    
    def _denoise(self, img: np.ndarray, strength: float) -> np.ndarray:
        """Apply denoising."""
        # Non-local means denoising
        denoised = cv2.fastNlMeansDenoisingColored(
            img,
            None,
            h=10 * strength,
            hColor=10 * strength,
            templateWindowSize=7,
            searchWindowSize=21
        )
        return denoised
    
    def _restore_faces(self, img: np.ndarray) -> np.ndarray:
        """Restore faces in the image."""
        if self.face_restorer is None:
            return img
        
        try:
            # GFPGAN expects BGR
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # Restore faces
            _, _, restored_img = self.face_restorer.enhance(
                img_bgr,
                has_aligned=False,
                only_center_face=False,
                paste_back=True,
                weight=0.5
            )
            
            # Convert back to RGB
            restored_img = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
            
            return restored_img
            
        except Exception as e:
            logger.warning(f"Face restoration failed: {e}")
            return img
    
    def _upscale(self, img: np.ndarray, factor: float) -> np.ndarray:
        """Upscale image."""
        if self.upscaler is None or callable(self.upscaler):
            # Use fallback
            return self._fallback_upscale(img, factor)
        
        try:
            # Real-ESRGAN expects BGR
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # Upscale
            output, _ = self.upscaler.enhance(img_bgr, outscale=factor)
            
            # Convert back to RGB
            output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            
            return output
            
        except Exception as e:
            logger.warning(f"Upscaling failed: {e}, using fallback")
            return self._fallback_upscale(img, factor)
    
    def _color_correct(self, img: Image.Image, config: EnhancementConfig) -> Image.Image:
        """Apply color corrections."""
        # Brightness
        if config.contrast != 1.0:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(config.contrast)
        
        # Saturation
        if config.saturation != 1.0:
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(config.saturation)
        
        # Sharpness
        if config.sharpening > 0:
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.0 + config.sharpening)
        
        return img
    
    def batch_enhance(self,
                     images: List[Image.Image],
                     config: Optional[EnhancementConfig] = None,
                     parallel: bool = True) -> List[Image.Image]:
        """Enhance multiple images."""
        
        if parallel and len(images) > 1:
            # Could use multiprocessing here
            # For now, sequential
            pass
        
        enhanced_images = []
        for i, img in enumerate(images):
            logger.info(f"Enhancing image {i+1}/{len(images)}...")
            enhanced = self.enhance_image(img, config)
            enhanced_images.append(enhanced)
        
        return enhanced_images


class StyleEnhancer:
    """Style-specific enhancements for different image types."""
    
    STYLE_PRESETS = {
        "photorealistic": EnhancementConfig(
            upscale_factor=2.0,
            face_restoration=True,
            detail_enhancement=True,
            color_correction=True,
            sharpening=0.3,
            contrast=1.05,
            saturation=1.0,
            denoise_strength=0.2
        ),
        "artistic": EnhancementConfig(
            upscale_factor=2.0,
            face_restoration=False,
            detail_enhancement=True,
            color_correction=True,
            sharpening=0.5,
            contrast=1.2,
            saturation=1.3,
            denoise_strength=0.1
        ),
        "anime": EnhancementConfig(
            upscale_factor=2.0,
            face_restoration=False,
            detail_enhancement=True,
            color_correction=True,
            sharpening=0.7,
            contrast=1.15,
            saturation=1.4,
            denoise_strength=0.0
        ),
        "3d_render": EnhancementConfig(
            upscale_factor=2.0,
            face_restoration=False,
            detail_enhancement=True,
            color_correction=False,
            sharpening=0.8,
            contrast=1.0,
            saturation=1.0,
            denoise_strength=0.0
        ),
        "product": EnhancementConfig(
            upscale_factor=2.0,
            face_restoration=False,
            detail_enhancement=True,
            color_correction=True,
            sharpening=0.6,
            contrast=1.1,
            saturation=0.95,
            denoise_strength=0.3
        )
    }
    
    @classmethod
    def get_preset(cls, style: str) -> EnhancementConfig:
        """Get enhancement preset for style."""
        return cls.STYLE_PRESETS.get(style, cls.STYLE_PRESETS["photorealistic"])
    
    @classmethod
    def auto_detect_style(cls, image: Image.Image, prompt: Optional[str] = None) -> str:
        """Auto-detect image style for appropriate enhancement."""
        # Simple heuristic based on image characteristics
        img_array = np.array(image)
        
        # Check color distribution
        std_dev = np.std(img_array)
        mean_saturation = cls._calculate_saturation(img_array)
        
        # Check prompt hints
        if prompt:
            prompt_lower = prompt.lower()
            if any(word in prompt_lower for word in ["photo", "realistic", "real"]):
                return "photorealistic"
            elif any(word in prompt_lower for word in ["anime", "manga", "cartoon"]):
                return "anime"
            elif any(word in prompt_lower for word in ["3d", "render", "cgi"]):
                return "3d_render"
            elif any(word in prompt_lower for word in ["product", "commercial"]):
                return "product"
        
        # Fallback to image analysis
        if mean_saturation > 0.7 and std_dev > 50:
            return "artistic"
        elif mean_saturation > 0.8:
            return "anime"
        elif std_dev < 30:
            return "product"
        else:
            return "photorealistic"
    
    @staticmethod
    def _calculate_saturation(img_array: np.ndarray) -> float:
        """Calculate mean saturation of image."""
        # Convert to HSV
        img_hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # Get saturation channel
        saturation = img_hsv[:, :, 1]
        
        # Calculate mean (normalized to 0-1)
        mean_sat = np.mean(saturation) / 255.0
        
        return mean_sat


class CompositeEnhancer:
    """Combine multiple enhancement techniques."""
    
    def __init__(self, pipeline: PostProcessingPipeline):
        self.pipeline = pipeline
    
    def progressive_upscale(self,
                           image: Image.Image,
                           target_scale: float = 4.0,
                           stages: int = 2) -> Image.Image:
        """Progressive upscaling for better quality."""
        
        if stages < 1:
            stages = 1
        
        scale_per_stage = target_scale ** (1.0 / stages)
        
        result = image
        for stage in range(stages):
            logger.info(f"Progressive upscale stage {stage + 1}/{stages}")
            
            config = EnhancementConfig(
                upscale_factor=scale_per_stage,
                face_restoration=(stage == stages - 1),  # Only on last stage
                detail_enhancement=True,
                sharpening=0.3 * (stage + 1) / stages  # Increase sharpening
            )
            
            result = self.pipeline.enhance_image(result, config)
        
        return result
    
    def smart_enhance(self,
                     image: Image.Image,
                     prompt: Optional[str] = None,
                     auto_detect: bool = True) -> Image.Image:
        """Smart enhancement with auto-detection."""
        
        if auto_detect:
            style = StyleEnhancer.auto_detect_style(image, prompt)
            logger.info(f"Auto-detected style: {style}")
            config = StyleEnhancer.get_preset(style)
        else:
            config = EnhancementConfig()
        
        return self.pipeline.enhance_image(image, config)
    
    def prepare_for_3d(self, image: Image.Image) -> Image.Image:
        """Prepare image for 3D reconstruction."""
        
        # Specific enhancements for 3D
        config = EnhancementConfig(
            upscale_factor=1.0,  # Don't upscale
            face_restoration=False,
            detail_enhancement=True,
            color_correction=False,
            sharpening=0.8,  # High sharpening for edges
            contrast=1.2,    # Increase contrast
            saturation=0.8,  # Reduce saturation
            denoise_strength=0.5  # Strong denoising
        )
        
        enhanced = self.pipeline.enhance_image(image, config)
        
        # Additional processing for 3D
        # Center crop to square
        width, height = enhanced.size
        min_dim = min(width, height)
        left = (width - min_dim) // 2
        top = (height - min_dim) // 2
        enhanced = enhanced.crop((left, top, left + min_dim, top + min_dim))
        
        return enhanced


# Utility functions
def compare_enhancement_methods(image: Image.Image) -> Dict[str, Image.Image]:
    """Compare different enhancement methods."""
    pipeline = PostProcessingPipeline()
    
    results = {
        "original": image
    }
    
    # Try different presets
    for style_name in ["photorealistic", "artistic", "anime"]:
        config = StyleEnhancer.get_preset(style_name)
        results[style_name] = pipeline.enhance_image(image, config)
    
    # Try progressive upscale
    composite = CompositeEnhancer(pipeline)
    results["progressive_4x"] = composite.progressive_upscale(image, 4.0, stages=2)
    
    return results


def create_enhancement_report(image: Image.Image, enhanced: Image.Image) -> Dict[str, Any]:
    """Create detailed enhancement report."""
    
    # Calculate metrics
    original_size = image.size
    enhanced_size = enhanced.size
    
    # Convert to numpy for analysis
    orig_np = np.array(image)
    enh_np = np.array(enhanced)
    
    # Calculate statistics
    report = {
        "size_change": {
            "original": original_size,
            "enhanced": enhanced_size,
            "scale_factor": enhanced_size[0] / original_size[0]
        },
        "quality_metrics": {
            "sharpness_increase": calculate_sharpness(enh_np) / calculate_sharpness(orig_np),
            "detail_increase": calculate_detail_level(enh_np) / calculate_detail_level(orig_np),
            "color_vibrancy": calculate_vibrancy(enh_np) / calculate_vibrancy(orig_np)
        },
        "file_size": {
            "original_kb": image.tell() / 1024 if hasattr(image, 'tell') else 0,
            "enhanced_kb": enhanced.tell() / 1024 if hasattr(enhanced, 'tell') else 0
        }
    }
    
    return report


def calculate_sharpness(img: np.ndarray) -> float:
    """Calculate image sharpness metric."""
    # Laplacian variance method
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return laplacian.var()


def calculate_detail_level(img: np.ndarray) -> float:
    """Calculate detail level metric."""
    # High-pass filter response
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return np.mean(edges)


def calculate_vibrancy(img: np.ndarray) -> float:
    """Calculate color vibrancy metric."""
    # Saturation in HSV space
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    return np.mean(hsv[:, :, 1])