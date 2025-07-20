"""Hunyuan3D Model Wrapper for integration with the app"""

import os
import sys
import torch
import logging
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

class Hunyuan3DModelWrapper:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = Path(model_path)
        self.device = device
        self.shape_pipeline = None
        self.texture_pipeline = None
        self.model_loaded = False
        
        # Add Hunyuan3D to Python path
        hunyuan3d_path = Path(__file__).parent.parent.parent.parent / "Hunyuan3D"
        if hunyuan3d_path.exists():
            sys.path.insert(0, str(hunyuan3d_path))
            # Add subdirectories
            sys.path.insert(0, str(hunyuan3d_path / "hy3dshape"))
            sys.path.insert(0, str(hunyuan3d_path / "hy3dpaint"))
            logger.info(f"Added Hunyuan3D paths: {hunyuan3d_path}")
        
    def load_model(self) -> bool:
        """Load both shape and texture generation pipelines"""
        try:
            logger.info(f"Loading Hunyuan3D model from {self.model_path}")
            
            # Load shape generation pipeline
            shape_success = self._load_shape_pipeline()
            
            # Load texture generation pipeline (optional)
            texture_success = self._load_texture_pipeline()
            
            self.model_loaded = shape_success
            
            if shape_success:
                logger.info("Hunyuan3D model loaded successfully with shape pipeline")
            else:
                logger.error("Failed to load shape pipeline")
                
            return self.model_loaded
            
        except Exception as e:
            logger.error(f"Failed to load Hunyuan3D model: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def _load_shape_pipeline(self) -> bool:
        """Load the shape generation pipeline"""
        try:
            # Import Hunyuan3D pipeline
            logger.info("Attempting to import Hunyuan3D shape pipeline...")
            
            # Add Hunyuan3D paths to sys.path if needed
            import sys
            hunyuan_path = Path(__file__).parent.parent.parent.parent / "Hunyuan3D"
            hy3dshape_path = hunyuan_path / "hy3dshape"
            
            if str(hy3dshape_path) not in sys.path:
                sys.path.insert(0, str(hy3dshape_path))
                logger.info(f"Added {hy3dshape_path} to sys.path")
            
            # Now import
            from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
            logger.info("Successfully imported Hunyuan3DDiTFlowMatchingPipeline")
            
            # Find model checkpoint
            model_dir = self.model_path
            
            # Look for HuggingFace cache structure
            if not (model_dir / "hunyuan3d-dit-v2-1").exists():
                # Try to find in HuggingFace cache structure
                for subdir in model_dir.rglob("models--tencent--Hunyuan3D-*"):
                    snapshots = subdir / "snapshots"
                    if snapshots.exists():
                        snapshot_dirs = [d for d in snapshots.iterdir() if d.is_dir()]
                        if snapshot_dirs:
                            model_dir = snapshot_dirs[0]
                            logger.info(f"Found model in HuggingFace cache: {model_dir}")
                            break
            
            # Try to load from pretrained
            try:
                logger.info(f"Loading shape pipeline from {model_dir}")
                
                # Check if we have the checkpoint file
                ckpt_path = model_dir / "hunyuan3d-dit-v2-1" / "model.fp16.ckpt"
                if not ckpt_path.exists():
                    ckpt_path = model_dir / "hunyuan3d-dit-v2-1" / "model.ckpt"
                
                if ckpt_path.exists():
                    # Load using from_single_file method
                    config_path = model_dir / "hunyuan3d-dit-v2-1" / "config.yaml"
                    if config_path.exists():
                        logger.info(f"Loading with from_single_file: ckpt={ckpt_path}, config={config_path}")
                        self.shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_single_file(
                            ckpt_path=str(ckpt_path),
                            config_path=str(config_path),
                            device=self.device,
                            dtype=torch.float16 if "fp16" in str(ckpt_path) else torch.float32
                        )
                    else:
                        logger.error(f"Config file not found at {config_path}")
                        raise FileNotFoundError(f"Config file not found: {config_path}")
                else:
                    logger.error(f"Checkpoint file not found at {ckpt_path}")
                    raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
                
                if self.shape_pipeline is not None:
                    logger.info("Shape generation pipeline loaded successfully")
                    return True
                else:
                    logger.error("Shape pipeline is None after loading")
                    return False
                    
            except Exception as e:
                logger.error(f"Failed to load shape pipeline: {e}")
                logger.error(traceback.format_exc())
                return False
                
        except ImportError as e:
            logger.error(f"Failed to import Hunyuan3D modules: {e}")
            logger.error(f"Import error details: {traceback.format_exc()}")
            logger.error("Make sure Hunyuan3D repository is cloned and dependencies are installed")
            logger.info("The hy3dshape module should be in the Hunyuan3D directory")
            logger.info("Try running: pip install -e ./Hunyuan3D")
            return False
        except Exception as e:
            logger.error(f"Error loading shape pipeline: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def _load_texture_pipeline(self) -> bool:
        """Load the texture generation pipeline (optional)"""
        try:
            # Check if xatlas is available for texture unwrapping
            try:
                import xatlas
                logger.info("xatlas available for texture generation")
            except ImportError:
                logger.warning("xatlas not available, texture generation will be limited")
                return False
            
            # Try to load texture pipeline
            try:
                # Add hy3dpaint to path if needed
                import sys
                hunyuan_path = Path(__file__).parent.parent.parent.parent / "Hunyuan3D"
                hy3dpaint_path = hunyuan_path / "hy3dpaint"
                
                if str(hy3dpaint_path) not in sys.path:
                    sys.path.insert(0, str(hy3dpaint_path))
                    logger.info(f"Added {hy3dpaint_path} to sys.path")
                
                from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
                
                # Configure texture pipeline
                max_num_view = 6
                resolution = 512
                conf = Hunyuan3DPaintConfig(max_num_view, resolution)
                
                # Set paths
                hunyuan3d_path = Path(__file__).parent.parent.parent.parent / "Hunyuan3D"
                conf.realesrgan_ckpt_path = str(hunyuan3d_path / "hy3dpaint/ckpt/RealESRGAN_x4plus.pth")
                conf.multiview_cfg_path = str(hunyuan3d_path / "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml")
                conf.custom_pipeline = str(hunyuan3d_path / "hy3dpaint/hunyuanpaintpbr")
                
                self.texture_pipeline = Hunyuan3DPaintPipeline(conf)
                logger.info("Texture generation pipeline loaded successfully")
                return True
                
            except Exception as e:
                logger.warning(f"Failed to load texture pipeline: {e}")
                self.texture_pipeline = None
                return False
            
        except Exception as e:
            logger.warning(f"Texture generation pipeline not available: {e}")
            return False
    
    def has_shape_pipeline(self) -> bool:
        """Check if shape pipeline is loaded and ready"""
        return self.shape_pipeline is not None
    
    def has_texture_pipeline(self) -> bool:
        """Check if texture pipeline is loaded and ready"""
        return self.texture_pipeline is not None
    
    def generate_3d_from_image(self, image: Image.Image, **kwargs) -> Dict[str, Any]:
        """Generate 3D model from input image using the shape pipeline"""
        logger.info(f"[HUNYUAN3D_WRAPPER.generate_3d_from_image] Called")
        logger.info(f"[HUNYUAN3D_WRAPPER.generate_3d_from_image] Input image: size={image.size}, mode={image.mode}")
        logger.info(f"[HUNYUAN3D_WRAPPER.generate_3d_from_image] shape_pipeline type: {type(self.shape_pipeline)}")
        
        if not self.has_shape_pipeline():
            raise RuntimeError("Shape pipeline not loaded or not ready")
        
        try:
            logger.info("[HUNYUAN3D_WRAPPER.generate_3d_from_image] Starting 3D generation with Hunyuan3D")
            
            # Preprocess image
            processed_image = self._preprocess_image(image)
            logger.info(f"[HUNYUAN3D_WRAPPER.generate_3d_from_image] Preprocessed image: size={processed_image.size}, mode={processed_image.mode}")
            
            # Generate 3D shape using the pipeline
            logger.info("[HUNYUAN3D_WRAPPER.generate_3d_from_image] Calling shape pipeline...")
            
            # The pipeline expects PIL Image and returns trimesh
            with torch.no_grad():
                logger.info(f"[HUNYUAN3D_WRAPPER.generate_3d_from_image] Calling pipeline with:")
                logger.info(f"  - num_inference_steps: {kwargs.get('num_inference_steps', 50)}")
                logger.info(f"  - guidance_scale: {kwargs.get('guidance_scale', 7.5)}")
                logger.info(f"  - seed: {kwargs.get('seed', None)}")
                
                mesh_outputs = self.shape_pipeline(
                    image=processed_image,
                    num_inference_steps=kwargs.get('num_inference_steps', 50),
                    guidance_scale=kwargs.get('guidance_scale', 7.5),
                    seed=kwargs.get('seed', None)
                )
                
                logger.info(f"[HUNYUAN3D_WRAPPER.generate_3d_from_image] Pipeline returned: {type(mesh_outputs)}")
            
            # Handle output
            if isinstance(mesh_outputs, list) and len(mesh_outputs) > 0:
                mesh = mesh_outputs[0]
            else:
                mesh = mesh_outputs
            
            if mesh is None:
                logger.error("Pipeline returned None")
                return {'success': False, 'error': 'Pipeline returned no mesh'}
            
            # Extract mesh data
            vertices = mesh.vertices
            faces = mesh.faces
            
            logger.info(f"[HUNYUAN3D_WRAPPER.generate_3d_from_image] Generated mesh with {len(vertices)} vertices and {len(faces)} faces")
            logger.info(f"[HUNYUAN3D_WRAPPER.generate_3d_from_image] Vertices shape: {vertices.shape}")
            logger.info(f"[HUNYUAN3D_WRAPPER.generate_3d_from_image] Faces shape: {faces.shape}")
            
            return {
                'mesh': mesh,  # Return the trimesh object directly
                'vertices': vertices,
                'faces': faces,
                'success': True,
                'has_texture': False  # Shape pipeline doesn't generate textures
            }
                
        except Exception as e:
            logger.error(f"3D generation failed: {e}")
            logger.error(traceback.format_exc())
            return {'success': False, 'error': str(e)}
    
    def generate_mesh(self, image: Image.Image, progress_callback=None) -> Any:
        """Generate 3D mesh from a single image (compatibility method)"""
        logger.info(f"[HUNYUAN3D_WRAPPER.generate_mesh] Called with image: {image.size if image else 'None'}")
        logger.info(f"[HUNYUAN3D_WRAPPER.generate_mesh] Image mode: {image.mode if image else 'None'}")
        
        if progress_callback:
            progress_callback(0.1, "Preprocessing image...")
        
        result = self.generate_3d_from_image(image)
        logger.info(f"[HUNYUAN3D_WRAPPER.generate_mesh] Result from generate_3d_from_image: success={result.get('success')}, has_mesh={'mesh' in result}")
        
        if result['success']:
            if progress_callback:
                progress_callback(1.0, "Mesh generation complete!")
            return result['mesh']
        else:
            raise RuntimeError(f"Mesh generation failed: {result['error']}")
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for Hunyuan3D input"""
        logger.info(f"[HUNYUAN3D_WRAPPER._preprocess_image] Input: size={image.size}, mode={image.mode}")
        
        # Ensure RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')
            logger.info(f"[HUNYUAN3D_WRAPPER._preprocess_image] Converted to RGB")
        
        # Resize to expected input size
        target_size = (512, 512)  # Hunyuan3D input size
        if image.size != target_size:
            image = image.resize(target_size, Image.Resampling.LANCZOS)
            logger.info(f"[HUNYUAN3D_WRAPPER._preprocess_image] Resized to {target_size}")
        
        logger.info(f"[HUNYUAN3D_WRAPPER._preprocess_image] Output: size={image.size}, mode={image.mode}")
        return image

    # Legacy compatibility methods
    def load(self, progress_callback=None) -> bool:
        """Legacy load method for compatibility"""
        return self.load_model()
    
    @property
    def loaded(self):
        """Legacy loaded property for compatibility"""
        return self.model_loaded


def load_hunyuan3d_model(model_path: Path, version: str, device: str = "cuda", 
                        progress_callback=None) -> Tuple[Optional[Hunyuan3DModelWrapper], str]:
    """Load a Hunyuan3D model with the proper wrapper
    
    Args:
        model_path: Path to model files
        version: Model version (e.g., "2.1", "2.0")
        device: Device to load on
        progress_callback: Optional progress callback
        
    Returns:
        Tuple of (model_wrapper, status_message)
    """
    try:
        logger.info(f"Loading Hunyuan3D model version {version} from {model_path}")
        
        # Create wrapper
        wrapper = Hunyuan3DModelWrapper(str(model_path), device)
        
        # Load model
        if progress_callback:
            progress_callback(0.5, "Loading Hunyuan3D model components...")
        
        success = wrapper.load_model()
        
        if success and wrapper.has_shape_pipeline():
            return wrapper, f"Hunyuan3D {version} loaded successfully with full inference support"
        else:
            return None, "Failed to load Hunyuan3D model components"
            
    except Exception as e:
        logger.error(f"Failed to create Hunyuan3D wrapper: {e}")
        logger.error(traceback.format_exc())
        return None, f"Failed to load Hunyuan3D model: {str(e)}"