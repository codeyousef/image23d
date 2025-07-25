"""Intermediate processing components for 3D generation

Implements the processing steps defined in the 3D Implementation Guide:
- Depth estimation
- Normal map generation  
- UV unwrapping
- Texture synthesis
- PBR material generation
"""

import logging
from typing import List, Dict, Optional, Tuple, Any
from io import BytesIO
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
import trimesh

from .base import IntermediateProcessor, IntermediateFormat

logger = logging.getLogger(__name__)


class DepthEstimator(IntermediateProcessor):
    """Estimate depth maps from images using MiDaS or DPT"""
    
    def __init__(self, model_type: str = "MiDaS", device: str = "cuda"):
        super().__init__(device)
        self.model_type = model_type
        self.model = None
        self.transform = None
        
    def load_model(self):
        """Load depth estimation model"""
        if self.model is not None:
            return
            
        try:
            if self.model_type == "MiDaS":
                # Use MiDaS for depth estimation
                from transformers import AutoImageProcessor, AutoModelForDepthEstimation
                
                model_id = "Intel/dpt-large"
                self.transform = AutoImageProcessor.from_pretrained(model_id)
                self.model = AutoModelForDepthEstimation.from_pretrained(model_id)
                self.model.to(self.device)
                self.model.eval()
                
            logger.info(f"Loaded {self.model_type} depth estimation model")
            
        except Exception as e:
            logger.error(f"Failed to load depth model: {e}")
            # Fallback to simple depth estimation
            self.model = None
            
    def process(
        self,
        image: Image.Image,
        normalize: bool = True,
        colormap: bool = False,
        **kwargs
    ) -> np.ndarray:
        """Estimate depth map from image"""
        
        # Load model if needed
        if self.model is None:
            self.load_model()
            
        if self.model is None:
            # Fallback: create simple gradient depth
            return self._fallback_depth(image)
            
        try:
            # Prepare image
            inputs = self.transform(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                predicted_depth = outputs.predicted_depth
                
            # Interpolate to original size
            prediction = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=image.size[::-1],
                mode="bicubic",
                align_corners=False,
            )
            
            # Convert to numpy
            depth = prediction.squeeze().cpu().numpy()
            
            # Normalize if requested
            if normalize:
                depth = (depth - depth.min()) / (depth.max() - depth.min())
                
            # Apply colormap if requested
            if colormap:
                depth_uint8 = (depth * 255).astype(np.uint8)
                depth = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_VIRIDIS)
                
            return depth
            
        except Exception as e:
            logger.error(f"Depth estimation failed: {e}")
            return self._fallback_depth(image)
            
    def _fallback_depth(self, image: Image.Image) -> np.ndarray:
        """Simple fallback depth estimation"""
        # Convert to grayscale and use as rough depth
        gray = np.array(image.convert('L'))
        # Invert so brighter = closer
        depth = 1.0 - (gray / 255.0)
        return depth
        
    def estimate_multiview_depth(
        self,
        views: List[Image.Image],
        **kwargs
    ) -> List[np.ndarray]:
        """Estimate depth for multiple views"""
        depths = []
        for view in views:
            depth = self.process(view, **kwargs)
            depths.append(depth)
        return depths


class NormalEstimator(IntermediateProcessor):
    """Estimate normal maps from images or depth maps"""
    
    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        
    def process(
        self,
        input_data: Any,
        mode: str = "from_depth",
        **kwargs
    ) -> np.ndarray:
        """Estimate normal map
        
        Args:
            input_data: Either Image (for direct estimation) or depth map
            mode: "from_depth" or "from_image"
        """
        
        if mode == "from_depth":
            return self._normals_from_depth(input_data)
        else:
            return self._normals_from_image(input_data)
            
    def _normals_from_depth(self, depth: np.ndarray) -> np.ndarray:
        """Compute normal map from depth map"""
        
        # Ensure float
        if depth.dtype != np.float32:
            depth = depth.astype(np.float32)
            
        # Compute gradients
        zy, zx = np.gradient(depth)
        
        # Construct normal vectors
        normal = np.dstack((-zx, -zy, np.ones_like(depth)))
        
        # Normalize
        n = np.linalg.norm(normal, axis=2, keepdims=True)
        normal = normal / (n + 1e-8)
        
        # Convert to 0-255 range
        normal = (normal + 1.0) * 127.5
        normal = np.clip(normal, 0, 255).astype(np.uint8)
        
        return normal
        
    def _normals_from_image(self, image: Image.Image) -> np.ndarray:
        """Estimate normals directly from image"""
        
        # Convert to grayscale
        gray = np.array(image.convert('L')).astype(np.float32) / 255.0
        
        # Apply Sobel filters
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        
        # Create normal map
        normal = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.float32)
        normal[:, :, 0] = -sobel_x
        normal[:, :, 1] = -sobel_y
        normal[:, :, 2] = 1.0
        
        # Normalize
        norm = np.linalg.norm(normal, axis=2, keepdims=True)
        normal = normal / (norm + 1e-8)
        
        # Convert to image format
        normal = (normal + 1.0) * 127.5
        normal = np.clip(normal, 0, 255).astype(np.uint8)
        
        return normal


class UVUnwrapper(IntermediateProcessor):
    """UV unwrapping for 3D meshes"""
    
    def __init__(self, method: str = "smart", device: str = "cuda"):
        super().__init__(device)
        self.method = method
        
    def process(
        self,
        mesh: trimesh.Trimesh,
        texture_size: int = 1024,
        **kwargs
    ) -> Dict[str, Any]:
        """Unwrap mesh UVs
        
        Returns:
            Dict with UV coordinates and atlas layout
        """
        
        try:
            # Check if mesh already has UVs
            if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'uv'):
                logger.info("Mesh already has UV coordinates")
                return {
                    "uv": mesh.visual.uv,
                    "faces": mesh.faces,
                    "texture_size": texture_size
                }
                
            # Use xatlas for UV unwrapping if available
            try:
                import xatlas
                
                # Prepare mesh data
                vertices = mesh.vertices.astype(np.float32)
                faces = mesh.faces.astype(np.uint32)
                
                # Create atlas
                atlas = xatlas.Atlas()
                atlas.add_mesh(vertices, faces)
                
                # Generate with options
                options = xatlas.PackOptions()
                options.resolution = texture_size
                options.padding = 4
                
                atlas.generate(pack_options=options)
                
                # Get the output
                vmapping, indices, uvs = atlas.get_mesh(0)
                
                return {
                    "uv": uvs,
                    "faces": indices.reshape(-1, 3),
                    "vmapping": vmapping,
                    "texture_size": texture_size
                }
                
            except ImportError:
                logger.warning("xatlas not available, using simple unwrapping")
                return self._simple_unwrap(mesh, texture_size)
                
        except Exception as e:
            logger.error(f"UV unwrapping failed: {e}")
            return self._simple_unwrap(mesh, texture_size)
            
    def _simple_unwrap(self, mesh: trimesh.Trimesh, texture_size: int) -> Dict[str, Any]:
        """Simple UV unwrapping fallback using spherical mapping"""
        
        try:
            vertices = mesh.vertices
            faces = mesh.faces
            
            # Validate mesh
            if len(vertices) == 0:
                logger.warning("Mesh has no vertices, creating default UV mapping")
                return {
                    "uv": np.array([[0.5, 0.5]]),
                    "faces": np.array([[0, 0, 0]]),
                    "texture_size": texture_size
                }
            
            # Create UV coordinates using spherical mapping for better coverage
            uv = np.zeros((len(vertices), 2))
            
            # Normalize vertices to unit sphere
            center = vertices.mean(axis=0)
            centered = vertices - center
            distances = np.linalg.norm(centered, axis=1, keepdims=True)
            distances[distances == 0] = 1  # Avoid division by zero
            normalized = centered / distances
            
            # Spherical mapping with better handling of poles
            # U coordinate from azimuth angle
            uv[:, 0] = 0.5 + np.arctan2(normalized[:, 2], normalized[:, 0]) / (2 * np.pi)
            
            # V coordinate from elevation angle with clamping
            # Clamp Y values to avoid arcsin domain errors
            y_clamped = np.clip(normalized[:, 1], -0.999, 0.999)
            uv[:, 1] = 0.5 + np.arcsin(y_clamped) / np.pi
            
            # Fix seam issues by wrapping U coordinates
            # Detect vertices on the seam (where U jumps from ~1 to ~0)
            for face in faces:
                u_values = uv[face, 0]
                if np.max(u_values) - np.min(u_values) > 0.5:
                    # Face crosses the seam, adjust U coordinates
                    for i, vertex_idx in enumerate(face):
                        if uv[vertex_idx, 0] < 0.25:
                            # Create a duplicate vertex with wrapped U
                            new_vertex_idx = len(uv)
                            new_uv = uv[vertex_idx].copy()
                            new_uv[0] += 1.0
                            uv = np.vstack([uv, new_uv.reshape(1, -1)])
                            # Update face to use new vertex
                            face[i] = new_vertex_idx
            
            # Ensure UV coordinates are in [0, 1] range
            uv = np.clip(uv, 0, 1)
            
            logger.info(f"Simple UV unwrap complete: {len(uv)} UV coordinates for {len(vertices)} vertices")
            
            return {
                "uv": uv,
                "faces": faces,
                "texture_size": texture_size
            }
            
        except Exception as e:
            logger.error(f"Error in simple UV unwrapping: {e}")
            # Return basic planar mapping as last resort
            vertices = mesh.vertices
            if len(vertices) > 0:
                # Simple planar projection
                uv = vertices[:, :2] - vertices[:, :2].min(axis=0)
                uv = uv / (uv.max() + 1e-8)
            else:
                uv = np.array([[0.5, 0.5]])
                
            return {
                "uv": uv,
                "faces": mesh.faces,
                "texture_size": texture_size
            }


class TextureSynthesizer(IntermediateProcessor):
    """Synthesize textures from multiple views"""
    
    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        
    def process(
        self,
        mesh: trimesh.Trimesh,
        views: List[Image.Image],
        uv_data: Dict[str, Any],
        resolution: int = 1024,
        **kwargs
    ) -> Image.Image:
        """Synthesize texture from multiple views
        
        Args:
            mesh: 3D mesh
            views: Multiple view images
            uv_data: UV unwrapping data
            resolution: Output texture resolution
        """
        
        # Create texture atlas with better default color
        texture = Image.new('RGB', (resolution, resolution), color=(200, 200, 200))  # Light gray instead of dark
        
        try:
            # Get UV coordinates
            uv = uv_data.get("uv", None)
            if uv is None:
                logger.error("No UV coordinates provided")
                return texture
                
            # Simple texture projection
            # In practice, this would use view-dependent texture mapping
            if views and len(views) > 0:
                # Use first view as base texture
                base_view = views[0]
                
                # Ensure proper color mode
                if base_view.mode == 'RGBA':
                    # Convert RGBA to RGB with white background
                    background = Image.new('RGB', base_view.size, (255, 255, 255))
                    background.paste(base_view, mask=base_view.split()[3] if base_view.mode == 'RGBA' else None)
                    base_view = background
                elif base_view.mode != 'RGB':
                    base_view = base_view.convert('RGB')
                    
                # Resize to texture resolution
                texture = base_view.resize((resolution, resolution), Image.Resampling.LANCZOS)
                
            # TODO: Implement proper multi-view texture synthesis
            # This would involve:
            # 1. Camera pose estimation for each view
            # 2. Visibility computation
            # 3. View blending based on normal angles
            # 4. Seam reduction
            
            return texture
            
        except Exception as e:
            logger.error(f"Texture synthesis failed: {e}")
            return texture
            
    def blend_textures(
        self,
        textures: List[Image.Image],
        weights: Optional[List[float]] = None
    ) -> Image.Image:
        """Blend multiple textures"""
        
        if not textures:
            return Image.new('RGB', (1024, 1024))
            
        if weights is None:
            weights = [1.0 / len(textures)] * len(textures)
            
        # Convert to numpy for blending
        arrays = [np.array(tex).astype(np.float32) for tex in textures]
        
        # Weighted blend
        result = np.zeros_like(arrays[0])
        for arr, weight in zip(arrays, weights):
            result += arr * weight
            
        # Convert back to image
        result = np.clip(result, 0, 255).astype(np.uint8)
        return Image.fromarray(result)


class PBRMaterialGenerator(IntermediateProcessor):
    """Generate PBR material maps"""
    
    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        
    def process(
        self,
        base_texture: Image.Image,
        mesh: Optional[trimesh.Trimesh] = None,
        **kwargs
    ) -> Dict[str, Image.Image]:
        """Generate PBR maps from base texture
        
        Returns:
            Dict containing:
            - albedo: Base color map
            - normal: Normal map
            - metallic: Metallic map
            - roughness: Roughness map
            - ao: Ambient occlusion map
        """
        
        size = base_texture.size
        
        # Albedo is the base texture
        albedo = base_texture
        
        # Generate normal map from texture
        normal_estimator = NormalEstimator(self.device)
        normal_array = normal_estimator.process(base_texture, mode="from_image")
        normal = Image.fromarray(normal_array)
        
        # Generate simple metallic map (non-metallic for now)
        metallic = Image.new('L', size, color=0)
        
        # Generate roughness map from texture variance
        roughness = self._estimate_roughness(base_texture)
        
        # Generate simple AO map
        ao = self._estimate_ao(base_texture, mesh)
        
        return {
            "albedo": albedo,
            "normal": normal,
            "metallic": metallic,
            "roughness": roughness,
            "ao": ao
        }
        
    def _estimate_roughness(self, texture: Image.Image) -> Image.Image:
        """Estimate roughness from texture"""
        
        # Convert to grayscale
        gray = np.array(texture.convert('L')).astype(np.float32)
        
        # Compute local variance as roughness indicator
        kernel_size = 5
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
        
        # Compute local mean
        mean = cv2.filter2D(gray, -1, kernel)
        
        # Compute local variance
        sqr_mean = cv2.filter2D(gray ** 2, -1, kernel)
        variance = sqr_mean - mean ** 2
        
        # Normalize and invert (high variance = rough)
        roughness = np.sqrt(variance)
        roughness = (roughness - roughness.min()) / (roughness.max() - roughness.min() + 1e-8)
        
        # Convert to grayscale image
        roughness = (roughness * 255).astype(np.uint8)
        
        return Image.fromarray(roughness, mode='L')
        
    def _estimate_ao(
        self,
        texture: Image.Image,
        mesh: Optional[trimesh.Trimesh] = None
    ) -> Image.Image:
        """Estimate ambient occlusion"""
        
        # Simple AO estimation from texture darkness
        gray = np.array(texture.convert('L')).astype(np.float32) / 255.0
        
        # Invert and adjust contrast
        ao = 1.0 - gray
        ao = np.power(ao, 2.0)  # Increase contrast
        
        # Blur for smoother AO
        ao = cv2.GaussianBlur(ao, (15, 15), 0)
        
        # Normalize
        ao = 0.3 + 0.7 * ao  # Keep some ambient light
        
        ao = (ao * 255).astype(np.uint8)
        
        return Image.fromarray(ao, mode='L')


class IntermediateOutputHandler:
    """Handles saving and loading of intermediate outputs during 3D generation."""
    
    def __init__(self, output_dir: Optional[Any] = None):
        """Initialize the intermediate output handler.
        
        Args:
            output_dir: Directory to save intermediate outputs
        """
        from pathlib import Path
        
        if output_dir is None:
            output_dir = Path("outputs/intermediate")
        elif not isinstance(output_dir, Path):
            output_dir = Path(output_dir)
            
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def save_multiview_images(self, images: List[Image.Image], prefix: str = "view") -> List[str]:
        """Save multi-view images.
        
        Args:
            images: List of PIL images
            prefix: Filename prefix
            
        Returns:
            List of saved file paths
        """
        paths = []
        for i, img in enumerate(images):
            path = self.output_dir / f"{prefix}_{i:03d}.png"
            
            # Handle different image types
            if hasattr(img, 'save'):
                # PIL Image
                img.save(path)
            elif isinstance(img, np.ndarray):
                # NumPy array - convert to PIL
                if img.dtype != np.uint8:
                    # Normalize to 0-255 if needed
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)
                Image.fromarray(img).save(path)
            elif isinstance(img, torch.Tensor):
                # PyTorch tensor - convert to PIL
                img_np = img.detach().cpu().numpy()
                if img_np.ndim == 3 and img_np.shape[0] in [1, 3, 4]:
                    # CHW format - convert to HWC
                    img_np = np.transpose(img_np, (1, 2, 0))
                if img_np.ndim == 3 and img_np.shape[2] == 1:
                    # Single channel - squeeze
                    img_np = img_np.squeeze(2)
                # Normalize to 0-255
                if img_np.max() <= 1.0:
                    img_np = (img_np * 255).astype(np.uint8)
                else:
                    img_np = img_np.astype(np.uint8)
                Image.fromarray(img_np).save(path)
            elif hasattr(img, 'vertices') and hasattr(img, 'faces'):
                # Trimesh object - render it to an image
                logger.warning(f"Received Trimesh object instead of image. Attempting to render...")
                try:
                    # Try to render the mesh to an image
                    # This is a temporary fix - the proper solution is to use the correct pipeline
                    
                    # Create a scene with the mesh
                    scene = img.scene()
                    
                    # Set camera position for a good view
                    # Get mesh bounds for proper camera positioning
                    bounds = img.bounds
                    center = img.centroid
                    scale = (bounds[1] - bounds[0]).max()
                    
                    # Position camera to view the mesh
                    camera_distance = scale * 2.5
                    camera_pos = center + np.array([camera_distance, camera_distance, camera_distance])
                    
                    # Set camera transform to look at the mesh
                    scene.set_camera(center=center, distance=camera_distance)
                    
                    # Render with higher resolution
                    resolution = (512, 512)
                    try:
                        # Try using pyrender if available
                        rendered = scene.save_image(resolution=resolution, visible=False)
                        
                        # Check if rendered is bytes (PNG data)
                        if isinstance(rendered, bytes):
                            # Decode PNG bytes to numpy array
                            rendered_img = Image.open(BytesIO(rendered))
                            rendered = np.array(rendered_img)
                            logger.debug("Decoded PNG bytes to numpy array")
                    except Exception as e:
                        # Fallback: save as a simple wireframe representation
                        logger.warning(f"Advanced rendering failed: {e}, creating simple representation")
                        rendered = np.ones((512, 512, 3), dtype=np.uint8) * 128  # Gray background
                    
                    if rendered is not None and hasattr(rendered, 'shape') and len(rendered.shape) >= 2:
                        # Convert to PIL Image
                        if len(rendered.shape) == 3:
                            rendered_img = Image.fromarray(rendered.astype(np.uint8))
                        else:
                            # Convert grayscale to RGB
                            rendered_rgb = np.stack([rendered] * 3, axis=-1)
                            rendered_img = Image.fromarray(rendered_rgb.astype(np.uint8))
                        rendered_img.save(path)
                        logger.info(f"Successfully rendered Trimesh to image: {path}")
                    else:
                        logger.error(f"Failed to render Trimesh object - no image data")
                        continue
                except Exception as e:
                    logger.error(f"Failed to render Trimesh object: {e}")
                    # Create a placeholder image indicating this was a 3D mesh
                    try:
                        from PIL import ImageDraw, ImageFont
                        placeholder = Image.new('RGB', (512, 512), color='lightgray')
                        draw = ImageDraw.Draw(placeholder)
                        text = f"3D Mesh\n{img.vertices.shape[0]} vertices\n{img.faces.shape[0]} faces"
                        draw.text((256, 256), text, fill='black', anchor='mm')
                        placeholder.save(path)
                        logger.info(f"Created placeholder image for Trimesh: {path}")
                    except Exception as e2:
                        logger.error(f"Failed to create placeholder image: {e2}")
                        continue
            else:
                logger.error(f"Unknown image type: {type(img)}. Expected PIL Image, numpy array, torch tensor, or Trimesh object.")
                # Try to convert to string representation for debugging
                logger.debug(f"Image object: {img}")
                continue
                
            paths.append(str(path))
            
        logger.debug(f"Saved {len(paths)} multi-view images to {self.output_dir}")
        return paths
        
    def save_depth_maps(self, depth_maps: List[np.ndarray], prefix: str = "depth") -> List[str]:
        """Save depth maps.
        
        Args:
            depth_maps: List of depth arrays
            prefix: Filename prefix
            
        Returns:
            List of saved file paths
        """
        paths = []
        for i, depth in enumerate(depth_maps):
            path = self.output_dir / f"{prefix}_{i:03d}.npy"
            np.save(path, depth)
            paths.append(str(path))
        logger.debug(f"Saved {len(depth_maps)} depth maps to {self.output_dir}")
        return paths
        
    def save_mesh(self, mesh: trimesh.Trimesh, filename: str = "mesh.obj") -> str:
        """Save mesh to file.
        
        Args:
            mesh: Trimesh object
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        path = self.output_dir / filename
        mesh.export(path)
        logger.debug(f"Saved mesh to {path}")
        return str(path)
        
    def save_texture(self, texture: Image.Image, filename: str = "texture.png") -> str:
        """Save texture image.
        
        Args:
            texture: PIL Image
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        path = self.output_dir / filename
        texture.save(path)
        logger.debug(f"Saved texture to {path}")
        return str(path)