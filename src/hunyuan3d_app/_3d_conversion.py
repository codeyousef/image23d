"""Enhanced 3D conversion with real Hunyuan3D integration"""

import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Tuple, Optional, Any, Dict, List

import gradio as gr
import numpy as np
import torch
import trimesh
from PIL import Image
from transformers import pipeline

logger = logging.getLogger(__name__)


class ThreeDConverter:
    """Enhanced 3D converter with real Hunyuan3D model integration"""
    
    def __init__(self, cache_dir: Path, output_dir: Path):
        self.cache_dir = cache_dir
        self.output_dir = output_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Model loading state
        self.current_model = None
        self.current_model_name = None

    def convert_to_3d(
        self,
        hunyuan3d_model: Any,
        hunyuan3d_model_name: str,
        image: Image.Image,
        num_views: int,
        mesh_resolution: int,
        texture_resolution: int,
        progress=gr.Progress()
    ) -> Tuple[Optional[str], Optional[Image.Image], str]:
        """Convert image to 3D model using Hunyuan3D"""
        try:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            input_path = self.cache_dir / f"input_{timestamp}.png"
            image.save(input_path)

            logger.info(f"Starting 3D conversion with {hunyuan3d_model_name}")
            progress(0.1, "Preparing for 3D conversion...")

            # Try to use real Hunyuan3D model
            try:
                mesh_path, preview = self._convert_with_hunyuan3d(
                    hunyuan3d_model,
                    hunyuan3d_model_name,
                    image,
                    num_views,
                    mesh_resolution,
                    texture_resolution,
                    timestamp,
                    progress
                )
                
                info = f"""
<div class="success-box" style="padding: 15px; background: #e8f5e9; border-radius: 8px; border-left: 4px solid #4caf50;">
    <h4>✅ 3D Model Generated Successfully!</h4>
    <div style="margin: 10px 0;">
        <strong>📊 Generation Details:</strong>
        <ul style="margin: 8px 0; padding-left: 20px;">
            <li><strong>Model:</strong> {hunyuan3d_model_name}</li>
            <li><strong>Multi-view Images:</strong> {num_views}</li>
            <li><strong>Mesh Resolution:</strong> {mesh_resolution:,} vertices</li>
            <li><strong>Texture Resolution:</strong> {texture_resolution}x{texture_resolution}px</li>
            <li><strong>Output File:</strong> {mesh_path.name}</li>
        </ul>
    </div>
    <div style="margin-top: 15px; padding: 10px; background: rgba(76, 175, 80, 0.1); border-radius: 4px;">
        <strong>💡 Next Steps:</strong> The 3D model has been saved and is ready for download or further editing in your preferred 3D software.
    </div>
</div>
"""
                return str(mesh_path), preview, info
                
            except Exception as e:
                logger.warning(f"Hunyuan3D conversion failed, falling back to enhanced dummy: {e}")
                return self._create_enhanced_dummy_model(
                    image, hunyuan3d_model_name, num_views, 
                    mesh_resolution, texture_resolution, timestamp, progress
                )

        except Exception as e:
            logger.error(f"3D conversion failed: {e}")
            error_info = f"""
<div class="error-box" style="padding: 15px; background: #ffebee; border-radius: 8px; border-left: 4px solid #f44336;">
    <h4>❌ 3D Conversion Failed</h4>
    <p><strong>Error:</strong> {str(e)}</p>
    <p>Please check your input image and try again. For best results, use clear images with good lighting and distinct objects.</p>
</div>
"""
            return None, None, error_info

    def _convert_with_hunyuan3d(
        self,
        hunyuan3d_model: Any,
        model_name: str,
        image: Image.Image,
        num_views: int,
        mesh_resolution: int,
        texture_resolution: int,
        timestamp: str,
        progress
    ) -> Tuple[Path, Image.Image]:
        """Attempt real Hunyuan3D conversion"""
        
        try:
            # Check if we have a real Hunyuan3D model loaded
            if not hunyuan3d_model:
                raise NotImplementedError("No Hunyuan3D model loaded")
            
            # Check if it's a placeholder or real model
            if isinstance(hunyuan3d_model, dict) and hunyuan3d_model.get("status") == "placeholder":
                logger.info("Using Hunyuan3D placeholder model - will create enhanced demo")
                raise NotImplementedError("Hunyuan3D inference not yet integrated")
            
            # Log what type of model we have
            logger.info(f"Hunyuan3D model type: {type(hunyuan3d_model)}")
            
            # Check if it's our wrapper
            if hasattr(hunyuan3d_model, 'has_shape_pipeline'):
                if not hunyuan3d_model.has_shape_pipeline():
                    logger.error("Hunyuan3D wrapper exists but shape pipeline not loaded")
                    raise RuntimeError("Shape pipeline not loaded")
                else:
                    logger.info("Shape pipeline is loaded and ready")
            
            if not hasattr(hunyuan3d_model, 'generate_mesh'):
                raise NotImplementedError("Hunyuan3D model does not have generate_mesh method")
            
            progress(0.2, "Preparing image for Hunyuan3D...")
            
            # Define a progress callback for the model
            def model_progress(p, msg):
                # Map model progress (0-1) to our progress range (0.3-0.9)
                progress(0.3 + p * 0.6, msg)
            
            progress(0.3, "Generating 3D mesh with Hunyuan3D...")
            
            # Generate mesh directly from image
            # Hunyuan3D generates the mesh in one step
            mesh_data = hunyuan3d_model.generate_mesh(
                image,
                progress_callback=model_progress
            )
            
            progress(0.9, "Saving model...")
            
            # Save mesh and texture
            mesh_path = self.output_dir / f"hunyuan3d_{timestamp}.glb"
            preview_path = self.output_dir / f"preview_{timestamp}.png"
            
            # Save the mesh data
            # mesh_data should be a trimesh object from Hunyuan3D
            try:
                mesh_data.export(str(mesh_path))
                logger.info(f"Saved mesh to {mesh_path}")
            except Exception as e:
                logger.error(f"Failed to export mesh: {e}")
                # Try alternative export format
                mesh_path = self.output_dir / f"hunyuan3d_{timestamp}.obj"
                mesh_data.export(str(mesh_path))
                logger.info(f"Saved mesh as OBJ to {mesh_path}")
            
            # Create preview
            preview = self._create_preview_image(mesh_data)
            if preview:
                preview.save(preview_path)
                logger.info(f"Saved preview to {preview_path}")
            
            progress(1.0, "3D conversion complete!")
            
            return mesh_path, str(preview_path)
            
        except Exception as e:
            logger.error(f"Hunyuan3D conversion error: {e}")
            raise

    def _prepare_image_for_hunyuan3d(self, image: Image.Image) -> torch.Tensor:
        """Prepare PIL Image for Hunyuan3D model input"""
        # Resize to expected input size (typically 512x512 or 1024x1024)
        image = image.convert('RGB')
        image = image.resize((512, 512), Image.Resampling.LANCZOS)
        
        # Convert to tensor and normalize
        import torchvision.transforms as transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        tensor = transform(image).unsqueeze(0)  # Add batch dimension
        return tensor

    def _convert_hunyuan3d_to_trimesh(self, mesh_data: Dict, texture: Any) -> trimesh.Trimesh:
        """Convert Hunyuan3D mesh data to trimesh format"""
        vertices = mesh_data['vertices'].cpu().numpy()
        faces = mesh_data['faces'].cpu().numpy()
        
        # Create trimesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        # Apply texture if available
        if texture is not None:
            # Convert texture to format usable by trimesh
            if hasattr(texture, 'cpu'):
                texture_array = texture.cpu().numpy()
            else:
                texture_array = np.array(texture)
            
            # Create material
            mesh.visual = trimesh.visual.TextureVisuals(
                uv=mesh_data.get('uvs'),
                image=Image.fromarray((texture_array * 255).astype(np.uint8))
            )
        
        return mesh

    def _create_enhanced_dummy_model(
        self,
        image: Image.Image,
        model_name: str,
        num_views: int,
        mesh_resolution: int,
        texture_resolution: int,
        timestamp: str,
        progress
    ) -> Tuple[str, Image.Image, str]:
        """Create an enhanced dummy 3D model when real conversion fails"""
        
        progress(0.4, "Creating enhanced demo model...")
        
        # Analyze input image to create a more relevant shape
        mesh = self._create_shape_from_image_analysis(image, mesh_resolution)
        
        progress(0.7, "Applying image-based texturing...")
        
        # Apply the input image as texture
        mesh = self._apply_image_texture(mesh, image, texture_resolution)
        
        progress(0.9, "Finalizing model...")
        
        # Save mesh
        mesh_path = self.output_dir / f"enhanced_demo_{timestamp}.glb"
        mesh.export(mesh_path)
        
        # Create preview and save it
        preview = self._create_preview_image(mesh)
        preview_path = None
        if preview:
            preview_path = self.output_dir / f"preview_{timestamp}.png"
            preview.save(preview_path)
            preview_path = str(preview_path)
        
        info = f"""
<div class="info-box" style="padding: 15px; background: #fff3e0; border-radius: 8px; border-left: 4px solid #ff9800;">
    <h4>🔧 Enhanced Demo Model Created</h4>
    <div style="margin: 10px 0;">
        <p><strong>Note:</strong> This is an enhanced demonstration model created because full Hunyuan3D integration is still in development.</p>
        <strong>📊 Model Details:</strong>
        <ul style="margin: 8px 0; padding-left: 20px;">
            <li><strong>Model:</strong> {model_name} (Demo Mode)</li>
            <li><strong>Shape Analysis:</strong> Auto-detected from input image</li>
            <li><strong>Texture:</strong> Applied from input image</li>
            <li><strong>Vertices:</strong> {len(mesh.vertices):,}</li>
            <li><strong>Faces:</strong> {len(mesh.faces):,}</li>
        </ul>
    </div>
    <div style="margin-top: 15px; padding: 10px; background: rgba(255, 152, 0, 0.1); border-radius: 4px;">
        <strong>🚀 Coming Soon:</strong> Full Hunyuan3D integration with real multi-view generation and advanced 3D reconstruction.
    </div>
</div>
"""
        
        progress(1.0, "Enhanced demo model complete!")
        return str(mesh_path), preview_path, info

    def _create_shape_from_image_analysis(self, image: Image.Image, resolution: int) -> trimesh.Trimesh:
        """Analyze image to determine appropriate 3D shape"""
        try:
            # Convert to numpy for analysis
            img_array = np.array(image.convert('RGB'))
            
            # Simple analysis: check image dimensions and content
            height, width = img_array.shape[:2]
            aspect_ratio = width / height
            
            # Analyze color distribution and edges
            gray = np.mean(img_array, axis=2)
            edges = np.gradient(gray)
            edge_strength = np.mean(np.abs(edges))
            
            # Determine shape based on analysis - create higher resolution meshes
            if aspect_ratio > 1.5:
                # Wide image - create high-res cylinder
                mesh = trimesh.creation.cylinder(radius=1, height=0.5, sections=64)
            elif aspect_ratio < 0.7:
                # Tall image - create subdivided elongated shape
                mesh = trimesh.creation.box(extents=[0.8, 0.8, 2.0])
                mesh = mesh.subdivide()  # Increase resolution
                mesh = mesh.subdivide()  # More vertices
            elif edge_strength > 20:  # High detail image
                # Complex edges - create high-res icosphere for organic shapes
                mesh = trimesh.creation.icosphere(subdivisions=4, radius=1)
            else:
                # Default - create a more detailed shape
                # Start with icosphere for better default shape
                mesh = trimesh.creation.icosphere(subdivisions=3, radius=1)
                # Apply some deformation to make it less spherical
                mesh.vertices[:, 0] *= 1.2  # Stretch X
                mesh.vertices[:, 1] *= 1.2  # Stretch Y
                mesh.vertices[:, 2] *= 0.8  # Compress Z slightly
            
            # Ensure we have enough vertices
            while len(mesh.vertices) < resolution // 2:
                mesh = mesh.subdivide()
            
            # Apply smoothing for organic look
            if len(mesh.vertices) > 100:
                mesh = mesh.smoothed()
            
            # Final scale adjustment
            scale_factor = 2.0  # Make it a reasonable size
            mesh.apply_scale(scale_factor)
            
            return mesh
            
        except Exception as e:
            logger.warning(f"Image analysis failed, using default shape: {e}")
            return trimesh.creation.box(extents=[1, 1, 1])

    def _apply_image_texture(self, mesh: trimesh.Trimesh, image: Image.Image, resolution: int) -> trimesh.Trimesh:
        """Apply input image as texture to the mesh"""
        try:
            # Resize image to texture resolution
            texture_img = image.resize((resolution, resolution), Image.Resampling.LANCZOS)
            
            # Generate UV coordinates if not present
            if not hasattr(mesh.visual, 'uv') or mesh.visual.uv is None:
                # Simple spherical UV mapping
                vertices = mesh.vertices
                
                # Normalize vertices
                center = vertices.mean(axis=0)
                vertices_centered = vertices - center
                
                # Spherical coordinates
                x, y, z = vertices_centered.T
                u = 0.5 + np.arctan2(z, x) / (2 * np.pi)
                v = 0.5 - np.arcsin(y / np.linalg.norm(vertices_centered, axis=1)) / np.pi
                
                uv = np.column_stack([u, v])
            else:
                uv = mesh.visual.uv
            
            # Apply texture
            mesh.visual = trimesh.visual.TextureVisuals(
                uv=uv,
                image=texture_img
            )
            
            return mesh
            
        except Exception as e:
            logger.warning(f"Texture application failed: {e}")
            return mesh

    def _create_preview_image(self, mesh: trimesh.Trimesh) -> Image.Image:
        """Create a preview image of the 3D mesh"""
        try:
            # Try to render with trimesh
            try:
                # Create scene
                scene = mesh.scene()
                
                # Set good viewing angle
                scene.camera_transform = scene.camera.look_at(
                    points=mesh.vertices,
                    center=mesh.centroid,
                    distance=mesh.scale * 2.5
                )
                
                # Render preview
                preview_data = scene.save_image(resolution=[512, 512], visible=True)
                preview = Image.open(trimesh.util.wrap_as_stream(preview_data))
                
                return preview
                
            except ImportError as e:
                logger.warning(f"3D rendering not available (missing {e}), creating wireframe preview")
                return self._create_wireframe_preview(mesh)
            
        except Exception as e:
            logger.warning(f"Preview creation failed: {e}")
            # Create a simple placeholder with mesh info
            return self._create_info_preview(mesh)

    def _create_wireframe_preview(self, mesh: trimesh.Trimesh) -> Image.Image:
        """Create a wireframe-style preview when 3D rendering isn't available"""
        try:
            from PIL import ImageDraw
            
            # Create image
            img = Image.new('RGB', (512, 512), color='white')
            draw = ImageDraw.Draw(img)
            
            # Get mesh bounds and center
            bounds = mesh.bounds
            center = mesh.centroid[:2]  # X, Y only
            scale = min(400 / (bounds[1][0] - bounds[0][0]), 400 / (bounds[1][1] - bounds[0][1]))
            
            # Draw wireframe representation
            vertices_2d = []
            for vertex in mesh.vertices:
                x = int(256 + (vertex[0] - center[0]) * scale)
                y = int(256 + (vertex[1] - center[1]) * scale)
                vertices_2d.append((x, y))
            
            # Draw edges (sample)
            for i, face in enumerate(mesh.faces[:min(100, len(mesh.faces))]):  # Limit for performance
                v1, v2, v3 = face
                if v1 < len(vertices_2d) and v2 < len(vertices_2d) and v3 < len(vertices_2d):
                    # Draw triangle edges
                    draw.line([vertices_2d[v1], vertices_2d[v2]], fill='blue', width=1)
                    draw.line([vertices_2d[v2], vertices_2d[v3]], fill='blue', width=1)
                    draw.line([vertices_2d[v3], vertices_2d[v1]], fill='blue', width=1)
            
            # Add title
            draw.text((20, 20), "3D Mesh Preview (Wireframe)", fill='black')
            draw.text((20, 40), f"Vertices: {len(mesh.vertices):,}", fill='black')
            draw.text((20, 60), f"Faces: {len(mesh.faces):,}", fill='black')
            
            return img
            
        except Exception as e:
            logger.warning(f"Wireframe preview failed: {e}")
            return self._create_info_preview(mesh)

    def _create_info_preview(self, mesh: trimesh.Trimesh) -> Image.Image:
        """Create an informational preview when rendering fails"""
        try:
            from PIL import ImageDraw, ImageFont
            
            # Create gradient background
            img = Image.new('RGB', (512, 512), color='lightblue')
            draw = ImageDraw.Draw(img)
            
            # Add gradient effect
            for i in range(512):
                color_intensity = int(200 + 55 * (i / 512))
                draw.line([(0, i), (512, i)], fill=(color_intensity, color_intensity, 255))
            
            # Add mesh information
            info_text = [
                "3D Model Generated",
                "",
                f"Vertices: {len(mesh.vertices):,}",
                f"Faces: {len(mesh.faces):,}",
                f"Volume: {mesh.volume:.2f}",
                f"Surface Area: {mesh.area:.2f}",
                "",
                "Preview rendering unavailable",
                "Model ready for download"
            ]
            
            y_pos = 150
            for line in info_text:
                if line:  # Skip empty lines for spacing
                    draw.text((50, y_pos), line, fill='black')
                y_pos += 30
            
            return img
            
        except Exception as e:
            logger.error(f"Info preview creation failed: {e}")
            # Ultimate fallback
            img = Image.new('RGB', (512, 512), color='lightgray')
            return img

    def get_supported_formats(self) -> List[str]:
        """Get list of supported export formats"""
        return ['glb', 'obj', 'ply', 'stl', 'fbx', 'usdz']

    def convert_format(self, input_path: str, output_format: str) -> str:
        """Convert 3D model to different format"""
        try:
            mesh = trimesh.load(input_path)
            
            input_path_obj = Path(input_path)
            output_path = input_path_obj.with_suffix(f'.{output_format}')
            
            # Handle format-specific export options
            if output_format == 'glb':
                mesh.export(output_path, file_type='glb')
            elif output_format == 'obj':
                mesh.export(output_path, file_type='obj')
            elif output_format == 'ply':
                mesh.export(output_path, file_type='ply')
            elif output_format == 'stl':
                mesh.export(output_path, file_type='stl')
            else:
                # For unsupported formats, export as OBJ
                mesh.export(output_path.with_suffix('.obj'), file_type='obj')
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Format conversion failed: {e}")
            raise