"""Enhanced 3D conversion with real Hunyuan3D integration"""

import logging
import os
import tempfile
import time
import traceback
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
            # Optimize PNG saving with compression level
            image.save(input_path, "PNG", compress_level=6, optimize=True)

            logger.info(f"\n{'='*80}")
            logger.info(f"[3D_CONVERSION] Starting 3D conversion")
            logger.info(f"[3D_CONVERSION] Model name: {hunyuan3d_model_name}")
            logger.info(f"[3D_CONVERSION] Model type: {type(hunyuan3d_model)}")
            logger.info(f"[3D_CONVERSION] Image type: {type(image)}")
            logger.info(f"[3D_CONVERSION] Image size: {image.size if hasattr(image, 'size') else 'N/A'}")
            logger.info(f"[3D_CONVERSION] Image mode: {image.mode if hasattr(image, 'mode') else 'N/A'}")
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
                logger.error(f"Hunyuan3D conversion failed: {e}")
                logger.error(f"Error type: {type(e).__name__}")
                logger.error(f"Full traceback: {traceback.format_exc()}")
                
                # No fallback - fail with clear message
                raise RuntimeError(
                    f"3D conversion failed: {str(e)}\n\n"
                    f"To use HunYuan3D {hunyuan3d_model_name}:\n"
                    f"1. Ensure Hunyuan3D repository is available\n"
                    f"2. Install dependencies: pip install -e ./Hunyuan3D\n"
                    f"3. Check that model weights are in models/3d/{hunyuan3d_model_name}/"
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
        """Use new 3D generation system for conversion"""
        
        try:
            # Import our new 3D generator
            from .generation.threed import get_3d_generator, generate_3d_model
            
            progress(0.2, "Initializing 3D generation system...")
            
            # Map old parameters to new system
            quality_preset = "standard"
            if mesh_resolution >= 1024:
                quality_preset = "high"
            elif mesh_resolution <= 256:
                quality_preset = "draft"
                
            # Define a progress callback for the model
            def model_progress(p, msg):
                # Map model progress (0-1) to our progress range (0.3-0.9)
                progress(0.3 + p * 0.6, msg)
            
            progress(0.3, "Generating 3D model...")
            
            # Check if we're dealing with an orchestrator
            if hasattr(hunyuan3d_model, 'generate') and not isinstance(hunyuan3d_model, dict):
                logger.info("Using orchestrator-based model")
                # Call orchestrator's generate method directly
                from ..models.threed.base import TaskRequirements
                
                requirements = TaskRequirements(
                    input_type="image",
                    quality_preset=quality_preset,
                    output_format="glb",
                    required_capabilities=["image_to_3d", "pbr_materials"] if texture_resolution >= 1024 else ["image_to_3d"]
                )
                
                # If model_name specified, set it as preferred
                if model_name != "auto":
                    requirements.preferred_model = model_name
                
                result = hunyuan3d_model.generate(
                    image,
                    requirements,
                    progress_callback=model_progress
                )
            else:
                # Use new 3D generation system
                result = generate_3d_model(
                    image=image,
                    model_type=model_name,
                    quality_preset=quality_preset,
                    output_format="glb",
                    enable_pbr=texture_resolution >= 1024,
                    enable_depth_refinement=True,
                    progress_callback=model_progress
                )
            
            progress(0.9, "Processing results...")
            
            # Extract results
            mesh_path = Path(result["output_path"])
            preview_image = result.get("preview_image")
            
            # Save preview if available
            preview_path = None
            if preview_image:
                preview_path = self.output_dir / f"preview_{timestamp}.png"
                preview_image.save(preview_path, "PNG", compress_level=6, optimize=True)
                logger.info(f"Saved preview to {preview_path}")
                preview_path = str(preview_path)
            
            progress(1.0, "3D conversion complete!")
            
            return mesh_path, preview_path, "3D model generated successfully"
            
        except ImportError as e:
            logger.error(f"Failed to import new 3D generation system: {e}")
            # Fall back to checking if we have old-style model
            if hasattr(hunyuan3d_model, 'generate_mesh'):
                logger.info("Falling back to legacy model conversion")
                return self._convert_with_legacy_model(
                    hunyuan3d_model, model_name, image, 
                    num_views, mesh_resolution, texture_resolution,
                    timestamp, progress
                )
            else:
                raise RuntimeError("3D generation system not available")
            
        except Exception as e:
            logger.error(f"3D generation error: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
            # Try legacy fallback if available
            if hasattr(hunyuan3d_model, 'generate_mesh'):
                logger.info("Error in new generation system, falling back to legacy model")
                return self._convert_with_legacy_model(
                    hunyuan3d_model, model_name, image, 
                    num_views, mesh_resolution, texture_resolution,
                    timestamp, progress
                )
            else:
                raise
    
    def _convert_with_legacy_model(
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
        """Legacy conversion for old model format"""
        
        try:
            progress(0.3, "Using legacy Hunyuan3D model...")
            
            # Define a progress callback for the model
            def model_progress(p, msg):
                progress(0.3 + p * 0.6, msg)
            
            # Generate mesh directly from image
            mesh_data = hunyuan3d_model.generate_mesh(
                image,
                progress_callback=model_progress
            )
            
            progress(0.9, "Saving model...")
            
            # Save mesh and texture
            mesh_path = self.output_dir / f"hunyuan3d_{timestamp}.glb"
            preview_path = self.output_dir / f"preview_{timestamp}.png"
            
            # Save the mesh data
            try:
                logger.info(f"[LEGACY] Attempting to export mesh to {mesh_path}")
                logger.info(f"[LEGACY] Mesh type: {type(mesh_data)}")
                logger.info(f"[LEGACY] Mesh has export method: {hasattr(mesh_data, 'export')}")
                
                if hasattr(mesh_data, 'export'):
                    mesh_data.export(str(mesh_path))
                    logger.info(f"[LEGACY] Saved mesh to {mesh_path}")
                    
                    # Verify the file was created
                    if mesh_path.exists():
                        logger.info(f"[LEGACY] Verified: File exists at {mesh_path}, size: {mesh_path.stat().st_size} bytes")
                    else:
                        logger.error(f"[LEGACY] ERROR: File was not created at {mesh_path}")
                        raise FileNotFoundError(f"Mesh file was not created at {mesh_path}")
                else:
                    logger.error(f"[LEGACY] Mesh data has no export method")
                    raise AttributeError(f"Mesh object of type {type(mesh_data)} has no export method")
                    
            except Exception as e:
                logger.error(f"[LEGACY] Failed to export mesh as GLB: {e}")
                logger.error(f"[LEGACY] Error type: {type(e).__name__}")
                
                # Try OBJ format as fallback
                try:
                    mesh_path = self.output_dir / f"hunyuan3d_{timestamp}.obj"
                    logger.info(f"[LEGACY] Attempting OBJ export to {mesh_path}")
                    mesh_data.export(str(mesh_path))
                    logger.info(f"[LEGACY] Saved mesh as OBJ to {mesh_path}")
                    
                    if mesh_path.exists():
                        logger.info(f"[LEGACY] Verified: OBJ file exists, size: {mesh_path.stat().st_size} bytes")
                    else:
                        logger.error(f"[LEGACY] ERROR: OBJ file was not created")
                        raise FileNotFoundError(f"OBJ file was not created at {mesh_path}")
                        
                except Exception as obj_error:
                    logger.error(f"[LEGACY] Failed to export as OBJ too: {obj_error}")
                    raise RuntimeError(f"Failed to export mesh in any format. GLB error: {e}, OBJ error: {obj_error}")
            
            # Create preview
            preview = self._create_preview_image(mesh_data)
            if preview:
                preview.save(preview_path, "PNG", compress_level=6, optimize=True)
                logger.info(f"Saved preview to {preview_path}")
            
            progress(1.0, "3D conversion complete!")
            
            return mesh_path, str(preview_path), "3D model generated with legacy model"
            
        except Exception as e:
            logger.error(f"Legacy model conversion error: {e}")
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

    def _create_preview_image(self, mesh: trimesh.Trimesh) -> Optional[Image.Image]:
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
            import numpy as np
            
            # Create image
            img = Image.new('RGB', (512, 512), color='white')
            draw = ImageDraw.Draw(img)
            
            # Check if mesh has vertices
            if len(mesh.vertices) == 0:
                draw.text((20, 20), "Empty mesh", fill='black')
                return img
            
            # Vectorized vertex transformation
            vertices = mesh.vertices
            bounds = vertices.max(axis=0) - vertices.min(axis=0)
            center = vertices.mean(axis=0)
            
            # Prevent division by zero and calculate scale
            scale = 1
            if bounds[0] > 0 and bounds[1] > 0:
                scale = min(400 / bounds[0], 400 / bounds[1])
            
            # Transform all vertices at once using NumPy (vectorized operation)
            vertices_2d = (vertices[:, :2] - center[:2]) * scale + 256
            vertices_2d = vertices_2d.astype(int)
            
            # Use edge deduplication to avoid drawing the same edge multiple times
            edges = set()
            faces_to_render = mesh.faces[:min(100, len(mesh.faces))]  # Limit for performance
            
            for face in faces_to_render:
                # Create edges from face, ensuring each edge is unique
                for i in range(3):
                    v1, v2 = face[i], face[(i+1)%3]
                    if v1 < len(vertices_2d) and v2 < len(vertices_2d):
                        # Sort vertices to ensure edge (1,2) and (2,1) are treated as the same
                        edge = tuple(sorted([v1, v2]))
                        edges.add(edge)
            
            # Draw unique edges only
            for v1, v2 in edges:
                try:
                    draw.line([
                        tuple(vertices_2d[v1]), 
                        tuple(vertices_2d[v2])
                    ], fill='blue', width=1)
                except:
                    continue  # Skip any problematic edges
            
            # Add mesh info
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
            from PIL import ImageDraw
            
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