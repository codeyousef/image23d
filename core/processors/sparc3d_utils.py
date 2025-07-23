"""
Utility functions for Sparc3D sparse cube processing
"""

import numpy as np
import cv2
import logging
from typing import List, Tuple
from PIL import Image

logger = logging.getLogger(__name__)

def estimate_depth_for_sparse(image_np: np.ndarray) -> np.ndarray:
    """Estimate depth map specifically for sparse cube placement"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        # Use advanced depth cues for sparse placement
        # 1. Gradient-based depth
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_depth = np.sqrt(grad_x**2 + grad_y**2)
        
        # 2. Intensity-based depth (darker = further)
        intensity_depth = 1.0 - gray
        
        # 3. Texture-based depth
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_depth = np.abs(laplacian)
        
        # Combine depth cues
        combined_depth = (
            0.4 * gradient_depth + 
            0.3 * intensity_depth + 
            0.3 * texture_depth
        )
        
        # Normalize and apply bilateral smoothing
        combined_depth = combined_depth / (combined_depth.max() + 1e-8)
        depth_smooth = cv2.bilateralFilter(
            combined_depth.astype(np.float32), 
            d=9, 
            sigmaColor=0.1, 
            sigmaSpace=10
        )
        
        return depth_smooth
        
    except Exception as e:
        logger.warning(f"Advanced depth estimation failed, using simple method: {e}")
        return np.ones_like(image_np[:,:,0]) * 0.5

def fill_sparse_grid(sparse_grid: np.ndarray, image: np.ndarray, depth_map: np.ndarray, resolution: int):
    """Fill sparse 3D grid with voxels based on image and depth"""
    h, w = image.shape[:2]
    
    # Create coordinate mappings
    y_coords, x_coords = np.meshgrid(
        np.linspace(0, resolution-1, h).astype(int),
        np.linspace(0, resolution-1, w).astype(int),
        indexing='ij'
    )
    
    # Map depth to Z coordinates
    z_coords = (depth_map * (resolution - 1)).astype(int)
    
    # Fill grid with color and opacity
    for i in range(h):
        for j in range(w):
            x, y, z = x_coords[i, j], y_coords[i, j], z_coords[i, j]
            
            # Ensure coordinates are within bounds
            if 0 <= x < resolution and 0 <= y < resolution and 0 <= z < resolution:
                # RGB color
                sparse_grid[x, y, z, :3] = image[i, j, :]
                # Alpha (opacity) based on depth confidence
                confidence = calculate_depth_confidence(depth_map[i, j], i, j, depth_map)
                sparse_grid[x, y, z, 3] = confidence

def calculate_depth_confidence(depth_value: float, i: int, j: int, depth_map: np.ndarray) -> float:
    """Calculate confidence for depth value based on local consistency"""
    try:
        h, w = depth_map.shape
        
        # Check local neighborhood consistency
        window_size = 5
        i_start = max(0, i - window_size // 2)
        i_end = min(h, i + window_size // 2 + 1)
        j_start = max(0, j - window_size // 2)
        j_end = min(w, j + window_size // 2 + 1)
        
        local_depths = depth_map[i_start:i_end, j_start:j_end]
        depth_variance = np.var(local_depths)
        
        # Higher confidence for consistent regions
        confidence = np.exp(-depth_variance * 10)  # Scale factor 10
        return np.clip(confidence, 0.1, 1.0)
        
    except:
        return 0.5  # Default confidence

def optimize_sparse_representation(sparse_grid: np.ndarray) -> np.ndarray:
    """Optimize sparse representation for memory efficiency and quality"""
    try:
        # Remove low-confidence voxels
        confidence_threshold = 0.3
        mask = sparse_grid[:, :, :, 3] > confidence_threshold
        
        # Create optimized sparse representation
        optimized_grid = sparse_grid.copy()
        optimized_grid[~mask] = 0  # Zero out low-confidence voxels
        
        # Apply morphological operations to clean up noise
        from scipy.ndimage import binary_opening, binary_closing
        
        # Work with binary mask
        binary_mask = mask.astype(bool)
        
        # Apply opening (erosion followed by dilation) to remove noise
        cleaned_mask = binary_opening(binary_mask, iterations=1)
        
        # Apply closing (dilation followed by erosion) to fill gaps
        final_mask = binary_closing(cleaned_mask, iterations=1)
        
        # Apply cleaned mask
        optimized_grid[~final_mask] = 0
        
        return optimized_grid
        
    except Exception as e:
        logger.warning(f"Sparse optimization failed: {e}")
        return sparse_grid

def render_view_from_angle(sparse_cubes: np.ndarray, azimuth: float, elevation: float) -> Image.Image:
    """Render a single view from sparse cubes at specified angle"""
    try:
        resolution = sparse_cubes.shape[0]
        render_size = 512
        
        # Create rotation matrix
        cos_az, sin_az = np.cos(azimuth), np.sin(azimuth)
        cos_el, sin_el = np.cos(elevation), np.sin(elevation)
        
        # Simple orthographic projection with rotation
        rendered_image = np.zeros((render_size, render_size, 3), dtype=np.float32)
        z_buffer = np.zeros((render_size, render_size), dtype=np.float32) - np.inf
        
        # Get non-zero voxel positions
        occupied_voxels = np.where(sparse_cubes[:, :, :, 3] > 0)
        
        for idx in range(len(occupied_voxels[0])):
            x, y, z = occupied_voxels[0][idx], occupied_voxels[1][idx], occupied_voxels[2][idx]
            
            # Normalize coordinates to [-1, 1]
            nx, ny, nz = (
                2 * x / resolution - 1,
                2 * y / resolution - 1,
                2 * z / resolution - 1
            )
            
            # Apply rotation
            rotated_x = cos_az * nx - sin_az * nz
            rotated_y = ny
            rotated_z = sin_az * nx + cos_az * nz
            
            # Apply elevation
            final_y = cos_el * rotated_y - sin_el * rotated_z
            final_z = sin_el * rotated_y + cos_el * rotated_z
            
            # Project to 2D screen coordinates
            screen_x = int((rotated_x + 1) * render_size / 2)
            screen_y = int((final_y + 1) * render_size / 2)
            
            # Check bounds and z-buffer
            if (0 <= screen_x < render_size and 0 <= screen_y < render_size and
                final_z > z_buffer[screen_y, screen_x]):
                
                z_buffer[screen_y, screen_x] = final_z
                
                # Get voxel color and alpha
                color = sparse_cubes[x, y, z, :3]
                alpha = sparse_cubes[x, y, z, 3]
                
                # Alpha blending
                rendered_image[screen_y, screen_x] = (
                    alpha * color + (1 - alpha) * rendered_image[screen_y, screen_x]
                )
        
        # Convert to PIL Image
        rendered_image = np.clip(rendered_image * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(rendered_image)
        
    except Exception as e:
        logger.warning(f"View rendering failed: {e}")
        # Return a placeholder image
        placeholder = np.ones((512, 512, 3), dtype=np.uint8) * 128
        return Image.fromarray(placeholder)

def create_cube_vertices(center: np.ndarray, size: float) -> List[List[float]]:
    """Create vertices for a cube centered at position with given size"""
    half_size = size / 2
    x, y, z = center
    
    return [
        [x - half_size, y - half_size, z - half_size],
        [x + half_size, y - half_size, z - half_size],
        [x + half_size, y + half_size, z - half_size],
        [x - half_size, y + half_size, z - half_size],
        [x - half_size, y - half_size, z + half_size],
        [x + half_size, y - half_size, z + half_size],
        [x + half_size, y + half_size, z + half_size],
        [x - half_size, y + half_size, z + half_size],
    ]

def create_cube_faces(vertex_offset: int) -> List[List[int]]:
    """Create face indices for a cube with given vertex offset"""
    return [
        [vertex_offset + 0, vertex_offset + 1, vertex_offset + 2],
        [vertex_offset + 0, vertex_offset + 2, vertex_offset + 3],
        [vertex_offset + 4, vertex_offset + 7, vertex_offset + 6],
        [vertex_offset + 4, vertex_offset + 6, vertex_offset + 5],
        [vertex_offset + 0, vertex_offset + 4, vertex_offset + 5],
        [vertex_offset + 0, vertex_offset + 5, vertex_offset + 1],
        [vertex_offset + 1, vertex_offset + 5, vertex_offset + 6],
        [vertex_offset + 1, vertex_offset + 6, vertex_offset + 2],
        [vertex_offset + 2, vertex_offset + 6, vertex_offset + 7],
        [vertex_offset + 2, vertex_offset + 7, vertex_offset + 3],
        [vertex_offset + 3, vertex_offset + 7, vertex_offset + 4],
        [vertex_offset + 3, vertex_offset + 4, vertex_offset + 0],
    ]