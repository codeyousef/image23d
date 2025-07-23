"""
Utility functions for Hi3DGen normal bridging processing
"""

import numpy as np
import cv2
import logging
from typing import List, Tuple, Optional
from PIL import Image
from scipy.ndimage import sobel, gaussian_filter

logger = logging.getLogger(__name__)

def compute_2_5d_normal_map(image_np: np.ndarray) -> np.ndarray:
    """Compute 2.5D normal map using Hi3DGen's bridging approach"""
    try:
        # Convert to grayscale for depth/normal estimation
        if len(image_np.shape) == 3:
            gray = np.mean(image_np, axis=2)
        else:
            gray = image_np
            
        # Hi3DGen approach: Decouple low and high frequency patterns
        
        # 1. Low-frequency component (overall shape)
        low_freq = gaussian_filter(gray, sigma=5.0)
        
        # 2. High-frequency component (fine details) 
        high_freq = gray - low_freq
        
        # 3. Compute gradients for normal estimation
        grad_x = sobel(low_freq, axis=1, mode='reflect')
        grad_y = sobel(low_freq, axis=0, mode='reflect')
        
        # 4. Add high-frequency detail back
        detail_grad_x = sobel(high_freq, axis=1, mode='reflect') * 0.5
        detail_grad_y = sobel(high_freq, axis=0, mode='reflect') * 0.5
        
        combined_grad_x = grad_x + detail_grad_x
        combined_grad_y = grad_y + detail_grad_y
        
        # 5. Compute normal vectors
        # Normal = (-dx, -dy, 1) normalized
        normals = np.zeros((gray.shape[0], gray.shape[1], 3))
        normals[:,:,0] = -combined_grad_x  # X component (red)
        normals[:,:,1] = -combined_grad_y  # Y component (green)
        normals[:,:,2] = 1.0              # Z component (blue)
        
        # 6. Normalize normal vectors
        norm = np.sqrt(np.sum(normals**2, axis=2, keepdims=True))
        normals = normals / (norm + 1e-8)
        
        # 7. Convert to [0,1] range for visualization (standard normal map format)
        normals = (normals + 1.0) / 2.0
        
        return normals
        
    except Exception as e:
        logger.warning(f"2.5D normal computation failed: {e}")
        # Fallback to simple normal map
        return np.ones((image_np.shape[0], image_np.shape[1], 3)) * 0.5

def enhance_normal_map(normal_map: np.ndarray) -> np.ndarray:
    """Enhance normal map using Hi3DGen regularization techniques"""
    try:
        # Apply Hi3DGen's normal regularization
        
        # 1. Noise injection for stability (key Hi3DGen technique)
        noise_strength = 0.02
        noise = np.random.normal(0, noise_strength, normal_map.shape)
        enhanced = normal_map + noise
        
        # 2. Bilateral filtering to preserve edges while smoothing
        for channel in range(3):
            enhanced[:,:,channel] = cv2.bilateralFilter(
                enhanced[:,:,channel].astype(np.float32),
                d=5,
                sigmaColor=0.1,
                sigmaSpace=5
            )
        
        # 3. Re-normalize after filtering
        # Convert back to normal vector space
        normal_vectors = (enhanced * 2.0) - 1.0
        norm = np.sqrt(np.sum(normal_vectors**2, axis=2, keepdims=True))
        normal_vectors = normal_vectors / (norm + 1e-8)
        
        # 4. Convert back to [0,1] range
        enhanced = (normal_vectors + 1.0) / 2.0
        
        # 5. Ensure Z component is always positive (facing outward)
        enhanced[:,:,2] = np.maximum(enhanced[:,:,2], 0.5)
        
        # 6. Clamp to valid range
        enhanced = np.clip(enhanced, 0.0, 1.0)
        
        return enhanced
        
    except Exception as e:
        logger.warning(f"Normal map enhancement failed: {e}")
        return normal_map

def warp_image_with_normals(image: np.ndarray, normal_map: np.ndarray, view_angle: float) -> np.ndarray:
    """Warp image based on normal map to simulate different viewpoints"""
    try:
        h, w = image.shape[:2]
        
        # Convert normal map back to normal vectors
        normals = (normal_map * 2.0) - 1.0
        
        # Create coordinate grids
        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        
        # Simulate viewpoint change based on normals
        cos_angle = np.cos(view_angle)
        sin_angle = np.sin(view_angle)
        
        # Apply normal-based displacement
        displacement_x = normals[:,:,0] * sin_angle * 10  # Scale factor for visibility
        displacement_y = normals[:,:,1] * cos_angle * 5
        
        # Create new coordinates
        new_x = x_coords + displacement_x
        new_y = y_coords + displacement_y
        
        # Clamp coordinates to image bounds
        new_x = np.clip(new_x, 0, w - 1)
        new_y = np.clip(new_y, 0, h - 1)
        
        # Sample the image at new coordinates using bilinear interpolation
        warped_image = np.zeros_like(image)
        
        for c in range(image.shape[2]):
            warped_image[:,:,c] = cv2.remap(
                image[:,:,c].astype(np.float32),
                new_x.astype(np.float32),
                new_y.astype(np.float32),
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT
            )
            
        return warped_image
        
    except Exception as e:
        logger.warning(f"Normal-guided warping failed: {e}")
        # Return original image as fallback
        return image

def integrate_normals_to_height(normal_map: np.ndarray) -> np.ndarray:
    """Integrate normal vectors to reconstruct height field"""
    try:
        # Convert normals back to [-1, 1] range
        normals = (normal_map * 2.0) - 1.0
        
        # Extract gradients
        grad_x = -normals[:,:,0]  # Negative because of normal map convention
        grad_y = -normals[:,:,1]
        
        # Simple integration using cumulative sum (Poisson integration would be better)
        height = np.zeros_like(grad_x)
        
        # Integrate in X direction
        height[:,1:] = np.cumsum(grad_x[:,:-1], axis=1)
        
        # Integrate in Y direction and average
        height_y = np.zeros_like(grad_y)
        height_y[1:,:] = np.cumsum(grad_y[:-1,:], axis=0)
        
        # Combine both integrations
        height = (height + height_y) / 2.0
        
        # Normalize height field
        height = (height - height.min()) / (height.max() - height.min() + 1e-8)
        
        return height
        
    except Exception as e:
        logger.warning(f"Normal integration failed: {e}")
        return np.ones_like(normal_map[:,:,0]) * 0.5

def generate_mesh_from_height_field(height_field: np.ndarray, target_resolution: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate 3D mesh vertices and faces from height field"""
    try:
        h, w = height_field.shape
        
        # Downsample if too high resolution
        if h > target_resolution or w > target_resolution:
            scale_factor = min(target_resolution / h, target_resolution / w)
            new_h = int(h * scale_factor)
            new_w = int(w * scale_factor)
            height_field = cv2.resize(height_field, (new_w, new_h))
            h, w = new_h, new_w
        
        # Generate vertices
        vertices = []
        for i in range(h):
            for j in range(w):
                x = (j / w) * 2 - 1  # [-1, 1]
                y = (i / h) * 2 - 1  # [-1, 1] 
                z = height_field[i, j] * 0.5  # Scale height
                vertices.append([x, y, z])
        
        # Generate faces (triangles)
        faces = []
        for i in range(h - 1):
            for j in range(w - 1):
                # Current quad vertices
                v0 = i * w + j
                v1 = i * w + (j + 1)
                v2 = (i + 1) * w + j
                v3 = (i + 1) * w + (j + 1)
                
                # Two triangles per quad
                faces.append([v0, v1, v2])
                faces.append([v1, v3, v2])
        
        return np.array(vertices), np.array(faces)
        
    except Exception as e:
        logger.error(f"Mesh generation from height field failed: {e}")
        # Return a simple quad as fallback
        vertices = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]])
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        return vertices, faces