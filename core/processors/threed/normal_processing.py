"""Normal map estimation for 3D surface detail"""

import asyncio
import logging
from typing import List
import numpy as np
from PIL import Image
import cv2

logger = logging.getLogger(__name__)


class NormalProcessor:
    """Handles normal map estimation from depth maps"""
    
    async def estimate_normal_maps(
        self,
        images: List[Image.Image],
        depth_maps: List[np.ndarray],
        progress_callback=None
    ) -> List[np.ndarray]:
        """Estimate normal maps from depth maps - mock implementation"""
        if progress_callback:
            progress_callback("postprocess", 0.5, "Computing surface normals...")
            
        normal_maps = []
        
        for i, (img, depth) in enumerate(zip(images, depth_maps)):
            if progress_callback:
                progress = 0.5 + (i / len(images)) * 0.1  # 50-60% range
                progress_callback("postprocess", progress, f"Computing normals for view {i+1}/{len(images)}")
                
            # Compute surface normals from depth
            normals = await asyncio.to_thread(self._compute_surface_normals, depth)
            normal_maps.append(normals)
            
            # Small delay to simulate processing
            await asyncio.sleep(0.1)
            
        return normal_maps
    
    def _compute_surface_normals(self, depth: np.ndarray) -> np.ndarray:
        """Compute surface normals from depth map"""
        h, w = depth.shape
        
        # Compute gradients
        zy, zx = np.gradient(depth)
        
        # Construct normal vectors
        normal = np.dstack((-zx, -zy, np.ones_like(depth)))
        
        # Normalize
        norm = np.sqrt(np.sum(normal**2, axis=2, keepdims=True))
        normal = normal / (norm + 1e-8)
        
        # Convert to 0-255 range for visualization
        # Map from [-1, 1] to [0, 255]
        normal_vis = (normal + 1) * 127.5
        normal_vis = np.clip(normal_vis, 0, 255).astype(np.uint8)
        
        return normal_vis