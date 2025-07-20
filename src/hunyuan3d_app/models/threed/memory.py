"""Memory management for 3D model generation

Based on the Managing Multiple AI Models guide:
- Complete model unloading before switching
- Intermediate result caching
- Memory-efficient model swapping
- GGUF quantization support
"""

import gc
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple
import torch
import psutil
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class IntermediateCache:
    """Cache for intermediate 3D processing results"""
    
    def __init__(self, cache_dir: Path, max_size_gb: float = 5.0):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_gb = max_size_gb
        self.cache_index: Dict[str, Dict[str, Any]] = {}
        
    def save_intermediate(self, key: str, data: Any, data_type: str):
        """Save intermediate data to cache"""
        cache_path = self.cache_dir / f"{key}_{data_type}"
        
        try:
            if data_type == "multiview_images":
                # Save list of images
                images_dir = cache_path
                images_dir.mkdir(exist_ok=True)
                for i, img in enumerate(data):
                    img.save(images_dir / f"view_{i:03d}.png")
                    
            elif data_type == "depth_map":
                # Save depth map as numpy array
                np.save(f"{cache_path}.npy", data)
                
            elif data_type == "normal_map":
                # Save normal map as image
                if isinstance(data, Image.Image):
                    data.save(f"{cache_path}.png")
                else:
                    Image.fromarray(data).save(f"{cache_path}.png")
                    
            elif data_type == "mesh_data":
                # Save mesh vertices and faces
                np.savez(f"{cache_path}.npz", 
                        vertices=data.get("vertices"),
                        faces=data.get("faces"))
                        
            self.cache_index[key] = {
                "type": data_type,
                "path": str(cache_path),
                "size": self._get_size(cache_path)
            }
            
            # Check cache size and cleanup if needed
            self._cleanup_if_needed()
            
        except Exception as e:
            logger.error(f"Failed to cache {data_type} for {key}: {e}")
            
    def load_intermediate(self, key: str, data_type: str) -> Optional[Any]:
        """Load intermediate data from cache"""
        if key not in self.cache_index:
            return None
            
        entry = self.cache_index[key]
        if entry["type"] != data_type:
            return None
            
        cache_path = Path(entry["path"])
        
        try:
            if data_type == "multiview_images":
                # Load list of images
                images = []
                if cache_path.is_dir():
                    for img_path in sorted(cache_path.glob("view_*.png")):
                        images.append(Image.open(img_path))
                return images
                
            elif data_type == "depth_map":
                # Load depth map
                return np.load(f"{cache_path}.npy")
                
            elif data_type == "normal_map":
                # Load normal map
                return Image.open(f"{cache_path}.png")
                
            elif data_type == "mesh_data":
                # Load mesh data
                data = np.load(f"{cache_path}.npz")
                return {
                    "vertices": data["vertices"],
                    "faces": data["faces"]
                }
                
        except Exception as e:
            logger.error(f"Failed to load {data_type} for {key}: {e}")
            return None
            
    def _get_size(self, path: Path) -> int:
        """Get size of file or directory in bytes"""
        if path.is_file():
            return path.stat().st_size
        elif path.is_dir():
            return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        return 0
        
    def _cleanup_if_needed(self):
        """Clean up old cache entries if size exceeds limit"""
        total_size = sum(entry["size"] for entry in self.cache_index.values())
        
        if total_size > self.max_size_gb * 1024**3:
            # Remove oldest entries
            sorted_entries = sorted(
                self.cache_index.items(),
                key=lambda x: Path(x[1]["path"]).stat().st_mtime
            )
            
            while total_size > self.max_size_gb * 1024**3 and sorted_entries:
                key, entry = sorted_entries.pop(0)
                path = Path(entry["path"])
                
                # Remove file/directory
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    import shutil
                    shutil.rmtree(path)
                    
                total_size -= entry["size"]
                del self.cache_index[key]
                
    def clear(self):
        """Clear all cache"""
        import shutil
        shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_index.clear()


class ModelSwapper:
    """Manages swapping between different 3D models"""
    
    def __init__(self):
        self.current_model = None
        self.current_model_name = None
        self.loaded_components: Dict[str, Any] = {}
        
    def swap_model(
        self,
        new_model_name: str,
        new_model_loader,
        force_unload: bool = True
    ) -> Tuple[bool, str]:
        """Swap to a new model, unloading current if needed"""
        
        # Check if same model
        if self.current_model_name == new_model_name and self.current_model is not None:
            return True, f"{new_model_name} is already loaded"
            
        # Unload current model if requested
        if force_unload and self.current_model is not None:
            success, msg = self.unload_current()
            if not success:
                return False, f"Failed to unload current model: {msg}"
                
        # Load new model
        try:
            new_model = new_model_loader()
            self.current_model = new_model
            self.current_model_name = new_model_name
            return True, f"Successfully loaded {new_model_name}"
            
        except Exception as e:
            logger.error(f"Failed to load {new_model_name}: {e}")
            return False, f"Failed to load {new_model_name}: {str(e)}"
            
    def unload_current(self) -> Tuple[bool, str]:
        """Unload current model and free memory"""
        if self.current_model is None:
            return True, "No model loaded"
            
        try:
            model_name = self.current_model_name
            
            # Call model's unload method if available
            if hasattr(self.current_model, 'unload'):
                self.current_model.unload()
                
            # Clear references
            self.current_model = None
            self.current_model_name = None
            self.loaded_components.clear()
            
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
            return True, f"Unloaded {model_name}"
            
        except Exception as e:
            logger.error(f"Error unloading model: {e}")
            return False, str(e)
            
    def register_component(self, name: str, component: Any):
        """Register a loaded component"""
        self.loaded_components[name] = component
        
    def get_component(self, name: str) -> Optional[Any]:
        """Get a loaded component"""
        return self.loaded_components.get(name)


class ThreeDMemoryManager:
    """Central memory manager for 3D generation"""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.intermediate_cache = IntermediateCache(cache_dir / "intermediate")
        self.model_swapper = ModelSwapper()
        
        # Memory thresholds (GB)
        self.critical_threshold = 2.0  # Force unload if less than this
        self.warning_threshold = 4.0   # Warn if less than this
        
    def check_memory_available(self, required_gb: float) -> Tuple[bool, str]:
        """Check if enough memory is available"""
        available = self.get_available_memory()
        
        if available < required_gb:
            # Try to free memory
            freed = self.free_memory(required_gb - available)
            available = self.get_available_memory()
            
            if available < required_gb:
                return False, (
                    f"Insufficient memory: {available:.1f}GB available, "
                    f"{required_gb:.1f}GB required"
                )
                
        return True, f"{available:.1f}GB available"
        
    def get_available_memory(self) -> float:
        """Get available GPU memory in GB"""
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            return free / (1024**3)
        else:
            # For CPU, use system RAM
            mem = psutil.virtual_memory()
            return mem.available / (1024**3)
            
    def get_memory_usage(self) -> Dict[str, float]:
        """Get detailed memory usage"""
        usage = {}
        
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            
            usage["gpu_total_gb"] = total / (1024**3)
            usage["gpu_free_gb"] = free / (1024**3)
            usage["gpu_allocated_gb"] = allocated / (1024**3)
            usage["gpu_reserved_gb"] = reserved / (1024**3)
        else:
            mem = psutil.virtual_memory()
            usage["ram_total_gb"] = mem.total / (1024**3)
            usage["ram_available_gb"] = mem.available / (1024**3)
            usage["ram_used_gb"] = mem.used / (1024**3)
            
        return usage
        
    def free_memory(self, target_gb: float) -> float:
        """Try to free at least target_gb of memory"""
        initial_free = self.get_available_memory()
        
        # Clear intermediate cache
        self.intermediate_cache.clear()
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        # Check if current model can be unloaded
        if self.model_swapper.current_model is not None:
            available = self.get_available_memory()
            if available < self.critical_threshold:
                logger.warning("Critical memory level - unloading current model")
                self.model_swapper.unload_current()
                
                # Another round of cleanup
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
        final_free = self.get_available_memory()
        freed = final_free - initial_free
        
        logger.info(f"Freed {freed:.1f}GB of memory")
        return freed
        
    def should_use_sequential_offload(self) -> bool:
        """Check if sequential CPU offload should be used"""
        available = self.get_available_memory()
        return available < self.warning_threshold
        
    def get_optimal_batch_size(self, base_batch_size: int = 1) -> int:
        """Get optimal batch size based on available memory"""
        available = self.get_available_memory()
        
        if available > 16:
            return base_batch_size * 4
        elif available > 12:
            return base_batch_size * 2
        elif available > 8:
            return base_batch_size
        else:
            return max(1, base_batch_size // 2)
            
    def log_memory_status(self):
        """Log current memory status"""
        usage = self.get_memory_usage()
        
        if "gpu_free_gb" in usage:
            logger.info(
                f"GPU Memory: {usage['gpu_free_gb']:.1f}GB free / "
                f"{usage['gpu_total_gb']:.1f}GB total"
            )
        else:
            logger.info(
                f"RAM: {usage['ram_available_gb']:.1f}GB available / "
                f"{usage['ram_total_gb']:.1f}GB total"
            )