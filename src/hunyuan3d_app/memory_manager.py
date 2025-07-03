"""Advanced memory management for optimal GPU utilization."""

import gc
import logging
from typing import Optional, Dict, Callable
import torch
import psutil
from functools import wraps
import time

logger = logging.getLogger(__name__)


class MemoryManager:
    """Manages GPU and system memory for optimal performance."""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.monitoring_enabled = True
        
    def get_memory_stats(self):
        """Get current memory statistics."""
        stats = {
            "ram": {
                "used_gb": psutil.virtual_memory().used / (1024**3),
                "total_gb": psutil.virtual_memory().total / (1024**3),
                "percent": psutil.virtual_memory().percent,
                "available_gb": psutil.virtual_memory().available / (1024**3)
            }
        }
        
        if torch.cuda.is_available():
            stats["vram"] = {
                "allocated_gb": torch.cuda.memory_allocated() / (1024**3),
                "reserved_gb": torch.cuda.memory_reserved() / (1024**3),
                "total_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
                "free_gb": torch.cuda.mem_get_info()[0] / (1024**3),
            }
            stats["vram"]["percent"] = (stats["vram"]["allocated_gb"] / stats["vram"]["total_gb"]) * 100
            
        return stats
        
    def optimize_memory(self):
        """Perform comprehensive memory optimization."""
        initial_stats = self.get_memory_stats()
        logger.info("Starting memory optimization...")
        
        # Python garbage collection
        gc.collect()
        
        # PyTorch specific cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        # Force garbage collection again
        gc.collect()
        
        final_stats = self.get_memory_stats()
        
        # Log improvements
        if "vram" in initial_stats and "vram" in final_stats:
            vram_freed = initial_stats["vram"]["allocated_gb"] - final_stats["vram"]["allocated_gb"]
            if vram_freed > 0.1:  # More than 100MB freed
                logger.info(f"Freed {vram_freed:.2f}GB of VRAM")
                
        ram_freed = initial_stats["ram"]["used_gb"] - final_stats["ram"]["used_gb"]
        if ram_freed > 0.5:  # More than 500MB freed
            logger.info(f"Freed {ram_freed:.2f}GB of RAM")
            
    def clear_cache_aggressive(self):
        """Aggressively clear all caches - use with caution."""
        # Clear Python caches
        gc.collect()
        gc.collect()  # Run twice to ensure cleanup
        
        if torch.cuda.is_available():
            # Clear PyTorch caches
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            
            # Reset peak memory stats
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()
            
        logger.info("Performed aggressive cache cleanup")
        
    def monitor_memory_usage(self, func):
        """Decorator to monitor memory usage of a function."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not self.monitoring_enabled:
                return func(*args, **kwargs)
                
            # Get initial memory state
            initial_stats = self.get_memory_stats()
            start_time = time.time()
            
            # Run the function
            try:
                result = func(*args, **kwargs)
            finally:
                # Get final memory state
                final_stats = self.get_memory_stats()
                duration = time.time() - start_time
                
                # Log memory changes
                ram_delta = final_stats["ram"]["used_gb"] - initial_stats["ram"]["used_gb"]
                logger.info(f"{func.__name__} - Duration: {duration:.2f}s, RAM delta: {ram_delta:+.2f}GB")
                
                if "vram" in final_stats:
                    vram_delta = final_stats["vram"]["allocated_gb"] - initial_stats["vram"]["allocated_gb"]
                    logger.info(f"{func.__name__} - VRAM delta: {vram_delta:+.2f}GB")
                    
            return result
        return wrapper
        
    def ensure_memory_available(self, required_gb, device):
        """Ensure sufficient memory is available, cleaning if necessary."""
        if device == "cuda" and torch.cuda.is_available():
            available = torch.cuda.mem_get_info()[0] / (1024**3)
            if available < required_gb:
                logger.warning(f"Insufficient VRAM: {available:.1f}GB available, {required_gb:.1f}GB required")
                self.optimize_memory()
                # Check again
                available = torch.cuda.mem_get_info()[0] / (1024**3)
                if available < required_gb:
                    logger.error(f"Still insufficient VRAM after cleanup: {available:.1f}GB available")
                    return False
        else:
            available = psutil.virtual_memory().available / (1024**3)
            if available < required_gb:
                logger.warning(f"Insufficient RAM: {available:.1f}GB available, {required_gb:.1f}GB required")
                self.optimize_memory()
                # Check again
                available = psutil.virtual_memory().available / (1024**3)
                if available < required_gb:
                    logger.error(f"Still insufficient RAM after cleanup: {available:.1f}GB available")
                    return False
                    
        return True
        
    def get_memory_summary(self):
        """Get a formatted memory summary string."""
        stats = self.get_memory_stats()
        summary = []
        
        # RAM summary
        ram = stats["ram"]
        summary.append(f"RAM: {ram['used_gb']:.1f}/{ram['total_gb']:.1f}GB ({ram['percent']:.1f}%)")
        
        # VRAM summary if available
        if "vram" in stats:
            vram = stats["vram"]
            summary.append(f"VRAM: {vram['allocated_gb']:.1f}/{vram['total_gb']:.1f}GB ({vram['percent']:.1f}%)")
            
        return " | ".join(summary)
        
    def optimize_model_for_inference(self, model):
        """Optimize a model for inference to reduce memory usage."""
        if model is None:
            return model
            
        # Set to eval mode
        model.eval()
        
        # Disable gradients
        for param in model.parameters():
            param.requires_grad = False
            
        # Enable memory efficient settings if available
        if hasattr(model, "enable_gradient_checkpointing"):
            model.enable_gradient_checkpointing()
            logger.info("Enabled gradient checkpointing")
            
        return model
        

# Global instance
_memory_manager = None

def get_memory_manager():
    """Get or create the global memory manager instance."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager