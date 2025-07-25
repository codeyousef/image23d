"""Memory optimization utilities for FLUX models.

This module implements memory management strategies from the Flux.1 Dev Implementation Guide
to prevent CUDA out of memory errors, especially for GGUF Q8 models on RTX 4090.
"""

import os
import gc
import torch
import logging
from typing import Optional, Dict, Any
import psutil

logger = logging.getLogger(__name__)


def setup_memory_optimization_env():
    """Set up environment variables for optimal CUDA memory management.
    
    Based on Flux.1 Dev Implementation Guide recommendations.
    """
    # Critical memory optimization settings
    env_vars = {
        # Prevent memory fragmentation with expandable segments
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,max_split_size_mb:128",
        
        # Enable CUDA memory caching
        "PYTORCH_NO_CUDA_MEMORY_CACHING": "0",
        
        # Optimize CUDA operations
        "CUDA_LAUNCH_BLOCKING": "0",
        
        # Enable cuDNN optimizations
        "TORCH_CUDNN_V8_API_ENABLED": "1",
        
        # Enable TensorCore operations
        "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "1",
        
        # Optimize memory allocator
        "TORCHINDUCTOR_CACHE_DIR": "/tmp/inductor-cache",
        "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
        
        # Enable HuggingFace transfer optimization
        "HF_HUB_ENABLE_HF_TRANSFER": "1"
    }
    
    for key, value in env_vars.items():
        if key not in os.environ:
            os.environ[key] = value
            logger.info(f"Set {key}={value}")
        else:
            logger.info(f"{key} already set to {os.environ[key]}")
    
    # Apply PyTorch settings
    if torch.cuda.is_available():
        # Set memory fraction to prevent OOM
        torch.cuda.set_per_process_memory_fraction(0.95)
        
        # Enable TF32 for Ampere GPUs (RTX 3000+)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Set optimal precision
        torch.set_float32_matmul_precision('high')
        
        logger.info("Applied PyTorch CUDA optimizations")


class MemoryManager:
    """Manages GPU memory for FLUX model operations."""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vram_gb = 0
        self.reserved_memory_gb = 2.0  # Reserve 2GB for system
        
        if self.device == "cuda":
            self.vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"Available VRAM: {self.vram_gb:.1f} GB")
    
    def get_available_memory(self) -> float:
        """Get currently available GPU memory in GB."""
        if self.device != "cuda":
            return 0
        
        free_memory = torch.cuda.mem_get_info()[0] / (1024**3)
        return free_memory
    
    def check_memory_for_model(self, model_size_gb: float, overhead_gb: float = 8.0) -> Dict[str, Any]:
        """Check if there's enough memory for a model.
        
        Args:
            model_size_gb: Size of the model in GB
            overhead_gb: Additional memory needed for generation (default 8GB)
        
        Returns:
            Dict with memory status and recommendations
        """
        total_required = model_size_gb + overhead_gb
        available = self.get_available_memory()
        
        result = {
            "has_enough_memory": available >= total_required,
            "available_gb": available,
            "required_gb": total_required,
            "model_size_gb": model_size_gb,
            "overhead_gb": overhead_gb,
            "recommendations": []
        }
        
        # Early return if enough memory
        if result["has_enough_memory"]:
            return result
            
        # Calculate deficit and add recommendations
        deficit = total_required - available
        result["deficit_gb"] = deficit
        
        # Add recommendations
        if model_size_gb >= 10:  # Q8 models
            result["recommendations"].append("Use sequential CPU offload for Q8 models")
            result["recommendations"].append("Consider using Q6 or Q5 quantization instead")
        
        result["recommendations"].append("Enable VAE slicing and tiling")
        result["recommendations"].append("Reduce batch size to 1")
        result["recommendations"].append("Lower resolution if possible")
        
        return result
    
    def aggressive_memory_clear(self):
        """Aggressively clear GPU memory."""
        if self.device == "cuda":
            # Clear PyTorch cache
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Reset memory stats
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()
            
        # Force garbage collection
        gc.collect()
        
        logger.info("Performed aggressive memory clearing")
    
    def optimize_model_for_memory(self, pipeline: Any) -> Any:
        """Apply memory optimizations to a pipeline.
        
        Args:
            pipeline: Diffusers pipeline to optimize
            
        Returns:
            Optimized pipeline
        """
        if not hasattr(pipeline, 'vae'):
            return pipeline
        
        # Enable VAE optimizations
        if hasattr(pipeline.vae, 'enable_slicing'):
            pipeline.vae.enable_slicing()
            logger.info("Enabled VAE slicing")
        
        if hasattr(pipeline.vae, 'enable_tiling'):
            pipeline.vae.enable_tiling()
            # Set optimal tile sizes
            if hasattr(pipeline.vae, 'tile_sample_min_size'):
                pipeline.vae.tile_sample_min_size = 512
            if hasattr(pipeline.vae, 'tile_overlap_factor'):
                pipeline.vae.tile_overlap_factor = 0.25
            logger.info("Enabled VAE tiling with optimal settings")
        
        # Enable attention slicing for transformer
        if hasattr(pipeline, 'enable_attention_slicing'):
            pipeline.enable_attention_slicing(slice_size=1)
            logger.info("Enabled attention slicing")
        
        return pipeline
    
    def should_use_sequential_offload(self, model_name: str, model_size_gb: float) -> bool:
        """Determine if sequential CPU offload should be used.
        
        Based on model size and available memory.
        """
        # Always use sequential offload for Q8 models
        if any(q in model_name.lower() for q in ["q8", "q6"]):
            available = self.get_available_memory()
            # Q8 needs ~10-12GB + 8-10GB overhead = ~20GB total
            # On 24GB GPU, this is tight, so use offload
            if available < 22:
                logger.info(f"Q8/Q6 model detected with {available:.1f}GB available - using sequential offload")
                return True
        
        # Check general memory requirements
        memory_check = self.check_memory_for_model(model_size_gb)
        return not memory_check["has_enough_memory"]
    
    def monitor_memory_usage(self, stage: str):
        """Log current memory usage."""
        if self.device != "cuda":
            return
        
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        free = torch.cuda.mem_get_info()[0] / (1024**3)
        
        logger.info(f"Memory usage at {stage}:")
        logger.info(f"  Allocated: {allocated:.2f} GB")
        logger.info(f"  Reserved: {reserved:.2f} GB")
        logger.info(f"  Free: {free:.2f} GB")
        
        # Warn if memory is low
        if free < 2.0:
            logger.warning(f"Low GPU memory: only {free:.2f} GB free!")


# Global memory manager instance
_memory_manager = None


def get_memory_manager() -> MemoryManager:
    """Get the global memory manager instance."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager


def setup_flux_memory_optimization():
    """Complete setup for FLUX memory optimization."""
    # Set environment variables
    setup_memory_optimization_env()
    
    # Get memory manager
    manager = get_memory_manager()
    
    # Clear memory initially
    manager.aggressive_memory_clear()
    
    # Log initial state
    manager.monitor_memory_usage("initialization")
    
    return manager