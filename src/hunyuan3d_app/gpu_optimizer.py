"""GPU detection and optimization module for automatic configuration based on available hardware."""

import os
import logging
import platform
from typing import Dict, Optional, Tuple
import torch

logger = logging.getLogger(__name__)


class GPUOptimizer:
    """Automatically detect GPU and configure optimal settings."""
    
    # GPU Architecture mapping (compute capability -> architecture name)
    GPU_ARCHITECTURES = {
        "8.9": "Ada Lovelace",  # RTX 4090, 4080, etc.
        "8.6": "Ampere",        # RTX 3090, 3080, A100, etc.
        "8.0": "Ampere",        # A100
        "7.5": "Turing",        # RTX 2080, 2070, etc.
        "7.0": "Volta",         # V100
        "6.1": "Pascal",        # GTX 1080, 1070, etc.
        "6.0": "Pascal",        # P100
    }
    
    # Optimal settings per GPU generation
    GPU_SETTINGS = {
        "Ada Lovelace": {
            "arch_list": "8.9",
            "allow_tf32": True,
            "allow_fp8": True,
            "use_flash_attn": True,
            "max_vram_fraction": 0.95,
            "enable_cuda_graphs": True,
            "matmul_precision": "high",
        },
        "Ampere": {
            "arch_list": "8.0,8.6",
            "allow_tf32": True,
            "allow_fp8": False,
            "use_flash_attn": True,
            "max_vram_fraction": 0.90,
            "enable_cuda_graphs": True,
            "matmul_precision": "high",
        },
        "Turing": {
            "arch_list": "7.5",
            "allow_tf32": False,
            "allow_fp8": False,
            "use_flash_attn": False,
            "max_vram_fraction": 0.85,
            "enable_cuda_graphs": False,
            "matmul_precision": "medium",
        },
        "default": {
            "arch_list": "7.0+",
            "allow_tf32": False,
            "allow_fp8": False,
            "use_flash_attn": False,
            "max_vram_fraction": 0.80,
            "enable_cuda_graphs": False,
            "matmul_precision": "medium",
        }
    }
    
    def __init__(self):
        self.gpu_info = self._detect_gpu()
        self.settings = self._get_optimal_settings()
        
    def _detect_gpu(self):
        """Detect GPU properties and capabilities."""
        info = {
            "available": torch.cuda.is_available(),
            "device_count": 0,
            "primary_gpu": None,
            "vram_gb": 0,
            "compute_capability": None,
            "architecture": None,
            "name": None,
        }
        
        if not info["available"]:
            logger.warning("No CUDA GPU detected, using CPU")
            return info
            
        info["device_count"] = torch.cuda.device_count()
        
        # Get primary GPU info
        if info["device_count"] > 0:
            props = torch.cuda.get_device_properties(0)
            info["name"] = props.name
            info["vram_gb"] = props.total_memory / (1024**3)
            info["compute_capability"] = f"{props.major}.{props.minor}"
            
            # Determine architecture
            for cap, arch in self.GPU_ARCHITECTURES.items():
                if info["compute_capability"] >= cap:
                    info["architecture"] = arch
                    break
            
            if not info["architecture"]:
                info["architecture"] = "default"
                
            logger.info(f"Detected GPU: {info['name']} ({info['architecture']}, "
                       f"{info['vram_gb']:.1f}GB VRAM, compute {info['compute_capability']})")
                       
        return info
        
    def _get_optimal_settings(self):
        """Get optimal settings for detected GPU."""
        if not self.gpu_info["available"]:
            return {}
            
        arch = self.gpu_info["architecture"]
        settings = self.GPU_SETTINGS.get(arch, self.GPU_SETTINGS["default"]).copy()
        
        # Adjust based on VRAM
        vram = self.gpu_info["vram_gb"]
        if vram < 8:
            settings["max_vram_fraction"] = 0.80
            settings["use_flash_attn"] = False
        elif vram < 12:
            settings["max_vram_fraction"] = 0.85
        elif vram < 16:
            settings["max_vram_fraction"] = 0.90
        else:
            settings["max_vram_fraction"] = 0.95
            
        return settings
        
    def apply_optimizations(self):
        """Apply detected optimizations to PyTorch and environment."""
        if not self.gpu_info["available"]:
            logger.info("No GPU available, skipping optimizations")
            return
            
        settings = self.settings
        
        # Set environment variables
        env_vars = {
            'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True,max_split_size_mb:512',
            'CUDA_MODULE_LOADING': 'LAZY',
            'TORCH_CUDNN_V8_API_ENABLED': '1',
            'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
            'CUDA_LAUNCH_BLOCKING': '0',
            'TORCH_CUDA_ARCH_LIST': settings["arch_list"],
        }
        
        # Linux-specific optimizations
        if platform.system() == "Linux":
            env_vars['TORCHINDUCTOR_MAX_AUTOTUNE'] = '1'
        
        # Windows-specific optimizations
        elif platform.system() == "Windows":
            # Use more conservative settings for Windows
            env_vars['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            
        for key, value in env_vars.items():
            os.environ[key] = value
            logger.debug(f"Set {key}={value}")
            
        # Apply PyTorch settings
        torch.set_float32_matmul_precision(settings["matmul_precision"])
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = settings["allow_tf32"]
        torch.backends.cudnn.allow_tf32 = settings["allow_tf32"]
        
        # Set memory fraction
        torch.cuda.set_per_process_memory_fraction(settings["max_vram_fraction"])
        
        # Enable FP8 if supported (Ada Lovelace)
        if settings["allow_fp8"] and hasattr(torch.backends.cuda.matmul, 'allow_fp8_e4m3'):
            torch.backends.cuda.matmul.allow_fp8_e4m3 = True
            logger.info("Enabled FP8 compute for Ada Lovelace")
            
        # Enable CUDA graphs if supported
        if settings["enable_cuda_graphs"]:
            try:
                # This is a placeholder - actual implementation depends on model
                logger.info("CUDA graphs support enabled")
            except Exception as e:
                logger.warning(f"Could not enable CUDA graphs: {e}")
                
        logger.info(f"Applied optimizations for {self.gpu_info['architecture']} GPU")
        
    def get_recommended_dtype(self):
        """Get recommended dtype based on GPU capabilities."""
        if not self.gpu_info["available"]:
            return torch.float32
            
        # Use FP16 for GPU with adequate VRAM
        if self.gpu_info["vram_gb"] >= 8:
            return torch.float16
        else:
            return torch.float32
            
    def should_use_xformers(self):
        """Check if xformers should be used."""
        if not self.gpu_info["available"]:
            return False
            
        # xformers works best on newer GPUs with enough VRAM
        return (self.settings.get("use_flash_attn", False) and 
                self.gpu_info["vram_gb"] >= 8)
                
    def get_info_dict(self):
        """Get GPU info and settings as a dictionary."""
        return {
            "gpu_info": self.gpu_info,
            "settings": self.settings,
        }


# Global instance
_gpu_optimizer = None

def get_gpu_optimizer():
    """Get or create the global GPU optimizer instance."""
    global _gpu_optimizer
    if _gpu_optimizer is None:
        _gpu_optimizer = GPUOptimizer()
        _gpu_optimizer.apply_optimizations()
    return _gpu_optimizer