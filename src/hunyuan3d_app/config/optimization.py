"""Advanced optimization configurations for HunYuan3D models"""

import os
import platform
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for model optimization settings"""
    
    # Memory optimizations
    use_cpu_offload: bool = False
    use_sequential_cpu_offload: bool = False
    enable_attention_slicing: bool = True
    enable_memory_efficient_attention: bool = True
    
    # Performance optimizations
    use_torch_compile: bool = False
    enable_flash_attention: bool = True
    use_channels_last: bool = True
    enable_tf32: bool = True
    
    # Quantization settings
    enable_quantization: bool = False
    quantization_level: str = "Q8_0"
    use_dynamic_quantization: bool = False
    
    # Platform-specific
    platform_optimized: bool = True
    disable_triton_on_windows: bool = True


class OptimizationManager:
    """Manages optimization settings based on system capabilities"""
    
    def __init__(self):
        self.platform = platform.system().lower()
        self.is_windows = self.platform == "windows"
        self.is_linux = self.platform == "linux"
        
        # Detect hardware capabilities
        self.has_cuda = self._detect_cuda()
        self.cuda_capability = self._get_cuda_capability()
        self.vram_gb = self._detect_vram()
        
        logger.info(f"OptimizationManager initialized: {self.platform} | CUDA: {self.has_cuda} | VRAM: {self.vram_gb:.1f}GB")
    
    def _detect_cuda(self) -> bool:
        """Detect CUDA availability"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _get_cuda_capability(self) -> Optional[tuple]:
        """Get CUDA compute capability"""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.get_device_capability(0)
        except:
            pass
        return None
    
    def _detect_vram(self) -> float:
        """Detect VRAM in GB"""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except:
            pass
        return 0.0
    
    def get_optimization_config(self, memory_profile: str = "auto") -> OptimizationConfig:
        """Get optimized configuration based on system capabilities
        
        Args:
            memory_profile: Memory profile (auto, minimal, standard, high, ultra)
            
        Returns:
            Optimized configuration
        """
        config = OptimizationConfig()
        
        # Auto-detect memory profile if needed
        if memory_profile == "auto":
            memory_profile = self._get_auto_memory_profile()
        
        # Apply platform-specific optimizations
        self._apply_platform_optimizations(config)
        
        # Apply memory-based optimizations
        self._apply_memory_optimizations(config, memory_profile)
        
        # Apply hardware-specific optimizations
        self._apply_hardware_optimizations(config)
        
        logger.info(f"Generated optimization config for {memory_profile} profile on {self.platform}")
        return config
    
    def _get_auto_memory_profile(self) -> str:
        """Auto-detect optimal memory profile"""
        if self.vram_gb >= 32:
            return "ultra"
        elif self.vram_gb >= 16:
            return "high"
        elif self.vram_gb >= 8:
            return "standard"
        else:
            return "minimal"
    
    def _apply_platform_optimizations(self, config: OptimizationConfig):
        """Apply platform-specific optimizations"""
        if self.is_windows:
            # Windows-specific optimizations
            config.use_torch_compile = False  # Triton issues on Windows
            config.disable_triton_on_windows = True
            
            # Set environment variables for Windows
            os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
            os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1") 
            os.environ.setdefault("TRITON_DISABLE", "1")
            
        elif self.is_linux:
            # Linux-specific optimizations
            config.use_torch_compile = True  # Triton works better on Linux
            config.disable_triton_on_windows = False
    
    def _apply_memory_optimizations(self, config: OptimizationConfig, profile: str):
        """Apply memory-based optimizations"""
        if profile == "minimal":
            config.use_cpu_offload = True
            config.use_sequential_cpu_offload = True
            config.enable_attention_slicing = True
            config.enable_memory_efficient_attention = True
            config.enable_quantization = True
            config.quantization_level = "Q4_K_M"
            
        elif profile == "standard":
            config.use_cpu_offload = False
            config.use_sequential_cpu_offload = False
            config.enable_attention_slicing = True
            config.enable_memory_efficient_attention = True
            config.enable_quantization = False
            
        elif profile == "high":
            config.use_cpu_offload = False
            config.use_sequential_cpu_offload = False
            config.enable_attention_slicing = False
            config.enable_memory_efficient_attention = True
            config.enable_quantization = False
            
        elif profile == "ultra":
            config.use_cpu_offload = False
            config.use_sequential_cpu_offload = False
            config.enable_attention_slicing = False
            config.enable_memory_efficient_attention = False
            config.enable_quantization = False
    
    def _apply_hardware_optimizations(self, config: OptimizationConfig):
        """Apply hardware-specific optimizations"""
        if self.has_cuda:
            # CUDA optimizations
            config.enable_tf32 = True
            config.use_channels_last = True
            
            # Flash Attention support (requires compatible GPU)
            if self.cuda_capability and self.cuda_capability >= (8, 0):
                config.enable_flash_attention = True
            else:
                config.enable_flash_attention = False
                logger.info("Flash Attention disabled - requires Ampere GPU or newer")
        else:
            # CPU-only optimizations
            config.enable_tf32 = False
            config.enable_flash_attention = False
            config.use_cpu_offload = True
    
    def apply_torch_optimizations(self, config: OptimizationConfig):
        """Apply PyTorch-level optimizations"""
        try:
            import torch
            
            # Enable TF32 for faster training on Ampere GPUs
            if config.enable_tf32 and self.has_cuda:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                logger.info("Enabled TF32 for faster computation")
            
            # Enable memory efficient attention
            if config.enable_memory_efficient_attention:
                try:
                    torch.backends.cuda.enable_math_sdp(True)
                    torch.backends.cuda.enable_flash_sdp(config.enable_flash_attention)
                    torch.backends.cuda.enable_mem_efficient_sdp(True)
                    logger.info("Enabled memory efficient attention")
                except AttributeError:
                    logger.warning("Memory efficient attention not available in this PyTorch version")
            
            # Set memory format
            if config.use_channels_last:
                # This will be applied per-tensor as needed
                logger.info("Channels-last memory format enabled")
                
        except ImportError:
            logger.warning("PyTorch not available for optimization")
    
    def get_environment_variables(self, config: OptimizationConfig) -> Dict[str, str]:
        """Get environment variables for optimization"""
        env_vars = {}
        
        # Triton/Torch compile settings
        if config.disable_triton_on_windows and self.is_windows:
            env_vars.update({
                "TORCH_COMPILE_DISABLE": "1",
                "TORCHINDUCTOR_DISABLE": "1",
                "TRITON_DISABLE": "1"
            })
        
        # Memory settings
        if config.enable_memory_efficient_attention:
            env_vars["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
        
        # Platform settings
        if self.is_windows:
            env_vars["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        
        return env_vars


# Global optimization manager instance
_optimization_manager = None

def get_optimization_manager() -> OptimizationManager:
    """Get global optimization manager instance"""
    global _optimization_manager
    if _optimization_manager is None:
        _optimization_manager = OptimizationManager()
    return _optimization_manager


def apply_global_optimizations(memory_profile: str = "auto"):
    """Apply global optimization settings"""
    manager = get_optimization_manager()
    config = manager.get_optimization_config(memory_profile)
    
    # Apply PyTorch optimizations
    manager.apply_torch_optimizations(config)
    
    # Set environment variables
    env_vars = manager.get_environment_variables(config)
    for key, value in env_vars.items():
        os.environ.setdefault(key, value)
    
    logger.info(f"Applied global optimizations for {memory_profile} profile")
    return config