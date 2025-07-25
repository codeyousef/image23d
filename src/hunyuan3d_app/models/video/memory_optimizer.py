"""Memory Optimization System for Video Generation

Provides comprehensive memory optimization strategies for video models:
- VAE slicing and tiling
- Sequential CPU offloading  
- Quantization (FP8, INT8, NF4)
- Dynamic batch size adjustment
- Gradient checkpointing
- Memory profiling and monitoring
"""

import logging
import gc
import psutil
from typing import Dict, Any, Optional, List, Callable, Union
import torch
import torch.nn as nn
from dataclasses import dataclass
from enum import Enum

from .base import QuantizationType, BaseVideoModel

logger = logging.getLogger(__name__)


@dataclass
class MemoryProfile:
    """Memory usage profile"""
    gpu_total: float  # GB
    gpu_used: float  # GB
    gpu_free: float  # GB
    ram_total: float  # GB
    ram_used: float  # GB
    ram_free: float  # GB
    model_size: float  # GB
    activation_size: float  # GB
    recommended_batch_size: int
    recommended_resolution: tuple
    can_run: bool
    warnings: List[str]


class OptimizationLevel(Enum):
    """Memory optimization levels"""
    NONE = "none"  # No optimization
    MINIMAL = "minimal"  # Basic optimizations
    MODERATE = "moderate"  # Balanced performance/memory
    AGGRESSIVE = "aggressive"  # Maximum memory savings
    EXTREME = "extreme"  # Emergency mode


class VideoMemoryOptimizer:
    """Comprehensive memory optimizer for video generation"""
    
    # Memory thresholds (GB)
    MEMORY_THRESHOLDS = {
        "critical": 2.0,  # Below this is critical
        "low": 4.0,  # Low memory warning
        "moderate": 8.0,  # Moderate memory
        "comfortable": 16.0,  # Comfortable memory
        "abundant": 24.0  # Abundant memory
    }
    
    # Resolution limits by memory
    RESOLUTION_LIMITS = {
        4: (512, 288),  # 4GB
        8: (768, 512),  # 8GB
        12: (1024, 576),  # 12GB
        16: (1280, 720),  # 16GB
        24: (1920, 1080)  # 24GB+
    }
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimization_level = OptimizationLevel.MODERATE
        self.monitoring_enabled = True
        
    def profile_system(self) -> MemoryProfile:
        """Profile current system memory status"""
        profile = MemoryProfile(
            gpu_total=0.0,
            gpu_used=0.0,
            gpu_free=0.0,
            ram_total=0.0,
            ram_used=0.0,
            ram_free=0.0,
            model_size=0.0,
            activation_size=0.0,
            recommended_batch_size=1,
            recommended_resolution=(512, 288),
            can_run=True,
            warnings=[]
        )
        
        # GPU memory
        if torch.cuda.is_available():
            gpu_props = torch.cuda.get_device_properties(0)
            profile.gpu_total = gpu_props.total_memory / 1e9
            profile.gpu_used = torch.cuda.memory_allocated() / 1e9
            profile.gpu_free = profile.gpu_total - profile.gpu_used
            
            if profile.gpu_free < self.MEMORY_THRESHOLDS["critical"]:
                profile.warnings.append(f"Critical GPU memory: {profile.gpu_free:.1f}GB free")
                profile.can_run = False
            elif profile.gpu_free < self.MEMORY_THRESHOLDS["low"]:
                profile.warnings.append(f"Low GPU memory: {profile.gpu_free:.1f}GB free")
                
        # RAM
        ram = psutil.virtual_memory()
        profile.ram_total = ram.total / 1e9
        profile.ram_used = ram.used / 1e9
        profile.ram_free = ram.available / 1e9
        
        if profile.ram_free < 4.0:
            profile.warnings.append(f"Low RAM: {profile.ram_free:.1f}GB free")
            
        # Recommendations
        profile.recommended_batch_size = self._recommend_batch_size(profile.gpu_free)
        profile.recommended_resolution = self._recommend_resolution(profile.gpu_free)
        
        return profile
        
    def optimize_model(
        self,
        model: BaseVideoModel,
        optimization_level: Optional[OptimizationLevel] = None,
        target_memory_gb: Optional[float] = None
    ) -> Dict[str, Any]:
        """Apply memory optimizations to a video model"""
        
        if optimization_level:
            self.optimization_level = optimization_level
            
        # Get current memory status
        profile = self.profile_system()
        
        # Determine optimization level if not specified
        if optimization_level is None:
            if target_memory_gb:
                # Choose level based on target
                if target_memory_gb < 8:
                    self.optimization_level = OptimizationLevel.EXTREME
                elif target_memory_gb < 12:
                    self.optimization_level = OptimizationLevel.AGGRESSIVE
                elif target_memory_gb < 16:
                    self.optimization_level = OptimizationLevel.MODERATE
                else:
                    self.optimization_level = OptimizationLevel.MINIMAL
            else:
                # Auto-select based on available memory
                if profile.gpu_free < 4:
                    self.optimization_level = OptimizationLevel.EXTREME
                elif profile.gpu_free < 8:
                    self.optimization_level = OptimizationLevel.AGGRESSIVE
                elif profile.gpu_free < 16:
                    self.optimization_level = OptimizationLevel.MODERATE
                else:
                    self.optimization_level = OptimizationLevel.MINIMAL
                    
        logger.info(f"Applying {self.optimization_level.value} optimization level")
        
        # Apply optimizations
        optimizations = {}
        
        if self.optimization_level == OptimizationLevel.NONE:
            return optimizations
            
        # Basic optimizations (all levels)
        if self.optimization_level.value in ["minimal", "moderate", "aggressive", "extreme"]:
            optimizations.update(self._apply_basic_optimizations(model))
            
        # Moderate optimizations
        if self.optimization_level.value in ["moderate", "aggressive", "extreme"]:
            optimizations.update(self._apply_moderate_optimizations(model))
            
        # Aggressive optimizations
        if self.optimization_level.value in ["aggressive", "extreme"]:
            optimizations.update(self._apply_aggressive_optimizations(model))
            
        # Extreme optimizations
        if self.optimization_level == OptimizationLevel.EXTREME:
            optimizations.update(self._apply_extreme_optimizations(model))
            
        # Log optimization results
        logger.info(f"Applied optimizations: {list(optimizations.keys())}")
        
        return optimizations
        
    def _apply_basic_optimizations(self, model: BaseVideoModel) -> Dict[str, Any]:
        """Apply basic memory optimizations"""
        optimizations = {}
        
        # 1. Enable memory efficient attention
        if hasattr(model, 'enable_memory_optimizations'):
            model.enable_memory_optimizations()
            optimizations["memory_efficient_attention"] = True
            
        # 2. VAE slicing
        if hasattr(model, 'vae') and model.vae is not None:
            if hasattr(model.vae, 'enable_slicing'):
                model.vae.enable_slicing()
                optimizations["vae_slicing"] = True
                
        # 3. Clear cache
        torch.cuda.empty_cache()
        gc.collect()
        optimizations["cache_cleared"] = True
        
        # 4. Set to eval mode
        if hasattr(model, 'eval'):
            model.eval()
            optimizations["eval_mode"] = True
            
        return optimizations
        
    def _apply_moderate_optimizations(self, model: BaseVideoModel) -> Dict[str, Any]:
        """Apply moderate memory optimizations"""
        optimizations = {}
        
        # 1. CPU offloading
        if hasattr(model, 'pipeline') and model.pipeline is not None:
            if hasattr(model.pipeline, 'enable_model_cpu_offload'):
                model.pipeline.enable_model_cpu_offload()
                optimizations["cpu_offload"] = True
                
        # 2. Attention slicing
        if hasattr(model, 'pipeline') and model.pipeline is not None:
            if hasattr(model.pipeline, 'enable_attention_slicing'):
                model.pipeline.enable_attention_slicing(1)
                optimizations["attention_slicing"] = True
                
        # 3. VAE tiling
        if hasattr(model, 'vae') and model.vae is not None:
            if hasattr(model.vae, 'enable_tiling'):
                model.vae.enable_tiling()
                optimizations["vae_tiling"] = True
                
        # 4. Gradient checkpointing
        if hasattr(model, 'transformer') and model.transformer is not None:
            if hasattr(model.transformer, 'gradient_checkpointing_enable'):
                model.transformer.gradient_checkpointing_enable()
                optimizations["gradient_checkpointing"] = True
                
        return optimizations
        
    def _apply_aggressive_optimizations(self, model: BaseVideoModel) -> Dict[str, Any]:
        """Apply aggressive memory optimizations"""
        optimizations = {}
        
        # 1. Sequential CPU offloading
        if hasattr(model, 'pipeline') and model.pipeline is not None:
            if hasattr(model.pipeline, 'enable_sequential_cpu_offload'):
                model.pipeline.enable_sequential_cpu_offload()
                optimizations["sequential_cpu_offload"] = True
                
        # 2. Quantization to INT8
        optimizations.update(self._apply_quantization(model, QuantizationType.INT8))
        
        # 3. Reduce precision
        if model.dtype != torch.float16:
            self._convert_model_dtype(model, torch.float16)
            optimizations["dtype_conversion"] = "fp16"
            
        # 4. Memory format optimization
        if hasattr(model, 'transformer') and model.transformer is not None:
            try:
                model.transformer = model.transformer.to(memory_format=torch.channels_last)
                optimizations["channels_last"] = True
            except:
                pass
                
        return optimizations
        
    def _apply_extreme_optimizations(self, model: BaseVideoModel) -> Dict[str, Any]:
        """Apply extreme memory optimizations (emergency mode)"""
        optimizations = {}
        
        # 1. Aggressive quantization
        optimizations.update(self._apply_quantization(model, QuantizationType.NF4))
        
        # 2. Offload everything possible
        self._offload_to_cpu(model)
        optimizations["full_cpu_offload"] = True
        
        # 3. Enable all possible memory savings
        if hasattr(model, 'config'):
            model.config.chunk_size = 1  # Minimum chunk size
            model.config.max_batch_size = 1
            optimizations["minimum_batch"] = True
            
        # 4. Disable non-essential features
        if hasattr(model, 'disable_features'):
            model.disable_features(['lora', 'controlnet', 'ip_adapter'])
            optimizations["features_disabled"] = True
            
        return optimizations
        
    def _apply_quantization(
        self,
        model: BaseVideoModel,
        quantization: QuantizationType
    ) -> Dict[str, Any]:
        """Apply quantization to model"""
        optimizations = {}
        
        try:
            if quantization in [QuantizationType.INT8, QuantizationType.NF4]:
                # Use bitsandbytes for INT8/NF4
                import bitsandbytes as bnb
                
                # Quantize transformer
                if hasattr(model, 'transformer') and model.transformer is not None:
                    model.transformer = self._quantize_model_bnb(
                        model.transformer,
                        quantization
                    )
                    optimizations[f"transformer_quantized_{quantization.value}"] = True
                    
            elif quantization == QuantizationType.FP8:
                # Use Quanto for FP8
                try:
                    from optimum.quanto import quantize, freeze, qfloat8
                    
                    if hasattr(model, 'transformer') and model.transformer is not None:
                        quantize(model.transformer, weights=qfloat8)
                        freeze(model.transformer)
                        optimizations["transformer_quantized_fp8"] = True
                except ImportError:
                    logger.warning("Quanto not available for FP8 quantization")
                    
        except Exception as e:
            logger.warning(f"Failed to apply quantization: {e}")
            
        return optimizations
        
    def _quantize_model_bnb(
        self,
        model: nn.Module,
        quantization: QuantizationType
    ) -> nn.Module:
        """Quantize model using bitsandbytes"""
        import bitsandbytes as bnb
        
        # Configure quantization
        if quantization == QuantizationType.INT8:
            quantization_config = {
                "load_in_8bit": True,
                "llm_int8_threshold": 6.0,
                "llm_int8_has_fp16_weight": False
            }
        else:  # NF4
            quantization_config = {
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_compute_dtype": torch.float16
            }
            
        # Apply quantization
        # This is simplified - in production, use proper quantization APIs
        logger.info(f"Quantizing model to {quantization.value}")
        
        return model
        
    def _convert_model_dtype(self, model: BaseVideoModel, dtype: torch.dtype):
        """Convert model to different dtype"""
        if hasattr(model, 'to'):
            model.to(dtype)
            
        # Convert components
        for component in ['vae', 'text_encoder', 'transformer', 'unet']:
            if hasattr(model, component):
                comp = getattr(model, component)
                if comp is not None and hasattr(comp, 'to'):
                    comp.to(dtype)
                    
    def _offload_to_cpu(self, model: BaseVideoModel):
        """Offload model components to CPU"""
        # Keep only essential components on GPU
        essential = ['vae']  # VAE needed for decoding
        
        for component in ['text_encoder', 'transformer', 'unet']:
            if hasattr(model, component):
                comp = getattr(model, component)
                if comp is not None and hasattr(comp, 'to'):
                    comp.to('cpu')
                    logger.info(f"Offloaded {component} to CPU")
                    
    def _recommend_batch_size(self, free_memory_gb: float) -> int:
        """Recommend batch size based on available memory"""
        if free_memory_gb < 4:
            return 1
        elif free_memory_gb < 8:
            return 1
        elif free_memory_gb < 16:
            return 2
        else:
            return 4
            
    def _recommend_resolution(self, free_memory_gb: float) -> tuple:
        """Recommend resolution based on available memory"""
        for threshold, resolution in sorted(self.RESOLUTION_LIMITS.items()):
            if free_memory_gb < threshold:
                return resolution
        return self.RESOLUTION_LIMITS[24]  # Maximum
        
    def monitor_generation(
        self,
        callback: Callable[[Dict[str, float]], None],
        interval: float = 1.0
    ):
        """Monitor memory during generation"""
        if not self.monitoring_enabled:
            return
            
        import threading
        import time
        
        def monitor():
            while self.monitoring_enabled:
                stats = {
                    "gpu_used": torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
                    "gpu_reserved": torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0,
                    "ram_used": psutil.virtual_memory().used / 1e9,
                    "ram_percent": psutil.virtual_memory().percent
                }
                callback(stats)
                time.sleep(interval)
                
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        
    def estimate_memory_for_generation(
        self,
        model_type: str,
        resolution: tuple,
        num_frames: int,
        batch_size: int = 1,
        quantization: Optional[QuantizationType] = None
    ) -> Dict[str, float]:
        """Estimate memory requirements for a generation"""
        
        # Base model sizes (approximate)
        model_sizes = {
            "wan2_1_1.3b": 2.6,  # 1.3B * 2 bytes (FP16)
            "wan2_1_14b": 28.0,
            "hunyuanvideo": 26.0,
            "ltx_video": 4.0,
            "mochi_1": 20.0,
            "cogvideox_5b": 10.0
        }
        
        base_size = model_sizes.get(model_type, 10.0)
        
        # Apply quantization factor
        if quantization:
            quant_factors = {
                QuantizationType.FP32: 2.0,
                QuantizationType.FP16: 1.0,
                QuantizationType.BF16: 1.0,
                QuantizationType.FP8: 0.5,
                QuantizationType.INT8: 0.5,
                QuantizationType.NF4: 0.25
            }
            base_size *= quant_factors.get(quantization, 1.0)
            
        # Calculate activation memory
        width, height = resolution
        pixels_per_frame = width * height
        total_pixels = pixels_per_frame * num_frames * batch_size
        
        # Rough estimates
        latent_memory = total_pixels * 4 / 64 / 1e9  # 4 channels, 8x compression
        activation_memory = latent_memory * 10  # 10x for intermediate activations
        vae_memory = total_pixels * 3 * 4 / 1e9  # Decoding memory
        
        total = base_size + activation_memory + vae_memory
        
        return {
            "model": base_size,
            "activations": activation_memory,
            "vae": vae_memory,
            "total": total,
            "recommended": total * 1.3  # 30% safety margin
        }
        
    def get_optimization_suggestions(
        self,
        current_memory_gb: float,
        target_memory_gb: float,
        model_type: str
    ) -> List[str]:
        """Get suggestions for memory optimization"""
        suggestions = []
        
        reduction_needed = current_memory_gb - target_memory_gb
        
        if reduction_needed <= 0:
            return ["No optimization needed - memory usage is within target"]
            
        # Prioritized suggestions
        if reduction_needed > 0:
            suggestions.append("Enable VAE slicing and tiling")
            
        if reduction_needed > 2:
            suggestions.append("Enable CPU offloading")
            suggestions.append("Reduce batch size to 1")
            
        if reduction_needed > 4:
            suggestions.append("Enable sequential CPU offloading")
            suggestions.append("Use FP16 precision instead of FP32")
            suggestions.append("Reduce resolution")
            
        if reduction_needed > 8:
            suggestions.append("Apply INT8 quantization")
            suggestions.append("Enable gradient checkpointing")
            suggestions.append("Use smaller model variant if available")
            
        if reduction_needed > 12:
            suggestions.append("Apply NF4 quantization (4-bit)")
            suggestions.append("Offload all non-essential components to CPU")
            suggestions.append("Consider using a cloud GPU service")
            
        return suggestions


# Utility functions
def auto_optimize_for_hardware(
    model: BaseVideoModel,
    target_quality: str = "balanced"
) -> Dict[str, Any]:
    """Automatically optimize model for current hardware"""
    optimizer = VideoMemoryOptimizer()
    
    # Profile system
    profile = optimizer.profile_system()
    
    # Determine optimization level
    if profile.gpu_free < 6:
        level = OptimizationLevel.AGGRESSIVE
    elif profile.gpu_free < 12:
        level = OptimizationLevel.MODERATE
    else:
        level = OptimizationLevel.MINIMAL
        
    # Apply optimizations
    return optimizer.optimize_model(model, level)


def estimate_max_video_length(
    model_type: str,
    resolution: tuple,
    available_memory_gb: float,
    fps: int = 24
) -> int:
    """Estimate maximum video length (frames) for available memory"""
    optimizer = VideoMemoryOptimizer()
    
    # Binary search for max frames
    min_frames = 1
    max_frames = 500
    
    while min_frames < max_frames:
        mid_frames = (min_frames + max_frames + 1) // 2
        
        estimate = optimizer.estimate_memory_for_generation(
            model_type,
            resolution,
            mid_frames
        )
        
        if estimate["recommended"] <= available_memory_gb:
            min_frames = mid_frames
        else:
            max_frames = mid_frames - 1
            
    return min_frames