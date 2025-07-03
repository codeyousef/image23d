"""Performance monitoring and profiling utilities."""

import time
import logging
from typing import Dict, Optional, Callable, Any
from functools import wraps
from contextlib import contextmanager
import torch
import psutil

try:
    import nvitop
    from nvitop import Device
    NVITOP_AVAILABLE = True
except ImportError:
    NVITOP_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("nvitop not available - GPU monitoring will be limited")

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitor and track performance metrics."""
    
    def __init__(self):
        self.metrics = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gpu_device = None
        
        if NVITOP_AVAILABLE and torch.cuda.is_available():
            try:
                self.gpu_device = Device(0)
            except:
                logger.warning("Could not initialize nvitop device")
                
    def get_gpu_stats(self):
        """Get detailed GPU statistics."""
        stats = {}
        
        if torch.cuda.is_available():
            # Basic PyTorch stats
            stats["pytorch"] = {
                "allocated_gb": torch.cuda.memory_allocated() / (1024**3),
                "reserved_gb": torch.cuda.memory_reserved() / (1024**3),
                "total_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
            }
            
            # NVITOP stats if available
            if NVITOP_AVAILABLE and self.gpu_device:
                try:
                    stats["nvitop"] = {
                        "gpu_utilization": self.gpu_device.gpu_utilization(),
                        "memory_used": self.gpu_device.memory_used_human(),
                        "memory_total": self.gpu_device.memory_total_human(),
                        "temperature": self.gpu_device.temperature(),
                        "power_draw": self.gpu_device.power_draw(),
                        "fan_speed": self.gpu_device.fan_speed(),
                    }
                except Exception as e:
                    logger.debug(f"Could not get nvitop stats: {e}")
                    
        return stats
        
    def get_system_stats(self):
        """Get system performance statistics."""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        stats = {
            "cpu": {
                "percent": cpu_percent,
                "count": psutil.cpu_count(),
            },
            "memory": {
                "used_gb": memory.used / (1024**3),
                "total_gb": memory.total / (1024**3),
                "percent": memory.percent,
            }
        }
        
        return stats
        
    @contextmanager
    def profile_operation(self, operation_name: str):
        """Context manager for profiling operations."""
        start_time = time.time()
        start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        logger.info(f"Starting {operation_name}...")
        
        try:
            yield
        finally:
            duration = time.time() - start_time
            
            if torch.cuda.is_available():
                end_memory = torch.cuda.memory_allocated()
                memory_delta = (end_memory - start_memory) / (1024**3)
                logger.info(f"{operation_name} completed in {duration:.2f}s, VRAM delta: {memory_delta:+.2f}GB")
            else:
                logger.info(f"{operation_name} completed in {duration:.2f}s")
                
            # Store metrics
            self.metrics[operation_name] = {
                "duration": duration,
                "memory_delta": memory_delta if torch.cuda.is_available() else 0,
                "timestamp": time.time()
            }
            
    def benchmark_function(self, func, *args, **kwargs):
        """Benchmark a function and return result with metrics."""
        start_time = time.time()
        start_gpu_stats = self.get_gpu_stats()
        start_system_stats = self.get_system_stats()
        
        # Run the function
        result = func(*args, **kwargs)
        
        # Collect end metrics
        duration = time.time() - start_time
        end_gpu_stats = self.get_gpu_stats()
        end_system_stats = self.get_system_stats()
        
        # Calculate deltas
        metrics = {
            "duration": duration,
            "gpu_memory_delta": 0,
            "gpu_utilization_avg": 0,
        }
        
        if "pytorch" in start_gpu_stats and "pytorch" in end_gpu_stats:
            metrics["gpu_memory_delta"] = (
                end_gpu_stats["pytorch"]["allocated_gb"] - 
                start_gpu_stats["pytorch"]["allocated_gb"]
            )
            
        return result, metrics
        
    def get_performance_summary(self):
        """Get a formatted performance summary."""
        lines = ["Performance Summary:"]
        
        # GPU stats
        gpu_stats = self.get_gpu_stats()
        if gpu_stats:
            if "pytorch" in gpu_stats:
                pytorch = gpu_stats["pytorch"]
                lines.append(f"GPU Memory: {pytorch['allocated_gb']:.1f}/{pytorch['total_gb']:.1f}GB")
                
            if "nvitop" in gpu_stats:
                nvitop = gpu_stats["nvitop"]
                lines.append(f"GPU Utilization: {nvitop['gpu_utilization']}%")
                lines.append(f"GPU Temperature: {nvitop['temperature']}°C")
                
        # System stats
        system_stats = self.get_system_stats()
        lines.append(f"CPU: {system_stats['cpu']['percent']:.1f}%")
        lines.append(f"RAM: {system_stats['memory']['used_gb']:.1f}/{system_stats['memory']['total_gb']:.1f}GB")
        
        # Recent operations
        if self.metrics:
            lines.append("\nRecent Operations:")
            for op_name, metrics in list(self.metrics.items())[-5:]:  # Last 5 operations
                lines.append(f"  {op_name}: {metrics['duration']:.2f}s")
                
        return "\n".join(lines)
        

# Global instance
_performance_monitor = None

def get_performance_monitor():
    """Get or create the global performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor
    

def profile_generation(func):
    """Decorator to profile image/3D generation functions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        monitor = get_performance_monitor()
        
        with monitor.profile_operation(f"{func.__name__}"):
            result = func(*args, **kwargs)
            
        # Log performance summary
        logger.info(monitor.get_performance_summary())
        
        return result
    return wrapper