"""Time estimation utilities for 3D generation parameters."""

import math
from typing import Dict, Any


def estimate_3d_generation_time(
    # Shape generation parameters
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    mesh_decode_resolution: int = 64,
    mesh_decode_batch_size: int = None,
    
    # Texture generation parameters  
    paint_max_num_view: int = 6,
    paint_resolution: int = 512,
    render_size: int = 1024,
    texture_size: int = 1024,
    
    # Other parameters
    num_views: int = 6,
    enable_texture: bool = True,
    model: str = "hunyuan3d-21"
) -> Dict[str, Any]:
    """
    Estimate total generation time based on parameters.
    
    Returns:
        Dict with total_time (seconds), breakdown by stage, and performance tips
    """
    
    # Base times (in seconds) for reference hardware (RTX 4090) - OPTIMIZED VALUES
    BASE_SHAPE_TIME = 5.0   # Optimized shape generation (was 15.0 - 3x faster)
    BASE_TEXTURE_TIME = 8.0  # Optimized texture generation (was 25.0 - 3x faster)
    
    # === Shape Generation Time ===
    
    # Inference steps impact (linear)
    steps_multiplier = num_inference_steps / 50.0
    
    # Guidance scale impact (CFG requires 2x forward passes)
    guidance_multiplier = 1.0 if guidance_scale <= 1.0 else 1.8
    
    # Mesh decode resolution impact (cubic scaling)
    resolution_factor = (mesh_decode_resolution / 64.0) ** 3
    
    # Auto batch size calculation
    if mesh_decode_batch_size is None:
        if mesh_decode_resolution >= 128:
            mesh_decode_batch_size = 8192
        elif mesh_decode_resolution >= 96:
            mesh_decode_batch_size = 16384
        elif mesh_decode_resolution >= 64:
            mesh_decode_batch_size = 32768
        else:
            mesh_decode_batch_size = 65536
    
    # Batch size impact (fewer batches = better GPU utilization)
    total_points = mesh_decode_resolution ** 3
    num_batches = math.ceil(total_points / mesh_decode_batch_size)
    batch_efficiency = min(1.0, 32768 / mesh_decode_batch_size)  # Sweet spot around 32k
    
    shape_time = (BASE_SHAPE_TIME * 
                  steps_multiplier * 
                  guidance_multiplier * 
                  resolution_factor / 
                  batch_efficiency)
    
    # === Texture Generation Time ===
    texture_time = 0.0
    if enable_texture:
        # Paint views impact (more views = more multiview diffusion time)
        paint_views_multiplier = paint_max_num_view / 6.0
        
        # Paint resolution impact (quadratic scaling)
        paint_res_multiplier = (paint_resolution / 512.0) ** 2
        
        # Render size impact (quadratic scaling for rendering)
        render_multiplier = (render_size / 1024.0) ** 2
        
        # Texture size impact (linear for final texture operations)
        texture_multiplier = texture_size / 1024.0
        
        texture_time = (BASE_TEXTURE_TIME * 
                       paint_views_multiplier * 
                       paint_res_multiplier * 
                       render_multiplier * 
                       texture_multiplier)
    
    # === Model-specific adjustments ===
    model_multipliers = {
        "hunyuan3d-21": 1.0,      # Full model - baseline
        "hunyuan3d-2mini": 0.6,   # Mini model - faster
        "hunyuan3d-2mv": 0.8,     # MV model - medium
        "hunyuan3d-2standard": 0.9  # Standard model
    }
    model_multiplier = model_multipliers.get(model, 1.0)
    
    # Apply model multiplier
    shape_time *= model_multiplier
    texture_time *= model_multiplier
    
    # Total time
    total_time = shape_time + texture_time
    
    # === Performance Tips ===
    tips = []
    
    if mesh_decode_resolution > 96:
        tips.append("⚠️ High mesh resolution will significantly increase generation time")
    
    if num_inference_steps > 60:
        tips.append("⚠️ High inference steps have diminishing returns after 50-60 steps")
        
    if paint_resolution > 512 and render_size > 1024:
        tips.append("⚠️ High paint + render resolution may cause GPU memory issues")
        
    if total_points / mesh_decode_batch_size > 100:
        tips.append("💡 Consider increasing batch size for better GPU utilization")
        
    if guidance_scale > 10:
        tips.append("💡 Very high guidance scale may produce over-saturated results")
        
    # Optimal settings suggestion
    if total_time > 120:  # > 2 minutes
        tips.append("🚀 For faster generation: reduce mesh resolution to 64, paint resolution to 512")
    
    return {
        "total_time": round(total_time, 1),
        "breakdown": {
            "shape_generation": round(shape_time, 1),
            "texture_generation": round(texture_time, 1)
        },
        "details": {
            "mesh_decode_points": total_points,
            "mesh_decode_batches": num_batches,
            "batch_size": mesh_decode_batch_size,
            "estimated_gpu_memory_gb": estimate_gpu_memory(
                mesh_decode_resolution, paint_resolution, render_size
            )
        },
        "performance_tips": tips,
        "quality_level": get_quality_assessment(
            mesh_decode_resolution, paint_resolution, render_size, texture_size
        )
    }


def estimate_gpu_memory(mesh_resolution: int, paint_resolution: int, render_size: int) -> float:
    """Estimate GPU memory usage in GB."""
    
    # Base model memory (varies by model)
    base_memory = 8.0  # GB for HunYuan3D
    
    # Mesh decoding memory (scales with resolution^3)
    mesh_memory = (mesh_resolution / 64.0) ** 3 * 2.0
    
    # Texture generation memory (scales with resolution^2)
    texture_memory = (paint_resolution / 512.0) ** 2 * 4.0
    texture_memory += (render_size / 1024.0) ** 2 * 2.0
    
    return round(base_memory + mesh_memory + texture_memory, 1)


def get_quality_assessment(
    mesh_resolution: int, paint_resolution: int, render_size: int, texture_size: int
) -> str:
    """Assess overall quality level based on parameters."""
    
    # Calculate quality scores (0-100)
    mesh_quality = min(100, (mesh_resolution / 128.0) * 100)
    texture_quality = min(100, (paint_resolution / 768.0) * 50 + (texture_size / 4096.0) * 50)
    
    overall_quality = (mesh_quality + texture_quality) / 2
    
    if overall_quality >= 80:
        return "🔥 Ultra High Quality"
    elif overall_quality >= 60:
        return "⭐ High Quality"
    elif overall_quality >= 40:
        return "✅ Standard Quality"  
    elif overall_quality >= 20:
        return "⚡ Performance Mode"
    else:
        return "🏃 Speed Mode"


def format_time(seconds: float) -> str:
    """Format time duration in a human-readable way."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        return f"{minutes:.0f}m {remaining_seconds:.0f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours:.0f}h {minutes:.0f}m"


# Preset configurations for quick selection
PERFORMANCE_PRESETS = {
    "speed": {
        "name": "⚡ Speed Mode",
        "description": "Fastest generation (~10s)",
        "params": {
            "steps": 30,
            "guidance_scale": 5.0,
            "mesh_decode_resolution": 48,
            "paint_resolution": 256,
            "render_size": 512,
            "texture_size": 512
        }
    },
    "balanced": {
        "name": "⚖️ Balanced",
        "description": "Good quality & speed (~20s)",
        "params": {
            "steps": 50,
            "guidance_scale": 7.5,
            "mesh_decode_resolution": 64,
            "paint_resolution": 512,
            "render_size": 1024,
            "texture_size": 1024
        }
    },
    "quality": {
        "name": "⭐ High Quality",
        "description": "Best results (~45s)",
        "params": {
            "steps": 75,
            "guidance_scale": 10.0,
            "mesh_decode_resolution": 96,
            "paint_resolution": 768,
            "render_size": 1536,
            "texture_size": 2048
        }
    },
    "ultra": {
        "name": "🔥 Ultra Quality",
        "description": "Maximum quality (~90s)",
        "params": {
            "steps": 100,
            "guidance_scale": 12.0,
            "mesh_decode_resolution": 128,
            "paint_resolution": 768,
            "render_size": 2048,
            "texture_size": 4096
        }
    }
}