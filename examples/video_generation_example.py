"""Example usage of the video generation system

This example demonstrates:
1. Basic text-to-video generation
2. Image-to-video animation
3. Memory optimization
4. Production pipeline usage
5. ComfyUI workflow
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from hunyuan3d_app.models.video import (
    VideoModelType,
    create_video_model,
    auto_optimize_for_hardware,
    VideoMemoryOptimizer,
    VIDEO_QUALITY_PRESETS
)
from hunyuan3d_app.models.video.production_pipeline import (
    VideoProductionPipeline,
    JobPriority
)
from hunyuan3d_app.generation.video import VideoGenerator, VideoModel, VideoGenerationParams
from PIL import Image


def example_1_basic_generation():
    """Example 1: Basic text-to-video generation"""
    print("\n=== Example 1: Basic Text-to-Video ===")
    
    # Create video generator (using existing interface)
    generator = VideoGenerator()
    
    # Load LTX-Video for real-time generation
    success, message = generator.load_model(
        VideoModel.LTXVIDEO,
        device="cuda",
        optimization_level="moderate"  # Auto-optimize for hardware
    )
    print(f"Model loading: {message}")
    
    if not success:
        return
        
    # Generate video
    params = VideoGenerationParams(
        prompt="A serene mountain lake at sunset with gentle ripples on the water",
        negative_prompt="blurry, low quality, distorted",
        duration_seconds=5.0,
        fps=30,
        width=1216,
        height=704,
        guidance_scale=7.5,
        num_inference_steps=25,
        seed=42
    )
    
    frames, info = generator.generate_video(params, quality_preset="standard")
    
    if frames:
        print(f"Generated {len(frames)} frames")
        print(f"Info: {info}")
        
        # Save video
        output_path = Path("outputs/example_video.mp4")
        generator.save_video(frames, output_path, fps=params.fps)
        print(f"Saved to: {output_path}")


def example_2_new_models():
    """Example 2: Using new state-of-the-art models"""
    print("\n=== Example 2: New Models (Wan2.1, HunyuanVideo, etc.) ===")
    
    # Check available memory
    optimizer = VideoMemoryOptimizer()
    profile = optimizer.profile_system()
    print(f"Available GPU memory: {profile.gpu_free:.1f}GB")
    
    # Choose model based on hardware
    if profile.gpu_free >= 24:
        model_type = VideoModelType.HUNYUANVIDEO
        print("Using HunyuanVideo for cinema quality")
    elif profile.gpu_free >= 16:
        model_type = VideoModelType.WAN2_1_14B
        print("Using Wan2.1 14B for professional quality")
    else:
        model_type = VideoModelType.WAN2_1_1_3B
        print("Using Wan2.1 1.3B for consumer GPU")
        
    # Create model directly
    model = create_video_model(
        model_type=model_type,
        device="cuda",
        dtype="fp16"
    )
    
    # Auto-optimize for hardware
    auto_optimize_for_hardware(model)
    
    # Load model
    print("Loading model...")
    success = model.load(lambda p, m: print(f"{m} - {p*100:.1f}%"))
    
    if not success:
        print("Failed to load model")
        return
        
    # Generate with visual text (Wan2.1 feature)
    result = model.generate(
        prompt="A neon sign saying 'HELLO WORLD' flickering in a rainy cyberpunk alley",
        negative_prompt="blurry, distorted text",
        num_frames=120,  # 5 seconds at 24fps
        height=720,
        width=1280,
        num_inference_steps=50,
        guidance_scale=6.0,
        fps=24,
        seed=42
    )
    
    print(f"Generated video: {result.duration}s at {result.fps}fps")
    print(f"Memory usage: {model.get_memory_usage()}")
    
    # Save frames
    if result.frames:
        output_path = Path("outputs/wan21_demo.mp4")
        save_video_frames(result.frames, output_path, result.fps)


def example_3_image_to_video():
    """Example 3: Image-to-video with CogVideoX-5B"""
    print("\n=== Example 3: Image-to-Video Animation ===")
    
    # Create CogVideoX-5B model (best for I2V)
    model = create_video_model(
        model_type=VideoModelType.COGVIDEOX_5B,
        device="cuda",
        dtype="fp16"
    )
    
    # Load model
    print("Loading CogVideoX-5B...")
    model.load()
    
    # Load or create an image
    image_path = Path("inputs/example_image.jpg")
    if image_path.exists():
        image = Image.open(image_path)
    else:
        # Create a simple test image
        image = Image.new("RGB", (720, 480), color="skyblue")
        
    # Animate the image
    result = model.image_to_video(
        image=image,
        prompt="Camera slowly zooms in while clouds drift across the sky",
        num_frames=48,  # 6 seconds at 8fps
        num_inference_steps=50,
        guidance_scale=6.0,
        fps=8,
        motion_bucket_id=127,  # Motion strength
        noise_aug_strength=0.02
    )
    
    print(f"Animated {len(result.frames)} frames")
    
    # Optionally interpolate to 24fps
    if result.metadata.get("interpolated", False):
        print("Frames interpolated to 24fps")


def example_4_production_pipeline():
    """Example 4: Production pipeline with queue"""
    print("\n=== Example 4: Production Pipeline ===")
    
    # Create production pipeline
    pipeline = VideoProductionPipeline(
        max_workers=2,  # Process 2 videos in parallel
        enable_monitoring=True
    )
    
    # Define multiple jobs
    jobs = [
        {
            "model_type": VideoModelType.LTX_VIDEO,
            "params": {
                "prompt": "A futuristic city with flying cars",
                "duration_seconds": 5.0,
                "fps": 30,
                "width": 1024,
                "height": 576
            },
            "metadata": {"project": "demo", "version": 1}
        },
        {
            "model_type": VideoModelType.WAN2_1_1_3B,
            "params": {
                "prompt": "Ocean waves crashing on a beach at sunset",
                "duration_seconds": 5.0,
                "fps": 24,
                "width": 832,
                "height": 480
            },
            "metadata": {"project": "demo", "version": 2}
        }
    ]
    
    # Submit jobs
    job_ids = []
    for job in jobs:
        job_id = pipeline.submit_job(
            model_type=job["model_type"],
            params=job["params"],
            priority=JobPriority.NORMAL,
            metadata=job["metadata"],
            callback=lambda info: print(f"Job update: {info}")
        )
        job_ids.append(job_id)
        print(f"Submitted job: {job_id}")
        
    # Check queue stats
    stats = pipeline.get_queue_stats()
    print(f"Queue stats: {stats}")
    
    # Wait for completion (in production, you'd poll or use callbacks)
    import time
    for job_id in job_ids:
        while True:
            status = pipeline.get_job_status(job_id)
            if status["status"] in ["completed", "failed"]:
                print(f"Job {job_id}: {status['status']}")
                break
            time.sleep(2)
            
    # Shutdown pipeline
    pipeline.shutdown()


def example_5_memory_optimization():
    """Example 5: Advanced memory optimization"""
    print("\n=== Example 5: Memory Optimization ===")
    
    # Create optimizer
    optimizer = VideoMemoryOptimizer()
    
    # Profile system
    profile = optimizer.profile_system()
    print(f"System profile:")
    print(f"  GPU: {profile.gpu_free:.1f}/{profile.gpu_total:.1f}GB free")
    print(f"  RAM: {profile.ram_free:.1f}/{profile.ram_total:.1f}GB free")
    print(f"  Recommended resolution: {profile.recommended_resolution}")
    
    # Estimate memory for different scenarios
    scenarios = [
        ("Wan2.1 1.3B 480p", "wan2_1_1.3b", (832, 480), 120),
        ("HunyuanVideo 720p", "hunyuanvideo", (1280, 720), 120),
        ("Mochi-1 480p 30fps", "mochi_1", (848, 480), 150)
    ]
    
    for name, model, resolution, frames in scenarios:
        estimate = optimizer.estimate_memory_for_generation(
            model, resolution, frames
        )
        print(f"\n{name}:")
        print(f"  Model memory: {estimate['model']:.1f}GB")
        print(f"  Activation memory: {estimate['activations']:.1f}GB")
        print(f"  Total required: {estimate['recommended']:.1f}GB")
        
    # Get optimization suggestions
    current_usage = 20.0  # GB
    target_usage = 12.0   # GB
    suggestions = optimizer.get_optimization_suggestions(
        current_usage, target_usage, "hunyuanvideo"
    )
    
    print(f"\nTo reduce memory from {current_usage}GB to {target_usage}GB:")
    for suggestion in suggestions:
        print(f"  - {suggestion}")


def save_video_frames(frames: list, output_path: Path, fps: int):
    """Helper to save frames as video"""
    import cv2
    import numpy as np
    
    if not frames:
        return
        
    height, width = frames[0].size[::-1]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height)
    )
    
    for frame in frames:
        frame_np = np.array(frame)
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
        
    out.release()
    print(f"Saved video to: {output_path}")


if __name__ == "__main__":
    print("Video Generation System Examples")
    print("================================")
    
    # Create output directory
    Path("outputs").mkdir(exist_ok=True)
    
    # Run examples
    try:
        example_1_basic_generation()
    except Exception as e:
        print(f"Example 1 error: {e}")
        
    try:
        example_2_new_models()
    except Exception as e:
        print(f"Example 2 error: {e}")
        
    try:
        example_3_image_to_video()
    except Exception as e:
        print(f"Example 3 error: {e}")
        
    try:
        example_4_production_pipeline()
    except Exception as e:
        print(f"Example 4 error: {e}")
        
    try:
        example_5_memory_optimization()
    except Exception as e:
        print(f"Example 5 error: {e}")
        
    print("\nExamples completed!")