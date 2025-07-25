# Video Generation System Documentation

## Overview

The HunyuanVideo generation system provides state-of-the-art text-to-video and image-to-video generation capabilities through five cutting-edge models:

- **Wan2.1** (1.3B/14B): Visual text generation, multilingual support
- **HunyuanVideo** (13B): Cinema-quality with dual-stream architecture
- **LTX-Video** (2B): Real-time generation (4s for 5s video)
- **Mochi-1** (10B): Smooth 30fps motion with AsymmDiT
- **CogVideoX-5B** (5B): Superior image-to-video specialist

## Features

### Core Capabilities
- Text-to-video generation
- Image-to-video animation
- Multiple quality presets (draft, standard, high, ultra)
- Memory optimization for consumer GPUs
- Production pipeline with queue management
- ComfyUI integration
- LoRA support for style control

### Memory Optimization
- Automatic hardware detection and optimization
- VAE slicing and tiling
- Sequential CPU offloading
- Quantization support (FP8, INT8, NF4)
- Dynamic batch size adjustment

### Production Features
- Priority queue management
- Batch processing
- Multi-GPU support
- Progress tracking
- Error handling and retry logic
- Job persistence with SQLite

## Installation

```bash
# Install the package
pip install -e .

# Install video-specific dependencies
pip install transformers>=4.30.0 diffusers>=0.21.0 accelerate>=0.20.0
pip install opencv-python imageio imageio-ffmpeg
pip install xformers  # For memory efficient attention
pip install bitsandbytes  # For quantization

# For ComfyUI integration
cd ComfyUI/custom_nodes/
ln -s /path/to/hunyuan3d-app/src/hunyuan3d_app/models/video/comfyui_nodes.py .
pip install -r comfyui_requirements.txt
```

## Quick Start

### Basic Text-to-Video

```python
from hunyuan3d_app.models.video import create_video_model, VideoModelType

# Create and load model
model = create_video_model(
    VideoModelType.WAN2_1_1_3B,  # 8GB VRAM friendly
    device="cuda",
    dtype="fp16"
)
model.load()

# Generate video
result = model.generate(
    prompt="A majestic eagle soaring through clouds",
    num_frames=120,  # 5 seconds at 24fps
    width=832,
    height=480,
    fps=24
)

# Save video
save_video(result.frames, "eagle.mp4", fps=24)
```

### Image-to-Video Animation

```python
from PIL import Image

# Use CogVideoX-5B for best I2V results
model = create_video_model(VideoModelType.COGVIDEOX_5B)
model.load()

# Animate image
image = Image.open("landscape.jpg")
result = model.image_to_video(
    image=image,
    prompt="Camera slowly pans across the landscape",
    num_frames=48,
    fps=8
)
```

### Memory-Optimized Generation

```python
from hunyuan3d_app.models.video import auto_optimize_for_hardware

# Auto-optimize for your hardware
model = create_video_model(VideoModelType.HUNYUANVIDEO)
auto_optimize_for_hardware(model)  # Applies best settings
model.load()

# Use quality presets
from hunyuan3d_app.models.video import VIDEO_QUALITY_PRESETS

preset = VIDEO_QUALITY_PRESETS["standard"]  # Balanced quality/speed
result = model.generate(
    prompt="Futuristic city",
    width=preset.resolution[0],
    height=preset.resolution[1],
    num_inference_steps=preset.inference_steps,
    fps=preset.fps
)
```

### Production Pipeline

```python
from hunyuan3d_app.models.video.production_pipeline import (
    VideoProductionPipeline, JobPriority
)

# Create pipeline
pipeline = VideoProductionPipeline(max_workers=2)

# Submit jobs
job_id = pipeline.submit_job(
    model_type=VideoModelType.LTX_VIDEO,
    params={
        "prompt": "Ocean waves at sunset",
        "duration_seconds": 5.0,
        "fps": 30
    },
    priority=JobPriority.HIGH,
    callback=lambda info: print(f"Progress: {info}")
)

# Check status
status = pipeline.get_job_status(job_id)
```

## Model Comparison

| Model | Parameters | VRAM | Best For | Special Features |
|-------|------------|------|----------|------------------|
| Wan2.1 1.3B | 1.3B | 8GB+ | Consumer GPUs | Visual text, multilingual |
| Wan2.1 14B | 14B | 16GB+ | Professional | 1080p support |
| HunyuanVideo | 13B | 24GB+ | Cinema quality | Dual-stream, 30fps |
| LTX-Video | 2B | 12GB+ | Real-time | 4s generation time |
| Mochi-1 | 10B | 24GB+ | Smooth motion | 30fps, AsymmDiT |
| CogVideoX-5B | 5B | 16GB+ | Image-to-video | LoRA support |

## Quality Presets

### Draft (Fast Preview)
- Resolution: 512x288
- FPS: 8
- Steps: 20
- Memory: ~4GB
- Use case: Quick iterations

### Standard (Balanced)
- Resolution: 768x512  
- FPS: 24
- Steps: 30
- Memory: ~8GB
- Use case: General content

### High (Quality)
- Resolution: 1024x576
- FPS: 30
- Steps: 50
- Memory: ~16GB
- Use case: Professional work

### Ultra (Maximum)
- Resolution: 1280x720
- FPS: 30
- Steps: 100
- Memory: ~24GB
- Use case: Final renders

## ComfyUI Integration

The system includes custom ComfyUI nodes:

1. **Load Video Model**: Load any supported model
2. **Generate Video**: Text-to-video generation
3. **Image to Video**: Animate static images
4. **Video Memory Optimizer**: Optimize VRAM usage
5. **Video LoRA Loader**: Load style LoRAs
6. **Frame Interpolator**: Increase FPS
7. **Video Saver**: Save with multiple formats

### ComfyUI Workflow Example

```
[Load Video Model] → [Generate Video] → [Frame Interpolator] → [Video Saver]
         ↓
[Video Memory Optimizer]
```

## Memory Optimization Guide

### For 8GB VRAM
- Use Wan2.1 1.3B
- Enable all optimizations
- Resolution: 512x288 or 768x432
- Use INT8 quantization if needed

### For 12GB VRAM  
- Use LTX-Video or Wan2.1 1.3B
- Standard optimizations
- Resolution: up to 1024x576
- FP16 precision

### For 16GB VRAM
- Use Wan2.1 14B or CogVideoX-5B
- Minimal optimizations needed
- Resolution: up to 1280x720
- Full features available

### For 24GB+ VRAM
- Use any model including HunyuanVideo
- No optimizations required
- Resolution: up to 1920x1080
- Maximum quality settings

## Troubleshooting

### Out of Memory
```python
# Enable aggressive optimizations
from hunyuan3d_app.models.video import OptimizationLevel

optimizer.optimize_model(model, OptimizationLevel.AGGRESSIVE)
```

### Slow Generation
```python
# Use draft preset for testing
preset = VIDEO_QUALITY_PRESETS["draft"]
# Or use LTX-Video for real-time generation
```

### Poor Quality
```python
# Increase steps and guidance
result = model.generate(
    num_inference_steps=50,  # More steps
    guidance_scale=8.0,      # Stronger guidance
    seed=42                  # Fixed seed for consistency
)
```

## Advanced Features

### Visual Text Generation (Wan2.1)
```python
# Generate readable text in videos
result = model.generate(
    prompt="A neon sign saying 'OPEN 24/7' flickering in the rain"
)
```

### Cinema Quality (HunyuanVideo)
```python
# Professional cinematography
result = model.generate(
    prompt="Cinematic shot: Camera dollies through misty forest",
    fps=30,  # High frame rate
    width=1920,
    height=1080
)
```

### LoRA Style Control
```python
# Add artistic styles
model.add_lora("anime_style", "path/to/anime.safetensors", alpha=0.8)
result = model.generate(prompt="Character walking through city")
```

## API Reference

See the [API documentation](API.md) for detailed class and method references.

## Examples

Check the `examples/` directory for:
- `video_generation_example.py`: Complete usage examples
- `comfyui_workflows/`: Example ComfyUI workflows
- `notebooks/`: Jupyter notebooks with tutorials

## Performance Tips

1. **Use appropriate models for your hardware**
   - 8GB: Wan2.1 1.3B
   - 12GB: LTX-Video
   - 16GB+: All models

2. **Enable optimizations proactively**
   ```python
   auto_optimize_for_hardware(model)
   ```

3. **Process in batches for efficiency**
   ```python
   pipeline = VideoProductionPipeline(max_workers=2)
   ```

4. **Use quality presets instead of manual settings**

5. **Monitor GPU memory during generation**
   ```python
   profile = optimizer.profile_system()
   print(f"Free VRAM: {profile.gpu_free}GB")
   ```

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines on adding new models or features.

## License

The video generation system is part of the Hunyuan3D project. Individual model weights may have their own licenses - check the respective model cards on HuggingFace.