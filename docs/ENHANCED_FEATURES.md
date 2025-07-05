# Enhanced Features Documentation

## Overview

The enhanced version of Hunyuan3D Studio includes comprehensive production-ready features for professional 3D asset generation workflows.

## New Features

### 1. Complete GGUF Quantization Support

- **13 Quantization Levels**: From Q2_K (4GB VRAM) to FP16 (24GB VRAM)
- **Dynamic Model Discovery**: Auto-detect available quantizations
- **Smart Recommendations**: VRAM-based model selection
- **Quality vs Performance**: Choose the right balance for your hardware

### 2. Civitai Integration

- **Model Search**: Search 100,000+ community models
- **Advanced Filters**: By type, base model, popularity
- **Fast Downloads**: Multi-connection with aria2c support
- **Automatic Compatibility**: Filter by FLUX/SDXL compatibility

### 3. LoRA Support

- **Multi-LoRA Stacking**: Combine multiple LoRAs with custom weights
- **Auto-Detection**: Scan directories for LoRA files
- **Compatibility Checking**: Automatic base model matching
- **LoRA Merging**: Combine multiple LoRAs into one

### 4. Secure Credential Management

- **System Keyring**: Uses OS-native secure storage
- **Encrypted Fallback**: For systems without keyring
- **Multi-Service Support**: HuggingFace, Civitai, OpenAI, etc.
- **UI Management**: Easy credential management interface

### 5. Batch Processing & Queue

- **Priority Queue**: Urgent, High, Normal, Low priorities
- **Multi-Worker**: Concurrent job processing
- **Progress Tracking**: Real-time status updates
- **Job History**: Track all generation jobs

### 6. Generation History

- **SQLite Backend**: Efficient metadata storage
- **Searchable Gallery**: Filter by model, prompt, date
- **Thumbnails**: Automatic thumbnail generation
- **Export/Import**: Backup and share history

### 7. Modern UI Components

- **Bento Grid Layout**: Responsive design
- **Real-time Stats**: GPU usage, memory, queue status
- **Dark Mode**: Eye-friendly interface
- **Smooth Animations**: Professional feel

### 8. Model Benchmarking

- **Performance Metrics**: Speed, memory, quality
- **Comparison Reports**: Side-by-side analysis
- **Sample Gallery**: Visual quality comparison
- **Export Reports**: HTML reports with charts

## Running the Enhanced Version

```bash
# Run with default settings
python run_enhanced.py

# Run with custom options
python run_enhanced.py --share --port 8080 --workers 4

# Or directly
python -m hunyuan3d_app.app_enhanced
```

## Usage Guide

### Quick Start

1. **Set up credentials**: Go to Settings → API Credentials
2. **Download models**: Use Model Hub to search and download
3. **Generate**: Use Quick Generate for simple workflows
4. **Advanced**: Use Advanced Pipeline for LoRA combinations

### Batch Processing

1. Submit multiple jobs to the queue
2. Monitor progress in Queue tab
3. View completed generations in History

### Model Comparison

1. Go to Benchmarks tab
2. Select models to compare
3. Run benchmark with test prompts
4. View performance and quality metrics

### LoRA Workflow

1. Download LoRAs from Civitai
2. Place in `models/loras` directory
3. Use Advanced Pipeline tab
4. Select up to 2 LoRAs with custom weights

## API Usage

The enhanced studio can also be used programmatically:

```python
from hunyuan3d_app.hunyuan3d_studio_enhanced import Hunyuan3DStudioEnhanced

# Create instance
studio = Hunyuan3DStudioEnhanced()

# Submit a job
job_id = studio.submit_generation_job(
    job_type="image",
    params={
        "model_name": "FLUX.1-dev-Q8",
        "prompt": "A magical crystal",
        "width": 1024,
        "height": 1024
    }
)

# Check history
history = studio.get_generation_history(limit=10)

# Benchmark a model
results = studio.benchmark_current_model()
```

## Performance Tips

1. **VRAM Management**:
   - Use GGUF models for limited VRAM
   - Enable CPU offload for large models
   - Monitor VRAM usage in dashboard

2. **Queue Optimization**:
   - Adjust worker count based on VRAM
   - Use priorities for important jobs
   - Batch similar jobs together

3. **Quality vs Speed**:
   - Q8 models: Best quality, more VRAM
   - Q4 models: Good balance
   - Q2 models: Maximum speed, lower quality

## Troubleshooting

### Common Issues

1. **Out of Memory**:
   - Use lower quantization
   - Reduce batch size
   - Enable CPU offload

2. **Slow Downloads**:
   - Install aria2c for faster speeds
   - Check network connection
   - Use different mirror if available

3. **LoRA Compatibility**:
   - Check base model compatibility
   - Verify LoRA file integrity
   - Update metadata if needed

### Getting Help

- Check system requirements in Settings → System Info
- View logs in console for detailed errors
- Report issues on GitHub with full error logs