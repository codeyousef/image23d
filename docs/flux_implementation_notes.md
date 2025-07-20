# FLUX.1 Dev Implementation Notes

## Overview

This document provides detailed implementation notes for the comprehensive FLUX.1 Dev pipeline integration in the Hunyuan3D app. The implementation follows the "Flux.1 Dev Implementation Guide" and addresses the specific issues with GGUF Q8/Q6 models on RTX 4090.

## Architecture

The FLUX implementation is modular and consists of the following components:

### 1. Core Components (`flux_core.py`)
- **FluxGenerator**: Base implementation for standard FLUX models
- **FluxGGUFGenerator**: Specialized implementation for GGUF quantized models
  - Proper device management for Q6/Q8 models
  - No `device_map` for large quantizations to avoid device conflicts
  - Direct `.to(device)` placement after loading

### 2. Guidance System (`flux_guidance.py`)
- **AdvancedFluxGuidance**: Implements distilled CFG (critical for FLUX.1-dev)
  - **IMPORTANT**: FLUX.1-dev uses distilled CFG, not regular CFG
  - Always set CFG=1.0 internally, use distilled_cfg_scale for control
- **FluxSamplerOptimizer**: Optimal sampler/scheduler combinations
- **Multi-stage generation**: Structure → Details → Polish phases

### 3. Acceleration (`flux_acceleration.py`)
- **AcceleratedFluxGenerator**: HyperFlux and FluxTurbo LoRA integration
  - 3-5x speedup (4-16 steps vs 28)
  - Optimal settings for each acceleration method
- **HybridAccelerationPipeline**: Combines multiple acceleration techniques

### 4. Prompt Engineering (`flux_prompts.py`)
- **FluxPromptOptimizer**: FLUX-specific prompt optimization
  - Style-aware enhancements
  - Automatic negative prompt generation
  - Token optimization for T5 encoder
- **PromptTemplate**: Structured prompt building

### 5. Performance Optimization (`flux_optimization.py`)
- **OptimizedFluxGenerator**: 
  - torch.compile with max-autotune (30-53% speedup)
  - Memory-efficient attention (xFormers/native)
  - CUDA optimization settings
- **MemoryOptimizer**: Dynamic memory management
  - Progressive optimization based on available VRAM
  - Automatic VAE slicing/tiling

### 6. ControlNet Integration (`flux_controlnet.py`)
- **FluxControlNetGenerator**: Multiple control modalities
  - Depth, Canny, Pose, Normal, Line Art
  - Multi-ControlNet support
  - 3D-optimized configurations

### 7. Post-Processing (`flux_enhance.py`)
- **PostProcessingPipeline**: Comprehensive enhancement
  - Real-ESRGAN upscaling
  - GFPGAN face restoration
  - Style-aware enhancements
- **Progressive upscaling**: Multi-stage for better quality

### 8. Production Pipeline (`flux_production.py`)
- **ProductionFluxPipeline**: Unified interface combining all components
  - Automatic model selection based on hardware
  - Request/response pattern with caching
  - Comprehensive error handling
- **GenerationRequest/Result**: Structured I/O

## Key Implementation Details

### GGUF Q8/Q6 Device Management

The critical fix for GGUF Q8/Q6 models on RTX 4090:

```python
# For Q6/Q8 models - DON'T use device_map
if is_large_quant:
    transformer = FluxTransformer2DModel.from_single_file(
        gguf_file,
        quantization_config=quantization_config,
        device_map=None,  # Explicitly no device_map
        low_cpu_mem_usage=True
    )
    # Move to GPU after loading
    transformer = transformer.to(device, dtype=compute_dtype)
```

### Distilled CFG Usage

Critical for FLUX.1-dev:

```python
# FLUX.1-dev uses distilled CFG
guided_scale=3.5  # This is distilled CFG scale, not regular CFG
# Regular CFG is always 1.0 internally
```

### Memory Optimization Strategy

1. **24GB+ VRAM**: Full model with torch.compile
2. **12-24GB VRAM**: GGUF Q8 with optimizations
3. **8-12GB VRAM**: GGUF Q6 or Q4 with VAE slicing
4. **<8GB VRAM**: GGUF Q4/Q3 with aggressive optimizations

### Generation Workflow

1. **Request Processing**:
   - Prompt optimization
   - Style detection
   - Model selection

2. **Generation**:
   - Multi-stage if configured
   - Progress tracking
   - Memory management

3. **Post-Processing**:
   - Optional enhancement
   - Upscaling if requested
   - Format conversion

## Integration with Hunyuan3D App

The integration in `generation/image.py`:

```python
if is_flux_model:
    # Use production pipeline
    if self.flux_pipeline is None:
        self.flux_pipeline = create_production_pipeline(model_variant=variant)
    
    request = GenerationRequest(...)
    result = self.flux_pipeline.generate(request)
```

## Configuration

New configuration sections in `config.py`:

- `FLUX_PIPELINE_CONFIG`: Main pipeline settings
- `FLUX_CONTROLNET_CONFIG`: ControlNet models and settings  
- `FLUX_MODEL_VARIANTS`: Available model variants with requirements

## Troubleshooting

Use the diagnostics module:

```python
from hunyuan3d_app.models.flux_diagnostics import FluxDiagnostics

# Run full diagnostic
results = FluxDiagnostics.run_full_diagnostic()

# Diagnose specific error
diagnosis = diagnose_generation_failure(error, request)
```

## Performance Tips

1. **First Run**: Expect slower performance due to:
   - Model compilation (if enabled)
   - CUDA kernel compilation
   - Memory allocation

2. **Optimal Settings**:
   - Q8 models: 15 steps, guidance=3.5
   - Q6 models: 20 steps, guidance=3.5
   - Base models: 28 steps, guidance=3.5

3. **Acceleration**:
   - Enable HyperFlux for 3-4x speedup
   - Use torch.compile for additional 30-50%
   - Batch processing for multiple images

## Known Issues and Solutions

1. **Device Mismatch Errors**:
   - Fixed by proper device management in FluxGGUFGenerator
   - No device_map for Q6/Q8 models

2. **OOM Errors**:
   - Use progressive memory optimization
   - Enable VAE slicing/tiling
   - Use lower quantization

3. **Slow Generation**:
   - Enable acceleration methods
   - Use appropriate quantization for hardware
   - Check if using GPU (not CPU)

## Future Enhancements

1. **LoRA Support**:
   - Dynamic LoRA loading
   - LoRA mixing/blending

2. **Advanced Sampling**:
   - Custom schedulers
   - Adaptive sampling

3. **Distributed Generation**:
   - Multi-GPU support
   - Model parallelism

## Testing

Run the test suite:

```bash
# Test basic setup
python -m hunyuan3d_app.models.flux_diagnostics

# Test generation
python -c "from hunyuan3d_app.models.flux_production import quick_generate; img = quick_generate('a cat'); img.show()"
```

## References

- [FLUX.1 Dev Implementation Guide](../docs/private/Flux.1%20Dev%20Implementation%20Guide.md)
- [Diffusers FLUX Documentation](https://huggingface.co/docs/diffusers/api/pipelines/flux)
- [GGUF Quantization Guide](https://github.com/city96/ComfyUI-GGUF)