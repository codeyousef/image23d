# Critical Fix: Image Conditioning Diagnostics for HunYuan3D

## Problem: Abstract Meshes Not Using Input Images

The user reports that HunYuan3D is generating abstract meshes that don't correspond to the input images, indicating a failure in image conditioning.

## Root Cause Investigation

The issue could be:

1. **Wrong Pipeline**: Not loading the actual HunYuan3D pipeline (using a fallback/dummy)
2. **Parameter Mismatch**: Image not passed with correct parameter name to pipeline
3. **Model Not Loaded**: Pipeline components (UNet/DiT, VAE) not properly loaded
4. **Placeholder Generation**: Pipeline generating generic shapes instead of image-conditioned ones

## Comprehensive Diagnostics Implemented

### 1. **Pipeline Validation**
```python
# Check if this is actually a real HunYuan3D pipeline or a fallback
expected_pipeline_indicators = [
    'Hunyuan3DDiTFlowMatchingPipeline',
    'FlowMatchingPipeline', 
    'DiTPipeline',
    'Hunyuan3D'
]

pipeline_name = type(self.pipeline).__name__
is_real_hunyuan3d = any(indicator in pipeline_name for indicator in expected_pipeline_indicators)

if not is_real_hunyuan3d:
    logger.error(f"🚨 CRITICAL: Pipeline type '{pipeline_name}' doesn't look like real HunYuan3D!")
```

### 2. **Image Parameter Validation**
```python
# Check for different image parameter names
image_param_names = ['image', 'input_image', 'images', 'pil_image', 'image_tensor']
image_param_used = None

for param_name in image_param_names:
    if param_name in available_params:
        if param_name != 'image':  # If it's not the default 'image'
            pipeline_kwargs[param_name] = pipeline_kwargs.pop('image', input_image)
            logger.info(f"✅ Using image parameter: '{param_name}'")
        image_param_used = param_name
        break

if not image_param_used:
    logger.error(f"❌ CRITICAL: Pipeline doesn't accept any known image parameters!")
```

### 3. **Image Content Validation**
```python
# Validate the image is actually being passed correctly
if input_image is None:
    logger.error(f"❌ CRITICAL: Input image is None! HunYuan3D requires an image for conditioning!")
    raise ValueError("Input image cannot be None for HunYuan3D generation")

# Check if image looks valid
if hasattr(input_image, 'size'):
    width, height = input_image.size
    if width == 0 or height == 0:
        logger.error(f"❌ CRITICAL: Input image has zero dimensions: {input_image.size}")
```

### 4. **Placeholder Mesh Detection**
```python
# Check for common placeholder mesh patterns
placeholder_indicators = []

# Exact vertex counts that are suspiciously round
common_placeholder_counts = [1024, 2048, 4096, 8192, 512, 256, 1000, 2000, 5000, 10000]
if vertex_count in common_placeholder_counts:
    placeholder_indicators.append(f"Round vertex count: {vertex_count}")

# Check if it's a perfect sphere (icosphere patterns)
icosphere_vertex_counts = [12, 42, 162, 642, 2562, 10242]
if vertex_count in icosphere_vertex_counts:
    placeholder_indicators.append(f"Icosphere pattern: {vertex_count} vertices")
```

### 5. **Component Verification**
```python
# Check if pipeline components are actually loaded
if hasattr(self.pipeline, 'unet') and self.pipeline.unet is not None:
    logger.info(f"✅ UNet model loaded: {type(self.pipeline.unet).__name__}")
elif hasattr(self.pipeline, 'dit_model') and self.pipeline.dit_model is not None:
    logger.info(f"✅ DiT model loaded: {type(self.pipeline.dit_model).__name__}")
else:
    logger.error(f"❌ No UNet or DiT model found in pipeline - this won't generate proper 3D!")
```

### 6. **Image Hash Tracking**
```python
# Create a hash of the image to track if same images produce same outputs
import hashlib
if hasattr(input_image, 'tobytes'):
    image_hash = hashlib.md5(input_image.tobytes()).hexdigest()[:8]
    logger.info(f"📸 Input image hash: {image_hash}")
```

## Expected Diagnostic Output

When you run generation, you should now see comprehensive logging like:

```
🔍 PIPELINE VALIDATION:
   - Pipeline type: Hunyuan3DDiTFlowMatchingPipeline
   - Pipeline module: src.hunyuan3d_app.models.threed.hunyuan3d.hy3dshape.pipelines
   - Pipeline has text_encoder: True
   - Pipeline has vae: True
   - Pipeline has unet: False
   - Pipeline has dit_model: True
✅ Pipeline appears to be legitimate HunYuan3D implementation
✅ DiT model loaded: DiTWrapper
✅ VAE loaded: AutoencoderKL

🔍 IMAGE CONDITIONING DEBUGGING:
   - Input image type: <class 'PIL.Image.Image'>
   - Input image size: (512, 512)
   - Input image mode: RGB
   - Input image is None: False
✅ Image dimensions valid: 512x512
✅ Image pixel access works, sample pixel: (255, 128, 64)
📸 Input image hash: a1b2c3d4

🔍 Pipeline parameters: ['image', 'guidance_scale', 'num_inference_steps', 'generator', 'box_v', 'output_type']
✅ Image will be passed as 'image' parameter

🔍 MESH OUTPUT DEBUG:
   - Vertex count: 2048
   - Face count: 4096
   - Vertex hash (first 10): 1234567890
   - Bounding box: [[-1. -1. -1.] [ 1.  1.  1.]]
🚨 POTENTIAL PLACEHOLDER MESH DETECTED:
   - Round vertex count: 2048
   - Geometric primitive ratio: 2.00
🚨 This suggests the pipeline may be generating generic shapes instead of using image conditioning!
```

## What to Look For

1. **Pipeline Type**: Should be `Hunyuan3DDiTFlowMatchingPipeline` or similar
2. **Components Loaded**: Should show UNet/DiT and VAE are loaded
3. **Image Parameter**: Should confirm image is passed correctly
4. **Placeholder Detection**: Should NOT show placeholder indicators for different inputs
5. **Hash Variation**: Different images should produce different mesh signatures

## Next Steps

1. **Run generation** with these diagnostics
2. **Check the logs** for the diagnostic output
3. **Identify the specific failure point** based on the error messages
4. **Report back** what the diagnostics reveal

This will definitively identify whether the issue is:
- A) Wrong/fallback pipeline being loaded
- B) Image not being passed correctly to pipeline
- C) Pipeline components not loaded properly
- D) Pipeline generating generic shapes despite proper inputs

The diagnostics will pinpoint exactly where the image conditioning is failing.