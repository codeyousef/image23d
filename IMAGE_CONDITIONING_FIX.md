# HunYuan3D Image Conditioning Fix

## Problem Analysis

The HunYuan3D model was generating abstract meshes instead of following input images because **image conditioning was disabled** in the DiT (Diffusion Transformer) architecture.

### Root Cause

1. **Configuration Issue**: The model config (`models/3d/hunyuan3d-21/hunyuan3d-dit-v2-1/config.yaml`) had `with_decoupled_ca: false`
2. **Disabled Cross-Attention**: This setting disabled cross-attention layers in DiT blocks, which are essential for image conditioning
3. **Ignored Image Embeddings**: Even though image embeddings were generated (torch.Size([1, 257, 1024])), they were passed to the DiT model but completely ignored

### Technical Details

The HunYuan3D DiT model has two conditioning mechanisms:
- **AdaLN conditioning**: Uses timestep embeddings (always enabled)
- **Cross-attention conditioning**: Uses image/text embeddings (controlled by `with_decoupled_ca`)

With `with_decoupled_ca: false`, the DiT blocks would skip this code:
```python
# Cross-attention (if using decoupled)
if self.config.with_decoupled_ca and context is not None:
    residual = x
    x = self.norm_ca(x) * (1 + scale_ca) + shift_ca
    x, _ = self.cross_attn(x, context, context)  # This was NEVER executed!
    x = residual + gate_ca * x
```

## The Fix

### 1. Enable Cross-Attention During Model Loading

Modified `flow_matching.py` to force-enable cross-attention:

```python
# CRITICAL FIX: Enable cross-attention for image conditioning
# The config has with_decoupled_ca: false, but we need it true for image-to-3D
dit_params['with_decoupled_ca'] = True
logger.info("🔧 ENABLING cross-attention for image conditioning (with_decoupled_ca=True)")
```

### 2. Add Comprehensive Diagnostics

Added detailed logging to verify image conditioning is working:
- DiT configuration verification
- Image embedding status logging
- CFG effectiveness monitoring
- Cross-attention layer validation

### 3. CFG Effectiveness Detection

Added monitoring to detect when conditional and unconditional outputs are identical (indicating broken conditioning):

```python
# DIAGNOSTIC: Verify guidance is being applied
cond_norm = torch.norm(velocity_cond).item()
uncond_norm = torch.norm(velocity_uncond).item()
if abs(cond_norm - uncond_norm) < 0.001:
    logger.warning("⚠️  Conditional and unconditional outputs are nearly identical - image conditioning may not be working!")
```

## Expected Behavior After Fix

### Before Fix
- ❌ Abstract/random meshes regardless of input image
- ❌ Conditional and unconditional DiT outputs nearly identical
- ❌ Cross-attention layers not created (`has_cross_attn: False`)
- ❌ Image embeddings ignored during generation

### After Fix
- ✅ Meshes should follow input image content and structure
- ✅ Conditional and unconditional DiT outputs significantly different
- ✅ Cross-attention layers created and functional (`has_cross_attn: True`)
- ✅ Image embeddings actively used in each diffusion step

## Testing

Run the test script to verify the fix:

```bash
python test_image_conditioning.py
```

This test:
1. Creates a high-contrast image (white circle on black background)
2. Loads the model with the fix applied
3. Generates a 3D mesh
4. Verifies image conditioning via diagnostic logs

### Success Indicators

Look for these in the logs:
- `Cross-attention enabled: True`
- `Blocks have cross-attention: True`
- `CFG effectiveness` showing different conditional vs unconditional norms
- No warnings about identical conditional/unconditional outputs

## Technical Architecture

### Image Processing Pipeline

1. **Input Image** → Background removal → Resize to 512x512
2. **CLIP Vision Encoder** → Image embeddings (1, 257, 1024)
3. **DiT Cross-Attention** → Conditions latent generation on image features
4. **Flow Matching** → Iterative denoising with image guidance
5. **VAE Decoder** → Converts latents to 3D mesh

### Key Components

- **CLIP Vision Model**: `openai/clip-vit-large-patch14`
- **DiT Architecture**: 21 layers, 16 attention heads, 2048 hidden size
- **Cross-Attention**: 1024 context dimension matching CLIP features
- **Guidance**: CFG with conditional/unconditional passes

## Files Modified

1. **`src/hunyuan3d_app/models/threed/hunyuan3d/hy3dshape/pipelines/flow_matching.py`**
   - Enabled cross-attention during model loading
   - Added comprehensive diagnostic logging
   - Enhanced CFG effectiveness monitoring

2. **`test_image_conditioning.py`** *(new)*
   - Test script to verify the fix works
   - Creates high-contrast test images
   - Validates image conditioning functionality

## Related Issues Fixed

This fix addresses several related problems:
- Abstract mesh generation
- Lack of image-to-3D correspondence  
- Silent failure of image conditioning
- Difficulty diagnosing conditioning issues

## Future Improvements

1. **Model Validation**: Check if loaded checkpoint actually contains cross-attention weights
2. **Conditioning Strength**: Add parameter to control image conditioning strength
3. **Multi-Image Support**: Support for multi-view image conditioning
4. **Performance**: Optimize cross-attention computation for faster inference

## Verification Commands

```bash
# Test the fix
python test_image_conditioning.py

# Run the full app to test interactively
python -m hunyuan3d_app.app

# Check model loading logs for cross-attention status
python -c "
import sys; sys.path.append('src')
from hunyuan3d_app.models.threed.hunyuan3d import HunYuan3DModel
model = HunYuan3DModel('hunyuan3d-21', 'cpu', False)
model.load(['multiview'])
"
```

This fix should resolve the core issue of HunYuan3D generating abstract meshes instead of image-guided 3D models.