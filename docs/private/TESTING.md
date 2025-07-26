# HunYuan3D Pipeline Testing

This document describes how to test the HunYuan3D pipeline without running the full NiceGUI application.

## Quick Start

### Basic Test
```bash
python test_hunyuan3d_pipeline.py --model hunyuan3d-21 --verbose
```

### Test with Different Models
```bash
# Fast mini model
python test_hunyuan3d_pipeline.py --model hunyuan3d-2mini --steps 20

# Skip texture generation for faster testing
python test_hunyuan3d_pipeline.py --model hunyuan3d-21 --skip-texture

# Test with custom image
python test_hunyuan3d_pipeline.py --image-path path/to/your/image.png
```

### Run All Tests
```bash
./run_tests.sh
```

## Test Script Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | Model variant (hunyuan3d-21, hunyuan3d-2mini, etc.) | hunyuan3d-21 |
| `--device` | Device to use (cuda/cpu) | cuda if available |
| `--prompt` | Text prompt for generation | "A cute cartoon character" |
| `--image-path` | Path to input image | None (generates test image) |
| `--steps` | Number of inference steps | 30 |
| `--seed` | Random seed for reproducibility | 42 |
| `--skip-texture` | Skip texture generation | False |
| `--verbose` | Enable detailed logging | False |
| `--cleanup` | Clean up models after test | True |

## What the Test Does

1. **Image Creation**: Creates or loads a test image
2. **Model Loading**: Initializes and loads the specified HunYuan3D model
3. **Pipeline Execution**: Runs the complete 3D generation pipeline
4. **Memory Tracking**: Reports CPU and GPU memory usage at each stage
5. **Timing**: Measures time for each pipeline stage
6. **Output Validation**: Checks if generation was successful and reports mesh statistics

## Interpreting Results

### Success Indicators
- ✅ "3D generation successful!" message
- Output file saved with valid mesh statistics
- No errors in the log

### Common Failure Points
- Model loading failures (check model availability)
- CUDA out of memory (try --device cpu or smaller model)
- Import errors (check dependencies)
- Torch.load security errors (check our recent fixes)

### Performance Metrics
The test reports timing for each stage:
- `image_creation`: Time to load/generate input image
- `orchestrator_init`: Time to initialize the pipeline
- `model_loading`: Time to load model weights
- `generation`: Time for actual 3D generation

### Memory Usage
Memory is reported at key stages:
- Initial: Baseline memory usage
- After init: Memory after pipeline initialization
- After model load: Memory after loading model weights
- After generation: Peak memory during generation
- After cleanup: Memory after cleanup (if enabled)

## Example Output

```
2025-01-26 10:30:00 - __main__ - INFO - Testing HunYuan3D pipeline with model: hunyuan3d-21
2025-01-26 10:30:00 - __main__ - INFO - Initial memory - CPU: 2.45GB, GPU: 0.00GB
2025-01-26 10:30:02 - __main__ - INFO - After model load - CPU: 4.23GB, GPU: 8.45GB
2025-01-26 10:32:15 - __main__ - INFO - ✅ 3D generation successful!
2025-01-26 10:32:15 - __main__ - INFO - Mesh stats - Vertices: 12845, Faces: 25690

==================================================
SUMMARY
==================================================
Total time: 135.23s

Stage times:
  image_creation: 1.23s
  orchestrator_init: 0.45s
  model_loading: 15.67s
  generation: 117.88s
==================================================
```

## Troubleshooting

### GPU Memory Issues
```bash
# Use CPU instead
python test_hunyuan3d_pipeline.py --device cpu

# Use smaller model
python test_hunyuan3d_pipeline.py --model hunyuan3d-2mini
```

### Import Errors
Make sure you're in the project root and have all dependencies installed:
```bash
pip install -e .
# or
uv pip install -e .
```

### Torch Security Errors
Our recent fixes should handle torch.load security issues. If you still see errors, the test will show the full traceback to help debug.

## Adding New Tests

To add new test scenarios, modify `test_hunyuan3d_pipeline.py`:

1. Add new command line arguments
2. Modify the `test_pipeline()` function
3. Add new test cases to `run_tests.sh`

The test script is designed to be easily extensible for different testing scenarios.