#!/bin/bash
# Example test script usage

echo "Testing HunYuan3D Pipeline"
echo "=========================="

# Test with different configurations
echo "1. Testing with HunYuan3D-2.1 (full pipeline)..."
python test_hunyuan3d_pipeline.py --model hunyuan3d-21 --verbose

echo ""
echo "2. Testing with HunYuan3D-2mini (faster)..."
python test_hunyuan3d_pipeline.py --model hunyuan3d-2mini --verbose --steps 20

echo ""
echo "3. Testing without texture generation..."
python test_hunyuan3d_pipeline.py --model hunyuan3d-21 --skip-texture --verbose

echo ""
echo "4. Testing with custom image (if available)..."
if [ -f "Hunyuan3D/assets/example_images/004.png" ]; then
    python test_hunyuan3d_pipeline.py --model hunyuan3d-21 --image-path "Hunyuan3D/assets/example_images/004.png" --verbose
else
    echo "Example image not found, skipping..."
fi

echo ""
echo "All tests completed!"