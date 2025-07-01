#!/bin/bash

# After installing base dependencies
# Install CUDA-specific PyTorch

echo "Which CUDA version?"
echo "1) CUDA 11.8"
echo "2) CUDA 12.1"
echo "3) CPU only"
read -p "Enter choice (1-3): " choice

case $choice in
    1)
        uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        ;;
    2)
        uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        ;;
    3)
        echo "Using CPU version (already installed)"
        ;;
    *)
        echo "Invalid choice"
        ;;
esac