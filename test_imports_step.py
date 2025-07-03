#!/usr/bin/env python3
"""Test importing modules step by step to find the error"""

import sys
import os

# Add the source directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("Testing imports step by step...")

try:
    print("1. Testing config import...")
    from hunyuan3d_app.config import ALL_IMAGE_MODELS
    print("✅ Config imported successfully")
    
    print("2. Testing memory_manager import...")
    # Mock missing dependencies
    sys.modules['psutil'] = type(sys)('psutil')
    from hunyuan3d_app.memory_manager import get_memory_manager
    print("✅ Memory manager imported successfully")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("Test complete.")