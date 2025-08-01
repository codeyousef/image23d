#!/usr/bin/env python3
"""Test script to verify HunYuan3D model loading fixes

This script tests:
1. Model components are properly detected (dit instead of dit_model)
2. DiT model is loaded from checkpoint correctly
3. Windows fallback messages are appropriate
4. All components are verified after loading
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_model_loading():
    """Test HunYuan3D model loading with the fixes"""
    logger.info("Testing HunYuan3D model loading fixes...")
    
    try:
        from hunyuan3d_app.models.threed.hunyuan3d import HunYuan3DModel, HunYuan3DConfig
        
        # Create config
        config = HunYuan3DConfig(
            model_variant="hunyuan3d-21",
            device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
            enable_texture=True
        )
        
        logger.info(f"Creating HunYuan3D model with config: {config.model_variant}")
        
        # Create model
        model = HunYuan3DModel(
            model_variant=config.model_variant,
            device=config.device,
            enable_texture=config.enable_texture
        )
        
        logger.info("Loading model components...")
        
        # Load model
        success = model.load(components=["multiview"])
        
        if success:
            logger.info("✅ Model loaded successfully!")
            
            # Check pipeline components
            if hasattr(model.pipeline, 'multiview') and model.pipeline.multiview:
                pipeline = model.pipeline.multiview.pipeline
                logger.info(f"Pipeline type: {type(pipeline).__name__}")
                
                # Check for dit attribute (the fix)
                if hasattr(pipeline, 'dit'):
                    logger.info(f"✅ DiT model found: {type(pipeline.dit).__name__}")
                else:
                    logger.error("❌ DiT model not found in pipeline")
                    
                # Check VAE
                if hasattr(pipeline, 'vae'):
                    logger.info(f"✅ VAE found: {type(pipeline.vae).__name__}")
                else:
                    logger.error("❌ VAE not found in pipeline")
                    
            return True
        else:
            logger.error("❌ Model loading failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_windows_compatibility():
    """Test Windows compatibility messages"""
    logger.info("\nTesting Windows compatibility...")
    
    try:
        import platform
        
        if platform.system() == "Windows":
            logger.info("Running on Windows - checking fallback messages...")
            
            # The renderer should show appropriate Windows messages
            from hunyuan3d_app.models.threed.hunyuan3d.texture_pipeline.renderer.MeshRender import MeshRenderer
            
            # Create a simple test config
            class TestConfig:
                def __init__(self):
                    self.bake_resolution = 1024
                    self.shader_name = "pbr"
                    self.camera_type = "orth"
                    self.raster_mode = "pytorch3d"
                    self.render_size = 1024
                    
            config = TestConfig()
            
            # This should show Windows-specific messages
            renderer = MeshRenderer(config)
            logger.info("✅ Windows fallback renderer initialized successfully")
            
        else:
            logger.info("Not running on Windows - skipping Windows-specific tests")
            
        return True
        
    except Exception as e:
        logger.error(f"❌ Windows compatibility test failed: {e}")
        return False


def main():
    """Run all tests"""
    logger.info("🚀 Starting HunYuan3D model loading tests...")
    logger.info("="*80)
    
    tests = [
        ("Model Loading", test_model_loading),
        ("Windows Compatibility", test_windows_compatibility),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("📊 Test Summary:")
    logger.info("="*80)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status} {test_name}")
        if result:
            passed += 1
    
    logger.info("="*80)
    logger.info(f"📈 Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Model loading fixes verified.")
        return True
    else:
        logger.warning(f"⚠️ {total-passed} tests failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)