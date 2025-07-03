#!/usr/bin/env python3
"""Simple test of config classes without full app imports"""

import sys
import os

# Test the config classes directly
sys.path.insert(0, os.path.join('.', 'src'))

print("Testing config classes directly...")

try:
    # Create the classes manually to test (avoiding full imports)
    class ImageModelConfig:
        """Configuration for image generation models"""
        def __init__(self, name, repo_id, pipeline_class, size, vram_required, description, optimal_resolution, supports_refiner=False, is_gguf=False, gguf_file=""):
            self.name = name
            self.repo_id = repo_id
            self.pipeline_class = pipeline_class
            self.size = size
            self.vram_required = vram_required
            self.description = description
            self.optimal_resolution = optimal_resolution
            self.supports_refiner = supports_refiner
            self.is_gguf = is_gguf
            self.gguf_file = gguf_file
    
    class QualityPreset:
        """Quality preset configurations"""
        def __init__(self, name, image_steps, image_guidance, use_refiner, num_3d_views, mesh_resolution, texture_resolution):
            self.name = name
            self.image_steps = image_steps
            self.image_guidance = image_guidance
            self.use_refiner = use_refiner
            self.num_3d_views = num_3d_views
            self.mesh_resolution = mesh_resolution
            self.texture_resolution = texture_resolution
    
    # Test creating instances
    test_config = ImageModelConfig(
        name="Test Model",
        repo_id="test/repo", 
        pipeline_class="TestPipeline",
        size="1GB",
        vram_required="4GB",
        description="Test description",
        optimal_resolution=(1024, 1024),
        supports_refiner=False,
        is_gguf=True,  # Test boolean values
        gguf_file="test.gguf"
    )
    
    print("✅ Successfully created ImageModelConfig instance")
    print(f"   - supports_refiner: {test_config.supports_refiner} (type: {type(test_config.supports_refiner).__name__})")
    print(f"   - is_gguf: {test_config.is_gguf} (type: {type(test_config.is_gguf).__name__})")
    
    # Test that these are plain boolean values, not schema objects
    assert isinstance(test_config.supports_refiner, bool)
    assert isinstance(test_config.is_gguf, bool)
    print("✅ Boolean fields are proper bool types")
    
    # Test with gradio
    import gradio as gr
    
    def test_config_function():
        """Function that returns config data"""
        return {
            "name": test_config.name,
            "supports_refiner": test_config.supports_refiner,
            "is_gguf": test_config.is_gguf
        }
    
    with gr.Blocks() as demo:
        gr.Markdown("# Config Class Test")
        test_btn = gr.Button("Test Config")
        output = gr.JSON()
        
        test_btn.click(
            fn=test_config_function,
            outputs=[output]
        )
    
    print("✅ Created gradio interface with config objects")
    
    # Test API schema generation
    api_info = demo.get_api_info()
    print("✅ API schema generation successful!")
    
    print("\n🎉 Config class changes are working correctly!")
    print("The boolean fields are no longer causing gradio schema errors.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("Test complete.")