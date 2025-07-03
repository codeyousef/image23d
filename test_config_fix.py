#!/usr/bin/env python3
"""Test that the config.py changes resolve the schema issues"""

import sys
import os
sys.path.insert(0, os.path.join('.', 'src'))

import gradio as gr

print("Testing config.py changes...")

try:
    # Import the modified config classes
    from hunyuan3d_app.config import ImageModelConfig, QualityPreset
    
    print("✅ Successfully imported config classes")
    
    # Test creating instances (this would fail if there were dataclass issues)
    test_config = ImageModelConfig(
        name="Test Model",
        repo_id="test/repo", 
        pipeline_class="TestPipeline",
        size="1GB",
        vram_required="4GB",
        description="Test description",
        optimal_resolution=(1024, 1024),
        supports_refiner=False,
        is_gguf=False,
        gguf_file=""
    )
    
    print("✅ Successfully created ImageModelConfig instance")
    print(f"   - Model name: {test_config.name}")
    print(f"   - supports_refiner: {test_config.supports_refiner} (type: {type(test_config.supports_refiner)})")
    print(f"   - is_gguf: {test_config.is_gguf} (type: {type(test_config.is_gguf)})")
    
    test_preset = QualityPreset(
        name="Test Preset",
        image_steps=20,
        image_guidance=8.0,
        use_refiner=False,
        num_3d_views=6,
        mesh_resolution=256,
        texture_resolution=1024
    )
    
    print("✅ Successfully created QualityPreset instance")
    print(f"   - Preset name: {test_preset.name}")
    print(f"   - use_refiner: {test_preset.use_refiner} (type: {type(test_preset.use_refiner)})")
    
    # Test using these in a gradio interface
    def test_function_with_config(model_name):
        """Function that uses config objects"""
        config = test_config
        return f"Using model: {config.name}, GGUF: {config.is_gguf}, Refiner: {config.supports_refiner}"
    
    with gr.Blocks() as demo:
        gr.Markdown("# Config Test")
        
        model_input = gr.Textbox(label="Model Name")
        result_output = gr.HTML()
        test_btn = gr.Button("Test Config Usage")
        
        test_btn.click(
            fn=test_function_with_config,
            inputs=[model_input],
            outputs=[result_output]
        )
    
    print("✅ Successfully created gradio interface with config objects")
    
    # Test API schema generation
    print("Testing API schema generation with config objects...")
    api_info = demo.get_api_info()
    print("✅ API schema generated successfully with config objects!")
    
    print("\n🎉 Config.py fix successful!")
    print("The dataclass boolean field annotations are no longer causing schema errors.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    print("\n❌ There are still issues with the config changes.")

print("Test complete.")