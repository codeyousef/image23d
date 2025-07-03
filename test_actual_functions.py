#!/usr/bin/env python3
"""Test actual functions to identify the gradio schema issue"""

import gradio as gr
import sys
import os

# Add the source directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("Testing actual functions...")

# Test importing the classes
try:
    print("1. Testing imports...")
    
    # Mock the missing dependencies
    class MockModel:
        def __init__(self):
            pass
    
    class MockModelManager:
        def __init__(self):
            self.models_dir = "/tmp"
        
        def _load_gguf_model(self, *args, **kwargs):
            return ("Mock result", None)
    
    # Test the actual download function
    print("2. Testing download_gguf_model function...")
    
    def mock_download_gguf_model(model_name, force_redownload=False, progress=None):
        """Mock download function"""
        return f"Downloaded {model_name} with force_redownload={force_redownload}"
    
    # Test with gradio
    with gr.Blocks() as demo:
        model_dropdown = gr.Dropdown(["test-model"], label="Model")
        force_checkbox = gr.Checkbox(label="Force redownload")
        download_btn = gr.Button("Download")
        status_output = gr.HTML()
        
        # Connect the function
        download_btn.click(
            fn=mock_download_gguf_model,
            inputs=[model_dropdown, force_checkbox],
            outputs=[status_output]
        )
    
    print("✅ Mock function works with gradio")
    
    # Try to trigger the schema generation
    print("3. Testing gradio schema generation...")
    
    # This should trigger the schema generation
    api_info = demo.get_api_info()
    print("✅ API info generated successfully")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("Test complete.")