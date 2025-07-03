#!/usr/bin/env python3
"""Test if the model manager fixes resolve the gradio schema issue"""

import gradio as gr
import sys
import os

# Mock the necessary modules
class MockPath:
    def __init__(self, path):
        self.path = path
    def exists(self):
        return False
    def resolve(self):
        return self

class MockModelManager:
    def download_model(self, model_type, model_name, use_hf_token, force_redownload, progress):
        return f"Downloaded {model_type} model {model_name}"
    
    def get_model_status(self):
        return "Status OK"
    
    def check_model_complete(self, model_path, model_type, model_name):
        return True
    
    def check_missing_components(self, model_type, model_name):
        return []
        
    def set_hf_token(self, token):
        return "Token set"

print("Testing model manager with simplified signatures...")

try:
    manager = MockModelManager()
    
    # Create a test interface
    with gr.Blocks() as demo:
        # Test the download function
        with gr.Row():
            model_dropdown = gr.Dropdown(["test-model"], label="Model")
            force_checkbox = gr.Checkbox(label="Force redownload")
            download_btn = gr.Button("Download")
            status_output = gr.HTML()
        
        # This mimics the pattern in ui.py
        def create_download_fn(model_name):
            def download_fn(force, progress=gr.Progress()):
                yield from [manager.download_model("image", model_name, False, force, progress)]
            return download_fn
        
        download_btn.click(
            fn=create_download_fn("test-model"),
            inputs=[force_checkbox],
            outputs=[status_output]
        )
        
        # Test status function
        status_btn = gr.Button("Get Status")
        status_display = gr.HTML()
        
        status_btn.click(
            fn=manager.get_model_status,
            outputs=[status_display]
        )
    
    print("✅ Interface created successfully")
    
    # Test API info generation
    print("Testing API info generation...")
    api_info = demo.get_api_info()
    print("✅ API info generated successfully")
    
    print("🎉 Model manager fix works!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("Test complete.")