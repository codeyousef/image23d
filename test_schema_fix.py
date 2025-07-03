#!/usr/bin/env python3
"""Test if the schema generation fix works"""

import gradio as gr
import sys
import os

# Add the source directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("Testing schema generation fix...")

# Test the simplified function signatures
def simplified_download_gguf_model(model_name, force_redownload, progress):
    """Download a GGUF model and components"""
    return f"Downloaded {model_name} with force={force_redownload}"

def simplified_download_component(component_name, progress):
    """Download a FLUX component"""
    return f"Downloaded component {component_name}"

def simplified_delete_model(model_type, model_name):
    """Delete a downloaded model"""
    return f"Deleted {model_type} model {model_name}"

try:
    print("1. Testing simplified function signatures...")
    
    with gr.Blocks() as demo:
        # Test GGUF download
        with gr.Row():
            model_dropdown = gr.Dropdown(["test-model"], label="Model")
            force_checkbox = gr.Checkbox(label="Force redownload")
            download_btn = gr.Button("Download")
            status_output = gr.HTML()
        
        download_btn.click(
            fn=simplified_download_gguf_model,
            inputs=[model_dropdown, force_checkbox],
            outputs=[status_output]
        )
        
        # Test component download
        with gr.Row():
            comp_dropdown = gr.Dropdown(["vae"], label="Component")
            comp_download_btn = gr.Button("Download Component")
            comp_status = gr.HTML()
        
        comp_download_btn.click(
            fn=simplified_download_component,
            inputs=[comp_dropdown],
            outputs=[comp_status]
        )
        
        # Test delete model
        with gr.Row():
            type_dropdown = gr.Dropdown(["image", "3d"], label="Type")
            name_input = gr.Textbox(label="Model Name")
            delete_btn = gr.Button("Delete")
            delete_status = gr.HTML()
        
        delete_btn.click(
            fn=simplified_delete_model,
            inputs=[type_dropdown, name_input],
            outputs=[delete_status]
        )
    
    print("✅ Interface created successfully")
    
    # Test API info generation
    print("2. Testing API info generation...")
    api_info = demo.get_api_info()
    print("✅ API info generated successfully")
    
    print("All tests passed! The schema fix should work.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("Test complete.")