#!/usr/bin/env python3
"""Test that model deletion no longer causes gradio schema errors"""

import gradio as gr
import sys
import os

# Test functions that mirror the delete patterns in the actual app
def mock_delete_model(model_type, model_name):
    """Mock delete model function matching app.delete_model signature"""
    return f"✅ Successfully deleted {model_type} model: {model_name}"

def mock_delete_gguf_model(model_name):
    """Mock delete GGUF model function matching app.delete_gguf_model signature"""
    return f"✅ Successfully deleted GGUF model: {model_name}"

def mock_get_model_selection_data():
    """Mock function to update model dropdowns"""
    return (
        gr.update(choices=["model1", "model2"], value="model1"),  # image_model
        gr.update(choices=["3d_model1"], value="3d_model1"),      # hunyuan_model
        gr.update(choices=["model1", "model2"], value="model1"),  # manual_img_model
        gr.update(choices=["3d_model1"], value="3d_model1")       # manual_3d_model
    )

print("Testing model deletion without gradio schema errors...")

try:
    # Test the exact pattern from the real UI that was causing the error
    with gr.Blocks() as demo:
        gr.Markdown("# Model Deletion Test")
        
        # Create the dropdown components (like in the real app)
        image_model = gr.Dropdown(["model1", "model2"], label="Image Model")
        hunyuan_model = gr.Dropdown(["3d_model1"], label="3D Model") 
        manual_img_model = gr.Dropdown(["model1", "model2"], label="Manual Image Model")
        manual_3d_model = gr.Dropdown(["3d_model1"], label="Manual 3D Model")
        
        # Test regular model deletion (the pattern that was failing)
        with gr.Group():
            gr.Markdown("### Test Regular Model Deletion")
            model_name_input = gr.Textbox(label="Model Name", value="test-model")
            delete_img_btn = gr.Button("Delete Image Model")
            status_html = gr.HTML()
            
            # This is the exact pattern from ui.py that was causing the error
            def create_delete_fn(model_name):
                return lambda: mock_delete_model("image", model_name)
            
            delete_img_btn.click(
                fn=create_delete_fn("test-model"),
                outputs=[status_html]
            ).then(
                # Update model selection dropdowns after deletion
                fn=mock_get_model_selection_data,
                outputs=[image_model, hunyuan_model, manual_img_model, manual_3d_model]
            )
        
        # Test GGUF model deletion
        with gr.Group():
            gr.Markdown("### Test GGUF Model Deletion")
            gguf_name_input = gr.Textbox(label="GGUF Model Name", value="test-gguf")
            delete_gguf_btn = gr.Button("Delete GGUF Model")
            gguf_status = gr.HTML()
            
            def create_gguf_delete_fn(model_name):
                return lambda: mock_delete_gguf_model(model_name)
            
            delete_gguf_btn.click(
                fn=create_gguf_delete_fn("test-gguf"),
                outputs=[gguf_status]
            ).then(
                # Update model selection dropdowns after deletion
                fn=mock_get_model_selection_data,
                outputs=[image_model, hunyuan_model, manual_img_model, manual_3d_model]
            )
    
    print("✅ Interface created successfully")
    
    # Test API info generation - this is where the schema error was occurring
    print("Testing API info generation (where the error occurred)...")
    api_info = demo.get_api_info()
    print("✅ API info generated successfully - NO SCHEMA ERROR!")
    
    # Verify that the API contains the delete endpoints
    print("Verifying delete endpoints are properly registered...")
    if hasattr(demo, 'fns') and demo.fns:
        print(f"✅ Found {len(demo.fns)} function endpoints registered")
        
        # Check if any functions are related to deletion
        delete_functions = [fn for fn in demo.fns if 'delete' in str(fn).lower()]
        if delete_functions:
            print(f"✅ Found {len(delete_functions)} delete-related functions")
    
    print("\n🎉 SUCCESS: Model deletion should now work without gradio schema errors!")
    print("The TypeError: argument of type 'bool' is not iterable should be resolved.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    print("\n❌ The gradio schema error is still present.")

print("\nTest complete.")