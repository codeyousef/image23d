#!/usr/bin/env python3
"""Test bypassing gradio API info generation"""

import gradio as gr
import sys

print(f"Testing gradio API bypass (version {gr.__version__})...")

def mock_delete_model(model_type, model_name):
    return f"Deleted {model_type} model: {model_name}"

try:
    # Create a minimal interface that matches the failing pattern
    with gr.Blocks() as demo:
        gr.Markdown("# Minimal Delete Test")
        
        # Simple delete button setup
        delete_btn = gr.Button("Delete Model")
        status_output = gr.HTML()
        
        # Use the exact pattern that's failing
        def create_delete_fn(model_name):
            return lambda: mock_delete_model("image", model_name)
        
        delete_btn.click(
            fn=create_delete_fn("test-model"),
            outputs=[status_output]
        )
    
    print("✅ Interface created")
    
    # Try to manually prevent API info generation
    try:
        # Override the problematic method temporarily
        original_get_api_info = demo.get_api_info
        demo.get_api_info = lambda: {"info": "disabled"}
        print("✅ API info generation bypassed")
        
        # Try to launch with disabled API
        print("Testing launch with disabled API...")
        # This is just a test, we won't actually launch
        
        # Restore original method
        demo.get_api_info = original_get_api_info
        
    except Exception as e:
        print(f"⚠️ Bypass attempt failed: {e}")
    
    # Test if the error occurs during API info generation
    print("Testing direct API info generation...")
    try:
        api_info = demo.get_api_info()
        print("✅ Direct API info generation succeeded!")
    except Exception as e:
        print(f"❌ Direct API info generation failed: {e}")
        print("This confirms the schema generation is the problem.")
        
        # Try to inspect the error more deeply
        print("\nTrying to get more details about the schema error...")
        import traceback
        traceback.print_exc()
    
except Exception as e:
    print(f"❌ Interface creation failed: {e}")
    import traceback
    traceback.print_exc()

print("Test complete.")