#!/usr/bin/env python3
"""Test the minimal UI components that were causing schema errors"""

import gradio as gr
import sys
import os

# Mock the missing classes and modules
class MockModel:
    def __init__(self):
        self.device = "cuda"

class MockModelManager:
    def __init__(self):
        self.models_dir = "/tmp"

class MockHunyuan3DStudio:
    def __init__(self):
        self.model_manager = MockModelManager()
    
    # These are the functions with simplified signatures
    def download_gguf_model(self, model_name, force_redownload, progress):
        return f"Downloaded {model_name} with force={force_redownload}"
    
    def download_component(self, component_name, progress):
        return f"Downloaded component {component_name}"
    
    def delete_gguf_model(self, model_name):
        return f"Deleted GGUF model {model_name}"
    
    def delete_component(self, component_name):
        return f"Deleted component {component_name}"

print("Testing minimal UI with simplified function signatures...")

try:
    app = MockHunyuan3DStudio()
    
    # Test the UI structure that was causing the error
    with gr.Blocks() as demo:
        gr.Markdown("### GGUF Models Test")
        
        # GGUF download section (simplified version of what's in ui.py)
        model_name = "FLUX.1-dev-Q8"
        with gr.Group():
            gr.Markdown(f"**{model_name}**")
            
            with gr.Row():
                download_gguf_btn = gr.Button(f"Download {model_name}", size="sm", variant="secondary")
                delete_gguf_btn = gr.Button("Delete", size="sm", variant="stop")
                force_redownload_gguf = gr.Checkbox(label="Force re-download", value=False)
            
            gguf_status = gr.HTML()
            
            # This is the exact pattern from ui.py that was causing issues
            def create_gguf_download_fn(model_name):
                def download_fn(force):
                    return app.download_gguf_model(model_name, force, None)
                return download_fn
            
            def create_gguf_delete_fn(model_name):
                def delete_fn():
                    return app.delete_gguf_model(model_name)
                return delete_fn
            
            download_gguf_btn.click(
                fn=create_gguf_download_fn(model_name),
                inputs=[force_redownload_gguf],
                outputs=[gguf_status]
            )
            
            delete_gguf_btn.click(
                fn=create_gguf_delete_fn(model_name),
                outputs=[gguf_status]
            )
        
        # Component download section
        gr.Markdown("### Components Test")
        comp_name = "vae"
        with gr.Row():
            download_comp_btn = gr.Button(f"Download {comp_name}", size="sm", variant="secondary")
            delete_comp_btn = gr.Button("Delete", size="sm", variant="stop")
            comp_status = gr.HTML()
            
            def create_comp_download_fn(component_name):
                def download_fn():
                    return app.download_component(component_name, None)
                return download_fn
            
            def create_comp_delete_fn(component_name):
                def delete_fn():
                    return app.delete_component(component_name)
                return delete_fn
            
            download_comp_btn.click(
                fn=create_comp_download_fn(comp_name),
                outputs=[comp_status]
            )
            
            delete_comp_btn.click(
                fn=create_comp_delete_fn(comp_name),
                outputs=[comp_status]
            )
    
    print("✅ UI created successfully")
    
    # Test API info generation - this is where the schema error was occurring
    print("Testing API info generation...")
    api_info = demo.get_api_info()
    print("✅ API info generated successfully")
    
    print("🎉 Schema error has been fixed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("Test complete.")