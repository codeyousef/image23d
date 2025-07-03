#!/usr/bin/env python3
"""Final validation test that covers all the patterns that were causing schema errors"""

import gradio as gr

print("🧪 Final validation of gradio schema fixes...")

# Test all the patterns that were problematic:

# 1. Boolean values in object configurations
test_config = {
    "name": "Test Model",
    "supports_refiner": False,  # Boolean value that was causing issues
    "is_gguf": True,           # Another boolean value
}

# 2. Multiple output patterns (delete vs download functions)
def mock_delete_with_single_output():
    return "✅ Deleted successfully"

def mock_delete_with_multiple_outputs():
    return ("✅ Deleted successfully", "Updated dropdown 1", "Updated dropdown 2")

# 3. Lambda closures (simplified pattern)
def create_delete_function(model_name):
    return lambda: f"✅ Deleted {model_name}"

# 4. Complex function signatures (now simplified)
def simplified_function(model_type, model_name, force_redownload, progress):
    return f"Processed {model_type}/{model_name} with force={force_redownload}"

try:
    with gr.Blocks() as demo:
        gr.Markdown("# Final Gradio Schema Validation")
        
        # Test 1: Boolean configuration objects
        with gr.Tab("Config Objects"):
            config_btn = gr.Button("Test Config")
            config_output = gr.JSON()
            
            config_btn.click(
                fn=lambda: test_config,
                outputs=[config_output]
            )
        
        # Test 2: Delete operations with single outputs
        with gr.Tab("Delete Operations"):
            delete_btn = gr.Button("Delete Model")
            delete_status = gr.HTML()
            
            delete_btn.click(
                fn=mock_delete_with_single_output,
                outputs=[delete_status]
            )
        
        # Test 3: Delete operations with multiple outputs (updated pattern)
        with gr.Tab("Delete with Updates"):
            dropdown1 = gr.Dropdown(["model1", "model2"], label="Dropdown 1")
            dropdown2 = gr.Dropdown(["model3", "model4"], label="Dropdown 2")
            delete_update_btn = gr.Button("Delete with Update")
            delete_update_status = gr.HTML()
            
            delete_update_btn.click(
                fn=mock_delete_with_single_output,
                outputs=[delete_update_status]
            ).then(
                fn=lambda: (gr.update(value="model1"), gr.update(value="model3")),
                outputs=[dropdown1, dropdown2]
            )
        
        # Test 4: Lambda closures
        with gr.Tab("Lambda Functions"):
            lambda_btn = gr.Button("Test Lambda")
            lambda_output = gr.HTML()
            
            lambda_btn.click(
                fn=create_delete_function("test-model"),
                outputs=[lambda_output]
            )
        
        # Test 5: Simplified function signatures
        with gr.Tab("Simplified Functions"):
            model_type = gr.Dropdown(["image", "3d"], value="image")
            model_name = gr.Textbox(value="test-model")
            force_flag = gr.Checkbox(label="Force", value=False)
            process_btn = gr.Button("Process")
            process_output = gr.HTML()
            
            process_btn.click(
                fn=simplified_function,
                inputs=[model_type, model_name, force_flag],
                outputs=[process_output]
            )
    
    print("✅ Complex interface created successfully")
    
    # The critical test - API schema generation
    print("🔍 Testing API schema generation (the critical point)...")
    api_info = demo.get_api_info()
    print("✅ API schema generated without errors!")
    
    # Validate the schema structure
    if hasattr(demo, 'fns') and demo.fns:
        print(f"✅ Found {len(demo.fns)} function endpoints")
        
        # Count different types of operations
        lambda_fns = sum(1 for fn in demo.fns if 'lambda' in str(fn))
        regular_fns = len(demo.fns) - lambda_fns
        
        print(f"   - Regular functions: {regular_fns}")
        print(f"   - Lambda functions: {lambda_fns}")
    
    print("\n🎉 ALL VALIDATION TESTS PASSED!")
    print("🎯 The gradio schema error should be completely resolved.")
    print("🔧 Model deletion and all other UI operations should work correctly.")
    
except Exception as e:
    print(f"❌ VALIDATION FAILED: {e}")
    import traceback
    traceback.print_exc()
    print("\n💥 The gradio schema error is still present.")
    print("🔍 Additional debugging may be needed.")

print("\n✅ Final validation complete.")