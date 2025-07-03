#!/usr/bin/env python3
"""Test gradio schema generation to identify the issue"""

import gradio as gr
import sys
import os

# Add the source directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_simple_function():
    """Test a simple function to ensure gradio works"""
    return "Hello World"

def test_function_with_boolean_param(force_redownload=False):
    """Test function with boolean parameter"""
    return f"Force redownload: {force_redownload}"

def test_function_with_complex_param(force_redownload: bool = False):
    """Test function with typed boolean parameter"""
    return f"Force redownload: {force_redownload}"

# Test each function individually
print("Testing gradio schema generation...")

try:
    # Test 1: Simple function
    print("1. Testing simple function...")
    with gr.Blocks() as demo1:
        btn1 = gr.Button("Test")
        out1 = gr.Textbox()
        btn1.click(test_simple_function, outputs=out1)
    print("✅ Simple function works")
    
    # Test 2: Function with boolean param (no type hint)
    print("2. Testing function with boolean param (no type hint)...")
    with gr.Blocks() as demo2:
        btn2 = gr.Button("Test")
        checkbox2 = gr.Checkbox(label="Force redownload")
        out2 = gr.Textbox()
        btn2.click(test_function_with_boolean_param, inputs=[checkbox2], outputs=out2)
    print("✅ Function with boolean param works")
    
    # Test 3: Function with typed boolean param
    print("3. Testing function with typed boolean param...")
    with gr.Blocks() as demo3:
        btn3 = gr.Button("Test")
        checkbox3 = gr.Checkbox(label="Force redownload")
        out3 = gr.Textbox()
        btn3.click(test_function_with_complex_param, inputs=[checkbox3], outputs=out3)
    print("✅ Function with typed boolean param works")
    
    print("All tests passed! The issue is likely in the hunyuan3d_app code.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()