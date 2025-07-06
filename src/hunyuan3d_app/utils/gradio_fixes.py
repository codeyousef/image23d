"""
Gradio schema generation fix for version 5.35.0
This file contains a monkey patch to fix the TypeError: argument of type 'bool' is not iterable error
"""

import gradio as gr
from gradio_client import utils as gradio_utils
try:
    from gradio_client.utils import APIInfoParseError
except ImportError:
    # Fallback if APIInfoParseError is not available
    class APIInfoParseError(Exception):
        pass

def patched_get_type(schema):
    """
    Patched version of gradio_client.utils.get_type that handles boolean schemas
    """
    # If schema is a boolean, return a default type
    if isinstance(schema, bool):
        return "any"
    
    # If schema is not a dict, convert it to one
    if not isinstance(schema, dict):
        return "any"
    
    # Original logic continues here
    if "const" in schema:
        return f'Literal[{schema["const"]}]'
    if "enum" in schema:
        return f'Literal[{", ".join([str(choice) for choice in schema["enum"]])}]'
    if "type" not in schema:
        return "any"
    
    type_mapping = {
        "string": "str",
        "number": "float", 
        "integer": "int",
        "boolean": "bool",
        "array": "List",
        "object": "Dict"
    }
    
    return type_mapping.get(schema["type"], "any")

def patched_json_schema_to_python_type(schema, defs=None):
    """
    Patched version that handles boolean schemas and parse errors gracefully
    """
    if isinstance(schema, bool):
        return "any"
    
    if not isinstance(schema, dict):
        return "any"
    
    # Call the original function but with protection
    try:
        return gradio_utils._json_schema_to_python_type(schema, defs)
    except TypeError as e:
        if "argument of type 'bool' is not iterable" in str(e):
            return "any"
        raise
    except APIInfoParseError as e:
        # Handle APIInfoParseError specifically
        print(f"⚠️ Gradio schema parsing error (handled): {e}")
        return "any"
    except Exception as e:
        # Handle other parsing errors
        if "Cannot parse schema" in str(e):
            print(f"⚠️ Gradio schema parsing error (handled): {e}")
            return "any"
        raise

def apply_gradio_fix():
    """
    Apply the monkey patch to fix gradio schema generation
    """
    print("🔧 Applying gradio schema generation fix...")
    
    # Monkey patch the problematic functions
    gradio_utils.get_type = patched_get_type
    
    # Store original for safety
    original_json_schema_to_python_type = gradio_utils.json_schema_to_python_type
    gradio_utils.json_schema_to_python_type = patched_json_schema_to_python_type
    
    print("✅ Gradio schema fix applied successfully")
    
    return original_json_schema_to_python_type

if __name__ == "__main__":
    # Test the fix
    apply_gradio_fix()
    print("✅ Gradio fix can be imported and applied")