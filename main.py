#!/usr/bin/env python3
"""
Main entry point for HunYuan3D App

Usage:
    python main.py          # Interactive menu
    python main.py desktop  # Run desktop interface directly  
    python main.py gradio   # Run Gradio interface directly
    python main.py api      # Run API server directly
"""

import sys
import os
from pathlib import Path

def main():
    """Main entry point with interface selection"""
    # Check for command line argument
    if len(sys.argv) > 1:
        choice = sys.argv[1].lower()
        if choice in ['desktop', '1']:
            run_desktop()
        elif choice in ['gradio', 'web', '2']:
            run_gradio()
        elif choice in ['api', 'server', '3']:
            run_api()
        else:
            print(f"Unknown option: {choice}")
            print("Valid options: desktop, gradio, api")
            sys.exit(1)
        return
    
    # Interactive menu
    print("🚀 HunYuan3D Studio Launcher")
    print("=" * 50)
    print("Choose your interface:")
    print("1. Desktop App (NiceGUI) - Modern desktop interface")
    print("2. Web App (Gradio) - Classic web interface")
    print("3. API Server (FastAPI) - REST API backend")
    print()
    
    choice = input("Enter your choice (1-3) [default: 1]: ").strip() or "1"
    
    if choice == "1":
        run_desktop()
    elif choice == "2":
        run_gradio()
    elif choice == "3":
        run_api()
    else:
        print("Invalid choice. Please run again and select 1, 2, or 3.")
        sys.exit(1)

def run_desktop():
    """Run the NiceGUI desktop interface"""
    print("\n🖥️  Starting Desktop Interface...")
    print("📍 Open your browser at http://localhost:8765")
    print("⌨️  Press Ctrl+C to stop the server\n")
    
    # Setup paths
    project_root = Path(__file__).parent.resolve()
    os.chdir(project_root)
    
    # Use desktop_simple.py which is more stable
    os.system(f'"{sys.executable}" desktop_simple.py')

def run_gradio():
    """Run the Gradio web interface"""
    print("\n🌐 Starting Gradio Web Interface...")
    print("📍 The interface will open in your browser automatically")
    print("⌨️  Press Ctrl+C to stop the server\n")
    
    # Setup paths
    project_root = Path(__file__).parent.resolve()
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "src"))
    
    # Import and run Gradio app
    try:
        # Try to find and run the actual Gradio app
        gradio_files = [
            "hunyuan3d_app.py",
            "app.py", 
            "src/app.py",
            "src/hunyuan3d_app/app.py"
        ]
        
        for file in gradio_files:
            file_path = project_root / file
            if file_path.exists():
                print(f"Found Gradio app at: {file_path}")
                os.system(f'"{sys.executable}" "{file_path}"')
                return
        
        # If no app found, create simple one
        print("Creating simple Gradio interface...")
        import gradio as gr
        
        def simple_3d_generation(prompt, model="hunyuan3d-2.1"):
            return f"Would generate 3D model with:\nPrompt: {prompt}\nModel: {model}"
        
        with gr.Blocks(title="HunYuan3D Studio") as demo:
            gr.Markdown("# 🎨 HunYuan3D Studio")
            gr.Markdown("Generate 3D models from text prompts")
            
            with gr.Row():
                with gr.Column():
                    prompt = gr.Textbox(label="Prompt", placeholder="A cute cat sitting")
                    model = gr.Dropdown(
                        choices=["hunyuan3d-2.1", "hunyuan3d-2.0", "hunyuan3d-2mini"],
                        value="hunyuan3d-2.1",
                        label="Model"
                    )
                    generate_btn = gr.Button("Generate", variant="primary")
                
                with gr.Column():
                    output = gr.Textbox(label="Output", lines=5)
            
            generate_btn.click(
                fn=simple_3d_generation,
                inputs=[prompt, model],
                outputs=output
            )
        
        demo.launch(server_port=7860, show_error=True)
        
    except ImportError as e:
        print(f"❌ Error loading Gradio: {e}")
        print("Please install gradio: pip install gradio")

def run_api():
    """Run the FastAPI backend server"""
    print("\n🚀 Starting API Server...")
    print("📍 API will be available at http://localhost:8000")
    print("📖 Documentation at http://localhost:8000/docs")
    print("⌨️  Press Ctrl+C to stop the server\n")
    
    # Run the backend
    project_root = Path(__file__).parent.resolve()
    backend_path = project_root / "backend" / "main.py"
    if backend_path.exists():
        os.system(f'"{sys.executable}" "{backend_path}"')
    else:
        print(f"❌ Backend not found at: {backend_path}")
        print("The API server requires the backend module.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down gracefully...")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)