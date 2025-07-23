#!/usr/bin/env python3
"""
Development helper script for NeuralForge Studio

Usage:
    python dev.py desktop    # Run desktop app only
    python dev.py gradio     # Run existing Gradio app  
    python dev.py web        # Run web app (backend + frontend)
    python dev.py all        # Run everything
    python dev.py test       # Run tests
"""

import subprocess
import sys
import os
import signal
import time
import asyncio
from pathlib import Path

class DevRunner:
    """Manages development processes"""
    
    def __init__(self):
        self.processes = []
        self.project_root = Path(__file__).parent
        
    def run_desktop(self):
        """Run desktop app"""
        print("🚀 Starting NeuralForge Desktop...")
        cmd = [sys.executable, "desktop/main.py"]
        p = subprocess.Popen(cmd, cwd=self.project_root)
        self.processes.append(p)
        print("✅ Desktop app started on http://localhost:8765")
        return p
        
    def run_gradio(self):
        """Run existing Gradio app"""
        print("🚀 Starting Gradio app...")
        cmd = [sys.executable, "-m", "hunyuan3d_app.app"]
        p = subprocess.Popen(cmd, cwd=self.project_root)
        self.processes.append(p)
        print("✅ Gradio app started on http://localhost:7860")
        return p
        
    def run_web_backend(self):
        """Run FastAPI backend"""
        print("🚀 Starting web backend...")
        backend_dir = self.project_root / "backend"
        if not backend_dir.exists():
            print("⚠️  Backend directory not found. Creating placeholder...")
            backend_dir.mkdir(exist_ok=True)
            # Create minimal FastAPI app
            main_py = backend_dir / "main.py"
            main_py.write_text('''from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="NeuralForge API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "NeuralForge API - Coming Soon"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}
''')
            
        cmd = ["uvicorn", "main:app", "--reload", "--port", "8000"]
        p = subprocess.Popen(cmd, cwd=backend_dir)
        self.processes.append(p)
        print("✅ Web backend started on http://localhost:8000")
        return p
        
    def run_web_frontend(self):
        """Run React frontend"""
        print("🚀 Starting web frontend...")
        frontend_dir = self.project_root / "frontend"
        if not frontend_dir.exists():
            print("⚠️  Frontend directory not found. Creating placeholder...")
            frontend_dir.mkdir(exist_ok=True)
            # Create minimal index.html
            index_html = frontend_dir / "index.html"
            index_html.write_text('''<!DOCTYPE html>
<html>
<head>
    <title>NeuralForge Studio</title>
    <style>
        body {
            margin: 0;
            padding: 0;
            background: #0A0A0A;
            color: #E5E5E5;
            font-family: system-ui, sans-serif;
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100vh;
        }
        .container {
            text-align: center;
        }
        h1 {
            color: #7C3AED;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>NeuralForge Studio</h1>
        <p>Web interface coming soon...</p>
        <p>Use the desktop app or API for now.</p>
    </div>
</body>
</html>''')
            # Simple Python HTTP server
            cmd = [sys.executable, "-m", "http.server", "3000"]
            p = subprocess.Popen(cmd, cwd=frontend_dir)
        else:
            # Assume npm/yarn project
            if (frontend_dir / "package.json").exists():
                cmd = ["npm", "run", "dev"]
            else:
                cmd = [sys.executable, "-m", "http.server", "3000"]
            p = subprocess.Popen(cmd, cwd=frontend_dir)
            
        self.processes.append(p)
        print("✅ Web frontend started on http://localhost:3000")
        return p
        
    def run_all(self):
        """Run all services"""
        print("🚀 Starting all services...")
        self.run_gradio()
        time.sleep(2)
        self.run_desktop()
        time.sleep(2)
        self.run_web_backend()
        time.sleep(2)
        self.run_web_frontend()
        
    def run_tests(self):
        """Run tests"""
        print("🧪 Running tests...")
        
        # Test prompt enhancement
        print("\n--- Testing Prompt Enhancement ---")
        test_cmd = [sys.executable, "test_prompt_enhancement.py"]
        subprocess.run(test_cmd, cwd=self.project_root)
        
        # Run pytest if available
        try:
            print("\n--- Running pytest ---")
            subprocess.run(["pytest", "tests/"], cwd=self.project_root)
        except FileNotFoundError:
            print("⚠️  pytest not found. Install with: pip install pytest")
            
    def cleanup(self):
        """Clean up all processes"""
        print("\n🛑 Shutting down all services...")
        for p in self.processes:
            try:
                p.terminate()
                p.wait(timeout=5)
            except:
                p.kill()
        print("✅ All services stopped")
        
    def wait(self):
        """Wait for interrupt"""
        try:
            print("\n✨ All services running. Press Ctrl+C to stop.\n")
            while True:
                time.sleep(1)
                # Check if any process died
                for p in self.processes:
                    if p.poll() is not None:
                        print(f"⚠️  Process {p.pid} died with code {p.returncode}")
                        self.processes.remove(p)
        except KeyboardInterrupt:
            pass

def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        mode = "all"
    else:
        mode = sys.argv[1]
        
    runner = DevRunner()
    
    # Set up signal handler
    def signal_handler(sig, frame):
        runner.cleanup()
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        if mode == "desktop":
            runner.run_desktop()
            runner.wait()
        elif mode == "gradio":
            runner.run_gradio()
            runner.wait()
        elif mode == "web":
            runner.run_web_backend()
            runner.run_web_frontend()
            runner.wait()
        elif mode == "all":
            runner.run_all()
            runner.wait()
        elif mode == "test":
            runner.run_tests()
        else:
            print(f"Unknown mode: {mode}")
            print(__doc__)
            sys.exit(1)
    finally:
        runner.cleanup()

if __name__ == "__main__":
    main()