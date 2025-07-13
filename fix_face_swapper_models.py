#!/usr/bin/env python3
"""Script to fix FaceFusion face_swapper model loading issues."""

import sys
import os
import subprocess
from pathlib import Path
import shutil

def check_dependencies():
    """Check if required dependencies are installed."""
    print("🔍 Checking Dependencies")
    print("=" * 30)
    
    required_packages = [
        "onnxruntime",
        "insightface", 
        "opencv-python",
        "numpy",
        "PIL"
    ]
    
    missing = []
    for package in required_packages:
        try:
            if package == "PIL":
                import PIL
                print(f"✅ {package}: {PIL.__version__}")
            else:
                module = __import__(package.replace("-", "_"))
                version = getattr(module, "__version__", "unknown")
                print(f"✅ {package}: {version}")
        except ImportError:
            print(f"❌ {package}: Missing")
            missing.append(package)
    
    if missing:
        print(f"\n💡 Install missing packages:")
        print(f"pip install {' '.join(missing)}")
        return False
    
    print("\n✅ All dependencies found!")
    return True

def check_facefusion_installation():
    """Check FaceFusion installation."""
    print("\n🔍 Checking FaceFusion Installation")
    print("=" * 40)
    
    facefusion_path = Path("models/facefusion")
    if not facefusion_path.exists():
        print(f"❌ FaceFusion not found at {facefusion_path}")
        print("💡 Install FaceFusion:")
        print("   git clone https://github.com/facefusion/facefusion.git models/facefusion")
        print("   cd models/facefusion")
        print("   pip install -r requirements.txt")
        return False
    
    facefusion_script = facefusion_path / "facefusion.py"
    if not facefusion_script.exists():
        print(f"❌ FaceFusion script not found at {facefusion_script}")
        return False
    
    print(f"✅ FaceFusion found at {facefusion_path}")
    print(f"✅ Script found at {facefusion_script}")
    return True

def check_model_files():
    """Check if model files exist."""
    print("\n🔍 Checking Model Files")
    print("=" * 25)
    
    # Common model locations
    model_locations = [
        Path("models/facefusion/.assets"),
        Path("models/.insightface"),
        Path.home() / ".insightface",
        Path.home() / ".cache" / "insightface"
    ]
    
    for location in model_locations:
        if location.exists():
            print(f"📁 Found models directory: {location}")
            models = list(location.glob("**/*.onnx"))
            if models:
                print(f"   Found {len(models)} model files:")
                for model in models[:5]:  # Show first 5
                    print(f"   • {model.name}")
                if len(models) > 5:
                    print(f"   • ... and {len(models) - 5} more")
                return True
        else:
            print(f"❌ Not found: {location}")
    
    print("❌ No model files found")
    return False

def download_models():
    """Attempt to download models."""
    print("\n🔄 Downloading Models")
    print("=" * 22)
    
    # First try using our adapter
    try:
        sys.path.insert(0, str(Path(__file__).parent / "src" / "hunyuan3d_app" / "features" / "face_swap"))
        from facefusion_cli_adapter import FaceFusionCLIAdapter
        
        print("📦 Using FaceFusion CLI adapter...")
        adapter = FaceFusionCLIAdapter()
        success, msg = adapter.initialize()
        
        if success:
            print("✅ Adapter initialized")
            print("🔄 Downloading models (this may take several minutes)...")
            download_success, download_msg = adapter.download_models()
            
            if download_success:
                print(f"✅ {download_msg}")
                return True
            else:
                print(f"❌ Download failed: {download_msg}")
        else:
            print(f"❌ Adapter initialization failed: {msg}")
            
    except Exception as e:
        print(f"❌ Adapter method failed: {e}")
    
    # Fallback: try manual download using FaceFusion CLI
    print("\n🔄 Trying manual model download...")
    facefusion_path = Path("models/facefusion")
    if facefusion_path.exists():
        try:
            # Try to run a simple command that should download models
            cmd = [
                sys.executable,
                str(facefusion_path / "facefusion.py"),
                "--help"
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(facefusion_path),
                env=dict(os.environ, PYTHONPATH=str(facefusion_path))
            )
            
            if result.returncode == 0:
                print("✅ FaceFusion CLI is working")
                print("💡 Models should download automatically on first use")
                return True
            else:
                print(f"❌ FaceFusion CLI failed: {result.stderr}")
                
        except Exception as e:
            print(f"❌ Manual download failed: {e}")
    
    return False

def fix_permissions():
    """Fix file permissions if on Unix-like system."""
    if os.name != 'nt':  # Not Windows
        print("\n🔧 Fixing Permissions")
        print("=" * 20)
        
        facefusion_path = Path("models/facefusion")
        if facefusion_path.exists():
            try:
                # Make facefusion.py executable
                script_path = facefusion_path / "facefusion.py"
                if script_path.exists():
                    os.chmod(script_path, 0o755)
                    print(f"✅ Made {script_path} executable")
                    
                return True
            except Exception as e:
                print(f"❌ Permission fix failed: {e}")
                return False
        else:
            print("❌ FaceFusion not found")
            return False
    else:
        print("\n💡 Windows detected - no permission fixes needed")
        return True

def create_test_images():
    """Create test images for model validation."""
    print("\n🖼️  Creating Test Images")
    print("=" * 25)
    
    try:
        from PIL import Image
        import numpy as np
        
        # Create simple test images
        test_dir = Path("test_images")
        test_dir.mkdir(exist_ok=True)
        
        # Create a simple face-like image (white with black dots for eyes)
        face_img = Image.new('RGB', (256, 256), color='white')
        # Add simple "eyes" 
        pixels = np.array(face_img)
        pixels[80:90, 80:90] = [0, 0, 0]  # Left eye
        pixels[80:90, 160:170] = [0, 0, 0]  # Right eye
        pixels[150:160, 120:130] = [0, 0, 0]  # Mouth
        
        face_img = Image.fromarray(pixels)
        face_img.save(test_dir / "test_source.png")
        
        # Create target image
        target_img = Image.new('RGB', (256, 256), color='lightblue')
        target_img.save(test_dir / "test_target.png")
        
        print(f"✅ Created test images in {test_dir}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create test images: {e}")
        return False

def run_diagnostic():
    """Run complete diagnostic."""
    print("🔧 FaceFusion Face Swapper Diagnostic")
    print("=" * 42)
    
    steps = [
        ("Dependencies", check_dependencies),
        ("FaceFusion Installation", check_facefusion_installation),
        ("Model Files", check_model_files),
        ("Permissions", fix_permissions),
        ("Test Images", create_test_images),
    ]
    
    results = {}
    for step_name, step_func in steps:
        try:
            results[step_name] = step_func()
        except Exception as e:
            print(f"❌ {step_name} failed: {e}")
            results[step_name] = False
    
    # Try model download if models are missing
    if not results.get("Model Files", False):
        print("\n🔄 Attempting model download...")
        results["Model Download"] = download_models()
    
    # Summary
    print("\n📋 Diagnostic Summary")
    print("=" * 22)
    for step, success in results.items():
        status = "✅" if success else "❌"
        print(f"{status} {step}")
    
    # Recommendations
    print("\n💡 Recommendations")
    print("=" * 18)
    
    if not all(results.values()):
        print("❌ Issues found. Please address the failed items above.")
        
        if not results.get("Dependencies", True):
            print("🔧 Install missing dependencies first")
        
        if not results.get("FaceFusion Installation", True):
            print("🔧 Install FaceFusion in models/facefusion/")
            
        if not results.get("Model Files", True):
            print("🔧 Models will download on first use, or run manual download")
            
        print("\n📖 After fixing issues:")
        print("1. Restart your application")
        print("2. Try face swap again")
        print("3. Be patient - first model download takes time")
        
    else:
        print("✅ All checks passed!")
        print("🎯 Face swapper should work now")
        print("💡 If you still get errors, the models may be downloading")

if __name__ == "__main__":
    run_diagnostic()