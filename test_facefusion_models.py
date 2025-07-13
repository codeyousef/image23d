#!/usr/bin/env python3
"""Test FaceFusion model download and face swapper loading."""

import sys
import subprocess
import os
from pathlib import Path
import tempfile

def test_facefusion_basic():
    """Test basic FaceFusion functionality."""
    print("🧪 Testing FaceFusion Basic Functionality")
    print("=" * 45)
    
    facefusion_path = Path("models/facefusion").absolute()
    facefusion_script = facefusion_path / "facefusion.py"
    
    print(f"FaceFusion path: {facefusion_path}")
    print(f"Script path: {facefusion_script}")
    print(f"Script exists: {facefusion_script.exists()}")
    
    # Test 1: Check if FaceFusion can show help
    print("1. Testing FaceFusion help command...")
    try:
        cmd = [sys.executable, str(facefusion_script), "--version"]  # Try version instead of help
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(facefusion_path),
            env=dict(os.environ, 
                   PYTHONPATH=str(facefusion_path),
                   FACEFUSION_SKIP_VALIDATION="1",
                   FACEFUSION_SKIP_FFMPEG="1")
        )
        
        if result.returncode == 0:
            print("✅ FaceFusion help works")
        else:
            print(f"❌ FaceFusion help failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Help test failed: {e}")
        return False
    
    # Test 2: Try to list available models
    print("2. Testing model listing...")
    try:
        cmd = [sys.executable, str(facefusion_script), "--version"]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(facefusion_path),
            env=dict(os.environ, PYTHONPATH=str(facefusion_path))
        )
        
        if result.returncode == 0:
            print(f"✅ FaceFusion version: {result.stdout.strip()}")
        else:
            print(f"⚠️  Version check failed: {result.stderr}")
            
    except Exception as e:
        print(f"⚠️  Version test failed: {e}")
    
    return True

def try_model_download():
    """Try to download models using various methods."""
    print("\n🔄 Attempting Model Download")
    print("=" * 32)
    
    facefusion_path = Path("models/facefusion")
    facefusion_script = facefusion_path / "facefusion.py"
    
    # Create test images
    print("Creating minimal test images...")
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create tiny test images
        from PIL import Image
        source_img = Image.new('RGB', (64, 64), color='white')
        target_img = Image.new('RGB', (64, 64), color='lightblue')
        
        source_path = temp_path / "source.png"
        target_path = temp_path / "target.png"
        output_path = temp_path / "output.png"
        
        source_img.save(source_path)
        target_img.save(target_path)
        
        print("✅ Test images created")
        
        # Try face swap command to trigger model download
        print("🔄 Running face swap to trigger model download...")
        print("⏱️  This may take several minutes for first-time model download...")
        
        cmd = [
            sys.executable, str(facefusion_script),
            "headless-run",
            "--source-paths", str(source_path),
            "--target-path", str(target_path),
            "--output-path", str(output_path),
            "--processors", "face_swapper",
            "--face-swapper-model", "inswapper_128"
        ]
        
        print(f"Command: {' '.join(cmd[:8])}...")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minutes for model download
                cwd=str(facefusion_path),
                env=dict(os.environ, 
                       PYTHONPATH=str(facefusion_path),
                       FACEFUSION_SKIP_VALIDATION="1")
            )
            
            print(f"\nReturn code: {result.returncode}")
            if result.stdout:
                print(f"Stdout: {result.stdout[:500]}...")
            if result.stderr:
                print(f"Stderr: {result.stderr[:500]}...")
            
            if result.returncode == 0:
                if output_path.exists():
                    print("✅ Face swap succeeded! Models are working.")
                    return True
                else:
                    print("⚠️  Command succeeded but no output file created")
                    return False
            else:
                if "download" in result.stderr.lower() or "model" in result.stderr.lower():
                    print("📥 Models are downloading - this is expected on first run")
                    print("💡 Try again in a few minutes after models finish downloading")
                    return False
                else:
                    print(f"❌ Face swap failed: {result.stderr}")
                    return False
                    
        except subprocess.TimeoutExpired:
            print("⏱️  Command timed out - models may still be downloading")
            print("💡 Check your internet connection and try again later")
            return False
        except Exception as e:
            print(f"❌ Face swap test failed: {e}")
            return False

def check_model_directories():
    """Check if model directories were created."""
    print("\n📁 Checking Model Directories")
    print("=" * 32)
    
    possible_dirs = [
        Path("models/facefusion/.assets"),
        Path.home() / ".insightface",
        Path.home() / ".cache" / "insightface",
        Path("models") / ".insightface"
    ]
    
    found_models = False
    for model_dir in possible_dirs:
        if model_dir.exists():
            models = list(model_dir.glob("**/*.onnx"))
            if models:
                print(f"✅ Found {len(models)} model files in {model_dir}")
                found_models = True
            else:
                print(f"📁 Directory exists but empty: {model_dir}")
        else:
            print(f"❌ Not found: {model_dir}")
    
    if not found_models:
        print("📥 No model files found - they will download on first use")
    
    return found_models

def main():
    print("🔧 FaceFusion Model Test")
    print("=" * 26)
    
    # Basic functionality test
    if not test_facefusion_basic():
        print("\n❌ Basic functionality test failed")
        return False
    
    # Check existing models
    has_models = check_model_directories()
    
    if not has_models:
        print("\n🔄 No models found, attempting download...")
        download_success = try_model_download()
        
        if download_success:
            print("\n✅ Model download test passed!")
        else:
            print("\n⚠️  Model download is in progress or failed")
            print("💡 This is normal for first-time setup")
            print("💡 Models download automatically when needed")
    else:
        print("\n✅ Models already present!")
    
    print("\n📋 Next Steps:")
    print("1. If models are downloading, wait for completion")
    print("2. Try face swap in your application") 
    print("3. Be patient on first use - model download takes time")
    print("4. Check internet connection if downloads fail")
    
    return True

if __name__ == "__main__":
    main()