#!/usr/bin/env python3
"""Windows-specific FFmpeg installation helper."""

import subprocess
import sys
import os
import zipfile
import urllib.request
from pathlib import Path
import shutil

def check_ffmpeg():
    """Check if FFmpeg is installed."""
    try:
        result = subprocess.run(["ffmpeg", "-version"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            return True, version_line
        else:
            return False, "FFmpeg command failed"
    except FileNotFoundError:
        return False, "FFmpeg not found in PATH"
    except Exception as e:
        return False, f"Error: {e}"

def check_winget():
    """Check if winget is available."""
    try:
        result = subprocess.run(["winget", "--version"], 
                              capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except:
        return False

def install_with_winget():
    """Install FFmpeg using winget."""
    try:
        print("🔄 Installing FFmpeg with winget...")
        result = subprocess.run(["winget", "install", "FFmpeg"], 
                              capture_output=False, text=True, timeout=300)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Winget installation failed: {e}")
        return False

def manual_install_guide():
    """Provide manual installation instructions."""
    print("\n📋 Manual FFmpeg Installation for Windows:")
    print("=" * 50)
    print("1. Download FFmpeg:")
    print("   🔗 https://www.gyan.dev/ffmpeg/builds/")
    print("   📦 Download the 'release builds' > 'ffmpeg-release-essentials.zip'")
    print()
    print("2. Extract the zip file:")
    print("   📁 Extract to C:\\ffmpeg\\")
    print("   📁 You should have C:\\ffmpeg\\bin\\ffmpeg.exe")
    print()
    print("3. Add to PATH:")
    print("   ⚙️  Press Win + R, type 'sysdm.cpl', press Enter")
    print("   ⚙️  Click 'Environment Variables'")
    print("   ⚙️  Under 'System Variables', find 'Path', click 'Edit'")
    print("   ⚙️  Click 'New', add: C:\\ffmpeg\\bin")
    print("   ⚙️  Click OK on all dialogs")
    print()
    print("4. Test installation:")
    print("   🔄 Open a NEW command prompt")
    print("   🧪 Type: ffmpeg -version")
    print()
    print("🔄 Alternative: Use chocolatey package manager:")
    print("   choco install ffmpeg")

def try_auto_install():
    """Try to automatically install FFmpeg."""
    print("🤖 Attempting automatic FFmpeg installation...")
    
    # Try winget first
    if check_winget():
        print("✅ Winget found, trying installation...")
        if install_with_winget():
            print("✅ FFmpeg installed successfully with winget!")
            return True
        else:
            print("❌ Winget installation failed")
    else:
        print("ℹ️  Winget not available")
    
    print("ℹ️  Automatic installation not possible, providing manual instructions...")
    return False

def main():
    print("🎬 Windows FFmpeg Installation Helper")
    print("=" * 45)
    
    # Check current status
    ffmpeg_available, status_msg = check_ffmpeg()
    
    if ffmpeg_available:
        print(f"✅ FFmpeg is already installed: {status_msg}")
        print("\n🎯 FFmpeg is ready for FaceFusion!")
        print("You should no longer see FFmpeg-related errors.")
        return
    
    print(f"❌ FFmpeg not found: {status_msg}")
    print()
    
    # Ask user preference
    print("🔧 Installation Options:")
    print("1. Try automatic installation (winget)")
    print("2. Show manual installation instructions")
    print("3. Exit")
    
    try:
        choice = input("\nChoose an option (1-3): ").strip()
        
        if choice == "1":
            success = try_auto_install()
            if not success:
                print("\n" + "="*50)
                manual_install_guide()
        elif choice == "2":
            manual_install_guide()
        elif choice == "3":
            print("👋 Exiting...")
            return
        else:
            print("❌ Invalid choice")
            manual_install_guide()
            
    except KeyboardInterrupt:
        print("\n👋 Installation cancelled")
        return
    
    print("\n" + "="*50)
    print("🔄 After installing FFmpeg:")
    print("1. ⚠️  IMPORTANT: Restart your terminal/command prompt")
    print("2. 🧪 Test with: ffmpeg -version")
    print("3. 🎯 Try face swap in the application again")
    print()
    print("💡 If you still get errors after installation:")
    print("   • Make sure you restarted your terminal")
    print("   • Check that C:\\ffmpeg\\bin is in your PATH")
    print("   • Try running 'where ffmpeg' to verify location")

if __name__ == "__main__":
    main()