#!/usr/bin/env python3
"""Simple Windows FFmpeg installation guide."""

import subprocess
import sys

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

def main():
    print("🎬 Windows FFmpeg Installation Guide")
    print("=" * 42)
    
    # Check current status
    ffmpeg_available, status_msg = check_ffmpeg()
    
    if ffmpeg_available:
        print(f"✅ FFmpeg is already installed: {status_msg}")
        print("\n🎯 Your face swap should work now!")
        return
    
    print(f"❌ FFmpeg not found: {status_msg}")
    print("\n📋 Quick Installation for Windows:")
    print("=" * 40)
    
    print("\n🚀 Method 1: Using Windows Package Manager (Recommended)")
    print("1. Open Command Prompt or PowerShell as Administrator")
    print("2. Run: winget install FFmpeg")
    print("3. Restart your terminal")
    print("4. Test: ffmpeg -version")
    
    print("\n🔧 Method 2: Manual Installation")
    print("1. 🔗 Go to: https://www.gyan.dev/ffmpeg/builds/")
    print("2. 📦 Download 'ffmpeg-release-essentials.zip'")
    print("3. 📁 Extract to C:\\ffmpeg\\")
    print("4. ⚙️  Add C:\\ffmpeg\\bin to your PATH:")
    print("   • Press Win + X, select 'System'")
    print("   • Click 'Advanced system settings'")  
    print("   • Click 'Environment Variables'")
    print("   • Under 'System Variables', find 'Path', click 'Edit'")
    print("   • Click 'New', add: C:\\ffmpeg\\bin")
    print("   • Click OK on all dialogs")
    print("5. 🔄 Restart your terminal/application")
    print("6. 🧪 Test: ffmpeg -version")
    
    print("\n🍫 Method 3: Using Chocolatey")
    print("1. Install Chocolatey from: https://chocolatey.org/install")
    print("2. Run: choco install ffmpeg")
    print("3. Restart your terminal")
    
    print("\n" + "="*50)
    print("⚠️  IMPORTANT NOTES:")
    print("• You MUST restart your terminal/command prompt after installation")
    print("• You MUST restart the application after installation")
    print("• Test with 'ffmpeg -version' before trying face swap again")
    
    print("\n🎯 After successful installation:")
    print("The face swap feature should work without FFmpeg errors!")

if __name__ == "__main__":
    main()