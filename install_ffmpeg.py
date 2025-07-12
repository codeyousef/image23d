#!/usr/bin/env python3
"""Helper script to detect and guide FFmpeg installation."""

import subprocess
import sys
import platform

def check_ffmpeg():
    """Check if FFmpeg is installed and accessible."""
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
    except subprocess.TimeoutExpired:
        return False, "FFmpeg command timed out"
    except Exception as e:
        return False, f"Error checking FFmpeg: {e}"

def get_install_instructions():
    """Get platform-specific installation instructions."""
    system = platform.system().lower()
    
    if system == "linux":
        return {
            "platform": "Linux",
            "commands": [
                "sudo apt update",
                "sudo apt install ffmpeg"
            ],
            "alternative": "Or for other distributions: sudo yum install ffmpeg (CentOS/RHEL) or pacman -S ffmpeg (Arch)"
        }
    elif system == "darwin":
        return {
            "platform": "macOS", 
            "commands": [
                "brew install ffmpeg"
            ],
            "alternative": "If you don't have Homebrew: /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        }
    elif system == "windows":
        return {
            "platform": "Windows",
            "commands": [
                "Download from https://ffmpeg.org/download.html#build-windows",
                "Extract to C:\\ffmpeg\\",
                "Add C:\\ffmpeg\\bin to your PATH environment variable"
            ],
            "alternative": "Or use Windows Package Manager: winget install FFmpeg"
        }
    else:
        return {
            "platform": "Unknown",
            "commands": ["Visit https://ffmpeg.org/download.html for your platform"],
            "alternative": ""
        }

def main():
    print("🎬 FFmpeg Installation Helper")
    print("=" * 40)
    
    # Check current status
    ffmpeg_available, status_msg = check_ffmpeg()
    
    if ffmpeg_available:
        print(f"✅ FFmpeg is installed: {status_msg}")
        print("\n🎯 FFmpeg is ready for FaceFusion!")
        print("You should no longer see FFmpeg-related errors.")
        return
    
    print(f"❌ FFmpeg not found: {status_msg}")
    print()
    
    # Get installation instructions
    instructions = get_install_instructions()
    
    print(f"📋 Installation Instructions for {instructions['platform']}:")
    print("-" * 50)
    
    for i, cmd in enumerate(instructions['commands'], 1):
        if cmd.startswith("http"):
            print(f"{i}. {cmd}")
        else:
            print(f"{i}. {cmd}")
    
    if instructions['alternative']:
        print(f"\n💡 Alternative: {instructions['alternative']}")
    
    print("\n🔧 After installing FFmpeg:")
    print("1. Restart your terminal/command prompt")
    print("2. Test with: ffmpeg -version")
    print("3. Try face swap in the application again")
    
    print("\n📝 Note: Even without FFmpeg, our FaceFusion adapter includes")
    print("workarounds that may allow face swapping to work for images.")

if __name__ == "__main__":
    main()