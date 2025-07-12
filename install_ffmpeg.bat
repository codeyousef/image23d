@echo off
echo 🎬 Windows FFmpeg Quick Installer
echo ================================

echo.
echo Checking if FFmpeg is already installed...
ffmpeg -version >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ FFmpeg is already installed!
    echo.
    ffmpeg -version | findstr "ffmpeg version"
    echo.
    echo 🎯 Your face swap should work now!
    pause
    exit /b 0
)

echo ❌ FFmpeg not found. Installing...
echo.

echo 🚀 Attempting installation with winget...
winget install FFmpeg
if %errorlevel% == 0 (
    echo.
    echo ✅ FFmpeg installed successfully!
    echo.
    echo ⚠️  IMPORTANT: 
    echo    • Close this window
    echo    • Restart your terminal/command prompt  
    echo    • Restart your Python application
    echo    • Test with: ffmpeg -version
    echo.
    pause
    exit /b 0
) else (
    echo.
    echo ❌ winget installation failed. 
    echo.
    echo 📋 Manual Installation Required:
    echo 1. Go to: https://www.gyan.dev/ffmpeg/builds/
    echo 2. Download 'ffmpeg-release-essentials.zip'
    echo 3. Extract to C:\ffmpeg\
    echo 4. Add C:\ffmpeg\bin to your PATH
    echo 5. Restart your terminal and application
    echo.
    echo 💡 Or run: python windows_ffmpeg_guide.py
    echo    for detailed instructions
    echo.
    pause
    exit /b 1
)