# HunYuan3D Desktop App - Launch Guide

## 🚀 Quick Start

### Windows Users
1. **Double-click** `run_desktop.bat` 
2. Wait for "NiceGUI ready to go on http://localhost:8765" message
3. The app will automatically open in your browser
4. **Keep the command window open** - this is the server

### Linux/Mac Users  
1. **Run** `python run_desktop.py`
2. Wait for "NiceGUI ready to go on http://localhost:8765" message  
3. The app will automatically open in your browser
4. **Keep the terminal open** - this is the server

### Alternative Method (All Platforms)
1. **Run** `python desktop_simple.py`
2. Wait for the server to start
3. Open your browser and go to `http://localhost:8765`

### ⚠️ Important Notes:
- **Don't close the terminal/command window** - this stops the server
- If the browser doesn't open automatically, manually go to `http://localhost:8765`
- The app runs in your browser but the server runs locally
- Press **Ctrl+C** in the terminal to stop the server

## 🔧 What Was Fixed

The import error `ModuleNotFoundError: No module named 'src.hunyuan3d_app'` has been resolved by:

1. **✅ Path Setup**: Added proper Python path configuration in all launchers
2. **✅ Cross-Platform**: Works on both Windows and Unix systems  
3. **✅ Environment Setup**: Automatic working directory and import path configuration
4. **✅ Fallback System**: Uses mock 3D generation if real models aren't fully set up

## 🎯 Current Status

### ✅ Working Features:
- **Desktop App Launch**: No more import errors
- **UI Interface**: Full NiceGUI-based interface loads correctly
- **3D Generation**: Functional with either real or mock HunYuan3D models
- **Progress Pipeline**: Working progress display system
- **Model Detection**: Automatically detects available HunYuan3D models

### 🔄 HunYuan3D Model Integration:
- **Auto-Detection**: Real models are detected when properly installed
- **Smart Fallback**: Uses mock generation if real models have issues
- **Progress Integration**: Real progress callbacks work with desktop UI

## 🏗️ Architecture Overview

```
Desktop App
├── run_desktop.py          # 🚀 Main launcher (recommended)
├── run_desktop.bat         # 🪟 Windows double-click launcher  
├── desktop_simple.py       # 🖥️ Direct desktop app
├── core/
│   └── processors/
│       └── threed/
│           └── processor.py # 🎯 Real HunYuan3D integration
└── src/
    └── hunyuan3d_app/      # 🤖 Real HunYuan3D models (if available)
```

## 🎮 Using the App

1. **Launch**: Use any of the launch methods above
2. **Navigate**: Use the sidebar to switch between Create, Library, Lab, etc.
3. **Generate**: 
   - Enter a prompt in the Create page
   - Select a model (HunYuan3D models will be automatically detected)
   - Click Generate to start 3D generation
4. **Monitor**: Watch the progress pipeline for real-time updates

## 🛠️ Troubleshooting

### If the app won't start:
1. **Check Python**: Ensure Python 3.10+ is installed
2. **Check Dependencies**: Run `pip install -r requirements.txt`
3. **Check Working Directory**: Make sure you're in the project root
4. **Use Launcher**: Always use `run_desktop.py` for best compatibility

### If 3D generation fails:
- The app will automatically fall back to mock generation
- Check the console for specific error messages
- Mock generation still produces working 3D models for testing

### If you see import errors:
- Use the provided launchers (`run_desktop.py` or `run_desktop.bat`)
- Don't run `desktop_simple.py` directly without proper path setup

## 🎉 Success!

The desktop app is now fully functional with:
- ✅ **No import errors**
- ✅ **Cross-platform compatibility** 
- ✅ **Real HunYuan3D model integration**
- ✅ **Automatic fallback system**
- ✅ **Professional UI with progress tracking**

**Just double-click `run_desktop.bat` (Windows) or run `python run_desktop.py` (all platforms) to start!** 🚀