# NeuralForge Studio Desktop Application

A native desktop application for AI-powered creative content generation, featuring image synthesis, 3D model creation, and video generation capabilities.

## Features

- **Native Desktop Experience**: Runs as a standalone desktop application, not in a browser
- **Multi-Modal Generation**: Create images, 3D models, and videos from text prompts
- **Advanced Enhancement**: AI-powered prompt enhancement using Ollama/Mistral
- **Model Management**: Download and manage various AI models
- **Library System**: Organize and browse your generated assets
- **Lab Tools**: Advanced features for power users

## Installation

1. **Prerequisites**:
   - Python 3.10 or higher
   - CUDA-capable GPU (recommended) or CPU-only mode
   - Display environment (for Linux: X11 or Wayland)

2. **Install Dependencies**:
   ```bash
   # Clone the repository
   git clone <repository-url>
   cd hunyuan3d-app

   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install dependencies
   pip install -e .
   pip install pywebview  # Required for native desktop mode
   ```

3. **Install Ollama** (for prompt enhancement):
   ```bash
   # Linux/Mac
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Pull the Mistral model
   ollama pull mistral:7b-instruct
   ```

## Running the Application

### Windows
Double-click `NeuralForge.bat` or run:
```cmd
python neuralforge.py
```

### Linux/Mac
```bash
./neuralforge.sh
# or
python3 neuralforge.py
```

### Direct Launch
```bash
python -m desktop.main
```

## Native Desktop Mode

The application runs exclusively as a native desktop application:
- No web browser is opened
- No web server is started
- All UI is rendered in a native window using pywebview
- Secure local storage in `~/.neuralforge/`

## Configuration

The app stores configuration and data in:
- **Windows**: `%USERPROFILE%\.neuralforge\`
- **Linux/Mac**: `~/.neuralforge/`

## Troubleshooting

### "No module named 'pywebview'"
Install pywebview: `pip install pywebview`

### Linux: "No display found"
- Ensure X11 or Wayland is running
- For SSH: Use X11 forwarding with `ssh -X`

### Windows: Application doesn't start
- Check if Python is in PATH
- Run from command prompt to see error messages

### GPU not detected
- Install CUDA toolkit and appropriate PyTorch version
- Check GPU drivers are up to date

## Architecture

The desktop app uses:
- **NiceGUI**: Modern Python UI framework
- **pywebview**: Native window rendering
- **Shared Core**: Common business logic with web version
- **Async Processing**: Non-blocking UI during generation

## Development

To run in development mode with hot reload:
```python
# Edit desktop/main.py and set:
reload=True  # in ui.run()
```

## Security

- No web server exposure
- Local file system access only
- Credentials stored securely in user directory
- No external network access except for model downloads

## License

See LICENSE file in the root directory.