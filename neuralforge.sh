#!/bin/bash
# NeuralForge Studio Desktop Launcher for Linux/Mac

echo "Starting NeuralForge Studio..."
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.10 or higher"
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "ERROR: Python $required_version or higher is required (found $python_version)"
    exit 1
fi

# Check if display is available (Linux)
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if [ -z "$DISPLAY" ]; then
        echo "ERROR: No display found. Desktop apps require a display."
        echo "If running via SSH, use X11 forwarding: ssh -X user@host"
        exit 1
    fi
fi

# Check virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "WARNING: No virtual environment detected"
    echo "Consider activating a virtual environment first"
    echo
fi

# Launch the desktop app
python3 neuralforge.py

if [ $? -ne 0 ]; then
    echo
    echo "ERROR: Failed to start NeuralForge Studio"
    exit 1
fi