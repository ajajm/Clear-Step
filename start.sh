#!/bin/bash

# ClearStep Navigation Assistant - Startup Script

echo "========================================="
echo "  ClearStep Navigation Assistant"
echo "========================================="
echo ""

# Find best Python command (prefer Python 3.12 with packages installed)
if command -v python &> /dev/null; then
    PY_CMD="python"
elif command -v python3 &> /dev/null; then
    PY_CMD="python3"
else
    echo "Error: Python is not installed or not in PATH."
    echo "Please install Python 3.8+."
    exit 1
fi

# Load environment variables from .env if present
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
    echo "✅ Environment variables loaded from .env"
fi

# Check if required packages are already available in current Python
$PY_CMD -c "import flask, ultralytics, cv2" &> /dev/null
HAS_DEPS=$?

if [ $HAS_DEPS -ne 0 ]; then
    echo "Checking virtual environment..."
    # Determine correct venv activation path for Windows vs Linux
    if [ -f "venv/Scripts/activate" ]; then
        source "venv/Scripts/activate"
    elif [ -f "venv/bin/activate" ]; then
        source "venv/bin/activate"
    else
        echo "Creating virtual environment..."
        $PY_CMD -m venv venv
        if [ -f "venv/Scripts/activate" ]; then
            source "venv/Scripts/activate"
            $PY_CMD -m pip install -r requirements.txt
        elif [ -f "venv/bin/activate" ]; then
            source "venv/bin/activate"
            $PY_CMD -m pip install -r requirements.txt
        fi
    fi
fi

# Check OpenRouter API Key
if [ -n "$OPENROUTER_API_KEY" ]; then
    echo "✅ OpenRouter API Key configured"
else
    echo "ℹ️  OPENROUTER_API_KEY not set (Running in Offline Assistive AI mode)"
fi

# Check YOLO Model
if [ -f "yolo11n-seg.pt" ] || [ -f "server/yolo11n-seg.pt" ]; then
    echo "✅ YOLO11 segmentation model detected"
fi

echo ""
echo "🚀 Starting server at: http://localhost:5000"
echo "Press Ctrl+C to stop"
echo ""

cd server
$PY_CMD client_camera_server.py
