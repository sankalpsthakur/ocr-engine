#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d "ocr_env" ]; then
    echo "Virtual environment 'ocr_env' not found. Please create it first with:"
    echo "python3 -m venv ocr_env"
    echo "source ocr_env/bin/activate"
    echo "pip install -r api/requirements.txt"
    exit 1
fi

# Activate virtual environment
source ocr_env/bin/activate

# Set environment variables
export PYTHONUNBUFFERED=1
export PORT=${PORT:-8080}

# Start the API server
nohup python api/main.py > api_server.log 2>&1 &
echo $! > api.pid
echo "API server started with PID $(cat api.pid) on port $PORT"
echo "Logs: tail -f api_server.log"