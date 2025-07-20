#!/bin/bash
cd /Users/sankalpthakur/Projects/Projects\ -\ Emtribe/ocr-engine
source ocr_env/bin/activate
export PYTHONUNBUFFERED=1
nohup python api/main.py > api_server.log 2>&1 &
echo $! > api.pid
echo "API server started with PID $(cat api.pid)"
echo "Logs: tail -f api_server.log"