#!/bin/bash

# OCR Engine Microservices Startup Script
# Starts Surya OCR, Qwen VL, and API Gateway services

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Starting OCR Engine Microservices..."

# Function to check if a service is healthy
check_service_health() {
    local url=$1
    local service_name=$2
    local max_attempts=30
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -s -f "$url" > /dev/null 2>&1; then
            echo "✓ $service_name is healthy"
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 2
    done
    
    echo "✗ $service_name failed to start after $max_attempts attempts"
    return 1
}

# Function to stop all services
stop_services() {
    echo "Stopping all OCR services..."
    pkill -f surya_service.py 2>/dev/null || true
    pkill -f qwen_service.py 2>/dev/null || true
    pkill -f api_gateway.py 2>/dev/null || true
    rm -f services/surya/surya.pid services/qwen/qwen.pid gateway.pid 2>/dev/null || true
}

# Handle script interruption (only stop on INT/TERM, not EXIT)
trap stop_services INT TERM

# Check for required virtual environments
if [ ! -d "surya_env" ]; then
    echo "Error: surya_env virtual environment not found"
    echo "Please run: python3 -m venv surya_env && source surya_env/bin/activate && pip install -r services/surya/requirements.txt"
    exit 1
fi

if [ ! -d "qwen_env" ]; then
    echo "Error: qwen_env virtual environment not found"
    echo "Please run: python3 -m venv qwen_env && source qwen_env/bin/activate && pip install -r services/qwen/requirements.txt"
    exit 1
fi

if [ ! -d "gateway_env" ]; then
    echo "Error: gateway_env virtual environment not found"
    echo "Please run: python3 -m venv gateway_env && source gateway_env/bin/activate && pip install -r gateway_requirements.txt"
    exit 1
fi

# Create required temp directory
mkdir -p /tmp/surya_ocr_api

# Stop any existing services
stop_services

echo "Starting services..."

# Start Surya OCR Service (port 8001)
echo "Starting Surya OCR service on port 8001..."
cd services/surya
source ../../surya_env/bin/activate
nohup python surya_service.py > "$SCRIPT_DIR/services/surya/surya.log" 2>&1 &
echo $! > surya.pid
cd ../..

# Start Qwen VL Service (port 8002)
echo "Starting Qwen VL service on port 8002..."
cd services/qwen
source ../../qwen_env/bin/activate
nohup python qwen_service.py > "$SCRIPT_DIR/services/qwen/qwen.log" 2>&1 &
echo $! > qwen.pid
cd ../..

# Start API Gateway (port 8080)
echo "Starting API Gateway on port 8080..."
source gateway_env/bin/activate
nohup python api_gateway.py > "$SCRIPT_DIR/gateway.log" 2>&1 &
echo $! > gateway.pid

# Wait for services to start
echo "Waiting for services to start..."
sleep 5

# Check service health
echo "Checking service health..."

if ! check_service_health "http://localhost:8001/health" "Surya OCR"; then
    echo "Failed to start Surya OCR service"
    stop_services
    exit 1
fi

if ! check_service_health "http://localhost:8002/health" "Qwen VL"; then
    echo "Failed to start Qwen VL service"
    stop_services
    exit 1
fi

if ! check_service_health "http://localhost:8080/health" "API Gateway"; then
    echo "Failed to start API Gateway"
    stop_services
    exit 1
fi

echo ""
echo "🚀 All OCR Engine services are running successfully!"
echo ""
echo "Service Status:"
echo "- Surya OCR:   http://localhost:8001/health"
echo "- Qwen VL:     http://localhost:8002/health"
echo "- API Gateway: http://localhost:8080/health"
echo ""
echo "API Endpoints:"
echo "- Basic OCR:           POST http://localhost:8080/ocr"
echo "- Qwen VL Processing:  POST http://localhost:8080/ocr/qwen-vl/process"
echo "- Batch Processing:    POST http://localhost:8080/ocr/batch"
echo ""
echo "Test with:"
echo "curl -X POST -F 'file=@test_bills/DEWA.png' http://localhost:8080/ocr/qwen-vl/process"
echo ""
echo "Logs:"
echo "- Surya:   tail -f services/surya/surya.log"
echo "- Qwen:    tail -f services/qwen/qwen.log"
echo "- Gateway: tail -f gateway.log"
echo ""
echo "To stop all services: pkill -f 'surya_service.py|qwen_service.py|api_gateway.py'"