#!/bin/bash
set -e

echo "Starting OCR Engine Microservices..."

# Export environment variables
export SURYA_SERVICE_URL=${SURYA_SERVICE_URL:-http://localhost:8001}
export QWEN_SERVICE_URL=${QWEN_SERVICE_URL:-http://localhost:8002}
export PORT=${PORT:-8080}

# Function to wait for a service to be ready
wait_for_service() {
    local url=$1
    local service_name=$2
    local max_attempts=30
    local attempt=0
    
    echo "Waiting for $service_name to be ready..."
    while [ $attempt -lt $max_attempts ]; do
        if curl -s -f "$url/health" > /dev/null 2>&1; then
            echo "✓ $service_name is ready"
            return 0
        fi
        attempt=$((attempt + 1))
        echo "Waiting for $service_name... (attempt $attempt/$max_attempts)"
        sleep 2
    done
    
    echo "✗ $service_name failed to start after $max_attempts attempts"
    return 1
}

# Start Surya OCR service
echo "Starting Surya OCR service on port 8001..."
cd /app/services/surya
python surya_service.py &
SURYA_PID=$!

# Start Qwen VL service
echo "Starting Qwen VL service on port 8002..."
cd /app/services/qwen
python qwen_service.py &
QWEN_PID=$!

# Wait for services to be ready
if ! wait_for_service "$SURYA_SERVICE_URL" "Surya OCR"; then
    echo "ERROR: Surya OCR service failed to start"
    kill $QWEN_PID 2>/dev/null || true
    exit 1
fi

if ! wait_for_service "$QWEN_SERVICE_URL" "Qwen VL"; then
    echo "ERROR: Qwen VL service failed to start"
    kill $SURYA_PID 2>/dev/null || true
    exit 1
fi

# Function to handle shutdown
shutdown() {
    echo "Shutting down services..."
    kill $SURYA_PID $QWEN_PID 2>/dev/null || true
    wait $SURYA_PID $QWEN_PID 2>/dev/null || true
    exit 0
}

# Set up signal handlers
trap shutdown SIGTERM SIGINT

# Start API Gateway
echo "Starting API Gateway on port $PORT..."
cd /app
python api_gateway.py &
GATEWAY_PID=$!

# Wait for all background processes
wait $GATEWAY_PID