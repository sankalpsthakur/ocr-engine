# OCR Engine Docker Deployment Guide

This guide provides comprehensive instructions for building, running, and deploying the OCR Engine microservices using Docker.

## Table of Contents
- [Architecture Overview](#architecture-overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Building Docker Images](#building-docker-images)
- [Running Locally](#running-locally)
- [Production Deployment](#production-deployment)
- [Testing with cURL](#testing-with-curl)
- [Security Considerations](#security-considerations)
- [Troubleshooting](#troubleshooting)

## Architecture Overview

The OCR Engine consists of three microservices:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Surya OCR     │     │    Qwen VL      │     │  API Gateway    │
│   Container     │     │   Container     │     │   Container     │
│   Port: 8001    │◄────│   Port: 8002    │◄────│   Port: 8080    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
     (Internal)              (Internal)            (Public Facing)
```

- **Surya OCR Service**: High-accuracy text extraction (internal only)
- **Qwen VL Service**: Vision-language processing for structured data (internal only)
- **API Gateway**: Public-facing API that orchestrates the other services

## Prerequisites

- Docker Engine 20.10+ installed
- Docker Compose v2.0+ installed
- 8GB+ RAM available
- 20GB+ disk space for Docker images and model caches
- (Optional) NVIDIA GPU with Docker GPU support for faster processing

### Verify Prerequisites
```bash
# Check Docker version
docker --version

# Check Docker Compose version
docker compose version

# Check available disk space
df -h

# Check available memory
free -h
```

## Quick Start

```bash
# Clone the repository
git clone https://github.com/sankalpsthakur/ocr-engine.git
cd ocr-engine

# Build all images
docker compose build

# Start all services
docker compose up -d

# Check service health
curl http://localhost:8080/health

# Test OCR processing
curl -X POST -F 'file=@test_bills/DEWA.png' http://localhost:8080/ocr
```

## Building Docker Images

### Option 1: Build All Images with Docker Compose
```bash
# Build all services defined in docker-compose.yml
docker compose build

# Build with no cache (fresh build)
docker compose build --no-cache

# Build specific service only
docker compose build surya-ocr
docker compose build qwen-vl
docker compose build api-gateway
```

### Option 2: Build Individual Images Manually
```bash
# Build Surya OCR image
docker build -t ocr-engine/surya:latest -f services/surya/Dockerfile services/surya/

# Build Qwen VL image
docker build -t ocr-engine/qwen:latest -f services/qwen/Dockerfile services/qwen/

# Build API Gateway image
docker build -t ocr-engine/gateway:latest .

# List built images
docker images | grep ocr-engine
```

## Running Locally

### Using Docker Compose (Recommended)
```bash
# Start all services in background
docker compose up -d

# View logs for all services
docker compose logs -f

# View logs for specific service
docker compose logs -f surya-ocr
docker compose logs -f qwen-vl
docker compose logs -f api-gateway

# Stop all services
docker compose down

# Stop and remove volumes (cleans model cache)
docker compose down -v
```

### Running Individual Containers (Advanced)
```bash
# Create network for containers
docker network create ocr-network

# Run Surya OCR container
docker run -d \
  --name surya-ocr \
  --network ocr-network \
  -v $(pwd)/model_cache:/app/.cache \
  ocr-engine/surya:latest

# Run Qwen VL container
docker run -d \
  --name qwen-vl \
  --network ocr-network \
  -v $(pwd)/model_cache:/app/.cache \
  ocr-engine/qwen:latest

# Run API Gateway container
docker run -d \
  --name api-gateway \
  --network ocr-network \
  -p 8080:8080 \
  -e SURYA_SERVICE_URL=http://surya-ocr:8001 \
  -e QWEN_SERVICE_URL=http://qwen-vl:8002 \
  ocr-engine/gateway:latest

# Check container status
docker ps

# Clean up
docker stop surya-ocr qwen-vl api-gateway
docker rm surya-ocr qwen-vl api-gateway
docker network rm ocr-network
```

## Production Deployment

### 1. Server Preparation
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker (Ubuntu/Debian)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Configure firewall (only expose gateway port)
sudo ufw allow 8080/tcp
sudo ufw enable
```

### 2. Deploy with Docker Compose
```bash
# Clone repository on server
git clone https://github.com/sankalpsthakur/ocr-engine.git
cd ocr-engine

# Create production docker-compose override (optional)
cat > docker-compose.prod.yml << EOF
version: '3.8'
services:
  api-gateway:
    restart: always
    environment:
      - LOG_LEVEL=info
  surya-ocr:
    restart: always
  qwen-vl:
    restart: always
EOF

# Deploy with production settings
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Enable auto-start on reboot
sudo systemctl enable docker
```

### 3. Configure Reverse Proxy (Nginx)
```bash
# Install Nginx
sudo apt install nginx -y

# Create configuration
sudo tee /etc/nginx/sites-available/ocr-api << EOF
server {
    listen 80;
    server_name api.yourdomain.com;
    
    client_max_body_size 50M;
    
    location / {
        proxy_pass http://localhost:8080;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }
}
EOF

# Enable site
sudo ln -s /etc/nginx/sites-available/ocr-api /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx

# Add SSL with Let's Encrypt (optional)
sudo apt install certbot python3-certbot-nginx -y
sudo certbot --nginx -d api.yourdomain.com
```

## Testing with cURL

### Local Testing (Development)
```bash
# 1. Health Check
curl http://localhost:8080/health

# Expected response:
# {
#   "status": "healthy",
#   "services": {
#     "surya_ocr": "healthy",
#     "qwen_vl": "healthy"
#   }
# }

# 2. Basic OCR (Text Extraction Only)
curl -X POST -F 'file=@test_bills/DEWA.png' http://localhost:8080/ocr

# 3. Qwen VL Processing (Structured Extraction)
curl -X POST -F 'file=@test_bills/DEWA.png' http://localhost:8080/ocr/qwen-vl/process

# 4. Process with Specific Resource Type
curl -X POST -F 'file=@test_bills/DEWA.png' \
  "http://localhost:8080/ocr/qwen-vl/process?resource_type=water&enable_reasoning=true"

# 5. Batch Processing (Multiple Files)
curl -X POST \
  -F 'files=@test_bills/DEWA.png' \
  -F 'files=@test_bills/SEWA.png' \
  http://localhost:8080/ocr/batch

# 6. Get Schema Information
curl http://localhost:8080/ocr/qwen-vl/schema/DEWA
curl http://localhost:8080/ocr/qwen-vl/schema/SEWA

# 7. PDF Processing
curl -X POST -F 'file=@test_bills/Bill-4.pdf' http://localhost:8080/ocr

# Save response to file
curl -X POST -F 'file=@test_bills/DEWA.png' \
  http://localhost:8080/ocr/qwen-vl/process -o output.json
```

### Production Testing (Remote Server)
```bash
# Replace YOUR_SERVER_IP with actual server IP or domain

# 1. Health Check
curl http://YOUR_SERVER_IP:8080/health

# 2. Test with timeout settings
curl -X POST -F 'file=@test_bills/DEWA.png' \
  --max-time 300 \
  http://YOUR_SERVER_IP:8080/ocr/qwen-vl/process

# 3. Test through Nginx (if configured)
curl -X POST -F 'file=@test_bills/DEWA.png' \
  https://api.yourdomain.com/ocr/qwen-vl/process

# 4. Monitor response time
time curl -X POST -F 'file=@test_bills/DEWA.png' \
  http://YOUR_SERVER_IP:8080/ocr

# 5. Test with authentication (if configured)
curl -X POST -F 'file=@test_bills/DEWA.png' \
  -H "Authorization: Bearer YOUR_API_KEY" \
  http://YOUR_SERVER_IP:8080/ocr
```

## Security Considerations

### 1. Network Security
- Only the API Gateway port (8080) is exposed to the public
- Surya OCR and Qwen VL services are only accessible within Docker network
- Use firewall rules to restrict access

### 2. Container Security
```bash
# Run containers as non-root user (add to Dockerfile)
USER 1000:1000

# Set resource limits in docker-compose.yml
services:
  api-gateway:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
```

### 3. API Security (Optional)
- Implement rate limiting
- Add API key authentication
- Use HTTPS with SSL certificates
- Monitor and log access

### 4. Data Security
- Don't persist uploaded files longer than necessary
- Sanitize file uploads
- Implement file size limits (currently 50MB)

## Troubleshooting

### Common Issues

#### 1. Services Not Starting
```bash
# Check container status
docker compose ps

# View detailed logs
docker compose logs surya-ocr
docker compose logs qwen-vl
docker compose logs api-gateway

# Check resource usage
docker stats
```

#### 2. Model Download Issues
```bash
# Models are downloaded on first run, this can take time
# Monitor download progress
docker compose logs -f surya-ocr | grep -i "download"
docker compose logs -f qwen-vl | grep -i "download"

# If download fails, restart the service
docker compose restart surya-ocr
docker compose restart qwen-vl
```

#### 3. Out of Memory Errors
```bash
# Check memory usage
docker stats --no-stream

# Increase Docker memory limit
# Edit Docker Desktop settings or /etc/docker/daemon.json

# Use smaller batch sizes
# Reduce concurrent requests
```

#### 4. Slow Response Times
```bash
# Check if GPU is being used (if available)
docker exec surya-ocr nvidia-smi
docker exec qwen-vl nvidia-smi

# Monitor container resources
docker stats

# Check network latency between containers
docker exec api-gateway ping surya-ocr
```

#### 5. Port Conflicts
```bash
# Check if ports are in use
sudo lsof -i :8080
sudo lsof -i :8001
sudo lsof -i :8002

# Change ports in docker-compose.yml if needed
```

### Debugging Commands
```bash
# Enter container shell
docker exec -it api-gateway /bin/bash
docker exec -it surya-ocr /bin/bash
docker exec -it qwen-vl /bin/bash

# Check container file system
docker exec api-gateway ls -la /app
docker exec surya-ocr ls -la /app/.cache

# Test internal connectivity
docker exec api-gateway curl http://surya-ocr:8001/health
docker exec api-gateway curl http://qwen-vl:8002/health

# Export container logs
docker compose logs > ocr-engine-logs.txt

# Monitor real-time logs
docker compose logs -f --tail=100
```

### Performance Optimization

1. **Enable GPU Support** (if available):
```yaml
# Add to docker-compose.yml
services:
  surya-ocr:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

2. **Persistent Model Cache**:
```yaml
# Models are cached in volumes to avoid re-downloading
volumes:
  model_cache:
    driver: local
```

3. **Health Check Configuration**:
```yaml
# Adjust health check timing for slow systems
healthcheck:
  interval: 60s
  timeout: 30s
  retries: 5
  start_period: 300s
```

## Maintenance

### Regular Tasks
```bash
# Update images
docker compose pull
docker compose up -d

# Clean up old images
docker image prune -a

# Monitor disk usage
docker system df

# Backup model cache
tar -czf model_cache_backup.tar.gz model_cache/

# View resource usage over time
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"
```

### Monitoring
- Set up monitoring with Prometheus/Grafana
- Use Docker health checks
- Monitor API response times
- Track resource usage

## Support

For issues or questions:
- Check logs first: `docker compose logs`
- Review this troubleshooting guide
- Submit issues to: https://github.com/sankalpsthakur/ocr-engine/issues