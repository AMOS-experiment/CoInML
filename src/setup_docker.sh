#!/bin/bash

# Setup script for SCULPT with Docker and Prefect

set -e  # Exit on error

echo "🚀 SCULPT Docker Setup Script"
echo "============================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    echo "Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    echo "Visit: https://docs.docker.com/compose/install/"
    exit 1
fi

print_status "Docker and Docker Compose are installed"

# Check if Docker daemon is running
if ! docker info &> /dev/null; then
    print_error "Docker daemon is not running. Please start Docker."
    exit 1
fi

print_status "Docker daemon is running"

# Create necessary directories
echo ""
echo "Creating project directories..."
mkdir -p data notebooks
print_status "Created data/ and notebooks/ directories"

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cat > .env << EOF
# Prefect Configuration
PREFECT_API_URL=http://prefect-server:4200/api
POSTGRES_USER=prefect
POSTGRES_PASSWORD=prefect_password
POSTGRES_DB=prefect

# SCULPT Configuration
SCULPT_DATA_DIR=/app/data
SCULPT_CACHE_DIR=/app/.cache
EOF
    print_status "Created .env file"
else
    print_warning ".env file already exists, skipping creation"
fi

# Build Docker images
echo ""
echo "Building Docker images..."
docker-compose build --no-cache

print_status "Docker images built successfully"

# Start services
echo ""
echo "Starting services..."
docker-compose up -d

print_status "Services starting..."

# Wait for services to be healthy
echo ""
echo "Waiting for services to be ready..."
sleep 10

# Check service health
echo ""
echo "Checking service status..."

if docker-compose ps | grep -q "Up"; then
    print_status "All services are running"
else
    print_error "Some services failed to start. Check logs with: docker-compose logs"
    exit 1
fi

# Display service URLs
echo ""
echo "========================================="
echo "🎉 SCULPT is ready!"
echo "========================================="
echo ""
echo "📊 SCULPT Dashboard: http://localhost:9000"
echo "🔄 Prefect UI: http://localhost:4200"
echo "📓 Jupyter Lab: http://localhost:8888"
echo ""
echo "========================================="
echo ""
echo "Useful commands:"
echo "  • View logs: docker-compose logs -f [service-name]"
echo "  • Stop all: docker-compose down"
echo "  • Restart: docker-compose restart"
echo "  • Shell into container: docker exec -it sculpt-app-1 /bin/bash"
echo ""

# Optional: Open browser
read -p "Would you like to open SCULPT in your browser? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if command -v open &> /dev/null; then
        open http://localhost:9000
        open http://localhost:4200
    elif command -v xdg-open &> /dev/null; then
        xdg-open http://localhost:9000
        xdg-open http://localhost:4200
    else
        print_warning "Could not detect browser command. Please open manually."
    fi
fi

print_status "Setup complete!"