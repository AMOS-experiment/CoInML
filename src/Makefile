# Makefile for SCULPT Docker deployment

.PHONY: help build up down restart logs shell clean status test

# Default target
help:
	@echo "SCULPT Docker Management"
	@echo "========================"
	@echo ""
	@echo "Available commands:"
	@echo "  make build    - Build Docker images"
	@echo "  make up       - Start all services"
	@echo "  make down     - Stop all services"
	@echo "  make restart  - Restart all services"
	@echo "  make logs     - View logs (all services)"
	@echo "  make logs-app - View SCULPT app logs"
	@echo "  make logs-prefect - View Prefect server logs"
	@echo "  make shell    - Open shell in app container"
	@echo "  make clean    - Remove containers and volumes"
	@echo "  make status   - Show service status"
	@echo "  make test     - Run tests in container"
	@echo ""

# Build Docker images
build:
	@echo "Building Docker images..."
	docker-compose build

# Start services
up:
	@echo "Starting SCULPT services..."
	docker-compose up -d
	@echo ""
	@echo "Services started!"
	@echo "  SCULPT: http://localhost:9000"
	@echo "  Prefect: http://localhost:4200"
	@echo "  Jupyter: http://localhost:8888"

# Stop services
down:
	@echo "Stopping services..."
	docker-compose down

# Restart services
restart:
	@echo "Restarting services..."
	docker-compose restart

# View logs
logs:
	docker-compose logs -f

logs-app:
	docker-compose logs -f sculpt-app

logs-prefect:
	docker-compose logs -f prefect-server

logs-worker:
	docker-compose logs -f prefect-worker

# Open shell in container
shell:
	docker exec -it sculpt-app-1 /bin/bash

shell-worker:
	docker exec -it prefect-worker-1 /bin/bash

# Clean up
clean:
	@echo "Cleaning up containers and volumes..."
	docker-compose down -v
	@echo "Cleaned!"

# Show status
status:
	@echo "Service Status:"
	@docker-compose ps

# Run tests
test:
	@echo "Running tests in container..."
	docker exec sculpt-app-1 python -m pytest tests/

# Development mode - mount local code
dev:
	@echo "Starting in development mode with live code reload..."
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

# Production build
prod:
	@echo "Building for production..."
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml build

# Database backup
backup-db:
	@echo "Backing up Prefect database..."
	docker exec postgres-1 pg_dump -U prefect prefect > backup_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "Backup complete!"

# Restore database
restore-db:
	@echo "Restoring Prefect database from backup..."
	@read -p "Enter backup filename: " file; \
	docker exec -i postgres-1 psql -U prefect prefect < $$file
	@echo "Restore complete!"