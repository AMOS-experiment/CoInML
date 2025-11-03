# Dockerfile.app - For SCULPT Dash application
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir prefect>=2.14.0

# Copy application code - FIXED PATH
COPY sculpt /app/sculpt

# Create necessary directories
RUN mkdir -p /app/data /app/.cache

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Expose Dash port
EXPOSE 9000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:9000')" || exit 1

CMD ["python", "-m", "sculpt.app"]