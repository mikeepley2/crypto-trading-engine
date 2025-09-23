# Enhanced Signal Bridge with Real-Time Redis Streaming
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY *.py ./

# Create logs directory
RUN mkdir -p /app/logs

# Expose port
EXPOSE 8022

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8022/health || exit 1

# Run the enhanced signal bridge
CMD ["python", "enhanced_signal_bridge.py"]