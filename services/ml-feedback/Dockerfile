# ML Performance Feedback Service
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
EXPOSE 8035

# Health check endpoint
HEALTHCHECK --interval=60s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8035/health || exit 1

# Run the ML feedback service
CMD ["python", "ml_performance_feedback_service.py"]