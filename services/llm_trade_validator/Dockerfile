# Dockerfile for LLM Trade Validation Service
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY simple_llm_validation_service.py .
COPY symbol_utils.py .

# Expose port
EXPOSE 8035

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8035/health || exit 1

# Run the service
CMD ["python", "simple_llm_validation_service.py"]