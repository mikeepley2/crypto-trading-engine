FROM python:3.11-slim

WORKDIR /app

# Install required packages
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create logs directory
RUN mkdir -p /app/logs

# Health check
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8026/health || exit 1

# Run as non-root user
RUN useradd -m -u 1000 optimizer && \
    chown -R optimizer:optimizer /app
USER optimizer

EXPOSE 8026

CMD ["python", "portfolio_optimization_service.py"]
