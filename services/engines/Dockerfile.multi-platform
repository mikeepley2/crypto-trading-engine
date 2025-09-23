FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements-multi-platform.txt .
RUN pip install --no-cache-dir -r requirements-multi-platform.txt

# Create necessary directories
RUN mkdir -p /app/config /app/logs

# Copy multi-platform trading engine files
COPY multi_platform_engine.py .
COPY multi_platform_database.py .
COPY platforms/ ./platforms/

# Create a non-root user
RUN useradd -m -u 1000 trader && \
    chown -R trader:trader /app
USER trader

# Expose the trading engine port
EXPOSE 8024

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8024/health || exit 1

# Start the multi-platform trading engine
CMD ["python", "multi_platform_engine.py"]
