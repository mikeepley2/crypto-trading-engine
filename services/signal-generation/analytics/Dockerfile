FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy service files
COPY signal_gen_analytics.py .

# Set environment variables
ENV PYTHONPATH=/app
ENV ANALYTICS_PORT=8056

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8056/health || exit 1

# Expose port
EXPOSE 8056

# Run the service
CMD ["python", "signal_gen_analytics.py"]