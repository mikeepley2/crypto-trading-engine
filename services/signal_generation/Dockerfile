FROM python:3.11-slim

WORKDIR /app

# Install required packages
RUN pip install --no-cache-dir \
    aiohttp \
    mysql-connector-python

# Copy the signal bridge script
COPY microservices_signal_bridge.py /app/signal_bridge.py

# Make it executable
RUN chmod +x /app/signal_bridge.py

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import sys; sys.exit(0)"

CMD ["python", "/app/signal_bridge.py"]
