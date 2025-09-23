FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy service files
COPY signal_gen_risk_mgmt.py .

# Set environment variables
ENV PYTHONPATH=/app
ENV RISK_MGMT_PORT=8055

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8055/health || exit 1

# Expose port
EXPOSE 8055

# Run the service
CMD ["python", "signal_gen_risk_mgmt.py"]