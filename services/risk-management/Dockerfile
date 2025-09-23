FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY advanced_risk_service.py .

# Create non-root user for security
RUN useradd -m -u 1000 riskuser
USER riskuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8027/health', timeout=5)"

EXPOSE 8027

CMD ["python", "advanced_risk_service.py"]
