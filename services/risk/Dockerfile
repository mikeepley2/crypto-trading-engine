FROM python:3.11-slim
WORKDIR /app

# Install dependencies
COPY risk/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Create the backend directory structure for proper imports
RUN mkdir -p /app/backend/services/trading

# Copy the shared modules with proper structure
COPY shared ./backend/services/trading/shared

# Copy the main service file
COPY risk/risk_service.py .

# Set Python path to include the app directory
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1

EXPOSE 8025
CMD ["uvicorn", "risk_service:app", "--host", "0.0.0.0", "--port", "8025"]
