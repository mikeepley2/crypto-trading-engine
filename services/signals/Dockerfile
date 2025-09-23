FROM python:3.11-slim
WORKDIR /app

# Install dependencies
COPY backend/services/trading/signals/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Create the backend directory structure for proper imports
RUN mkdir -p /app/backend/services/trading

# Copy the shared modules with proper structure
COPY backend/services/trading/shared ./backend/services/trading/shared

# Copy the main service file and dependencies
COPY backend/services/trading/signals/automated_signal_bridge_service.py .
COPY symbol_utils.py .
COPY coinbase_asset_filter.py .

# Set Python path to include the app directory
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1

EXPOSE 8028
CMD ["python", "automated_signal_bridge_service.py"]
