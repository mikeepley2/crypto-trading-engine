@echo off
echo Starting Automated Live Trading System...

cd /d "e:\git\aitest\backend\services\trading"

echo Stopping existing containers...
docker-compose down

echo Building updated containers...
docker-compose build

echo Starting trade recommendations service...
docker-compose up -d trade-recommendations

echo Waiting for recommendations service to start...
timeout /t 10

echo Generating fresh trading signals...
python generate_fresh_signals.py

echo Starting automated live trader...
docker-compose up -d automated-live-trader

echo Checking container status...
docker-compose ps

echo Checking for fresh recommendations...
curl "http://localhost:8022/recommendations?is_mock=false&status=pending&limit=3"

echo.
echo ✅ Live trading system is now running!
echo Monitor logs with: docker-compose logs -f automated-live-trader
pause
