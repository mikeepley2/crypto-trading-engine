@echo off
echo === FINAL TRADING SYSTEM FIX ===

echo Step 1: Stopping all trading containers to free database connections...
cd /d "e:\git\aitest\backend\services\trading"
docker-compose down

cd /d "e:\git\aitest\backend\services\trading\trade-execution-engine"
docker-compose down

echo Step 2: Killing any lingering Python processes...
taskkill /F /IM python.exe 2>nul || echo No Python processes found

echo Step 3: Attempting to restart MySQL service (requires admin)...
net stop mysql80 2>nul
timeout /t 5
net start mysql80 2>nul

echo Step 4: Waiting for MySQL to be ready...
timeout /t 10

echo Step 5: Starting execution engine in LIVE mode...
cd /d "e:\git\aitest\backend\services\trading\trade-execution-engine"
set EXECUTION_MODE=live
set TRADE_EXECUTION_ENABLED=true
docker-compose up -d

echo Step 6: Waiting for execution engine...
timeout /t 15

echo Step 7: Starting trading services...
cd /d "e:\git\aitest\backend\services\trading"
docker-compose up -d

echo Step 8: Waiting for services to start...
timeout /t 10

echo Step 9: Testing connectivity...
curl -s http://localhost:8024/health
echo.
curl -s http://localhost:8022/health

echo Step 10: Generating fresh trading signals...
python generate_fresh_signals.py

echo.
echo === SYSTEM FIX COMPLETE ===
echo Monitor logs with: docker-compose logs -f automated-live-trader
echo Check status: python final_status_check.py
pause
