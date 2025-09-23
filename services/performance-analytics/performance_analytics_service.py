#!/usr/bin/env python3
"""
Performance Analytics Dashboard Service
Real-time analytics for trading performance including Sharpe ratio, win/loss tracking, drawdown analysis
Provides web interface and API endpoints for performance metrics
"""

import os
import sys
import asyncio
import logging
import mysql.connector
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
import json
import math
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Performance Analytics Dashboard", version="1.0.0")

# Set up templates directory
import tempfile
import pathlib
templates_dir = pathlib.Path(__file__).parent / "templates"
templates_dir.mkdir(exist_ok=True)
templates = Jinja2Templates(directory=str(templates_dir))

class PerformanceMetrics(BaseModel):
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    current_streak: Dict
    largest_win: float
    largest_loss: float

class PerformanceAnalytics:
    def __init__(self):
        self.db_config = {
            'host': os.environ.get('DATABASE_HOST', 'host.docker.internal'),
            'user': os.environ.get('DATABASE_USER', 'news_collector'),
            'password': os.environ.get('DATABASE_PASSWORD', '99Rules!'),
            'database': os.environ.get('DATABASE_NAME', 'crypto_transactions'),
            'port': int(os.environ.get('DATABASE_PORT', 3306))
        }
        
        # Performance tracking
        self.portfolio_history = []
        self.trade_history = []
        self.daily_returns = []
        
        # Risk-free rate for Sharpe ratio (3% annual)
        self.risk_free_rate = 0.03
        
        logger.info("📊 Performance Analytics initialized")
    
    async def get_portfolio_history(self, days: int = 30) -> List[Dict]:
        """Get portfolio value history"""
        try:
            conn = mysql.connector.connect(**self.db_config)
            
            # Get transaction history for portfolio value calculation
            query = """
            SELECT 
                DATE(created_at) as date,
                symbol,
                type,
                amount_usd,
                quantity,
                price,
                created_at
            FROM transactions 
            WHERE created_at >= DATE_SUB(NOW(), INTERVAL %s DAY)
            ORDER BY created_at ASC
            """
            
            df = pd.read_sql(query, conn, params=[days])
            conn.close()
            
            if df.empty:
                return []
            
            # Calculate daily portfolio values
            portfolio_history = []
            running_cash = 0
            running_positions = defaultdict(float)
            
            # Group by date and calculate daily portfolio value
            for date, day_data in df.groupby('date'):
                for _, trade in day_data.iterrows():
                    if trade['type'] == 'buy':
                        running_cash -= trade['amount_usd']
                        running_positions[trade['symbol']] += trade['quantity']
                    elif trade['type'] == 'sell':
                        running_cash += trade['amount_usd']
                        running_positions[trade['symbol']] -= trade['quantity']
                
                # Calculate portfolio value (simplified - using last known prices)
                portfolio_value = running_cash
                for symbol, quantity in running_positions.items():
                    if quantity > 0:
                        # Estimate value using average trade price for the day
                        symbol_trades = day_data[day_data['symbol'] == symbol]
                        if not symbol_trades.empty:
                            avg_price = symbol_trades['price'].mean()
                            portfolio_value += quantity * avg_price
                
                portfolio_history.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'portfolio_value': portfolio_value,
                    'cash': running_cash,
                    'positions_count': len([k for k, v in running_positions.items() if v > 0])
                })
            
            return portfolio_history
            
        except Exception as e:
            logger.error(f"Error getting portfolio history: {e}")
            return []
    
    async def get_trade_analytics(self, days: int = 30) -> Dict:
        """Get trade analytics and performance metrics"""
        try:
            conn = mysql.connector.connect(**self.db_config)
            
            # Get completed trade pairs (buy + sell)
            query = """
            WITH trade_pairs AS (
                SELECT 
                    b.symbol,
                    b.created_at as buy_date,
                    b.price as buy_price,
                    b.quantity as buy_quantity,
                    b.amount_usd as buy_amount,
                    s.created_at as sell_date,
                    s.price as sell_price,
                    s.quantity as sell_quantity,
                    s.amount_usd as sell_amount,
                    (s.amount_usd - b.amount_usd) as pnl,
                    ((s.price - b.price) / b.price * 100) as return_percent
                FROM transactions b
                JOIN transactions s ON b.symbol = s.symbol 
                WHERE b.type = 'buy' 
                AND s.type = 'sell'
                AND s.created_at > b.created_at
                AND b.created_at >= DATE_SUB(NOW(), INTERVAL %s DAY)
            )
            SELECT * FROM trade_pairs ORDER BY sell_date ASC
            """
            
            df = pd.read_sql(query, conn, params=[days])
            conn.close()
            
            if df.empty:
                return {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'win_rate': 0.0,
                    'total_pnl': 0.0,
                    'avg_win': 0.0,
                    'avg_loss': 0.0,
                    'profit_factor': 0.0,
                    'largest_win': 0.0,
                    'largest_loss': 0.0,
                    'avg_trade_duration_hours': 0.0
                }
            
            # Calculate metrics
            total_trades = len(df)
            winning_trades = len(df[df['pnl'] > 0])
            losing_trades = len(df[df['pnl'] < 0])
            
            total_pnl = df['pnl'].sum()
            total_wins = df[df['pnl'] > 0]['pnl'].sum()
            total_losses = abs(df[df['pnl'] < 0]['pnl'].sum())
            
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            avg_win = total_wins / winning_trades if winning_trades > 0 else 0
            avg_loss = total_losses / losing_trades if losing_trades > 0 else 0
            profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
            
            largest_win = df['pnl'].max() if not df.empty else 0
            largest_loss = df['pnl'].min() if not df.empty else 0
            
            # Calculate average trade duration
            df['duration'] = pd.to_datetime(df['sell_date']) - pd.to_datetime(df['buy_date'])
            avg_duration_hours = df['duration'].dt.total_seconds().mean() / 3600 if not df.empty else 0
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'largest_win': largest_win,
                'largest_loss': largest_loss,
                'avg_trade_duration_hours': avg_duration_hours
            }
            
        except Exception as e:
            logger.error(f"Error getting trade analytics: {e}")
            return {}
    
    async def calculate_portfolio_metrics(self, days: int = 30) -> Dict:
        """Calculate comprehensive portfolio performance metrics"""
        try:
            portfolio_history = await self.get_portfolio_history(days)
            
            if len(portfolio_history) < 2:
                return {
                    'total_return': 0.0,
                    'annualized_return': 0.0,
                    'volatility': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'current_drawdown': 0.0,
                    'calmar_ratio': 0.0
                }
            
            # Calculate daily returns
            values = [entry['portfolio_value'] for entry in portfolio_history]
            returns = [0]  # First day has 0 return
            
            for i in range(1, len(values)):
                if values[i-1] > 0:
                    daily_return = (values[i] - values[i-1]) / values[i-1]
                    returns.append(daily_return)
                else:
                    returns.append(0)
            
            returns = np.array(returns)
            
            # Calculate metrics with reset baseline
            baseline_value = 2571.86  # Current portfolio value becomes new baseline
            total_return = (values[-1] - baseline_value) / baseline_value * 100 if baseline_value > 0 else 0
            
            # Annualized return using reset baseline
            if len(values) > 1:
                periods_per_year = 365 / len(values)
                annualized_return = ((values[-1] / baseline_value) ** periods_per_year - 1) * 100 if baseline_value > 0 else 0
            else:
                annualized_return = 0
            
            # Volatility (annualized)
            if len(returns) > 1:
                volatility = np.std(returns) * np.sqrt(365) * 100
            else:
                volatility = 0
            
            # Sharpe ratio
            if volatility > 0:
                excess_return = annualized_return / 100 - self.risk_free_rate
                sharpe_ratio = excess_return / (volatility / 100)
            else:
                sharpe_ratio = 0
            
            # Maximum drawdown
            peak = values[0]
            max_drawdown = 0
            current_drawdown = 0
            
            for value in values:
                if value > peak:
                    peak = value
                
                drawdown = (peak - value) / peak * 100 if peak > 0 else 0
                max_drawdown = max(max_drawdown, drawdown)
            
            # Current drawdown
            current_peak = max(values)
            current_drawdown = (current_peak - values[-1]) / current_peak * 100 if current_peak > 0 else 0
            
            # Calmar ratio
            calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
            
            return {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'current_drawdown': current_drawdown,
                'calmar_ratio': calmar_ratio
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {e}")
            return {}
    
    async def get_current_portfolio_status(self) -> Dict:
        """Get current portfolio status"""
        try:
            import requests
            
            # Try to get portfolio from trading engine
            response = requests.get("http://localhost:8024/portfolio", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'total_value': data.get('total_value_usd', 0),
                    'cash_balance': data.get('usd_balance', 0),
                    'positions_count': len(data.get('positions', {})),
                    'largest_position': max(
                        [pos.get('value_usd', 0) for pos in data.get('positions', {}).values()],
                        default=0
                    )
                }
            else:
                return {
                    'total_value': 0,
                    'cash_balance': 0,
                    'positions_count': 0,
                    'largest_position': 0
                }
                
        except Exception as e:
            logger.error(f"Error getting current portfolio status: {e}")
            return {
                'total_value': 0,
                'cash_balance': 0,
                'positions_count': 0,
                'largest_position': 0
            }
    
    async def get_trading_performance_summary(self, days: int = 30) -> Dict:
        """Get comprehensive trading performance summary"""
        try:
            # Get all metrics
            portfolio_metrics = await self.calculate_portfolio_metrics(days)
            trade_analytics = await self.get_trade_analytics(days)
            current_status = await self.get_current_portfolio_status()
            portfolio_history = await self.get_portfolio_history(days)
            
            # Calculate current streak
            if trade_analytics and trade_analytics.get('total_trades', 0) > 0:
                conn = mysql.connector.connect(**self.db_config)
                
                # Get recent trades to determine current streak
                recent_query = """
                SELECT 
                    (s.amount_usd - b.amount_usd) as pnl
                FROM transactions b
                JOIN transactions s ON b.symbol = s.symbol 
                WHERE b.type = 'buy' 
                AND s.type = 'sell'
                AND s.created_at > b.created_at
                ORDER BY s.created_at DESC
                LIMIT 10
                """
                
                streak_df = pd.read_sql(recent_query, conn)
                conn.close()
                
                if not streak_df.empty:
                    current_streak = {'type': 'none', 'count': 0}
                    streak_count = 0
                    
                    for _, trade in streak_df.iterrows():
                        if streak_count == 0:
                            # First trade determines streak type
                            current_streak['type'] = 'winning' if trade['pnl'] > 0 else 'losing'
                            streak_count = 1
                        elif (current_streak['type'] == 'winning' and trade['pnl'] > 0) or \
                             (current_streak['type'] == 'losing' and trade['pnl'] <= 0):
                            streak_count += 1
                        else:
                            break
                    
                    current_streak['count'] = streak_count
                else:
                    current_streak = {'type': 'none', 'count': 0}
            else:
                current_streak = {'type': 'none', 'count': 0}
            
            # Combine all metrics
            performance_summary = {
                'current_status': current_status,
                'portfolio_metrics': portfolio_metrics,
                'trade_analytics': trade_analytics,
                'current_streak': current_streak,
                'portfolio_history': portfolio_history,
                'analysis_period_days': days,
                'last_updated': datetime.now().isoformat()
            }
            
            return performance_summary
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {}

# Global analytics instance
analytics = PerformanceAnalytics()

# Create dashboard HTML template
dashboard_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trading Performance Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }
        .dashboard {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .metric {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 0;
            border-bottom: 1px solid #eee;
        }
        .metric:last-child {
            border-bottom: none;
        }
        .metric-value {
            font-weight: bold;
            color: #333;
        }
        .positive {
            color: #27ae60;
        }
        .negative {
            color: #e74c3c;
        }
        .neutral {
            color: #95a5a6;
        }
        .chart-container {
            position: relative;
            height: 300px;
            margin-top: 20px;
        }
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 10px;
        }
        .status-active { background-color: #27ae60; }
        .status-warning { background-color: #f39c12; }
        .status-error { background-color: #e74c3c; }
        .refresh-btn {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 50px;
            padding: 15px 25px;
            cursor: pointer;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        .refresh-btn:hover {
            background: #5a6fd8;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 Trading Performance Dashboard</h1>
        <p>Real-time analytics for automated cryptocurrency trading system</p>
    </div>

    <div class="dashboard" id="dashboard">
        <div class="card">
            <h3>📊 Current Portfolio Status</h3>
            <div id="currentStatus">Loading...</div>
        </div>

        <div class="card">
            <h3>📈 Performance Metrics</h3>
            <div id="performanceMetrics">Loading...</div>
        </div>

        <div class="card">
            <h3>💹 Trading Analytics</h3>
            <div id="tradingAnalytics">Loading...</div>
        </div>

        <div class="card">
            <h3>🎯 Risk Metrics</h3>
            <div id="riskMetrics">Loading...</div>
        </div>
    </div>

    <div class="card">
        <h3>📈 Portfolio Value History</h3>
        <div class="chart-container">
            <canvas id="portfolioChart"></canvas>
        </div>
    </div>

    <button class="refresh-btn" onclick="refreshDashboard()">🔄 Refresh</button>

    <script>
        let portfolioChart;

        function formatCurrency(value) {
            return new Intl.NumberFormat('en-US', {
                style: 'currency',
                currency: 'USD'
            }).format(value);
        }

        function formatPercentage(value) {
            return `${value.toFixed(2)}%`;
        }

        function getValueClass(value) {
            if (value > 0) return 'positive';
            if (value < 0) return 'negative';
            return 'neutral';
        }

        function createMetric(label, value, isPercentage = false, isCurrency = false) {
            const formattedValue = isCurrency ? formatCurrency(value) : 
                                 isPercentage ? formatPercentage(value) : value;
            const valueClass = typeof value === 'number' ? getValueClass(value) : 'neutral';
            
            return `
                <div class="metric">
                    <span>${label}</span>
                    <span class="metric-value ${valueClass}">${formattedValue}</span>
                </div>
            `;
        }

        function updateCurrentStatus(data) {
            const statusHtml = `
                ${createMetric('Total Portfolio Value', data.total_value, false, true)}
                ${createMetric('Cash Balance', data.cash_balance, false, true)}
                ${createMetric('Active Positions', data.positions_count)}
                ${createMetric('Largest Position', data.largest_position, false, true)}
            `;
            document.getElementById('currentStatus').innerHTML = statusHtml;
        }

        function updatePerformanceMetrics(data) {
            const metricsHtml = `
                ${createMetric('Total Return', data.total_return, true)}
                ${createMetric('Annualized Return', data.annualized_return, true)}
                ${createMetric('Sharpe Ratio', data.sharpe_ratio.toFixed(2))}
                ${createMetric('Volatility', data.volatility, true)}
            `;
            document.getElementById('performanceMetrics').innerHTML = metricsHtml;
        }

        function updateTradingAnalytics(data) {
            const analyticsHtml = `
                ${createMetric('Total Trades', data.total_trades)}
                ${createMetric('Win Rate', data.win_rate, true)}
                ${createMetric('Profit Factor', data.profit_factor.toFixed(2))}
                ${createMetric('Average Win', data.avg_win, false, true)}
                ${createMetric('Average Loss', data.avg_loss, false, true)}
            `;
            document.getElementById('tradingAnalytics').innerHTML = analyticsHtml;
        }

        function updateRiskMetrics(data) {
            const riskHtml = `
                ${createMetric('Max Drawdown', data.max_drawdown, true)}
                ${createMetric('Current Drawdown', data.current_drawdown, true)}
                ${createMetric('Calmar Ratio', data.calmar_ratio.toFixed(2))}
            `;
            document.getElementById('riskMetrics').innerHTML = riskHtml;
        }

        function updatePortfolioChart(history) {
            const ctx = document.getElementById('portfolioChart').getContext('2d');
            
            if (portfolioChart) {
                portfolioChart.destroy();
            }

            const labels = history.map(entry => entry.date);
            const values = history.map(entry => entry.portfolio_value);

            portfolioChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Portfolio Value',
                        data: values,
                        borderColor: '#667eea',
                        backgroundColor: 'rgba(102, 126, 234, 0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: false,
                            ticks: {
                                callback: function(value) {
                                    return formatCurrency(value);
                                }
                            }
                        }
                    },
                    plugins: {
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    return `Portfolio Value: ${formatCurrency(context.raw)}`;
                                }
                            }
                        }
                    }
                }
            });
        }

        async function refreshDashboard() {
            try {
                const response = await fetch('/performance_summary');
                const data = await response.json();

                updateCurrentStatus(data.current_status);
                updatePerformanceMetrics(data.portfolio_metrics);
                updateTradingAnalytics(data.trade_analytics);
                updateRiskMetrics(data.portfolio_metrics);
                updatePortfolioChart(data.portfolio_history);

                console.log('Dashboard updated:', new Date().toLocaleTimeString());
            } catch (error) {
                console.error('Error refreshing dashboard:', error);
            }
        }

        // Initial load
        refreshDashboard();

        // Auto-refresh every 60 seconds
        setInterval(refreshDashboard, 60000);
    </script>
</body>
</html>
"""

# Save the HTML template
templates_dir = pathlib.Path(__file__).parent / "templates"
try:
    templates_dir.mkdir(exist_ok=True)
    with open(templates_dir / "dashboard.html", "w") as f:
        f.write(dashboard_html)
    print("📊 Dashboard template created successfully")
except PermissionError as e:
    print(f"⚠️ Warning: Could not create dashboard template file: {e}")
    print("📊 Will serve dashboard content dynamically")
except Exception as e:
    print(f"❌ Error creating dashboard template: {e}")
    print("📊 Will serve dashboard content dynamically")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "performance-analytics", "timestamp": datetime.now().isoformat()}

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Serve the dashboard HTML"""
    try:
        return templates.TemplateResponse("dashboard.html", {"request": request})
    except Exception as e:
        # If template file doesn't exist, serve the dashboard content directly
        logger.warning(f"Template file not found, serving dashboard directly: {e}")
        return HTMLResponse(content=dashboard_html)

@app.get("/performance_summary")
async def get_performance_summary(days: int = 30):
    """Get comprehensive performance summary"""
    try:
        summary = await analytics.get_trading_performance_summary(days)
        return summary
        
    except Exception as e:
        logger.error(f"Error getting performance summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/portfolio_metrics")
async def get_portfolio_metrics(days: int = 30):
    """Get portfolio performance metrics"""
    try:
        metrics = await analytics.calculate_portfolio_metrics(days)
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting portfolio metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/trade_analytics")
async def get_trade_analytics(days: int = 30):
    """Get trading analytics"""
    try:
        analytics_data = await analytics.get_trade_analytics(days)
        return analytics_data
        
    except Exception as e:
        logger.error(f"Error getting trade analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/portfolio_history")
async def get_portfolio_history(days: int = 30):
    """Get portfolio value history"""
    try:
        history = await analytics.get_portfolio_history(days)
        return history
        
    except Exception as e:
        logger.error(f"Error getting portfolio history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/current_status")
async def get_current_status():
    """Get current portfolio status"""
    try:
        status = await analytics.get_current_portfolio_status()
        return status
        
    except Exception as e:
        logger.error(f"Error getting current status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/risk_analysis")
async def get_risk_analysis(days: int = 30):
    """Get detailed risk analysis"""
    try:
        portfolio_metrics = await analytics.calculate_portfolio_metrics(days)
        trade_analytics = await analytics.get_trade_analytics(days)
        
        # Calculate additional risk metrics
        risk_analysis = {
            'sharpe_ratio': portfolio_metrics.get('sharpe_ratio', 0),
            'max_drawdown': portfolio_metrics.get('max_drawdown', 0),
            'current_drawdown': portfolio_metrics.get('current_drawdown', 0),
            'volatility': portfolio_metrics.get('volatility', 0),
            'win_rate': trade_analytics.get('win_rate', 0),
            'profit_factor': trade_analytics.get('profit_factor', 0),
            'avg_trade_duration_hours': trade_analytics.get('avg_trade_duration_hours', 0),
            'risk_score': 'LOW',  # Simplified risk scoring
            'recommendations': []
        }
        
        # Add risk recommendations
        if risk_analysis['max_drawdown'] > 20:
            risk_analysis['risk_score'] = 'HIGH'
            risk_analysis['recommendations'].append('Consider reducing position sizes - high drawdown detected')
        elif risk_analysis['win_rate'] < 40:
            risk_analysis['risk_score'] = 'MEDIUM'
            risk_analysis['recommendations'].append('Review trading strategy - low win rate')
        elif risk_analysis['volatility'] > 50:
            risk_analysis['risk_score'] = 'MEDIUM'
            risk_analysis['recommendations'].append('High volatility detected - consider risk management')
        
        if not risk_analysis['recommendations']:
            risk_analysis['recommendations'].append('Risk levels appear manageable')
        
        return risk_analysis
        
    except Exception as e:
        logger.error(f"Error getting risk analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "performance_analytics_service:app",
        host="0.0.0.0",
        port=8031,
        log_level="info"
    )
