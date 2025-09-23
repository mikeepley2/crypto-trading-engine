"""
Enhanced Portfolio Analytics Service
Real-time portfolio tracking, P&L analysis, risk metrics, and performance attribution
"""

import asyncio
import logging
from datetime import datetime, timedelta, date
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import numpy as np
import pandas as pd
from statistics import mean, stdev
import mysql.connector
import aiohttp
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

class PerformanceMetric(Enum):
    TOTAL_RETURN = "total_return"
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    VOLATILITY = "volatility"
    ALPHA = "alpha"
    BETA = "beta"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"

@dataclass
class PortfolioPosition:
    """Individual portfolio position"""
    symbol: str
    quantity: Decimal
    avg_cost: Decimal
    current_price: Decimal
    market_value: Decimal
    unrealized_pnl: Decimal
    unrealized_pnl_percent: float
    day_change: Decimal
    day_change_percent: float
    position_size_percent: float
    last_updated: datetime = field(default_factory=datetime.utcnow)

@dataclass
class PortfolioMetrics:
    """Comprehensive portfolio metrics"""
    total_value: Decimal
    total_cost: Decimal
    total_pnl: Decimal
    total_pnl_percent: float
    day_change: Decimal
    day_change_percent: float
    cash_balance: Decimal
    invested_amount: Decimal
    
    # Risk metrics
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    volatility: Optional[float] = None
    var_95: Optional[Decimal] = None  # Value at Risk
    
    # Performance metrics
    win_rate: Optional[float] = None
    profit_factor: Optional[float] = None
    avg_win: Optional[Decimal] = None
    avg_loss: Optional[Decimal] = None
    
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class TradeAnalysis:
    """Analysis of individual trade"""
    trade_id: str
    symbol: str
    side: str
    quantity: Decimal
    entry_price: Decimal
    exit_price: Optional[Decimal] = None
    entry_time: datetime = field(default_factory=datetime.utcnow)
    exit_time: Optional[datetime] = None
    pnl: Optional[Decimal] = None
    pnl_percent: Optional[float] = None
    holding_period: Optional[timedelta] = None
    strategy: Optional[str] = None
    fees: Decimal = Decimal('0')
    status: str = "open"  # 'open', 'closed', 'partial'

class PortfolioAnalytics:
    """Advanced portfolio analytics engine"""
    
    def __init__(self, db_config: Dict[str, str]):
        self.db_config = db_config
        self.logger = logging.getLogger(__name__)
        self.price_cache: Dict[str, Decimal] = {}
        self.cache_timeout = timedelta(minutes=5)
        self.last_price_update = datetime.utcnow()
    
    async def get_portfolio_summary(self, portfolio_type: str = "live") -> Dict[str, Any]:
        """Get comprehensive portfolio summary"""
        
        try:
            # Get positions
            positions = await self.get_current_positions(portfolio_type)
            
            # Get current prices
            await self.update_current_prices([pos.symbol for pos in positions])
            
            # Calculate metrics
            metrics = await self.calculate_portfolio_metrics(positions, portfolio_type)
            
            # Get performance history
            performance_history = await self.get_performance_history(portfolio_type, days=30)
            
            # Get trade analysis
            trade_analysis = await self.analyze_recent_trades(portfolio_type, days=7)
            
            return {
                'portfolio_type': portfolio_type,
                'metrics': metrics,
                'positions': [self._position_to_dict(pos) for pos in positions],
                'performance_history': performance_history,
                'trade_analysis': trade_analysis,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting portfolio summary: {e}")
            raise
    
    async def get_current_positions(self, portfolio_type: str) -> List[PortfolioPosition]:
        """Get current portfolio positions"""
        
        positions = []
        
        try:
            connection = mysql.connector.connect(**self.db_config)
            cursor = connection.cursor(dictionary=True)
            
            # Select appropriate table based on portfolio type
            table_name = f"{portfolio_type}_holdings"
            
            query = f"""
            SELECT symbol, SUM(quantity) as total_quantity, 
                   AVG(purchase_price) as avg_cost
            FROM {table_name}
            WHERE quantity > 0
            GROUP BY symbol
            """
            
            cursor.execute(query)
            holdings = cursor.fetchall()
            
            for holding in holdings:
                symbol = holding['symbol']
                quantity = Decimal(str(holding['total_quantity']))
                avg_cost = Decimal(str(holding['avg_cost']))
                
                # Get current price
                current_price = await self.get_current_price(symbol)
                
                # Calculate position metrics
                market_value = quantity * current_price
                cost_basis = quantity * avg_cost
                unrealized_pnl = market_value - cost_basis
                unrealized_pnl_percent = float((unrealized_pnl / cost_basis) * 100) if cost_basis > 0 else 0.0
                
                # Get daily change (simplified - would use price history in production)
                day_change = current_price * Decimal('0.01')  # Placeholder
                day_change_percent = 1.0  # Placeholder
                
                position = PortfolioPosition(
                    symbol=symbol,
                    quantity=quantity,
                    avg_cost=avg_cost,
                    current_price=current_price,
                    market_value=market_value,
                    unrealized_pnl=unrealized_pnl,
                    unrealized_pnl_percent=unrealized_pnl_percent,
                    day_change=day_change,
                    day_change_percent=day_change_percent,
                    position_size_percent=0.0  # Will calculate after getting total portfolio value
                )
                
                positions.append(position)
            
            cursor.close()
            connection.close()
            
            # Calculate position size percentages
            if positions:
                total_value = sum(pos.market_value for pos in positions)
                for position in positions:
                    position.position_size_percent = float((position.market_value / total_value) * 100)
            
            return positions
            
        except Exception as e:
            self.logger.error(f"Error getting current positions: {e}")
            return []
    
    async def calculate_portfolio_metrics(self, positions: List[PortfolioPosition], portfolio_type: str) -> PortfolioMetrics:
        """Calculate comprehensive portfolio metrics"""
        
        try:
            # Basic calculations
            total_value = sum(pos.market_value for pos in positions)
            total_cost = sum(pos.quantity * pos.avg_cost for pos in positions)
            total_pnl = total_value - total_cost
            total_pnl_percent = float((total_pnl / total_cost) * 100) if total_cost > 0 else 0.0
            
            # Daily change
            day_change = sum(pos.day_change * pos.quantity for pos in positions)
            day_change_percent = float((day_change / total_value) * 100) if total_value > 0 else 0.0
            
            # Get cash balance
            cash_balance = await self.get_cash_balance(portfolio_type)
            invested_amount = total_cost
            
            # Calculate advanced metrics
            performance_data = await self.get_daily_returns(portfolio_type, days=252)  # 1 year
            
            sharpe_ratio = self.calculate_sharpe_ratio(performance_data) if performance_data else None
            max_drawdown = self.calculate_max_drawdown(performance_data) if performance_data else None
            volatility = self.calculate_volatility(performance_data) if performance_data else None
            var_95 = self.calculate_var(performance_data, confidence=0.95) if performance_data else None
            
            # Trade analysis metrics
            trades = await self.get_closed_trades(portfolio_type, days=30)
            win_rate = self.calculate_win_rate(trades) if trades else None
            profit_factor = self.calculate_profit_factor(trades) if trades else None
            avg_win, avg_loss = self.calculate_avg_win_loss(trades) if trades else (None, None)
            
            return PortfolioMetrics(
                total_value=total_value,
                total_cost=total_cost,
                total_pnl=total_pnl,
                total_pnl_percent=total_pnl_percent,
                day_change=day_change,
                day_change_percent=day_change_percent,
                cash_balance=cash_balance,
                invested_amount=invested_amount,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                volatility=volatility,
                var_95=var_95,
                win_rate=win_rate,
                profit_factor=profit_factor,
                avg_win=avg_win,
                avg_loss=avg_loss
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio metrics: {e}")
            # Return basic metrics on error
            return PortfolioMetrics(
                total_value=Decimal('0'),
                total_cost=Decimal('0'),
                total_pnl=Decimal('0'),
                total_pnl_percent=0.0,
                day_change=Decimal('0'),
                day_change_percent=0.0,
                cash_balance=Decimal('0'),
                invested_amount=Decimal('0')
            )
    
    async def get_performance_history(self, portfolio_type: str, days: int = 30) -> List[Dict[str, Any]]:
        """Get historical performance data"""
        
        try:
            connection = mysql.connector.connect(**self.db_config)
            cursor = connection.cursor(dictionary=True)
            
            table_name = f"{portfolio_type}_performance_history"
            
            query = f"""
            SELECT date, portfolio_value, daily_pnl, daily_pnl_percent,
                   total_pnl, total_pnl_percent
            FROM {table_name}
            WHERE date >= DATE_SUB(CURDATE(), INTERVAL %s DAY)
            ORDER BY date DESC
            """
            
            cursor.execute(query, (days,))
            history = cursor.fetchall()
            
            cursor.close()
            connection.close()
            
            # Convert to list of dictionaries with proper formatting
            formatted_history = []
            for record in history:
                formatted_history.append({
                    'date': record['date'].isoformat(),
                    'portfolio_value': float(record['portfolio_value']),
                    'daily_pnl': float(record['daily_pnl']),
                    'daily_pnl_percent': float(record['daily_pnl_percent']),
                    'total_pnl': float(record['total_pnl']),
                    'total_pnl_percent': float(record['total_pnl_percent'])
                })
            
            return formatted_history
            
        except Exception as e:
            self.logger.error(f"Error getting performance history: {e}")
            return []
    
    async def analyze_recent_trades(self, portfolio_type: str, days: int = 7) -> Dict[str, Any]:
        """Analyze recent trading activity"""
        
        try:
            trades = await self.get_closed_trades(portfolio_type, days)
            
            if not trades:
                return {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'win_rate': 0.0,
                    'avg_win': 0.0,
                    'avg_loss': 0.0,
                    'profit_factor': 0.0,
                    'total_pnl': 0.0
                }
            
            # Analyze trades
            winning_trades = [t for t in trades if t.pnl and t.pnl > 0]
            losing_trades = [t for t in trades if t.pnl and t.pnl < 0]
            
            total_wins = sum(float(t.pnl) for t in winning_trades)
            total_losses = abs(sum(float(t.pnl) for t in losing_trades))
            
            analysis = {
                'total_trades': len(trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': (len(winning_trades) / len(trades)) * 100 if trades else 0.0,
                'avg_win': total_wins / len(winning_trades) if winning_trades else 0.0,
                'avg_loss': total_losses / len(losing_trades) if losing_trades else 0.0,
                'profit_factor': (total_wins / total_losses) if total_losses > 0 else float('inf'),
                'total_pnl': sum(float(t.pnl) for t in trades if t.pnl),
                'best_trade': max((float(t.pnl) for t in trades if t.pnl), default=0.0),
                'worst_trade': min((float(t.pnl) for t in trades if t.pnl), default=0.0)
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing recent trades: {e}")
            return {}
    
    async def get_current_price(self, symbol: str) -> Decimal:
        """Get current market price for symbol"""
        
        # Check cache first
        if symbol in self.price_cache:
            cache_age = datetime.utcnow() - self.last_price_update
            if cache_age < self.cache_timeout:
                return self.price_cache[symbol]
        
        try:
            # Fetch from Coinbase Pro API (example)
            async with aiohttp.ClientSession() as session:
                url = f"https://api.exchange.coinbase.com/products/{symbol}-USD/ticker"
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        price = Decimal(str(data['price']))
                        self.price_cache[symbol] = price
                        return price
                    else:
                        # Fallback price
                        return Decimal('50000')  # Placeholder
        except:
            # Fallback price
            return Decimal('50000')  # Placeholder
    
    async def update_current_prices(self, symbols: List[str]):
        """Update prices for multiple symbols"""
        
        tasks = [self.get_current_price(symbol) for symbol in symbols]
        await asyncio.gather(*tasks, return_exceptions=True)
        self.last_price_update = datetime.utcnow()
    
    async def get_cash_balance(self, portfolio_type: str) -> Decimal:
        """Get cash balance for portfolio"""
        
        try:
            connection = mysql.connector.connect(**self.db_config)
            cursor = connection.cursor()
            
            table_name = f"{portfolio_type}_portfolio"
            
            query = f"""
            SELECT cash_balance FROM {table_name} 
            ORDER BY last_updated DESC LIMIT 1
            """
            
            cursor.execute(query)
            result = cursor.fetchone()
            
            cursor.close()
            connection.close()
            
            return Decimal(str(result[0])) if result else Decimal('100000')  # Default
            
        except Exception as e:
            self.logger.error(f"Error getting cash balance: {e}")
            return Decimal('100000')  # Default
    
    async def get_daily_returns(self, portfolio_type: str, days: int = 252) -> List[float]:
        """Get daily returns for performance calculations"""
        
        try:
            connection = mysql.connector.connect(**self.db_config)
            cursor = connection.cursor()
            
            table_name = f"{portfolio_type}_performance_history"
            
            query = f"""
            SELECT daily_pnl_percent FROM {table_name}
            WHERE date >= DATE_SUB(CURDATE(), INTERVAL %s DAY)
            ORDER BY date ASC
            """
            
            cursor.execute(query, (days,))
            results = cursor.fetchall()
            
            cursor.close()
            connection.close()
            
            return [float(row[0]) / 100.0 for row in results]  # Convert percentage to decimal
            
        except Exception as e:
            self.logger.error(f"Error getting daily returns: {e}")
            return []
    
    async def get_closed_trades(self, portfolio_type: str, days: int = 30) -> List[TradeAnalysis]:
        """Get closed trades for analysis"""
        
        trades = []
        
        try:
            connection = mysql.connector.connect(**self.db_config)
            cursor = connection.cursor(dictionary=True)
            
            table_name = f"{portfolio_type}_trades"
            
            query = f"""
            SELECT trade_id, symbol, side, quantity, price, timestamp, pnl, fees
            FROM {table_name}
            WHERE timestamp >= DATE_SUB(NOW(), INTERVAL %s DAY)
            AND pnl IS NOT NULL
            ORDER BY timestamp DESC
            """
            
            cursor.execute(query, (days,))
            results = cursor.fetchall()
            
            for row in results:
                trade = TradeAnalysis(
                    trade_id=row['trade_id'],
                    symbol=row['symbol'],
                    side=row['side'],
                    quantity=Decimal(str(row['quantity'])),
                    entry_price=Decimal(str(row['price'])),
                    entry_time=row['timestamp'],
                    pnl=Decimal(str(row['pnl'])) if row['pnl'] else None,
                    fees=Decimal(str(row['fees'])) if row['fees'] else Decimal('0'),
                    status='closed'
                )
                trades.append(trade)
            
            cursor.close()
            connection.close()
            
            return trades
            
        except Exception as e:
            self.logger.error(f"Error getting closed trades: {e}")
            return []
    
    def calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        
        if len(returns) < 2:
            return 0.0
        
        try:
            excess_returns = [r - risk_free_rate/252 for r in returns]  # Daily risk-free rate
            avg_excess_return = mean(excess_returns)
            std_excess_return = stdev(excess_returns)
            
            if std_excess_return == 0:
                return 0.0
            
            # Annualized Sharpe ratio
            return (avg_excess_return / std_excess_return) * np.sqrt(252)
            
        except Exception as e:
            self.logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    def calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown"""
        
        if not returns:
            return 0.0
        
        try:
            cumulative_returns = np.cumprod([1 + r for r in returns])
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - running_max) / running_max
            
            return float(abs(min(drawdowns))) * 100  # Return as percentage
            
        except Exception as e:
            self.logger.error(f"Error calculating max drawdown: {e}")
            return 0.0
    
    def calculate_volatility(self, returns: List[float]) -> float:
        """Calculate annualized volatility"""
        
        if len(returns) < 2:
            return 0.0
        
        try:
            return stdev(returns) * np.sqrt(252) * 100  # Annualized percentage
        except Exception as e:
            self.logger.error(f"Error calculating volatility: {e}")
            return 0.0
    
    def calculate_var(self, returns: List[float], confidence: float = 0.95) -> Decimal:
        """Calculate Value at Risk"""
        
        if not returns:
            return Decimal('0')
        
        try:
            sorted_returns = sorted(returns)
            index = int((1 - confidence) * len(sorted_returns))
            var_return = sorted_returns[index] if index < len(sorted_returns) else sorted_returns[0]
            
            return Decimal(str(abs(var_return)))
            
        except Exception as e:
            self.logger.error(f"Error calculating VaR: {e}")
            return Decimal('0')
    
    def calculate_win_rate(self, trades: List[TradeAnalysis]) -> float:
        """Calculate win rate percentage"""
        
        if not trades:
            return 0.0
        
        winning_trades = sum(1 for trade in trades if trade.pnl and trade.pnl > 0)
        return (winning_trades / len(trades)) * 100
    
    def calculate_profit_factor(self, trades: List[TradeAnalysis]) -> float:
        """Calculate profit factor"""
        
        if not trades:
            return 0.0
        
        total_wins = sum(float(trade.pnl) for trade in trades if trade.pnl and trade.pnl > 0)
        total_losses = abs(sum(float(trade.pnl) for trade in trades if trade.pnl and trade.pnl < 0))
        
        return total_wins / total_losses if total_losses > 0 else float('inf')
    
    def calculate_avg_win_loss(self, trades: List[TradeAnalysis]) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        """Calculate average win and loss"""
        
        if not trades:
            return None, None
        
        wins = [trade.pnl for trade in trades if trade.pnl and trade.pnl > 0]
        losses = [abs(trade.pnl) for trade in trades if trade.pnl and trade.pnl < 0]
        
        avg_win = Decimal(str(mean(float(w) for w in wins))) if wins else None
        avg_loss = Decimal(str(mean(float(l) for l in losses))) if losses else None
        
        return avg_win, avg_loss
    
    def _position_to_dict(self, position: PortfolioPosition) -> Dict[str, Any]:
        """Convert position to dictionary for JSON serialization"""
        
        return {
            'symbol': position.symbol,
            'quantity': float(position.quantity),
            'avg_cost': float(position.avg_cost),
            'current_price': float(position.current_price),
            'market_value': float(position.market_value),
            'unrealized_pnl': float(position.unrealized_pnl),
            'unrealized_pnl_percent': position.unrealized_pnl_percent,
            'day_change': float(position.day_change),
            'day_change_percent': position.day_change_percent,
            'position_size_percent': position.position_size_percent,
            'last_updated': position.last_updated.isoformat()
        }

# FastAPI service for portfolio analytics
app = FastAPI(title="Portfolio Analytics Service", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database configuration
DB_CONFIG = {
    'host': 'host.docker.internal',
    'user': 'news_collector',
    'password': '99Rules!',
    'database': 'crypto_analysis'
}

# Initialize analytics engine
analytics = PortfolioAnalytics(DB_CONFIG)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "portfolio_analytics"}

@app.get("/portfolio/{portfolio_type}/summary")
async def get_portfolio_summary(portfolio_type: str):
    """Get comprehensive portfolio summary"""
    
    if portfolio_type not in ['mock', 'live']:
        raise HTTPException(status_code=400, detail="Portfolio type must be 'mock' or 'live'")
    
    try:
        summary = await analytics.get_portfolio_summary(portfolio_type)
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/portfolio/{portfolio_type}/positions")
async def get_positions(portfolio_type: str):
    """Get current portfolio positions"""
    
    if portfolio_type not in ['mock', 'live']:
        raise HTTPException(status_code=400, detail="Portfolio type must be 'mock' or 'live'")
    
    try:
        positions = await analytics.get_current_positions(portfolio_type)
        return {
            'portfolio_type': portfolio_type,
            'positions': [analytics._position_to_dict(pos) for pos in positions],
            'timestamp': datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/portfolio/{portfolio_type}/performance")
async def get_performance(portfolio_type: str, days: int = 30):
    """Get portfolio performance history"""
    
    if portfolio_type not in ['mock', 'live']:
        raise HTTPException(status_code=400, detail="Portfolio type must be 'mock' or 'live'")
    
    try:
        performance = await analytics.get_performance_history(portfolio_type, days)
        return {
            'portfolio_type': portfolio_type,
            'days': days,
            'performance_history': performance,
            'timestamp': datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/portfolio/{portfolio_type}/trades")
async def get_trade_analysis(portfolio_type: str, days: int = 7):
    """Get trade analysis"""
    
    if portfolio_type not in ['mock', 'live']:
        raise HTTPException(status_code=400, detail="Portfolio type must be 'mock' or 'live'")
    
    try:
        analysis = await analytics.analyze_recent_trades(portfolio_type, days)
        return {
            'portfolio_type': portfolio_type,
            'days': days,
            'trade_analysis': analysis,
            'timestamp': datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/portfolio/comparison")
async def compare_portfolios():
    """Compare mock vs live portfolio performance"""
    
    try:
        mock_summary = await analytics.get_portfolio_summary('mock')
        live_summary = await analytics.get_portfolio_summary('live')
        
        comparison = {
            'mock_portfolio': mock_summary,
            'live_portfolio': live_summary,
            'comparison_metrics': {
                'value_difference': float(
                    live_summary['metrics'].total_value - mock_summary['metrics'].total_value
                ),
                'pnl_difference': float(
                    live_summary['metrics'].total_pnl - mock_summary['metrics'].total_pnl
                ),
                'performance_difference': (
                    live_summary['metrics'].total_pnl_percent - mock_summary['metrics'].total_pnl_percent
                )
            },
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return comparison
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8027)
