#!/usr/bin/env python3
"""
Portfolio Optimization Service
Automatically consolidates small positions and focuses on highest-conviction assets
Designed to run every 4 hours to rebalance portfolio for maximum efficiency
"""

import os
import sys
import asyncio
import logging
import aiohttp
import mysql.connector
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Portfolio Optimization Service", version="1.0.0")

class PortfolioOptimizer:
    def __init__(self):
        self.db_config = {
            'host': os.environ.get('DATABASE_HOST', 'host.docker.internal'),
            'user': os.environ.get('DATABASE_USER', 'news_collector'),
            'password': os.environ.get('DATABASE_PASSWORD', '99Rules!'),
            'database': os.environ.get('DATABASE_NAME_TRANSACTIONS', 'crypto_transactions'),
            'port': int(os.environ.get('DATABASE_PORT', 3306))
        }
        
        # Optimization parameters
        self.min_position_size = float(os.environ.get('MIN_POSITION_SIZE', '50.0'))  # Consolidate below $50
        self.max_positions = int(os.environ.get('MAX_POSITIONS', '15'))  # Focus on top 15
        self.rebalance_threshold = float(os.environ.get('REBALANCE_THRESHOLD', '0.20'))  # 20% deviation triggers rebalance
        
        # Trading engine endpoint
        self.trading_engine_url = os.environ.get('TRADING_ENGINE_URL', 'http://trade-execution-engine:8024')
        
    async def get_current_portfolio(self) -> Dict:
        """Get current portfolio from trading engine"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.trading_engine_url}/portfolio") as response:
                    if response.status == 200:
                        portfolio = await response.json()
                        logger.info(f"📊 Retrieved portfolio: ${portfolio.get('total_portfolio_value', 0):.2f}")
                        return portfolio
                    else:
                        logger.error(f"Failed to get portfolio: {response.status}")
                        return {}
        except Exception as e:
            logger.error(f"Error getting portfolio: {e}")
            return {}
    
    def calculate_position_scores(self, positions: Dict) -> List[Tuple[str, float, Dict]]:
        """Calculate conviction scores for all positions"""
        try:
            conn = mysql.connector.connect(**self.db_config)
            
            position_scores = []
            
            for symbol, position_data in positions.items():
                # Get recent signals and performance for this symbol
                query = """
                SELECT signal_type, confidence, created_at
                FROM trade_recommendations 
                WHERE symbol = %s 
                AND created_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
                ORDER BY created_at DESC
                LIMIT 20
                """
                
                df = pd.read_sql(query, conn, params=[symbol])
                
                if df.empty:
                    conviction_score = 0.1  # Low conviction for no recent signals
                else:
                    # Calculate conviction based on recent signals
                    buy_signals = len(df[df['signal_type'] == 'BUY'])
                    total_signals = len(df)
                    avg_confidence = df['confidence'].mean()
                    
                    # Recent performance weight
                    recent_weight = len(df[df['created_at'] >= datetime.now() - timedelta(days=3)]) / max(1, total_signals)
                    
                    conviction_score = (
                        (buy_signals / max(1, total_signals)) * 0.4 +  # Buy signal ratio
                        avg_confidence * 0.4 +  # Average confidence
                        recent_weight * 0.2  # Recent activity weight
                    )
                
                position_scores.append((symbol, conviction_score, position_data))
            
            conn.close()
            
            # Sort by conviction score (highest first)
            position_scores.sort(key=lambda x: x[1], reverse=True)
            
            logger.info(f"📈 Calculated conviction scores for {len(position_scores)} positions")
            return position_scores
            
        except Exception as e:
            logger.error(f"Error calculating position scores: {e}")
            return []
    
    async def execute_consolidation_trade(self, symbol: str, action: str, amount: float) -> bool:
        """Execute a consolidation trade through the trading engine"""
        try:
            trade_request = {
                "symbol": symbol,
                "action": action,  # 'buy' or 'sell'
                "size_usd": amount,
                "order_type": "MARKET",
                "reason": f"portfolio_consolidation_{action}"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.trading_engine_url}/execute_trade", 
                    json=trade_request
                ) as response:
                    result = await response.json()
                    
                    if result.get('success'):
                        logger.info(f"✅ Consolidation trade executed: {action.upper()} ${amount:.2f} of {symbol}")
                        return True
                    else:
                        logger.warning(f"❌ Consolidation trade failed: {result.get('error', 'Unknown error')}")
                        return False
                        
        except Exception as e:
            logger.error(f"Error executing consolidation trade: {e}")
            return False
    
    async def optimize_portfolio(self) -> Dict:
        """Main portfolio optimization logic"""
        try:
            logger.info("🚀 Starting portfolio optimization...")
            
            # Get current portfolio
            portfolio = await self.get_current_portfolio()
            if not portfolio or not portfolio.get('positions'):
                logger.warning("No portfolio data available")
                return {'success': False, 'error': 'No portfolio data'}
            
            positions = portfolio['positions']
            total_value = portfolio['total_portfolio_value']
            
            # Calculate conviction scores
            position_scores = self.calculate_position_scores(positions)
            
            if not position_scores:
                logger.warning("No position scores calculated")
                return {'success': False, 'error': 'No position scores'}
            
            optimization_actions = []
            positions_to_sell = []
            positions_to_keep = []
            
            # Phase 1: Identify positions to consolidate (small + low conviction)
            for symbol, conviction_score, position_data in position_scores:
                position_value = position_data.get('value_usd', 0)
                
                # Consolidate if position is too small OR conviction is very low
                if position_value < self.min_position_size or conviction_score < 0.2:
                    positions_to_sell.append({
                        'symbol': symbol,
                        'value': position_value,
                        'conviction': conviction_score,
                        'reason': 'small_position' if position_value < self.min_position_size else 'low_conviction'
                    })
                else:
                    positions_to_keep.append({
                        'symbol': symbol,
                        'value': position_value,
                        'conviction': conviction_score
                    })
            
            # Phase 2: Focus on top positions
            if len(positions_to_keep) > self.max_positions:
                # Sort by conviction and keep only top positions
                positions_to_keep.sort(key=lambda x: x['conviction'], reverse=True)
                positions_to_sell.extend([
                    {
                        'symbol': pos['symbol'],
                        'value': pos['value'],
                        'conviction': pos['conviction'],
                        'reason': 'excess_positions'
                    }
                    for pos in positions_to_keep[self.max_positions:]
                ])
                positions_to_keep = positions_to_keep[:self.max_positions]
            
            # Phase 3: Execute consolidation trades
            total_liquidated = 0.0
            successful_sells = 0
            
            for position in positions_to_sell:
                if position['value'] > 5.0:  # Only sell if worth more than $5
                    success = await self.execute_consolidation_trade(
                        position['symbol'], 
                        'sell', 
                        position['value']
                    )
                    if success:
                        total_liquidated += position['value']
                        successful_sells += 1
                        optimization_actions.append({
                            'action': 'sell',
                            'symbol': position['symbol'],
                            'amount': position['value'],
                            'reason': position['reason']
                        })
            
            # Phase 4: Reinvest in top conviction positions
            if total_liquidated > 25.0 and positions_to_keep:  # Only reinvest if significant amount
                # Distribute liquidated funds among top 3 conviction positions
                top_positions = sorted(positions_to_keep, key=lambda x: x['conviction'], reverse=True)[:3]
                reinvestment_per_position = total_liquidated / len(top_positions)
                
                for position in top_positions:
                    if reinvestment_per_position > 25.0:  # Minimum $25 per position
                        success = await self.execute_consolidation_trade(
                            position['symbol'],
                            'buy',
                            reinvestment_per_position
                        )
                        if success:
                            optimization_actions.append({
                                'action': 'buy',
                                'symbol': position['symbol'],
                                'amount': reinvestment_per_position,
                                'reason': 'concentration_boost'
                            })
            
            # Phase 5: Log results
            result = {
                'success': True,
                'timestamp': datetime.now().isoformat(),
                'positions_sold': len([a for a in optimization_actions if a['action'] == 'sell']),
                'positions_bought': len([a for a in optimization_actions if a['action'] == 'buy']),
                'total_liquidated': total_liquidated,
                'optimization_actions': optimization_actions,
                'portfolio_summary': {
                    'total_positions_before': len(positions),
                    'positions_to_keep': len(positions_to_keep),
                    'positions_sold': len(positions_to_sell),
                    'concentration_improved': len(positions_to_keep) <= self.max_positions
                }
            }
            
            logger.info(f"✅ Portfolio optimization completed: {successful_sells} positions sold, ${total_liquidated:.2f} liquidated")
            return result
            
        except Exception as e:
            logger.error(f"Error in portfolio optimization: {e}")
            return {'success': False, 'error': str(e)}

# Global optimizer instance
optimizer = PortfolioOptimizer()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "portfolio-optimization", "timestamp": datetime.now().isoformat()}

@app.post("/optimize")
async def optimize_portfolio_endpoint(background_tasks: BackgroundTasks):
    """Trigger portfolio optimization"""
    background_tasks.add_task(run_optimization)
    return {"message": "Portfolio optimization started", "timestamp": datetime.now().isoformat()}

@app.get("/status")
async def get_optimization_status():
    """Get optimization status and last run info"""
    try:
        # Get current portfolio for status
        portfolio = await optimizer.get_current_portfolio()
        positions = portfolio.get('positions', {})
        
        return {
            'service': 'portfolio-optimization',
            'status': 'active',
            'current_positions': len(positions),
            'small_positions': len([p for p in positions.values() if p.get('value_usd', 0) < optimizer.min_position_size]),
            'parameters': {
                'min_position_size': optimizer.min_position_size,
                'max_positions': optimizer.max_positions,
                'rebalance_threshold': optimizer.rebalance_threshold
            },
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        return {"error": str(e), "timestamp": datetime.now().isoformat()}

async def run_optimization():
    """Background task to run optimization"""
    try:
        result = await optimizer.optimize_portfolio()
        logger.info(f"🎯 Optimization result: {result}")
    except Exception as e:
        logger.error(f"Optimization task error: {e}")

async def scheduled_optimization():
    """Run optimization every 4 hours"""
    while True:
        try:
            logger.info("⏰ Scheduled portfolio optimization starting...")
            await run_optimization()
            logger.info("😴 Sleeping for 4 hours until next optimization...")
            await asyncio.sleep(4 * 60 * 60)  # 4 hours
        except Exception as e:
            logger.error(f"Scheduled optimization error: {e}")
            await asyncio.sleep(300)  # Wait 5 minutes before retry

@app.on_event("startup")
async def startup_event():
    """Start the scheduled optimization task"""
    logger.info("🚀 Portfolio Optimization Service starting...")
    asyncio.create_task(scheduled_optimization())

if __name__ == "__main__":
    uvicorn.run(
        "portfolio_optimization_service:app",
        host="0.0.0.0",
        port=8026,
        log_level="info"
    )
