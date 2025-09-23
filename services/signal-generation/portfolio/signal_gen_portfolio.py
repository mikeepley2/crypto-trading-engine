#!/usr/bin/env python3
"""
Signal Generation Portfolio Service
Handles portfolio analysis, Kelly sizing, and cash deployment logic

This microservice extracts portfolio management functionality from enhanced_signal_generator.py
Responsibilities:
- Portfolio position analysis and weighting
- Kelly Criterion position sizing calculations
- Cash deployment pressure analysis
- Position rebalancing recommendations
- Risk-adjusted position management
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import asyncio
import aiohttp
from fastapi import FastAPI, HTTPException, Request
from prometheus_fastapi_instrumentator import Instrumentator
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel
import mysql.connector
from mysql.connector import pooling
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PortfolioAnalysisRequest(BaseModel):
    """Request model for portfolio analysis"""
    symbol: str
    signal_type: str  # BUY, SELL, HOLD
    confidence: float
    current_price: float
    timestamp: Optional[str] = None

class KellySizingRequest(BaseModel):
    """Request model for Kelly position sizing"""
    symbol: str
    confidence: float
    total_portfolio_value: float
    cash_balance: float
    win_rate_override: Optional[float] = None

class PortfolioPositionData(BaseModel):
    """Portfolio position information"""
    symbol: str
    balance: float
    value_usd: float
    current_price: float
    weight_percent: float

class PortfolioData(BaseModel):
    """Complete portfolio data"""
    total_value: float
    cash_balance: float
    cash_percentage: float
    positions: List[PortfolioPositionData]
    position_count: int

class KellySizingResponse(BaseModel):
    """Response for Kelly position sizing"""
    symbol: str
    kelly_position_size: float
    kelly_weight_percent: float
    confidence_adjusted_win_rate: float
    conservative_kelly_fraction: float
    recommendation: str

class PortfolioAnalysisResponse(BaseModel):
    """Response model for portfolio analysis"""
    symbol: str
    should_generate_signal: bool
    reason: str
    portfolio_data: PortfolioData
    kelly_sizing: Optional[KellySizingResponse] = None
    rebalancing_pressure: float
    cash_deployment_pressure: float
    position_analysis: Dict[str, Any]
    timestamp: str

class SignalGenPortfolio:
    """Portfolio Analysis Service for signal generation"""
    
    def __init__(self):
        self.app = FastAPI(
            title="Signal Generation Portfolio Service",
            description="Portfolio analysis and position management for trading signals",
            version="1.0.0"
        )

        # Setup Prometheus metrics
        self.instrumentator = Instrumentator()
        self.instrumentator.instrument(self.app).expose(self.app)
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self.db_pool = None
        self.trading_engine_url = os.getenv('TRADING_ENGINE_URL', 'http://host.docker.internal:8024')
        
        self.setup_database()
        self.setup_routes()
    
    def setup_database(self):
        """Setup database connection pool"""
        try:
            db_config = {
                'host': os.getenv('DATABASE_HOST', 'localhost'),
                'port': int(os.getenv('DATABASE_PORT', 3306)),
                'user': os.getenv('DATABASE_USER', 'news_collector'),
                'password': os.getenv('DATABASE_PASSWORD', '99Rules!'),
                'database': os.getenv('DATABASE_NAME', 'crypto_transactions'),
                'pool_name': 'portfolio_pool',
                'pool_size': 5,
                'pool_reset_session': True,
                'autocommit': True
            }
            
            self.db_pool = pooling.MySQLConnectionPool(**db_config)
            logger.info("✅ Database connection pool initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup database: {e}")
            self.db_pool = None
    
    async def get_current_portfolio(self) -> PortfolioData:
        """Get current portfolio positions from trading engine"""
        try:
            timeout = aiohttp.ClientTimeout(total=25)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(f"{self.trading_engine_url}/portfolio?fresh=true") as response:
                    if response.status == 200:
                        portfolio_data = await response.json()
                        
                        # Process positions
                        positions = []
                        total_value = float(portfolio_data.get('total_portfolio_value', 0))
                        
                        for position in portfolio_data.get('positions', []):
                            symbol = position.get('currency', '')
                            if symbol:
                                value_usd = float(position.get('value', 0))
                                weight_percent = (value_usd / total_value * 100) if total_value > 0 else 0
                                
                                positions.append(PortfolioPositionData(
                                    symbol=symbol,
                                    balance=float(position.get('available_balance', 0)),
                                    value_usd=value_usd,
                                    current_price=float(position.get('current_price', 0)),
                                    weight_percent=weight_percent
                                ))
                        
                        cash_balance = float(portfolio_data.get('usd_balance', 0))
                        cash_percentage = (cash_balance / total_value * 100) if total_value > 0 else 0
                        
                        portfolio = PortfolioData(
                            total_value=total_value,
                            cash_balance=cash_balance,
                            cash_percentage=cash_percentage,
                            positions=positions,
                            position_count=len(positions)
                        )
                        
                        logger.info(f"✅ Retrieved portfolio: ${total_value:.2f} total, "
                                   f"${cash_balance:.2f} cash ({cash_percentage:.1f}%), "
                                   f"{len(positions)} positions")
                        return portfolio
                    
                    else:
                        logger.warning(f"Trading engine unavailable: {response.status}")
        
        except Exception as e:
            logger.warning(f"Error getting portfolio: {e}")
        
        # Return empty portfolio on error
        return PortfolioData(
            total_value=0,
            cash_balance=0,
            cash_percentage=0,
            positions=[],
            position_count=0
        )
    
    def calculate_kelly_position_size(self, symbol: str, confidence: float, 
                                    total_value: float, cash_balance: float,
                                    win_rate_override: Optional[float] = None) -> KellySizingResponse:
        """Calculate Kelly Criterion-based position size"""
        try:
            # Conservative Kelly parameters for crypto
            base_win_rate = win_rate_override or 0.55  # 55% base win rate
            confidence_adjusted_win_rate = min(0.8, base_win_rate + (confidence - 0.5) * 0.4)
            
            avg_win = 0.08  # 8% average win
            avg_loss = 0.05  # 5% average loss
            
            # Kelly Formula: f* = (bp - q) / b
            b = avg_win / avg_loss  # 1.6 odds ratio
            p = confidence_adjusted_win_rate
            q = 1 - p
            
            kelly_fraction = (b * p - q) / b
            conservative_kelly = max(0, min(0.25, kelly_fraction * 0.5))  # Cap at 25%, use 50% of Kelly
            
            # Calculate position size
            kelly_position_size = total_value * conservative_kelly
            
            # Apply confidence-based scaling
            if confidence > 0.8:
                position_multiplier = 1.2  # Boost high-confidence positions
            elif confidence < 0.65:
                position_multiplier = 0.7  # Reduce low-confidence positions
            else:
                position_multiplier = 1.0
            
            final_position_size = kelly_position_size * position_multiplier
            
            # Ensure we don't exceed available cash
            final_position_size = min(final_position_size, cash_balance * 0.9)  # Leave 10% cash buffer
            
            kelly_weight_percent = (final_position_size / total_value * 100) if total_value > 0 else 0
            
            # Generate recommendation
            if final_position_size >= 25 and cash_balance >= final_position_size:
                recommendation = f"Kelly-sized position: ${final_position_size:.0f} ({kelly_weight_percent:.1f}%)"
            elif cash_balance < 25:
                recommendation = "Insufficient cash for minimum position"
            else:
                recommendation = f"Position too small: ${final_position_size:.0f} < $25 minimum"
            
            logger.debug(f"💰 Kelly sizing for {symbol}: kelly_frac={conservative_kelly:.3f}, "
                        f"base_size=${kelly_position_size:.0f}, final=${final_position_size:.0f}")
            
            return KellySizingResponse(
                symbol=symbol,
                kelly_position_size=final_position_size,
                kelly_weight_percent=kelly_weight_percent,
                confidence_adjusted_win_rate=confidence_adjusted_win_rate,
                conservative_kelly_fraction=conservative_kelly,
                recommendation=recommendation
            )
            
        except Exception as e:
            logger.warning(f"Error calculating Kelly size for {symbol}: {e}")
            fallback_size = min(100, cash_balance * 0.3)
            return KellySizingResponse(
                symbol=symbol,
                kelly_position_size=fallback_size,
                kelly_weight_percent=(fallback_size / total_value * 100) if total_value > 0 else 0,
                confidence_adjusted_win_rate=0.55,
                conservative_kelly_fraction=0.05,
                recommendation=f"Fallback conservative sizing: ${fallback_size:.0f}"
            )
    
    def should_generate_signal_for_position(self, symbol: str, signal_type: str, 
                                          confidence: float, portfolio: PortfolioData,
                                          kelly_sizing: KellySizingResponse) -> Tuple[bool, str]:
        """Determine if we should generate a signal based on current portfolio position"""
        try:
            # Find current position
            current_position = None
            for pos in portfolio.positions:
                if pos.symbol == symbol:
                    current_position = pos
                    break
            
            has_position = current_position is not None and current_position.value_usd > 1.0
            position_value = current_position.value_usd if has_position else 0
            position_weight = current_position.weight_percent if has_position else 0
            
            # Signal decision logic
            if signal_type == 'BUY':
                max_position_weight = 20.0  # 20% max per position
                
                if has_position:
                    if position_weight >= max_position_weight:
                        return False, f"Position at max weight ({position_weight:.1f}% >= {max_position_weight}%)"
                    elif kelly_sizing.kelly_position_size < 25:
                        return False, f"Kelly size too small (${kelly_sizing.kelly_position_size:.0f} < $25)"
                    elif portfolio.cash_balance < kelly_sizing.kelly_position_size:
                        return False, f"Insufficient cash (${portfolio.cash_balance:.0f} < ${kelly_sizing.kelly_position_size:.0f} Kelly size)"
                    else:
                        return True, f"Kelly-sized addition: ${kelly_sizing.kelly_position_size:.0f} (confidence {confidence:.3f})"
                else:
                    if kelly_sizing.kelly_position_size >= 25 and portfolio.cash_balance >= kelly_sizing.kelly_position_size:
                        return True, f"Kelly-sized new position: ${kelly_sizing.kelly_position_size:.0f} (confidence {confidence:.3f})"
                    elif portfolio.cash_balance >= 25 and confidence >= 0.8:
                        fallback_size = min(portfolio.cash_balance * 0.3, 100)
                        return True, f"High-confidence override: ${fallback_size:.0f} (confidence {confidence:.3f})"
                    else:
                        return False, f"Kelly size insufficient (${kelly_sizing.kelly_position_size:.0f}) or low cash (${portfolio.cash_balance:.0f})"
            
            elif signal_type == 'SELL':
                if has_position:
                    if confidence >= 0.7:
                        sell_fraction = min(1.0, confidence * 1.2)
                        sell_value = position_value * sell_fraction
                        return True, f"Confidence-based SELL: ${sell_value:.0f} ({sell_fraction:.1%} of position)"
                    elif position_weight > 15.0 and confidence >= 0.6:
                        return True, f"Rebalance large position ({position_weight:.1f}%) with moderate confidence"
                    else:
                        return False, f"SELL confidence too low ({confidence:.3f}) for position"
                else:
                    return False, f"No {symbol} position to sell"
            
            elif signal_type == 'HOLD':
                if has_position and position_weight > 25.0:
                    return True, f"HOLD/rebalance: Trim oversized position ({position_weight:.1f}%)"
                elif portfolio.cash_percentage > 15.0:
                    return False, f"HOLD signal but excess cash ({portfolio.cash_percentage:.1f}%) - prefer BUY signals"
                else:
                    return False, f"HOLD signal - position appropriately sized"
            
            return False, f"Unknown signal type: {signal_type}"
            
        except Exception as e:
            logger.error(f"❌ Error in position analysis for {symbol}: {e}")
            return False, f"Error in analysis: {str(e)}"
    
    def calculate_rebalancing_pressure(self, portfolio: PortfolioData) -> float:
        """Calculate rebalancing pressure from oversized positions"""
        try:
            max_position_weight = 20.0
            total_pressure = 0.0
            
            for position in portfolio.positions:
                if position.weight_percent > max_position_weight + 2.0:  # 2% buffer
                    excess_weight = position.weight_percent - max_position_weight
                    pressure = min(0.8, excess_weight / 10.0)  # Scale pressure
                    total_pressure += pressure
            
            return min(1.0, total_pressure)  # Cap at 1.0
            
        except Exception as e:
            logger.error(f"❌ Error calculating rebalancing pressure: {e}")
            return 0.0
    
    def calculate_cash_deployment_pressure(self, portfolio: PortfolioData) -> float:
        """Calculate cash deployment pressure from excess cash"""
        try:
            target_cash_percentage = 5.0
            
            if portfolio.cash_percentage > target_cash_percentage + 5.0:  # 5% buffer
                excess_cash_percentage = portfolio.cash_percentage - target_cash_percentage
                pressure = min(0.15, excess_cash_percentage / 20.0)  # Scale pressure
                return pressure
            
            return 0.0
            
        except Exception as e:
            logger.error(f"❌ Error calculating cash deployment pressure: {e}")
            return 0.0
    
    async def analyze_portfolio(self, request: PortfolioAnalysisRequest) -> PortfolioAnalysisResponse:
        """Main portfolio analysis endpoint"""
        try:
            # Get current portfolio
            portfolio = await self.get_current_portfolio()
            
            # Calculate Kelly sizing for BUY signals
            kelly_sizing = None
            if request.signal_type == 'BUY':
                kelly_sizing = self.calculate_kelly_position_size(
                    request.symbol,
                    request.confidence,
                    portfolio.total_value,
                    portfolio.cash_balance
                )
            
            # Determine if signal should be generated
            should_generate, reason = self.should_generate_signal_for_position(
                request.symbol,
                request.signal_type,
                request.confidence,
                portfolio,
                kelly_sizing or KellySizingResponse(
                    symbol=request.symbol,
                    kelly_position_size=0,
                    kelly_weight_percent=0,
                    confidence_adjusted_win_rate=0.55,
                    conservative_kelly_fraction=0,
                    recommendation="N/A for non-BUY signals"
                )
            )
            
            # Calculate pressures
            rebalancing_pressure = self.calculate_rebalancing_pressure(portfolio)
            cash_deployment_pressure = self.calculate_cash_deployment_pressure(portfolio)
            
            # Position analysis
            current_position = None
            for pos in portfolio.positions:
                if pos.symbol == request.symbol:
                    current_position = pos
                    break
            
            position_analysis = {
                "has_position": current_position is not None,
                "position_value": current_position.value_usd if current_position else 0,
                "position_weight": current_position.weight_percent if current_position else 0,
                "max_position_weight": 20.0,
                "remaining_capacity": max(0, 20.0 - (current_position.weight_percent if current_position else 0))
            }
            
            logger.info(f"📊 Portfolio analysis for {request.symbol}: {request.signal_type} "
                       f"-> {'APPROVED' if should_generate else 'REJECTED'} - {reason}")
            
            return PortfolioAnalysisResponse(
                symbol=request.symbol,
                should_generate_signal=should_generate,
                reason=reason,
                portfolio_data=portfolio,
                kelly_sizing=kelly_sizing,
                rebalancing_pressure=rebalancing_pressure,
                cash_deployment_pressure=cash_deployment_pressure,
                position_analysis=position_analysis,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"❌ Error analyzing portfolio for {request.symbol}: {e}")
            raise HTTPException(status_code=500, detail=f"Portfolio analysis failed: {str(e)}")
    
    def setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            db_status = "connected" if self.db_pool else "disconnected"
            
            return {
                "status": "healthy",
                "service": "signal-gen-portfolio",
                "database_status": db_status,
                "trading_engine": self.trading_engine_url,
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.post("/analyze", response_model=PortfolioAnalysisResponse)
        async def analyze_portfolio_endpoint(request: PortfolioAnalysisRequest):
            """Portfolio analysis endpoint"""
            return await self.analyze_portfolio(request)
        
        @self.app.post("/kelly_sizing", response_model=KellySizingResponse)
        async def kelly_sizing_endpoint(request: KellySizingRequest):
            """Kelly position sizing endpoint"""
            return self.calculate_kelly_position_size(
                request.symbol,
                request.confidence,
                request.total_portfolio_value,
                request.cash_balance,
                request.win_rate_override
            )
        
        @self.app.get("/portfolio", response_model=PortfolioData)
        async def get_portfolio_endpoint():
            """Get current portfolio data"""
            return await self.get_current_portfolio()
        
        @self.app.post("/batch_analyze")
        async def batch_analyze_endpoint(requests: List[PortfolioAnalysisRequest]):
            """Batch portfolio analysis endpoint"""
            results = []
            for request in requests:
                try:
                    result = await self.analyze_portfolio(request)
                    results.append(result)
                except Exception as e:
                    logger.error(f"❌ Error analyzing portfolio for {request.symbol}: {e}")
                    continue
            
            return {
                "results": results,
                "successful": len(results),
                "total": len(requests),
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.get("/status")
        async def get_status():
            """Detailed status information"""
            return {
                "service": "signal-gen-portfolio",
                "version": "1.0.0",
                "database_connected": bool(self.db_pool),
                "trading_engine_url": self.trading_engine_url,
                "features": [
                    "kelly_position_sizing",
                    "portfolio_analysis",
                    "rebalancing_pressure",
                    "cash_deployment_pressure",
                    "position_weight_management"
                ],
                "timestamp": datetime.now().isoformat()
            }

def main():
    """Main function to run the Portfolio service"""
    try:
        logger.info("🚀 Starting Signal Generation Portfolio Service...")
        
        portfolio_service = SignalGenPortfolio()
        
        # Get port from environment or use default
        port = int(os.getenv('PORTFOLIO_PORT', 8054))
        
        logger.info(f"💼 Portfolio service starting on port {port}")
        
        # Run the FastAPI application
        uvicorn.run(
            portfolio_service.app,
            host="0.0.0.0",
            port=port,
            log_level="info"
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to start Portfolio service: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
