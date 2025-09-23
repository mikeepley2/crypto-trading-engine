#!/usr/bin/env python3
"""
Signal Generation Risk Management Service
Handles risk controls, selloff protection, and risk adjustments

This microservice extracts risk management functionality from enhanced_signal_generator.py
Responsibilities:
- Selloff protection detection and activation
- Recovery enhancement logic
- Risk-adjusted signal modifications
- Global risk controls and limits
- Market condition risk assessments
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

class RiskAssessmentRequest(BaseModel):
    """Request model for risk assessment"""
    symbol: str
    signal_type: str  # BUY, SELL, HOLD
    confidence: float
    current_price: float
    position_size: Optional[float] = None
    timestamp: Optional[str] = None

class MarketConditionsData(BaseModel):
    """Market conditions assessment"""
    btc_price: float
    btc_24h_change: float
    fear_greed_index: Optional[int] = None
    volatility_index: float
    market_regime: str  # BULL, BEAR, SIDEWAYS, UNCERTAIN
    selloff_protection_active: bool
    recovery_enhancement_active: bool

class RiskMetrics(BaseModel):
    """Risk metrics for assessment"""
    portfolio_volatility: float
    max_drawdown_risk: float
    correlation_risk: float
    concentration_risk: float
    liquidity_risk: float
    overall_risk_score: float

class RiskAdjustments(BaseModel):
    """Risk-based adjustments to signals"""
    position_size_multiplier: float
    confidence_adjustment: float
    signal_block_reason: Optional[str] = None
    risk_level: str  # LOW, MEDIUM, HIGH, EXTREME
    recommended_action: str

class RiskAssessmentResponse(BaseModel):
    """Response model for risk assessment"""
    symbol: str
    signal_allowed: bool
    risk_adjustments: RiskAdjustments
    market_conditions: MarketConditionsData
    risk_metrics: RiskMetrics
    selloff_protection_details: Dict[str, Any]
    recovery_enhancement_details: Dict[str, Any]
    reasoning: str
    timestamp: str

class SignalGenRiskMgmt:
    """Risk Management Service for signal generation"""
    
    def __init__(self):
        self.app = FastAPI(
            title="Signal Generation Risk Management Service",
            description="Risk controls and protection for trading signals",
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
        self.market_context_url = os.getenv('MARKET_CONTEXT_URL', 'http://host.docker.internal:8053')
        
        # Risk thresholds
        self.max_portfolio_risk = 0.8
        self.max_single_position_risk = 0.25
        self.selloff_threshold = -8.0  # -8% BTC drop triggers selloff protection
        self.recovery_threshold = 3.0   # +3% BTC rise enables recovery enhancement
        
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
                'pool_name': 'risk_mgmt_pool',
                'pool_size': 5,
                'pool_reset_session': True,
                'autocommit': True
            }
            
            self.db_pool = pooling.MySQLConnectionPool(**db_config)
            logger.info("✅ Database connection pool initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup database: {e}")
            self.db_pool = None
    
    async def get_market_conditions(self) -> MarketConditionsData:
        """Get current market conditions and risk indicators"""
        try:
            # Get BTC price and conditions
            btc_data = await self.get_btc_market_data()
            
            # Get market context if available
            market_regime = "UNCERTAIN"
            volatility_index = 0.5
            
            try:
                timeout = aiohttp.ClientTimeout(total=10)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(f"{self.market_context_url}/sentiment") as response:
                        if response.status == 200:
                            sentiment_data = await response.json()
                            market_regime = sentiment_data.get('market_regime', 'UNCERTAIN')
                            volatility_index = sentiment_data.get('volatility_index', 0.5)
            except:
                logger.debug("Market context service unavailable, using defaults")
            
            # Determine protection states
            selloff_protection_active = btc_data['24h_change'] <= self.selloff_threshold
            recovery_enhancement_active = (
                btc_data['24h_change'] >= self.recovery_threshold and 
                not selloff_protection_active
            )
            
            return MarketConditionsData(
                btc_price=btc_data['price'],
                btc_24h_change=btc_data['24h_change'],
                fear_greed_index=btc_data.get('fear_greed_index'),
                volatility_index=volatility_index,
                market_regime=market_regime,
                selloff_protection_active=selloff_protection_active,
                recovery_enhancement_active=recovery_enhancement_active
            )
            
        except Exception as e:
            logger.warning(f"Error getting market conditions: {e}")
            return MarketConditionsData(
                btc_price=50000,
                btc_24h_change=0,
                fear_greed_index=50,
                volatility_index=0.5,
                market_regime="UNCERTAIN",
                selloff_protection_active=False,
                recovery_enhancement_active=False
            )
    
    async def get_btc_market_data(self) -> Dict[str, Any]:
        """Get BTC market data from trading engine"""
        try:
            timeout = aiohttp.ClientTimeout(total=15)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(f"{self.trading_engine_url}/market_data/BTC") as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            'price': float(data.get('price', 50000)),
                            '24h_change': float(data.get('change_24h', 0)),
                            'volume_24h': float(data.get('volume_24h', 0)),
                            'fear_greed_index': data.get('fear_greed_index')
                        }
        except Exception as e:
            logger.warning(f"Error getting BTC data: {e}")
        
        # Fallback data
        return {
            'price': 50000,
            '24h_change': 0,
            'volume_24h': 1000000000,
            'fear_greed_index': 50
        }
    
    async def calculate_risk_metrics(self) -> RiskMetrics:
        """Calculate current portfolio risk metrics"""
        try:
            # Get portfolio data
            portfolio_data = await self.get_portfolio_data()
            
            # Calculate risk metrics
            portfolio_volatility = self.calculate_portfolio_volatility(portfolio_data)
            max_drawdown_risk = self.calculate_max_drawdown_risk(portfolio_data)
            correlation_risk = self.calculate_correlation_risk(portfolio_data)
            concentration_risk = self.calculate_concentration_risk(portfolio_data)
            liquidity_risk = self.calculate_liquidity_risk(portfolio_data)
            
            # Overall risk score (0-1 scale)
            overall_risk_score = np.mean([
                portfolio_volatility,
                max_drawdown_risk,
                correlation_risk * 0.8,  # Lower weight
                concentration_risk,
                liquidity_risk * 0.6     # Lower weight
            ])
            
            return RiskMetrics(
                portfolio_volatility=portfolio_volatility,
                max_drawdown_risk=max_drawdown_risk,
                correlation_risk=correlation_risk,
                concentration_risk=concentration_risk,
                liquidity_risk=liquidity_risk,
                overall_risk_score=overall_risk_score
            )
            
        except Exception as e:
            logger.warning(f"Error calculating risk metrics: {e}")
            return RiskMetrics(
                portfolio_volatility=0.3,
                max_drawdown_risk=0.2,
                correlation_risk=0.4,
                concentration_risk=0.3,
                liquidity_risk=0.1,
                overall_risk_score=0.3
            )
    
    async def get_portfolio_data(self) -> Dict[str, Any]:
        """Get portfolio data from trading engine"""
        try:
            timeout = aiohttp.ClientTimeout(total=15)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(f"{self.trading_engine_url}/portfolio") as response:
                    if response.status == 200:
                        return await response.json()
        except Exception as e:
            logger.warning(f"Error getting portfolio data: {e}")
        
        return {}
    
    def calculate_portfolio_volatility(self, portfolio_data: Dict) -> float:
        """Calculate portfolio volatility estimate"""
        try:
            positions = portfolio_data.get('positions', [])
            if not positions:
                return 0.1
            
            # Simple volatility estimate based on position count and weights
            weights = [float(pos.get('value', 0)) for pos in positions]
            total_value = sum(weights)
            
            if total_value <= 0:
                return 0.1
            
            # Normalize weights
            weights = [w / total_value for w in weights]
            
            # Estimate volatility (higher for more concentrated portfolios)
            max_weight = max(weights) if weights else 0
            concentration_factor = max_weight * 2  # Scale concentration impact
            
            # Base crypto volatility + concentration adjustment
            base_volatility = 0.4  # 40% base crypto volatility
            portfolio_volatility = min(0.8, base_volatility * (1 + concentration_factor))
            
            return portfolio_volatility
            
        except Exception as e:
            logger.warning(f"Error calculating portfolio volatility: {e}")
            return 0.4
    
    def calculate_max_drawdown_risk(self, portfolio_data: Dict) -> float:
        """Calculate maximum drawdown risk estimate"""
        try:
            # Simple estimate based on current portfolio composition
            total_value = float(portfolio_data.get('total_portfolio_value', 0))
            cash_balance = float(portfolio_data.get('usd_balance', 0))
            
            if total_value <= 0:
                return 0.1
            
            cash_ratio = cash_balance / total_value
            
            # Higher cash = lower drawdown risk
            drawdown_risk = max(0.1, 0.6 - (cash_ratio * 0.4))
            
            return min(0.8, drawdown_risk)
            
        except Exception as e:
            logger.warning(f"Error calculating max drawdown risk: {e}")
            return 0.3
    
    def calculate_correlation_risk(self, portfolio_data: Dict) -> float:
        """Calculate correlation risk (all crypto positions are highly correlated)"""
        try:
            positions = portfolio_data.get('positions', [])
            non_stablecoin_positions = [
                pos for pos in positions 
                if pos.get('currency', '').upper() not in ['USDC', 'USDT', 'DAI', 'BUSD']
            ]
            
            if len(non_stablecoin_positions) <= 1:
                return 0.2  # Low correlation risk with few positions
            
            # High correlation risk with multiple crypto positions
            correlation_risk = min(0.8, 0.4 + (len(non_stablecoin_positions) * 0.1))
            
            return correlation_risk
            
        except Exception as e:
            logger.warning(f"Error calculating correlation risk: {e}")
            return 0.5
    
    def calculate_concentration_risk(self, portfolio_data: Dict) -> float:
        """Calculate concentration risk based on position weights"""
        try:
            positions = portfolio_data.get('positions', [])
            if not positions:
                return 0.1
            
            weights = [float(pos.get('value', 0)) for pos in positions]
            total_value = sum(weights)
            
            if total_value <= 0:
                return 0.1
            
            # Calculate Herfindahl index (concentration measure)
            normalized_weights = [w / total_value for w in weights]
            herfindahl_index = sum([w**2 for w in normalized_weights])
            
            # Convert to risk score (higher HHI = higher concentration risk)
            concentration_risk = min(0.9, herfindahl_index * 1.5)
            
            return concentration_risk
            
        except Exception as e:
            logger.warning(f"Error calculating concentration risk: {e}")
            return 0.4
    
    def calculate_liquidity_risk(self, portfolio_data: Dict) -> float:
        """Calculate liquidity risk based on position sizes and market caps"""
        try:
            positions = portfolio_data.get('positions', [])
            if not positions:
                return 0.1
            
            # Simple liquidity risk based on number of positions
            # More positions = better liquidity distribution
            num_positions = len(positions)
            
            if num_positions >= 5:
                return 0.1  # Good diversification
            elif num_positions >= 3:
                return 0.2  # Moderate liquidity risk
            else:
                return 0.4  # Higher liquidity risk with few positions
            
        except Exception as e:
            logger.warning(f"Error calculating liquidity risk: {e}")
            return 0.2
    
    def apply_selloff_protection(self, signal_type: str, confidence: float, 
                               btc_24h_change: float) -> Tuple[float, str, Optional[str]]:
        """Apply selloff protection adjustments"""
        try:
            if btc_24h_change > self.selloff_threshold:
                return 1.0, "NORMAL", None  # No selloff protection needed
            
            # Selloff protection active
            selloff_severity = abs(btc_24h_change) / abs(self.selloff_threshold)
            
            if signal_type == 'BUY':
                if selloff_severity >= 2.0:  # > -16% BTC drop
                    return 0.0, "EXTREME", "Extreme selloff - blocking all BUY signals"
                elif selloff_severity >= 1.5:  # > -12% BTC drop
                    return 0.2, "HIGH", f"Severe selloff ({btc_24h_change:.1f}%) - reducing BUY signals by 80%"
                else:  # -8% to -12% BTC drop
                    return 0.5, "MEDIUM", f"Moderate selloff ({btc_24h_change:.1f}%) - reducing BUY signals by 50%"
            
            elif signal_type == 'SELL':
                # Encourage SELL signals during selloffs
                if selloff_severity >= 1.5:
                    return 1.3, "HIGH", f"Severe selloff ({btc_24h_change:.1f}%) - enhancing SELL signals"
                else:
                    return 1.1, "MEDIUM", f"Moderate selloff ({btc_24h_change:.1f}%) - slightly enhancing SELL signals"
            
            return 1.0, "NORMAL", None
            
        except Exception as e:
            logger.error(f"❌ Error applying selloff protection: {e}")
            return 1.0, "NORMAL", None
    
    def apply_recovery_enhancement(self, signal_type: str, confidence: float,
                                 btc_24h_change: float) -> Tuple[float, str, Optional[str]]:
        """Apply recovery enhancement adjustments"""
        try:
            if btc_24h_change < self.recovery_threshold:
                return 1.0, "NORMAL", None  # No recovery enhancement
            
            # Recovery enhancement active
            recovery_strength = btc_24h_change / self.recovery_threshold
            
            if signal_type == 'BUY':
                if recovery_strength >= 3.0:  # > +9% BTC rise
                    return 1.4, "HIGH", f"Strong recovery ({btc_24h_change:.1f}%) - enhancing BUY signals by 40%"
                elif recovery_strength >= 2.0:  # > +6% BTC rise
                    return 1.2, "MEDIUM", f"Good recovery ({btc_24h_change:.1f}%) - enhancing BUY signals by 20%"
                else:  # +3% to +6% BTC rise
                    return 1.1, "LOW", f"Early recovery ({btc_24h_change:.1f}%) - slightly enhancing BUY signals"
            
            elif signal_type == 'SELL':
                # Reduce SELL signals during recoveries
                if recovery_strength >= 2.0:
                    return 0.7, "MEDIUM", f"Strong recovery ({btc_24h_change:.1f}%) - reducing SELL signals"
                else:
                    return 0.9, "LOW", f"Early recovery ({btc_24h_change:.1f}%) - slightly reducing SELL signals"
            
            return 1.0, "NORMAL", None
            
        except Exception as e:
            logger.error(f"❌ Error applying recovery enhancement: {e}")
            return 1.0, "NORMAL", None
    
    async def assess_risk(self, request: RiskAssessmentRequest) -> RiskAssessmentResponse:
        """Main risk assessment endpoint"""
        try:
            # Get market conditions
            market_conditions = await self.get_market_conditions()
            
            # Calculate risk metrics
            risk_metrics = await self.calculate_risk_metrics()
            
            # Initialize adjustments
            position_size_multiplier = 1.0
            confidence_adjustment = 0.0
            signal_block_reason = None
            risk_level = "MEDIUM"
            
            reasoning_parts = []
            
            # Apply selloff protection
            if market_conditions.selloff_protection_active:
                selloff_mult, selloff_risk, selloff_reason = self.apply_selloff_protection(
                    request.signal_type, request.confidence, market_conditions.btc_24h_change
                )
                position_size_multiplier *= selloff_mult
                if selloff_reason:
                    reasoning_parts.append(selloff_reason)
                if selloff_mult == 0.0:
                    signal_block_reason = selloff_reason
            
            # Apply recovery enhancement
            if market_conditions.recovery_enhancement_active:
                recovery_mult, recovery_risk, recovery_reason = self.apply_recovery_enhancement(
                    request.signal_type, request.confidence, market_conditions.btc_24h_change
                )
                position_size_multiplier *= recovery_mult
                if recovery_reason:
                    reasoning_parts.append(recovery_reason)
            
            # Risk-based adjustments
            if risk_metrics.overall_risk_score >= 0.7:
                risk_level = "HIGH"
                position_size_multiplier *= 0.7
                confidence_adjustment = -0.1
                reasoning_parts.append(f"High portfolio risk ({risk_metrics.overall_risk_score:.2f}) - reducing position sizes")
                
                if risk_metrics.overall_risk_score >= 0.85:
                    risk_level = "EXTREME"
                    if request.signal_type == 'BUY':
                        signal_block_reason = "Extreme portfolio risk - blocking new BUY signals"
                        position_size_multiplier = 0.0
            
            elif risk_metrics.overall_risk_score <= 0.3:
                risk_level = "LOW"
                position_size_multiplier *= 1.1
                confidence_adjustment = 0.05
                reasoning_parts.append("Low portfolio risk - allowing enhanced position sizing")
            
            # Market regime adjustments
            if market_conditions.market_regime == "BEAR":
                position_size_multiplier *= 0.8
                reasoning_parts.append("Bear market regime - reducing position sizes")
            elif market_conditions.market_regime == "BULL":
                position_size_multiplier *= 1.1
                reasoning_parts.append("Bull market regime - enhancing position sizes")
            
            # Determine if signal is allowed
            signal_allowed = signal_block_reason is None
            
            # Prepare detailed information
            selloff_protection_details = {
                "active": market_conditions.selloff_protection_active,
                "btc_24h_change": market_conditions.btc_24h_change,
                "threshold": self.selloff_threshold,
                "severity": abs(market_conditions.btc_24h_change) / abs(self.selloff_threshold) if market_conditions.selloff_protection_active else 0
            }
            
            recovery_enhancement_details = {
                "active": market_conditions.recovery_enhancement_active,
                "btc_24h_change": market_conditions.btc_24h_change,
                "threshold": self.recovery_threshold,
                "strength": market_conditions.btc_24h_change / self.recovery_threshold if market_conditions.recovery_enhancement_active else 0
            }
            
            # Final adjustments
            risk_adjustments = RiskAdjustments(
                position_size_multiplier=max(0.0, position_size_multiplier),
                confidence_adjustment=confidence_adjustment,
                signal_block_reason=signal_block_reason,
                risk_level=risk_level,
                recommended_action=f"Apply {position_size_multiplier:.1f}x size multiplier" if signal_allowed else "Block signal"
            )
            
            reasoning = " | ".join(reasoning_parts) if reasoning_parts else "No significant risk adjustments required"
            
            logger.info(f"🛡️ Risk assessment for {request.symbol}: {request.signal_type} "
                       f"-> {'ALLOWED' if signal_allowed else 'BLOCKED'} "
                       f"(size_mult: {position_size_multiplier:.2f}, risk: {risk_level})")
            
            return RiskAssessmentResponse(
                symbol=request.symbol,
                signal_allowed=signal_allowed,
                risk_adjustments=risk_adjustments,
                market_conditions=market_conditions,
                risk_metrics=risk_metrics,
                selloff_protection_details=selloff_protection_details,
                recovery_enhancement_details=recovery_enhancement_details,
                reasoning=reasoning,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"❌ Error assessing risk for {request.symbol}: {e}")
            raise HTTPException(status_code=500, detail=f"Risk assessment failed: {str(e)}")
    
    def setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            db_status = "connected" if self.db_pool else "disconnected"
            
            return {
                "status": "healthy",
                "service": "signal-gen-risk-mgmt",
                "database_status": db_status,
                "trading_engine": self.trading_engine_url,
                "market_context": self.market_context_url,
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.post("/assess", response_model=RiskAssessmentResponse)
        async def assess_risk_endpoint(request: RiskAssessmentRequest):
            """Risk assessment endpoint"""
            return await self.assess_risk(request)
        
        @self.app.get("/market_conditions", response_model=MarketConditionsData)
        async def get_market_conditions_endpoint():
            """Get current market conditions"""
            return await self.get_market_conditions()
        
        @self.app.get("/risk_metrics", response_model=RiskMetrics)
        async def get_risk_metrics_endpoint():
            """Get current risk metrics"""
            return await self.calculate_risk_metrics()
        
        @self.app.post("/batch_assess")
        async def batch_assess_endpoint(requests: List[RiskAssessmentRequest]):
            """Batch risk assessment endpoint"""
            results = []
            for request in requests:
                try:
                    result = await self.assess_risk(request)
                    results.append(result)
                except Exception as e:
                    logger.error(f"❌ Error assessing risk for {request.symbol}: {e}")
                    continue
            
            return {
                "results": results,
                "successful": len(results),
                "total": len(requests),
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.get("/protection_status")
        async def get_protection_status():
            """Get selloff protection and recovery enhancement status"""
            market_conditions = await self.get_market_conditions()
            
            return {
                "selloff_protection": {
                    "active": market_conditions.selloff_protection_active,
                    "btc_change": market_conditions.btc_24h_change,
                    "threshold": self.selloff_threshold
                },
                "recovery_enhancement": {
                    "active": market_conditions.recovery_enhancement_active,
                    "btc_change": market_conditions.btc_24h_change,
                    "threshold": self.recovery_threshold
                },
                "market_regime": market_conditions.market_regime,
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.get("/status")
        async def get_status():
            """Detailed status information"""
            return {
                "service": "signal-gen-risk-mgmt",
                "version": "1.0.0",
                "database_connected": bool(self.db_pool),
                "trading_engine_url": self.trading_engine_url,
                "market_context_url": self.market_context_url,
                "risk_thresholds": {
                    "max_portfolio_risk": self.max_portfolio_risk,
                    "max_single_position_risk": self.max_single_position_risk,
                    "selloff_threshold": self.selloff_threshold,
                    "recovery_threshold": self.recovery_threshold
                },
                "features": [
                    "selloff_protection",
                    "recovery_enhancement",
                    "portfolio_risk_assessment",
                    "market_condition_monitoring",
                    "signal_blocking",
                    "position_size_adjustment"
                ],
                "timestamp": datetime.now().isoformat()
            }

def main():
    """Main function to run the Risk Management service"""
    try:
        logger.info("🚀 Starting Signal Generation Risk Management Service...")
        
        risk_mgmt_service = SignalGenRiskMgmt()
        
        # Get port from environment or use default
        port = int(os.getenv('RISK_MGMT_PORT', 8055))
        
        logger.info(f"🛡️ Risk Management service starting on port {port}")
        
        # Run the FastAPI application
        uvicorn.run(
            risk_mgmt_service.app,
            host="0.0.0.0",
            port=port,
            log_level="info"
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to start Risk Management service: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
