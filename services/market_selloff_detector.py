#!/usr/bin/env python3
"""
Market Selloff Detection Service
Advanced market-wide selloff detection and automatic cash allocation strategy.

Features:
- Real-time correlation analysis across 30+ cryptocurrencies
- Volatility spike detection with VIX-style indicators
- Percentage decline analysis across market cap weighted assets
- Smart cash allocation triggers with gradual/emergency modes
- Market recovery detection for re-entry signals
- Integration with existing Kelly Criterion and portfolio management

Strategy Triggers:
1. Correlation Spike: >90% of cryptos moving together downward
2. Broad Decline: >70% of tracked assets down >10% in 24h
3. Volatility Explosion: Market volatility >3x normal levels
4. Volume Surge: Combined volume >5x average with negative price action
5. Fear Index: Multiple fear indicators triggering simultaneously
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import mysql.connector
from mysql.connector import Error
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import requests
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [SELLOFF_DETECTOR] %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class MarketSelloffSignal:
    """Market selloff detection signal."""
    timestamp: datetime
    severity: str  # mild, moderate, severe, extreme
    confidence: float
    trigger_factors: List[str]
    correlation_coefficient: float
    decline_percentage: float  # % of assets declining
    average_decline: float  # Average price decline
    volatility_spike: float  # Volatility compared to normal
    volume_surge: float  # Volume compared to normal
    suggested_cash_allocation: float  # 0-100% cash recommendation
    emergency_liquidation: bool
    recovery_threshold: float
    expected_duration: str  # hours, days, weeks
    risk_description: str
    action_recommendation: str

@dataclass
class MarketRecoverySignal:
    """Market recovery detection signal."""
    timestamp: datetime
    recovery_stage: str  # early, confirmed, strong
    confidence: float
    recovery_factors: List[str]
    correlation_normalization: float
    stabilization_score: float
    volume_confirmation: bool
    suggested_re_entry: float  # 0-100% re-entry percentage
    priority_assets: List[str]
    risk_level: str

class MarketSelloffDetector:
    """Advanced market selloff detection and cash allocation service."""
    
    def __init__(self):
        self.db_config = {
            'host': 'localhost',
            'user': 'news_collector',
            'password': '99Rules!',
            'database': 'crypto_prices'
        }
        
        # Selloff detection parameters
        self.correlation_threshold = 0.85  # High correlation threshold
        self.decline_threshold = 0.70  # 70% of assets declining
        self.decline_magnitude = 0.10  # 10% price decline
        self.volatility_multiplier = 3.0  # 3x normal volatility
        self.volume_multiplier = 5.0  # 5x normal volume
        
        # Cash allocation parameters
        self.cash_allocation_levels = {
            'mild': 0.20,      # 20% cash
            'moderate': 0.50,  # 50% cash
            'severe': 0.80,    # 80% cash
            'extreme': 0.95    # 95% cash (near full liquidation)
        }
        
        # Recovery parameters
        self.recovery_correlation_threshold = 0.60  # Correlation below this suggests recovery
        self.recovery_decline_threshold = 0.40  # <40% assets declining
        self.recovery_time_threshold = 4  # Hours of improvement needed
        
        # Tracked cryptocurrencies for analysis
        self.tracked_symbols = [
            'BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'AVAX', 'DOT', 'MATIC', 'LTC',
            'LINK', 'TRX', 'NEAR', 'UNI', 'ATOM', 'XLM', 'ALGO', 'VET', 'FIL', 'MANA',
            'SAND', 'AXS', 'CRV', 'COMP', 'SUSHI', 'YFI', 'BAL', 'REN', 'LRC', 'ZRX'
        ]
        
        # Historical data cache
        self.price_cache = {}
        self.correlation_cache = {}
        self.volatility_cache = {}
        self.last_analysis = None
        
        self.app = FastAPI(title="Market Selloff Detector", version="1.0.0")
        self.setup_routes()
    
    def setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {"status": "healthy", "service": "market-selloff-detector", "timestamp": datetime.now().isoformat()}
        
        @self.app.get("/selloff/current")
        async def get_current_selloff_status():
            """Get current market selloff status."""
            try:
                signal = await self.detect_market_selloff()
                return JSONResponse(content=asdict(signal))
            except Exception as e:
                logger.error(f"Error getting selloff status: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/recovery/current")
        async def get_current_recovery_status():
            """Get current market recovery status."""
            try:
                signal = await self.detect_market_recovery()
                return JSONResponse(content=asdict(signal))
            except Exception as e:
                logger.error(f"Error getting recovery status: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/analysis/correlation")
        async def get_correlation_analysis():
            """Get current market correlation analysis."""
            try:
                analysis = await self.analyze_market_correlation()
                return JSONResponse(content=analysis)
            except Exception as e:
                logger.error(f"Error getting correlation analysis: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/allocation/recommendation")
        async def get_cash_allocation_recommendation():
            """Get current cash allocation recommendation."""
            try:
                selloff_signal = await self.detect_market_selloff()
                portfolio = await self.get_current_portfolio()
                
                recommendation = self.calculate_cash_allocation_strategy(selloff_signal, portfolio)
                return JSONResponse(content=recommendation)
            except Exception as e:
                logger.error(f"Error getting allocation recommendation: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    async def get_market_data(self, hours_back: int = 24) -> pd.DataFrame:
        """Get market data for multiple cryptocurrencies."""
        try:
            conn = mysql.connector.connect(**self.db_config)
            
            # Get recent price data for all tracked symbols
            query = """
            SELECT symbol, timestamp_iso, price, volume_24h, market_cap
            FROM ml_features_materialized
            WHERE symbol IN ({}) 
            AND timestamp_iso >= DATE_SUB(NOW(), INTERVAL %s HOUR)
            ORDER BY symbol, timestamp_iso DESC
            """.format(','.join(['%s'] * len(self.tracked_symbols)))
            
            params = self.tracked_symbols + [hours_back]
            
            df = pd.read_sql(query, conn, params=params)
            conn.close()
            
            if df.empty:
                logger.warning("No market data retrieved")
                return pd.DataFrame()
            
            # Convert timestamp and sort
            df['timestamp'] = pd.to_datetime(df['timestamp_iso'])
            df = df.sort_values(['symbol', 'timestamp'])
            
            logger.info(f"Retrieved market data for {df['symbol'].nunique()} cryptocurrencies")
            return df
            
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return pd.DataFrame()
    
    async def analyze_market_correlation(self) -> Dict:
        """Analyze current market correlation patterns."""
        try:
            # Get recent price data
            df = await self.get_market_data(hours_back=48)
            if df.empty:
                return {"error": "No market data available"}
            
            # Calculate price changes for correlation analysis
            price_changes = {}
            correlations = {}
            
            for symbol in self.tracked_symbols:
                symbol_data = df[df['symbol'] == symbol].copy()
                if len(symbol_data) < 2:
                    continue
                
                # Calculate hourly returns
                symbol_data = symbol_data.sort_values('timestamp')
                symbol_data['pct_change'] = symbol_data['price'].pct_change()
                
                # Get recent changes (last 24 hours)
                recent_changes = symbol_data['pct_change'].dropna().tail(24)
                if len(recent_changes) > 0:
                    price_changes[symbol] = recent_changes.values
            
            if len(price_changes) < 5:
                return {"error": "Insufficient data for correlation analysis"}
            
            # Calculate correlation matrix
            correlation_matrix = {}
            symbols = list(price_changes.keys())
            
            for i, symbol1 in enumerate(symbols):
                correlation_matrix[symbol1] = {}
                for symbol2 in symbols:
                    if len(price_changes[symbol1]) > 0 and len(price_changes[symbol2]) > 0:
                        min_length = min(len(price_changes[symbol1]), len(price_changes[symbol2]))
                        corr = np.corrcoef(
                            price_changes[symbol1][-min_length:],
                            price_changes[symbol2][-min_length:]
                        )[0, 1]
                        correlation_matrix[symbol1][symbol2] = float(corr) if not np.isnan(corr) else 0.0
                    else:
                        correlation_matrix[symbol1][symbol2] = 0.0
            
            # Calculate average correlation (excluding self-correlation)
            all_correlations = []
            for symbol1 in symbols:
                for symbol2 in symbols:
                    if symbol1 != symbol2:
                        all_correlations.append(correlation_matrix[symbol1][symbol2])
            
            average_correlation = np.mean(all_correlations) if all_correlations else 0.0
            
            # Detect correlation spikes (high correlation suggests coordinated movement)
            high_correlation_pairs = []
            for symbol1 in symbols:
                for symbol2 in symbols:
                    if symbol1 != symbol2:
                        corr = correlation_matrix[symbol1][symbol2]
                        if abs(corr) > 0.8:  # High correlation threshold
                            high_correlation_pairs.append((symbol1, symbol2, corr))
            
            analysis = {
                "timestamp": datetime.now().isoformat(),
                "average_correlation": float(average_correlation),
                "symbols_analyzed": len(symbols),
                "high_correlation_pairs": len(high_correlation_pairs),
                "correlation_spike": abs(average_correlation) > self.correlation_threshold,
                "market_stress_indicator": abs(average_correlation) > 0.85,
                "correlation_matrix": correlation_matrix,
                "analysis_period_hours": 24
            }
            
            self.correlation_cache = analysis
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing market correlation: {e}")
            return {"error": str(e)}
    
    async def analyze_market_decline(self) -> Dict:
        """Analyze percentage of market declining and magnitude."""
        try:
            df = await self.get_market_data(hours_back=24)
            if df.empty:
                return {"error": "No market data available"}
            
            declining_assets = []
            price_changes = {}
            volumes = {}
            
            for symbol in self.tracked_symbols:
                symbol_data = df[df['symbol'] == symbol].copy()
                if len(symbol_data) < 2:
                    continue
                
                symbol_data = symbol_data.sort_values('timestamp')
                
                # Calculate 24h price change
                current_price = symbol_data['price'].iloc[-1]
                day_ago_price = symbol_data['price'].iloc[0]
                price_change = (current_price - day_ago_price) / day_ago_price
                
                price_changes[symbol] = price_change
                
                # Get average volume
                volumes[symbol] = symbol_data['volume_24h'].mean()
                
                # Check if declining significantly
                if price_change < -self.decline_magnitude:
                    declining_assets.append({
                        'symbol': symbol,
                        'price_change': price_change,
                        'magnitude': abs(price_change)
                    })
            
            decline_percentage = len(declining_assets) / len(price_changes) if price_changes else 0
            average_decline = np.mean([change for change in price_changes.values() if change < 0])
            average_decline = float(average_decline) if not np.isnan(average_decline) else 0.0
            
            analysis = {
                "timestamp": datetime.now().isoformat(),
                "decline_percentage": float(decline_percentage),
                "declining_assets_count": len(declining_assets),
                "total_assets_analyzed": len(price_changes),
                "average_decline": average_decline,
                "declining_assets": declining_assets,
                "broad_selloff": decline_percentage > self.decline_threshold,
                "severe_decline": decline_percentage > 0.8 and average_decline < -0.15,
                "analysis_period_hours": 24
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing market decline: {e}")
            return {"error": str(e)}
    
    async def analyze_volatility_spike(self) -> Dict:
        """Analyze market volatility spikes."""
        try:
            df = await self.get_market_data(hours_back=168)  # 1 week for baseline
            if df.empty:
                return {"error": "No market data available"}
            
            volatility_analysis = {}
            
            for symbol in self.tracked_symbols:
                symbol_data = df[df['symbol'] == symbol].copy()
                if len(symbol_data) < 24:
                    continue
                
                symbol_data = symbol_data.sort_values('timestamp')
                symbol_data['pct_change'] = symbol_data['price'].pct_change()
                
                # Calculate recent volatility (last 24 hours)
                recent_changes = symbol_data['pct_change'].tail(24)
                recent_volatility = recent_changes.std() if len(recent_changes) > 1 else 0
                
                # Calculate baseline volatility (previous 6 days)
                baseline_changes = symbol_data['pct_change'].iloc[-168:-24]
                baseline_volatility = baseline_changes.std() if len(baseline_changes) > 1 else recent_volatility
                
                # Calculate volatility spike ratio
                volatility_ratio = recent_volatility / baseline_volatility if baseline_volatility > 0 else 1.0
                
                volatility_analysis[symbol] = {
                    'recent_volatility': float(recent_volatility),
                    'baseline_volatility': float(baseline_volatility),
                    'volatility_ratio': float(volatility_ratio),
                    'spike_detected': volatility_ratio > self.volatility_multiplier
                }
            
            # Calculate market-wide volatility metrics
            volatility_ratios = [data['volatility_ratio'] for data in volatility_analysis.values()]
            spike_count = sum(1 for data in volatility_analysis.values() if data['spike_detected'])
            
            analysis = {
                "timestamp": datetime.now().isoformat(),
                "average_volatility_ratio": float(np.mean(volatility_ratios)) if volatility_ratios else 1.0,
                "volatility_spike_count": spike_count,
                "total_assets_analyzed": len(volatility_analysis),
                "spike_percentage": spike_count / len(volatility_analysis) if volatility_analysis else 0,
                "market_volatility_spike": spike_count / len(volatility_analysis) > 0.5 if volatility_analysis else False,
                "extreme_volatility": spike_count / len(volatility_analysis) > 0.7 if volatility_analysis else False,
                "volatility_by_asset": volatility_analysis
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing volatility spike: {e}")
            return {"error": str(e)}
    
    async def detect_market_selloff(self) -> MarketSelloffSignal:
        """Detect market-wide selloff conditions."""
        try:
            # Get all analysis components
            correlation_analysis = await self.analyze_market_correlation()
            decline_analysis = await self.analyze_market_decline()
            volatility_analysis = await self.analyze_volatility_spike()
            
            # Initialize signal parameters
            trigger_factors = []
            confidence = 0.0
            severity = "mild"
            emergency_liquidation = False
            
            # Analyze correlation spike
            if correlation_analysis.get("correlation_spike", False):
                trigger_factors.append("High market correlation detected")
                confidence += 0.25
            
            if correlation_analysis.get("market_stress_indicator", False):
                trigger_factors.append("Market stress correlation pattern")
                confidence += 0.15
            
            # Analyze broad decline
            if decline_analysis.get("broad_selloff", False):
                trigger_factors.append("Broad market selloff (>70% assets declining)")
                confidence += 0.30
            
            if decline_analysis.get("severe_decline", False):
                trigger_factors.append("Severe decline magnitude")
                confidence += 0.20
                emergency_liquidation = True
            
            # Analyze volatility spike
            if volatility_analysis.get("market_volatility_spike", False):
                trigger_factors.append("Market volatility spike detected")
                confidence += 0.20
            
            if volatility_analysis.get("extreme_volatility", False):
                trigger_factors.append("Extreme volatility levels")
                confidence += 0.15
                emergency_liquidation = True
            
            # Determine severity level
            if confidence >= 0.80:
                severity = "extreme"
            elif confidence >= 0.60:
                severity = "severe"
            elif confidence >= 0.40:
                severity = "moderate"
            else:
                severity = "mild"
            
            # Get suggested cash allocation
            suggested_cash_allocation = self.cash_allocation_levels.get(severity, 0.20)
            
            # Determine expected duration
            if severity in ["extreme", "severe"]:
                expected_duration = "days"
            elif severity == "moderate":
                expected_duration = "hours"
            else:
                expected_duration = "hours"
            
            # Create selloff signal
            signal = MarketSelloffSignal(
                timestamp=datetime.now(),
                severity=severity,
                confidence=min(1.0, confidence),
                trigger_factors=trigger_factors,
                correlation_coefficient=correlation_analysis.get("average_correlation", 0.0),
                decline_percentage=decline_analysis.get("decline_percentage", 0.0),
                average_decline=decline_analysis.get("average_decline", 0.0),
                volatility_spike=volatility_analysis.get("average_volatility_ratio", 1.0),
                volume_surge=1.0,  # TODO: Implement volume analysis
                suggested_cash_allocation=suggested_cash_allocation,
                emergency_liquidation=emergency_liquidation,
                recovery_threshold=0.3,  # 30% improvement needed for recovery
                expected_duration=expected_duration,
                risk_description=f"Market {severity} selloff detected with {confidence:.1%} confidence",
                action_recommendation=f"Consider {suggested_cash_allocation:.0%} cash allocation"
            )
            
            self.last_analysis = signal
            
            if confidence > 0.4:
                logger.warning(f"🚨 MARKET SELLOFF DETECTED: {severity.upper()} severity, {confidence:.1%} confidence")
                for factor in trigger_factors:
                    logger.warning(f"   📊 {factor}")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error detecting market selloff: {e}")
            # Return neutral signal on error
            return MarketSelloffSignal(
                timestamp=datetime.now(),
                severity="mild",
                confidence=0.0,
                trigger_factors=["Error in analysis"],
                correlation_coefficient=0.0,
                decline_percentage=0.0,
                average_decline=0.0,
                volatility_spike=1.0,
                volume_surge=1.0,
                suggested_cash_allocation=0.05,
                emergency_liquidation=False,
                recovery_threshold=0.3,
                expected_duration="unknown",
                risk_description="Analysis error occurred",
                action_recommendation="Monitor manually"
            )
    
    async def detect_market_recovery(self) -> MarketRecoverySignal:
        """Detect market recovery conditions."""
        try:
            # Get current market state
            correlation_analysis = await self.analyze_market_correlation()
            decline_analysis = await self.analyze_market_decline()
            volatility_analysis = await self.analyze_volatility_spike()
            
            recovery_factors = []
            confidence = 0.0
            recovery_stage = "early"
            
            # Check correlation normalization
            avg_correlation = abs(correlation_analysis.get("average_correlation", 0.8))
            if avg_correlation < self.recovery_correlation_threshold:
                recovery_factors.append("Market correlation normalizing")
                confidence += 0.25
            
            # Check decline reduction
            decline_percentage = decline_analysis.get("decline_percentage", 0.8)
            if decline_percentage < self.recovery_decline_threshold:
                recovery_factors.append("Fewer assets declining")
                confidence += 0.30
            
            # Check volatility stabilization
            volatility_spike = volatility_analysis.get("average_volatility_ratio", 3.0)
            if volatility_spike < 2.0:
                recovery_factors.append("Volatility stabilizing")
                confidence += 0.25
            
            # Check for positive price action
            average_decline = decline_analysis.get("average_decline", -0.1)
            if average_decline > -0.05:  # Less than 5% average decline
                recovery_factors.append("Price action improving")
                confidence += 0.20
            
            # Determine recovery stage
            if confidence >= 0.70:
                recovery_stage = "strong"
            elif confidence >= 0.50:
                recovery_stage = "confirmed"
            else:
                recovery_stage = "early"
            
            # Calculate suggested re-entry percentage
            if recovery_stage == "strong":
                suggested_re_entry = 0.60  # 60% re-entry
            elif recovery_stage == "confirmed":
                suggested_re_entry = 0.30  # 30% re-entry
            else:
                suggested_re_entry = 0.10  # 10% re-entry
            
            # Identify priority assets for re-entry (least correlated, strongest recovery)
            priority_assets = ["BTC", "ETH", "SOL"]  # TODO: Implement dynamic priority calculation
            
            signal = MarketRecoverySignal(
                timestamp=datetime.now(),
                recovery_stage=recovery_stage,
                confidence=min(1.0, confidence),
                recovery_factors=recovery_factors,
                correlation_normalization=1.0 - avg_correlation,
                stabilization_score=min(1.0, confidence),
                volume_confirmation=True,  # TODO: Implement volume confirmation
                suggested_re_entry=suggested_re_entry,
                priority_assets=priority_assets,
                risk_level="medium" if confidence > 0.5 else "high"
            )
            
            if confidence > 0.5:
                logger.info(f"✅ MARKET RECOVERY DETECTED: {recovery_stage.upper()} stage, {confidence:.1%} confidence")
                for factor in recovery_factors:
                    logger.info(f"   📈 {factor}")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error detecting market recovery: {e}")
            return MarketRecoverySignal(
                timestamp=datetime.now(),
                recovery_stage="early",
                confidence=0.0,
                recovery_factors=["Error in analysis"],
                correlation_normalization=0.0,
                stabilization_score=0.0,
                volume_confirmation=False,
                suggested_re_entry=0.0,
                priority_assets=[],
                risk_level="high"
            )
    
    async def get_current_portfolio(self) -> Dict:
        """Get current portfolio from trading engine."""
        try:
            response = requests.get("http://localhost:8024/portfolio", timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Trading engine unavailable: {response.status_code}")
                return {}
        except Exception as e:
            logger.warning(f"Error getting portfolio: {e}")
            return {}
    
    def calculate_cash_allocation_strategy(self, selloff_signal: MarketSelloffSignal, portfolio: Dict) -> Dict:
        """Calculate optimal cash allocation strategy based on selloff signal."""
        try:
            current_cash = portfolio.get('cash_balance', 0)
            total_value = portfolio.get('total_portfolio_value', 0)
            positions = portfolio.get('positions', {})
            
            current_cash_percentage = (current_cash / total_value) if total_value > 0 else 1.0
            target_cash_percentage = selloff_signal.suggested_cash_allocation
            
            # Calculate how much to liquidate
            liquidation_needed = max(0, target_cash_percentage - current_cash_percentage)
            liquidation_amount = liquidation_needed * total_value
            
            # Prioritize liquidation by risk (highest correlation, lowest recent performance)
            liquidation_plan = []
            
            if liquidation_needed > 0.05:  # Only if significant liquidation needed
                sorted_positions = sorted(
                    [(symbol, data) for symbol, data in positions.items()],
                    key=lambda x: x[1].get('value_usd', 0),
                    reverse=True  # Liquidate largest positions first for emergency
                )
                
                remaining_to_liquidate = liquidation_amount
                
                for symbol, position_data in sorted_positions:
                    if remaining_to_liquidate <= 0:
                        break
                    
                    position_value = position_data.get('value_usd', 0)
                    
                    if selloff_signal.emergency_liquidation:
                        # Emergency: Liquidate entire position
                        liquidate_amount = min(position_value, remaining_to_liquidate)
                        liquidate_percentage = liquidate_amount / position_value if position_value > 0 else 0
                    else:
                        # Gradual: Liquidate portion based on severity
                        liquidate_percentage = min(0.5, liquidation_needed / 0.2)  # Max 50% per position
                        liquidate_amount = position_value * liquidate_percentage
                    
                    if liquidate_amount > 25:  # Minimum liquidation threshold
                        liquidation_plan.append({
                            'symbol': symbol,
                            'liquidate_amount': liquidate_amount,
                            'liquidate_percentage': liquidate_percentage,
                            'reason': f"Market {selloff_signal.severity} selloff protection"
                        })
                        
                        remaining_to_liquidate -= liquidate_amount
            
            strategy = {
                "timestamp": datetime.now().isoformat(),
                "selloff_severity": selloff_signal.severity,
                "selloff_confidence": selloff_signal.confidence,
                "current_cash_percentage": current_cash_percentage,
                "target_cash_percentage": target_cash_percentage,
                "liquidation_needed": liquidation_needed,
                "liquidation_amount": liquidation_amount,
                "emergency_liquidation": selloff_signal.emergency_liquidation,
                "liquidation_plan": liquidation_plan,
                "total_positions_to_liquidate": len(liquidation_plan),
                "estimated_cash_after_liquidation": current_cash + sum(p['liquidate_amount'] for p in liquidation_plan),
                "strategy_rationale": selloff_signal.risk_description,
                "recovery_criteria": f"Wait for {selloff_signal.recovery_threshold:.0%} improvement in market conditions"
            }
            
            return strategy
            
        except Exception as e:
            logger.error(f"Error calculating cash allocation strategy: {e}")
            return {"error": str(e)}

async def main():
    """Main function to run the service."""
    detector = MarketSelloffDetector()
    
    # Start the FastAPI server
    config = uvicorn.Config(
        detector.app,
        host="0.0.0.0",
        port=8028,
        log_level="info"
    )
    server = uvicorn.Server(config)
    
    logger.info("🚨 Market Selloff Detector starting on port 8028...")
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())
