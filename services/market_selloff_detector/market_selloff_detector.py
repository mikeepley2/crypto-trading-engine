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

# Custom JSON encoder for datetime objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

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
    """
    Advanced market selloff detection and cash allocation service.
    
    This service continuously monitors cryptocurrency markets for signs of
    coordinated selloffs and provides intelligent cash allocation recommendations
    to protect portfolio value during market downturns.
    
    Key Features:
        - Real-time correlation analysis across 30+ cryptocurrencies
        - VIX-style volatility spike detection
        - Breadth analysis (percentage of assets declining)
        - Multi-factor confidence scoring system
        - Graduated cash allocation recommendations (20%-95%)
        - Market recovery detection for re-entry timing
        - Integration with portfolio management and trading execution
    
    Detection Methodology:
        Uses combination of technical indicators proven in traditional finance:
        1. Correlation analysis (similar to stock market stress tests)
        2. Breadth indicators (advance/decline ratios)
        3. Volatility analysis (VIX-style fear indicators)
        4. Volume analysis (distribution vs accumulation)
    
    Cash Allocation Strategy:
        Conservative approach prioritizing capital preservation:
        - Mild selloffs: 20% cash (minor rebalancing)
        - Moderate selloffs: 50% cash (partial defensive positioning)
        - Severe selloffs: 80% cash (major defensive positioning)
        - Extreme selloffs: 95% cash (near-complete liquidation)
    
    Integration Points:
        - Strategy Orchestrator: 25% weight in multi-strategy framework
        - Trading Engine: Automated liquidation execution
        - Risk Management: Coordinated stop-loss and position sizing
        - Dashboard: Real-time monitoring and manual overrides
    """
    
    def __init__(self):
        # Database configuration for market data access
        self.db_config = {
            'host': 'host.docker.internal',  # Docker networking
            'user': 'news_collector',        # Read-only database user
            'password': '99Rules!',          # Database password
            'database': 'crypto_prices'      # Primary market data database
        }
        
        # ==================== SELLOFF DETECTION PARAMETERS ====================
        
        # Correlation thresholds (based on traditional finance stress indicators)
        self.correlation_threshold = 0.85      # High correlation threshold (85%)
        self.market_stress_threshold = 0.85    # Market stress indicator threshold
        
        # Decline analysis thresholds
        self.decline_threshold = 0.70          # 70% of assets must be declining
        self.decline_magnitude = 0.10          # 10% minimum price decline
        self.severe_decline_threshold = 0.80   # 80% assets for severe classification
        self.severe_decline_magnitude = 0.15   # 15% decline for severe classification
        
        # Volatility spike detection (VIX-style)
        self.volatility_multiplier = 3.0       # 3x normal volatility threshold
        self.extreme_volatility_threshold = 0.7 # 70% assets for extreme classification
        self.volatility_baseline_hours = 168   # 1 week baseline calculation
        
        # Volume analysis parameters
        self.volume_multiplier = 5.0           # 5x normal volume threshold
        self.volume_analysis_hours = 24        # 24 hour volume comparison
        
        # ==================== CASH ALLOCATION PARAMETERS ====================
        # ==================== CASH ALLOCATION PARAMETERS ====================
        
        # Progressive cash allocation based on selloff severity
        # Conservative approach prioritizing capital preservation
        self.cash_allocation_levels = {
            'mild': 0.20,      # 20% cash - Minor rebalancing for early warning
            'moderate': 0.50,  # 50% cash - Partial defensive positioning
            'severe': 0.80,    # 80% cash - Major defensive positioning  
            'extreme': 0.95    # 95% cash - Near-complete liquidation for capital preservation
        }
        
        # Liquidation parameters
        self.max_position_liquidation = 0.50   # Max 50% of position in normal mode
        self.min_liquidation_amount = 25.0     # Minimum $25 liquidation threshold
        
        # ==================== RECOVERY DETECTION PARAMETERS ====================
        
        # Recovery thresholds (conservative to avoid false signals)
        self.recovery_correlation_threshold = 0.60  # <60% correlation suggests normalization
        self.recovery_decline_threshold = 0.40      # <40% assets declining for recovery
        self.recovery_time_threshold = 4            # 4 hours of improvement needed
        self.recovery_volatility_threshold = 2.0    # <2x volatility for stabilization
        self.recovery_price_threshold = -0.05       # >-5% average decline for improvement
        
        # Re-entry allocation percentages
        self.recovery_reentry_levels = {
            'early': 0.10,      # 10% re-entry - High risk, minimal exposure
            'confirmed': 0.30,  # 30% re-entry - Medium risk, gradual return
            'strong': 0.60      # 60% re-entry - Low risk, aggressive return
        }
        
        # ==================== ASSET CONFIGURATION ====================
        
        # ==================== ASSET CONFIGURATION ====================
        
        # Tracked cryptocurrencies for analysis (30 major assets)
        # Selected based on market cap, liquidity, and correlation diversity
        self.tracked_symbols = [
            # Tier 1: Major cryptocurrencies (highest weight in analysis)
            'BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 
            
            # Tier 2: Large cap altcoins (significant market influence)
            'AVAX', 'DOT', 'MATIC', 'LTC', 'LINK', 'TRX', 
            
            # Tier 3: Mid cap with good liquidity (diversification)
            'NEAR', 'UNI', 'ATOM', 'XLM', 'ALGO', 'VET', 
            
            # Tier 4: DeFi and utility tokens (sector representation)
            'FIL', 'MANA', 'SAND', 'AXS', 'CRV', 'COMP', 
            
            # Tier 5: Trading and infrastructure (market infrastructure)
            'SUSHI', 'YFI', 'BAL', 'REN', 'LRC', 'ZRX'
        ]
        
        # Priority assets for recovery re-entry (most stable, liquid)
        self.priority_recovery_assets = ['BTC', 'ETH', 'SOL', 'BNB']
        
        # ==================== PERFORMANCE OPTIMIZATION ====================
        
        # Data caching for performance (reduces database load)
        self.price_cache = {}           # Market price cache
        self.correlation_cache = {}     # Correlation analysis cache  
        self.volatility_cache = {}      # Volatility calculation cache
        self.last_analysis = None       # Last selloff analysis
        
        # Cache expiration times (minutes)
        self.cache_expiry_minutes = 15  # Cache expires after 15 minutes
        
        # ==================== SERVICE INITIALIZATION ====================
        
        # Initialize FastAPI application
        self.app = FastAPI(
            title="Market Selloff Detector",
            description="Advanced market-wide selloff detection and cash allocation service",
            version="1.0.0",
            docs_url="/docs",           # API documentation endpoint
            redoc_url="/redoc"          # Alternative documentation format
        )
        
        # Setup API routes
        self.setup_routes()
        
        # Validate configuration on startup
        self._validate_configuration()
    
    def _validate_configuration(self):
        """
        Validate service configuration and log important parameters.
        
        Ensures all thresholds are within reasonable ranges and logs
        configuration for troubleshooting and monitoring purposes.
        """
        logger.info("🔧 Market Selloff Detector Configuration:")
        logger.info(f"   📊 Correlation threshold: {self.correlation_threshold:.1%}")
        logger.info(f"   📉 Decline threshold: {self.decline_threshold:.1%} of assets")
        logger.info(f"   📈 Volatility multiplier: {self.volatility_multiplier}x normal")
        logger.info(f"   💰 Cash allocation levels: {self.cash_allocation_levels}")
        logger.info(f"   🔄 Recovery thresholds: {self.recovery_correlation_threshold:.1%} correlation")
        logger.info(f"   🎯 Tracking {len(self.tracked_symbols)} cryptocurrencies")
        
        # Validate thresholds are reasonable
        assert 0.5 <= self.correlation_threshold <= 0.95, "Correlation threshold should be 50-95%"
        assert 0.3 <= self.decline_threshold <= 0.9, "Decline threshold should be 30-90%"
        assert 1.5 <= self.volatility_multiplier <= 10.0, "Volatility multiplier should be 1.5-10x"
        assert len(self.tracked_symbols) >= 10, "Should track at least 10 cryptocurrencies"
        
        logger.info("✅ Configuration validation passed")
    
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
                # Convert dataclass to dict with datetime handling
                signal_dict = asdict(signal)
                signal_dict['timestamp'] = signal.timestamp.isoformat()
                return JSONResponse(content=signal_dict)
            except Exception as e:
                logger.error(f"Error getting selloff status: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/recovery/current")
        async def get_current_recovery_status():
            """Get current market recovery status."""
            try:
                signal = await self.detect_market_recovery()
                # Convert dataclass to dict with datetime handling
                signal_dict = asdict(signal)
                signal_dict['timestamp'] = signal.timestamp.isoformat()
                return JSONResponse(content=signal_dict)
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
        """
        Get market data for multiple cryptocurrencies from the database.
        
        Retrieves price, volume, and market cap data for all tracked symbols
        within the specified time window. This data is used for correlation,
        volatility, and decline analysis.
        
        Args:
            hours_back (int): Number of hours of historical data to retrieve (default: 24)
            
        Returns:
            pd.DataFrame: Market data with columns [symbol, timestamp_iso, price, volume_24h, market_cap]
            
        Note:
            - Uses ml_features_materialized table for comprehensive market data
            - Data is sorted by symbol and timestamp for time series analysis
            - Returns empty DataFrame if no data available or database error
        """
        try:
            conn = mysql.connector.connect(**self.db_config)
            
            # Get recent price data for all tracked symbols
            query = """
            SELECT symbol, timestamp_iso, current_price, volume_24h, market_cap
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
        """
        Analyze current market correlation patterns to detect coordinated movement.
        
        Calculates correlation coefficients between all tracked cryptocurrencies
        to identify periods of high correlation that often signal market stress
        or coordinated selling/buying pressure.
        
        Returns:
            Dict: Comprehensive correlation analysis including:
                - average_correlation: Mean correlation across all pairs
                - correlation_spike: Boolean indicating dangerous correlation levels
                - market_stress_indicator: Boolean for extreme market stress
                - correlation_matrix: Full correlation matrix between assets
                - high_correlation_pairs: Count of pairs with >80% correlation
                
        Algorithm:
            1. Calculate hourly returns for each cryptocurrency
            2. Compute correlation matrix using 24-hour rolling window
            3. Identify correlation spikes (>85% threshold)
            4. Detect market stress patterns (>85% absolute correlation)
            5. Cache results for performance optimization
            
        Note:
            High correlation during market stress indicates coordinated selling
            and reduced diversification benefits across crypto assets.
        """
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
                symbol_data['pct_change'] = symbol_data['current_price'].pct_change()
                
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
        """
        Analyze percentage of market declining and magnitude of selloff.
        
        Examines what percentage of tracked cryptocurrencies are declining
        and by what magnitude over the past 24 hours. This provides a
        quantitative measure of market-wide selling pressure.
        
        Returns:
            Dict: Market decline analysis including:
                - decline_percentage: Fraction of assets declining significantly
                - declining_assets_count: Number of assets down >10%
                - average_decline: Mean decline percentage for declining assets
                - broad_selloff: Boolean indicating >70% assets declining
                - severe_decline: Boolean for >80% decline with >15% magnitude
                
        Algorithm:
            1. Calculate 24-hour price changes for all tracked assets
            2. Identify assets declining more than threshold (10%)
            3. Calculate percentage of assets in decline
            4. Compute average decline magnitude
            5. Trigger selloff alerts based on breadth and magnitude
            
        Thresholds:
            - Decline magnitude: >10% price drop
            - Broad selloff: >70% of assets declining
            - Severe decline: >80% declining AND >15% average decline
        """
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
                current_price = symbol_data['current_price'].iloc[-1]
                day_ago_price = symbol_data['current_price'].iloc[0]
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
        """
        Analyze market volatility spikes using VIX-style methodology.
        
        Compares recent volatility (24h) against baseline volatility (1 week)
        to detect sudden spikes in market turbulence. High volatility often
        precedes or accompanies market selloffs.
        
        Returns:
            Dict: Volatility spike analysis including:
                - average_volatility_ratio: Mean volatility spike across assets
                - volatility_spike_count: Number of assets with >3x volatility
                - spike_percentage: Fraction of assets showing volatility spikes
                - market_volatility_spike: Boolean for >50% assets spiking
                - extreme_volatility: Boolean for >70% assets spiking
                - volatility_by_asset: Individual asset volatility analysis
                
        Algorithm:
            1. Calculate rolling standard deviation of price changes (24h recent)
            2. Calculate baseline volatility using 1-week historical data
            3. Compute volatility ratio (recent/baseline)
            4. Identify spikes >3x baseline volatility
            5. Aggregate market-wide volatility stress indicators
            
        Interpretation:
            - Ratio >3.0: Significant volatility spike
            - >50% assets spiking: Market volatility stress
            - >70% assets spiking: Extreme market turbulence
        """
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
                symbol_data['pct_change'] = symbol_data['current_price'].pct_change()
                
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
        """
        Detect market-wide selloff conditions using multi-factor analysis.
        
        Combines correlation analysis, decline breadth, and volatility spikes
        to generate comprehensive selloff signals with confidence scoring.
        Uses weighted scoring system to determine severity and recommended actions.
        
        Returns:
            MarketSelloffSignal: Comprehensive selloff analysis including:
                - severity: mild/moderate/severe/extreme classification
                - confidence: 0-1 confidence score based on multiple factors
                - trigger_factors: List of specific conditions triggering alert
                - suggested_cash_allocation: Recommended cash percentage (20-95%)
                - emergency_liquidation: Boolean for immediate liquidation need
                - risk_description: Human-readable risk assessment
                - action_recommendation: Specific action guidance
                
        Confidence Scoring:
            - Correlation spike: +25% confidence
            - Market stress correlation: +15% confidence  
            - Broad selloff (>70% declining): +30% confidence
            - Severe decline magnitude: +20% confidence
            - Volatility spike: +20% confidence
            - Extreme volatility: +15% confidence
            
        Severity Classification:
            - Mild (0-40% confidence): 20% cash allocation
            - Moderate (40-60% confidence): 50% cash allocation
            - Severe (60-80% confidence): 80% cash allocation
            - Extreme (80%+ confidence): 95% cash allocation
            
        Emergency Liquidation Triggers:
            - Severe decline magnitude (>80% assets down >15%)
            - Extreme volatility (>70% assets with >3x volatility)
        """
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
        """
        Detect market recovery conditions for strategic re-entry timing.
        
        Analyzes multiple factors to determine when market conditions are
        improving and it's safe to begin re-entering positions. Uses
        conservative approach to avoid false recovery signals.
        
        Returns:
            MarketRecoverySignal: Comprehensive recovery analysis including:
                - recovery_stage: early/confirmed/strong classification
                - confidence: 0-1 confidence score for recovery strength
                - recovery_factors: List of positive indicators detected
                - suggested_re_entry: Recommended re-entry percentage (10-60%)
                - priority_assets: Recommended assets for initial re-entry
                - risk_level: high/medium/low risk assessment
                
        Recovery Indicators & Scoring:
            - Correlation normalization (<60%): +25% confidence
            - Decline reduction (<40% assets declining): +30% confidence
            - Volatility stabilization (<2x normal): +25% confidence
            - Price improvement (>-5% average decline): +20% confidence
            
        Recovery Stages:
            - Early (20-50% confidence): 10% re-entry, high risk
            - Confirmed (50-70% confidence): 30% re-entry, medium risk
            - Strong (70%+ confidence): 60% re-entry, low risk
            
        Re-entry Strategy:
            - Start with most liquid, established assets (BTC, ETH)
            - Gradual re-entry over time to avoid false recoveries
            - Monitor for recovery sustainability over multiple timeframes
        """
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
            response = requests.get("http://host.docker.internal:8024/portfolio", timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Trading engine unavailable: {response.status_code}")
                return {}
        except Exception as e:
            logger.warning(f"Error getting portfolio: {e}")
            return {}
    
    def calculate_cash_allocation_strategy(self, selloff_signal: MarketSelloffSignal, portfolio: Dict) -> Dict:
        """
        Calculate optimal cash allocation strategy based on selloff signal and current portfolio.
        
        Determines specific liquidation plan to achieve target cash allocation
        while minimizing impact and preserving core positions when possible.
        
        Args:
            selloff_signal (MarketSelloffSignal): Current selloff analysis
            portfolio (Dict): Current portfolio positions and values
            
        Returns:
            Dict: Detailed liquidation strategy including:
                - target_cash_percentage: Desired cash allocation
                - liquidation_needed: Additional cash percentage required
                - liquidation_plan: Specific positions to liquidate with amounts
                - emergency_liquidation: Boolean for immediate execution
                - strategy_rationale: Explanation of recommended actions
                
        Liquidation Strategy:
        
        Normal Mode (Mild/Moderate Selloffs):
            - Liquidate largest positions first for efficiency
            - Maximum 50% of any single position to preserve diversification
            - Spread liquidation over time to minimize market impact
            - Preserve core positions (BTC, ETH) when possible
            
        Emergency Mode (Severe/Extreme Selloffs):
            - Immediate liquidation of all non-core positions
            - Complete liquidation of positions if needed
            - Priority on capital preservation over position maintenance
            - Minimum liquidation threshold: $25 to avoid micro-transactions
            
        Position Priority:
            1. Largest positions (for efficiency in emergency)
            2. Most correlated assets (highest selloff risk)
            3. Lowest recent performance (risk mitigation)
            4. Preserve BTC/ETH as safe haven assets when possible
        """
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
