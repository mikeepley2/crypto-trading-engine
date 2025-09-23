#!/usr/bin/env python3
"""
Market Regime Detection Service
Detects bull/bear/sideways market conditions using volatility and momentum indicators
Adjusts trading strategy based on market regime
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
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import requests
from collections import defaultdict, deque
import talib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Market Regime Detection Service", version="1.0.0")

class MarketRegimeRequest(BaseModel):
    symbols: List[str] = ['BTC', 'ETH', 'ADA']
    lookback_days: int = 30

class MarketRegimeResponse(BaseModel):
    overall_regime: str
    regime_confidence: float
    regime_strength: float
    symbol_regimes: Dict[str, Dict]
    market_indicators: Dict
    regime_duration_days: int
    timestamp: str

class MarketRegimeDetector:
    def __init__(self):
        self.db_config = {
            'host': os.environ.get('DATABASE_HOST', 'host.docker.internal'),
            'user': os.environ.get('DATABASE_USER', 'news_collector'),
            'password': os.environ.get('DATABASE_PASSWORD', '99Rules!'),
            'database': os.environ.get('DATABASE_NAME', 'crypto_prices'),
            'port': int(os.environ.get('DATABASE_PORT', 3306))
        }
        
        # Regime detection parameters
        self.trend_threshold = 0.05  # 5% for trend classification
        self.volatility_threshold = 0.02  # 2% daily volatility threshold
        self.momentum_lookback = 14  # Days for momentum calculation
        
        # Regime history for stability
        self.regime_history = deque(maxlen=10)
        self.current_regime = 'SIDEWAYS'
        self.regime_start_date = datetime.now()
        
        # Market indicators cache
        self.indicators_cache = {}
        self.cache_timestamp = None
        self.cache_duration = 300  # 5 minutes
        
        logger.info("📊 Market Regime Detection initialized")
    
    async def get_market_data(self, symbols: List[str], lookback_days: int = 30) -> Dict[str, pd.DataFrame]:
        """Get market data for multiple symbols"""
        try:
            conn = mysql.connector.connect(**self.db_config)
            
            market_data = {}
            
            for symbol in symbols:
                query = """
                SELECT 
                    timestamp_iso,
                    current_price,
                    volume_24h,
                    price_change_percentage_24h,
                    market_cap_usd
                FROM ml_features_materialized 
                WHERE symbol = %s 
                AND timestamp_iso >= DATE_SUB(NOW(), INTERVAL %s DAY)
                ORDER BY timestamp_iso ASC
                """
                
                df = pd.read_sql(query, conn, params=[symbol, lookback_days])
                
                if not df.empty:
                    df['timestamp_iso'] = pd.to_datetime(df['timestamp_iso'])
                    df = df.set_index('timestamp_iso')
                    market_data[symbol] = df
                
            conn.close()
            return market_data
            
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return {}
    
    def calculate_trend_indicators(self, price_series: pd.Series) -> Dict:
        """Calculate trend indicators for regime detection"""
        try:
            if len(price_series) < 20:
                return {}
            
            prices = price_series.values
            
            # Moving averages
            sma_20 = talib.SMA(prices, timeperiod=20)
            sma_50 = talib.SMA(prices, timeperiod=min(50, len(prices)-1))
            ema_12 = talib.EMA(prices, timeperiod=12)
            ema_26 = talib.EMA(prices, timeperiod=26)
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(prices, fastperiod=12, slowperiod=26, signalperiod=9)
            
            # ADX (trend strength)
            high = prices  # Simplified - using close as high/low
            low = prices
            adx = talib.ADX(high, low, prices, timeperiod=14)
            
            # Rate of Change
            roc = talib.ROC(prices, timeperiod=10)
            
            # Linear regression slope (trend direction)
            x = np.arange(len(prices))
            if len(prices) > 1:
                slope, intercept = np.polyfit(x[-20:], prices[-20:], 1)  # Last 20 periods
                trend_slope = slope / prices[-1] * 100  # Percentage slope
            else:
                trend_slope = 0
            
            return {
                'current_price': prices[-1],
                'sma_20': sma_20[-1] if not np.isnan(sma_20[-1]) else prices[-1],
                'sma_50': sma_50[-1] if not np.isnan(sma_50[-1]) else prices[-1],
                'ema_12': ema_12[-1] if not np.isnan(ema_12[-1]) else prices[-1],
                'ema_26': ema_26[-1] if not np.isnan(ema_26[-1]) else prices[-1],
                'macd': macd[-1] if not np.isnan(macd[-1]) else 0,
                'macd_signal': macd_signal[-1] if not np.isnan(macd_signal[-1]) else 0,
                'adx': adx[-1] if not np.isnan(adx[-1]) else 25,
                'roc': roc[-1] if not np.isnan(roc[-1]) else 0,
                'trend_slope': trend_slope
            }
            
        except Exception as e:
            logger.error(f"Error calculating trend indicators: {e}")
            return {}
    
    def calculate_volatility_indicators(self, price_series: pd.Series) -> Dict:
        """Calculate volatility indicators"""
        try:
            if len(price_series) < 10:
                return {}
            
            prices = price_series.values
            returns = np.diff(np.log(prices))
            
            # Volatility metrics
            daily_volatility = np.std(returns) * 100
            rolling_volatility = pd.Series(returns).rolling(window=10).std().iloc[-1] * 100
            
            # ATR (Average True Range) - simplified
            atr = talib.ATR(prices, prices, prices, timeperiod=14)
            atr_percent = (atr[-1] / prices[-1] * 100) if not np.isnan(atr[-1]) else daily_volatility
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(prices, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            bb_width = ((bb_upper[-1] - bb_lower[-1]) / bb_middle[-1] * 100) if not np.isnan(bb_upper[-1]) else 0
            
            return {
                'daily_volatility': daily_volatility,
                'rolling_volatility': rolling_volatility if not np.isnan(rolling_volatility) else daily_volatility,
                'atr_percent': atr_percent,
                'bb_width': bb_width
            }
            
        except Exception as e:
            logger.error(f"Error calculating volatility indicators: {e}")
            return {}
    
    def calculate_momentum_indicators(self, price_series: pd.Series) -> Dict:
        """Calculate momentum indicators"""
        try:
            if len(price_series) < 14:
                return {}
            
            prices = price_series.values
            
            # RSI
            rsi = talib.RSI(prices, timeperiod=14)
            
            # Stochastic
            stoch_k, stoch_d = talib.STOCH(prices, prices, prices, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
            
            # Williams %R
            willr = talib.WILLR(prices, prices, prices, timeperiod=14)
            
            # Commodity Channel Index
            cci = talib.CCI(prices, prices, prices, timeperiod=14)
            
            # Price momentum (simple)
            momentum_5d = (prices[-1] / prices[-6] - 1) * 100 if len(prices) > 5 else 0
            momentum_10d = (prices[-1] / prices[-11] - 1) * 100 if len(prices) > 10 else 0
            
            return {
                'rsi': rsi[-1] if not np.isnan(rsi[-1]) else 50,
                'stoch_k': stoch_k[-1] if not np.isnan(stoch_k[-1]) else 50,
                'stoch_d': stoch_d[-1] if not np.isnan(stoch_d[-1]) else 50,
                'willr': willr[-1] if not np.isnan(willr[-1]) else -50,
                'cci': cci[-1] if not np.isnan(cci[-1]) else 0,
                'momentum_5d': momentum_5d,
                'momentum_10d': momentum_10d
            }
            
        except Exception as e:
            logger.error(f"Error calculating momentum indicators: {e}")
            return {}
    
    def detect_symbol_regime(self, symbol: str, df: pd.DataFrame) -> Dict:
        """Detect market regime for a single symbol"""
        try:
            if df.empty or len(df) < 20:
                return {
                    'regime': 'UNKNOWN',
                    'confidence': 0.0,
                    'trend_score': 0.0,
                    'volatility_score': 0.0,
                    'momentum_score': 0.0
                }
            
            price_series = df['current_price']
            
            # Calculate indicators
            trend_indicators = self.calculate_trend_indicators(price_series)
            volatility_indicators = self.calculate_volatility_indicators(price_series)
            momentum_indicators = self.calculate_momentum_indicators(price_series)
            
            # Trend scoring
            trend_score = 0.0
            trend_signals = 0
            
            if trend_indicators:
                current_price = trend_indicators['current_price']
                sma_20 = trend_indicators['sma_20']
                sma_50 = trend_indicators['sma_50']
                macd = trend_indicators['macd']
                macd_signal = trend_indicators['macd_signal']
                adx = trend_indicators['adx']
                trend_slope = trend_indicators['trend_slope']
                
                # Price vs moving averages
                if current_price > sma_20:
                    trend_score += 0.25
                    trend_signals += 1
                elif current_price < sma_20:
                    trend_score -= 0.25
                    trend_signals += 1
                
                if sma_20 > sma_50:
                    trend_score += 0.25
                    trend_signals += 1
                elif sma_20 < sma_50:
                    trend_score -= 0.25
                    trend_signals += 1
                
                # MACD
                if macd > macd_signal:
                    trend_score += 0.2
                    trend_signals += 1
                elif macd < macd_signal:
                    trend_score -= 0.2
                    trend_signals += 1
                
                # Trend slope
                if trend_slope > 0.1:
                    trend_score += 0.3
                elif trend_slope < -0.1:
                    trend_score -= 0.3
                trend_signals += 1
            
            # Normalize trend score
            trend_score = trend_score / max(trend_signals, 1) if trend_signals > 0 else 0
            
            # Volatility scoring (high volatility = uncertain regime)
            volatility_score = 0.0
            if volatility_indicators:
                vol = volatility_indicators['daily_volatility']
                
                if vol < 2:  # Low volatility
                    volatility_score = 0.8
                elif vol < 4:  # Medium volatility
                    volatility_score = 0.6
                elif vol < 6:  # High volatility
                    volatility_score = 0.3
                else:  # Very high volatility
                    volatility_score = 0.1
            
            # Momentum scoring
            momentum_score = 0.0
            momentum_signals = 0
            
            if momentum_indicators:
                rsi = momentum_indicators['rsi']
                momentum_5d = momentum_indicators['momentum_5d']
                momentum_10d = momentum_indicators['momentum_10d']
                
                # RSI momentum
                if rsi > 60:
                    momentum_score += 0.4
                elif rsi > 50:
                    momentum_score += 0.2
                elif rsi < 40:
                    momentum_score -= 0.4
                elif rsi < 50:
                    momentum_score -= 0.2
                momentum_signals += 1
                
                # Price momentum
                if momentum_5d > 2:
                    momentum_score += 0.3
                elif momentum_5d < -2:
                    momentum_score -= 0.3
                momentum_signals += 1
                
                if momentum_10d > 5:
                    momentum_score += 0.3
                elif momentum_10d < -5:
                    momentum_score -= 0.3
                momentum_signals += 1
            
            momentum_score = momentum_score / max(momentum_signals, 1) if momentum_signals > 0 else 0
            
            # Combine scores to determine regime
            combined_score = (trend_score * 0.5 + momentum_score * 0.3) * volatility_score
            
            # Determine regime
            if combined_score > 0.3:
                regime = 'BULL'
                confidence = min(0.95, abs(combined_score) + 0.2)
            elif combined_score < -0.3:
                regime = 'BEAR'
                confidence = min(0.95, abs(combined_score) + 0.2)
            else:
                regime = 'SIDEWAYS'
                confidence = 0.7 - abs(combined_score)
            
            return {
                'regime': regime,
                'confidence': confidence,
                'trend_score': trend_score,
                'volatility_score': volatility_score,
                'momentum_score': momentum_score,
                'combined_score': combined_score,
                'indicators': {
                    'trend': trend_indicators,
                    'volatility': volatility_indicators,
                    'momentum': momentum_indicators
                }
            }
            
        except Exception as e:
            logger.error(f"Error detecting regime for {symbol}: {e}")
            return {
                'regime': 'UNKNOWN',
                'confidence': 0.0,
                'trend_score': 0.0,
                'volatility_score': 0.0,
                'momentum_score': 0.0
            }
    
    async def detect_market_regime(self, symbols: List[str], lookback_days: int = 30) -> Dict:
        """Detect overall market regime from multiple symbols"""
        try:
            # Check cache
            cache_key = f"{'-'.join(sorted(symbols))}_{lookback_days}"
            if (self.cache_timestamp and 
                (datetime.now() - self.cache_timestamp).total_seconds() < self.cache_duration and
                cache_key in self.indicators_cache):
                return self.indicators_cache[cache_key]
            
            # Get market data
            market_data = await self.get_market_data(symbols, lookback_days)
            
            if not market_data:
                return {
                    'overall_regime': 'UNKNOWN',
                    'regime_confidence': 0.0,
                    'regime_strength': 0.0,
                    'symbol_regimes': {},
                    'market_indicators': {},
                    'timestamp': datetime.now().isoformat()
                }
            
            # Analyze each symbol
            symbol_regimes = {}
            regime_scores = []
            confidence_scores = []
            
            for symbol, df in market_data.items():
                regime_data = self.detect_symbol_regime(symbol, df)
                symbol_regimes[symbol] = regime_data
                
                # Convert regime to numeric score for averaging
                regime = regime_data['regime']
                confidence = regime_data['confidence']
                
                if regime == 'BULL':
                    regime_scores.append(1.0 * confidence)
                elif regime == 'BEAR':
                    regime_scores.append(-1.0 * confidence)
                else:  # SIDEWAYS or UNKNOWN
                    regime_scores.append(0.0)
                
                confidence_scores.append(confidence)
            
            # Calculate overall regime
            if regime_scores:
                avg_regime_score = np.mean(regime_scores)
                avg_confidence = np.mean(confidence_scores)
                
                # Weight by market cap if available
                market_cap_weights = []
                for symbol in symbols:
                    if symbol in market_data and not market_data[symbol].empty:
                        market_cap = market_data[symbol]['market_cap_usd'].iloc[-1]
                        market_cap_weights.append(market_cap if not pd.isna(market_cap) else 1.0)
                    else:
                        market_cap_weights.append(1.0)
                
                # Normalize weights
                total_weight = sum(market_cap_weights)
                if total_weight > 0:
                    weights = [w/total_weight for w in market_cap_weights]
                    weighted_regime_score = sum(score * weight for score, weight in zip(regime_scores, weights))
                else:
                    weighted_regime_score = avg_regime_score
                
                # Determine overall regime
                if weighted_regime_score > 0.2:
                    overall_regime = 'BULL'
                elif weighted_regime_score < -0.2:
                    overall_regime = 'BEAR'
                else:
                    overall_regime = 'SIDEWAYS'
                
                # Calculate regime strength (agreement between symbols)
                regime_agreement = sum(1 for regime in [r['regime'] for r in symbol_regimes.values()] 
                                     if regime == overall_regime) / len(symbol_regimes)
                regime_strength = regime_agreement * avg_confidence
                
            else:
                overall_regime = 'UNKNOWN'
                avg_confidence = 0.0
                regime_strength = 0.0
                weighted_regime_score = 0.0
            
            # Update regime history
            self.regime_history.append(overall_regime)
            
            # Check for regime stability (prevent frequent changes)
            if len(self.regime_history) >= 3:
                recent_regimes = list(self.regime_history)[-3:]
                if recent_regimes.count(overall_regime) >= 2:
                    stable_regime = overall_regime
                else:
                    stable_regime = self.current_regime  # Keep current if unstable
            else:
                stable_regime = overall_regime
            
            # Update regime tracking
            if stable_regime != self.current_regime:
                self.current_regime = stable_regime
                self.regime_start_date = datetime.now()
            
            regime_duration_days = (datetime.now() - self.regime_start_date).days
            
            # Calculate market indicators summary
            market_indicators = {
                'avg_volatility': np.mean([r.get('volatility_score', 0) for r in symbol_regimes.values()]),
                'avg_trend_strength': np.mean([abs(r.get('trend_score', 0)) for r in symbol_regimes.values()]),
                'avg_momentum': np.mean([r.get('momentum_score', 0) for r in symbol_regimes.values()]),
                'regime_score': weighted_regime_score,
                'symbols_analyzed': len(symbol_regimes)
            }
            
            result = {
                'overall_regime': stable_regime,
                'regime_confidence': avg_confidence,
                'regime_strength': regime_strength,
                'symbol_regimes': symbol_regimes,
                'market_indicators': market_indicators,
                'regime_duration_days': regime_duration_days,
                'timestamp': datetime.now().isoformat()
            }
            
            # Cache result
            self.indicators_cache[cache_key] = result
            self.cache_timestamp = datetime.now()
            
            return result
            
        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            return {
                'overall_regime': 'UNKNOWN',
                'regime_confidence': 0.0,
                'regime_strength': 0.0,
                'symbol_regimes': {},
                'market_indicators': {},
                'regime_duration_days': 0,
                'timestamp': datetime.now().isoformat()
            }
    
    async def get_regime_based_strategy_adjustments(self, current_regime: str) -> Dict:
        """Get strategy adjustments based on current market regime"""
        try:
            if current_regime == 'BULL':
                return {
                    'position_size_multiplier': 1.2,  # Increase position sizes
                    'stop_loss_percent': 8.0,  # Wider stops in bull market
                    'take_profit_percent': 12.0,  # Higher targets
                    'signal_threshold': 0.6,  # Lower threshold for signals
                    'rebalance_frequency_hours': 8,  # More frequent rebalancing
                    'max_positions': 8,  # Allow more positions
                    'preferred_signal_types': ['BUY', 'STRONG_BUY'],
                    'risk_tolerance': 'AGGRESSIVE'
                }
            elif current_regime == 'BEAR':
                return {
                    'position_size_multiplier': 0.6,  # Reduce position sizes
                    'stop_loss_percent': 6.0,  # Tighter stops in bear market
                    'take_profit_percent': 6.0,  # Lower targets
                    'signal_threshold': 0.8,  # Higher threshold for signals
                    'rebalance_frequency_hours': 4,  # More frequent rebalancing
                    'max_positions': 4,  # Fewer positions
                    'preferred_signal_types': ['SELL', 'STRONG_SELL', 'HOLD'],
                    'risk_tolerance': 'CONSERVATIVE'
                }
            else:  # SIDEWAYS
                return {
                    'position_size_multiplier': 1.0,  # Normal position sizes
                    'stop_loss_percent': 10.0,  # Standard stops
                    'take_profit_percent': 8.0,  # Moderate targets
                    'signal_threshold': 0.7,  # Standard threshold
                    'rebalance_frequency_hours': 6,  # Normal rebalancing
                    'max_positions': 6,  # Moderate positions
                    'preferred_signal_types': ['BUY', 'SELL', 'HOLD'],
                    'risk_tolerance': 'MODERATE'
                }
                
        except Exception as e:
            logger.error(f"Error getting strategy adjustments: {e}")
            return {}

# Global regime detector instance
regime_detector = MarketRegimeDetector()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "market-regime-detection", "timestamp": datetime.now().isoformat()}

@app.post("/detect_regime", response_model=MarketRegimeResponse)
async def detect_regime_endpoint(request: MarketRegimeRequest):
    """Detect market regime for given symbols"""
    try:
        result = await regime_detector.detect_market_regime(
            request.symbols,
            request.lookback_days
        )
        
        return MarketRegimeResponse(
            overall_regime=result['overall_regime'],
            regime_confidence=result['regime_confidence'],
            regime_strength=result['regime_strength'],
            symbol_regimes=result['symbol_regimes'],
            market_indicators=result['market_indicators'],
            regime_duration_days=result.get('regime_duration_days', 0),
            timestamp=result['timestamp']
        )
        
    except Exception as e:
        logger.error(f"Error in regime detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/current_regime")
async def get_current_regime():
    """Get current market regime for major cryptos"""
    try:
        major_cryptos = ['BTC', 'ETH', 'ADA', 'SOL', 'DOT', 'MATIC']
        result = await regime_detector.detect_market_regime(major_cryptos, 30)
        return result
        
    except Exception as e:
        logger.error(f"Error getting current regime: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/strategy_adjustments/{regime}")
async def get_strategy_adjustments(regime: str):
    """Get strategy adjustments for specific regime"""
    try:
        adjustments = await regime_detector.get_regime_based_strategy_adjustments(regime.upper())
        return {
            'regime': regime.upper(),
            'adjustments': adjustments,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting strategy adjustments: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/regime_history")
async def get_regime_history():
    """Get recent regime history"""
    try:
        return {
            'current_regime': regime_detector.current_regime,
            'regime_start_date': regime_detector.regime_start_date.isoformat(),
            'regime_duration_days': (datetime.now() - regime_detector.regime_start_date).days,
            'recent_regimes': list(regime_detector.regime_history),
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting regime history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/symbol_regime/{symbol}")
async def get_symbol_regime(symbol: str, lookback_days: int = 30):
    """Get regime analysis for specific symbol"""
    try:
        market_data = await regime_detector.get_market_data([symbol.upper()], lookback_days)
        
        if symbol.upper() not in market_data:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
        
        regime_data = regime_detector.detect_symbol_regime(
            symbol.upper(),
            market_data[symbol.upper()]
        )
        
        return {
            'symbol': symbol.upper(),
            'regime_analysis': regime_data,
            'lookback_days': lookback_days,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting symbol regime: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/market_overview")
async def get_market_overview():
    """Get comprehensive market overview"""
    try:
        # Analyze major symbols
        major_symbols = ['BTC', 'ETH', 'ADA', 'SOL', 'DOT', 'MATIC', 'AVAX', 'ATOM']
        
        regime_analysis = await regime_detector.detect_market_regime(major_symbols, 30)
        strategy_adjustments = await regime_detector.get_regime_based_strategy_adjustments(
            regime_analysis['overall_regime']
        )
        
        # Count regimes by symbol
        regime_distribution = defaultdict(int)
        for symbol_data in regime_analysis['symbol_regimes'].values():
            regime_distribution[symbol_data['regime']] += 1
        
        return {
            'market_regime': regime_analysis['overall_regime'],
            'regime_confidence': regime_analysis['regime_confidence'],
            'regime_strength': regime_analysis['regime_strength'],
            'regime_duration_days': regime_analysis.get('regime_duration_days', 0),
            'regime_distribution': dict(regime_distribution),
            'market_indicators': regime_analysis['market_indicators'],
            'strategy_adjustments': strategy_adjustments,
            'symbols_analyzed': list(regime_analysis['symbol_regimes'].keys()),
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting market overview: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "market_regime_service:app",
        host="0.0.0.0",
        port=8032,
        log_level="info"
    )
