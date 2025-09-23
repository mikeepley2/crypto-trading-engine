#!/usr/bin/env python3
"""
Multi-Timeframe Signal Generator
Provides signal analysis across multiple timeframes (1-min, 15-min, 4-hour, daily)
Combines signals for comprehensive trading strategy
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
import ta
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Multi-Timeframe Signal Generator", version="1.0.0")

class TimeframeSignalRequest(BaseModel):
    symbol: str
    timeframes: List[str] = ['1m', '15m', '4h', '1d']

class TimeframeSignalResponse(BaseModel):
    symbol: str
    timeframe_signals: Dict[str, Dict]
    consensus_signal: Dict
    strength_score: float
    timestamp: str

class MultiTimeframeSignalGenerator:
    def __init__(self):
        self.db_config = {
            'host': os.environ.get('DATABASE_HOST', 'host.docker.internal'),
            'user': os.environ.get('DATABASE_USER', 'news_collector'),
            'password': os.environ.get('DATABASE_PASSWORD', '99Rules!'),
            'database': os.environ.get('DATABASE_NAME', 'crypto_prices'),
            'port': int(os.environ.get('DATABASE_PORT', 3306))
        }
        
        # Timeframe definitions
        self.timeframes = {
            '1m': {'minutes': 1, 'periods': 200},
            '15m': {'minutes': 15, 'periods': 100},
            '4h': {'minutes': 240, 'periods': 50},
            '1d': {'minutes': 1440, 'periods': 30}
        }
        
        # Signal weights by timeframe (longer = higher weight)
        self.timeframe_weights = {
            '1m': 0.1,   # Short-term noise
            '15m': 0.2,  # Short-term trend
            '4h': 0.35,  # Medium-term trend
            '1d': 0.35   # Long-term trend
        }
        
    def get_ohlcv_data(self, symbol: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
        """Get OHLCV data for specific timeframe"""
        try:
            conn = mysql.connector.connect(**self.db_config)
            
            minutes = self.timeframes[timeframe]['minutes']
            
            # Calculate the time interval for grouping
            if timeframe == '1m':
                # For 1-minute, use existing ml_features_materialized data
                query = """
                SELECT 
                    timestamp_iso as timestamp,
                    current_price as close,
                    current_price as open,
                    current_price as high,
                    current_price as low,
                    COALESCE(volume_24h, 0) as volume
                FROM ml_features_materialized 
                WHERE symbol = %s 
                AND timestamp_iso >= DATE_SUB(NOW(), INTERVAL %s MINUTE)
                ORDER BY timestamp_iso ASC
                """
                df = pd.read_sql(query, conn, params=[symbol, limit])
            else:
                # For other timeframes, aggregate from 1-minute data
                query = """
                WITH timeframe_data AS (
                    SELECT 
                        DATE_FORMAT(timestamp_iso, CASE 
                            WHEN %s = 15 THEN '%%Y-%%m-%%d %%H:%%i:00'
                            WHEN %s = 240 THEN '%%Y-%%m-%%d %%H:00:00'
                            WHEN %s = 1440 THEN '%%Y-%%m-%%d 00:00:00'
                        END) as timeframe_timestamp,
                        current_price,
                        COALESCE(volume_24h, 0) as volume
                    FROM ml_features_materialized 
                    WHERE symbol = %s 
                    AND timestamp_iso >= DATE_SUB(NOW(), INTERVAL %s MINUTE)
                    ORDER BY timestamp_iso ASC
                )
                SELECT 
                    timeframe_timestamp as timestamp,
                    (SELECT current_price FROM timeframe_data td2 
                     WHERE td2.timeframe_timestamp = td.timeframe_timestamp 
                     ORDER BY current_price LIMIT 1) as open,
                    MAX(current_price) as high,
                    MIN(current_price) as low,
                    (SELECT current_price FROM timeframe_data td3 
                     WHERE td3.timeframe_timestamp = td.timeframe_timestamp 
                     ORDER BY current_price DESC LIMIT 1) as close,
                    AVG(volume) as volume
                FROM timeframe_data td
                GROUP BY timeframe_timestamp
                ORDER BY timeframe_timestamp ASC
                """
                params = [minutes, minutes, minutes, symbol, limit * minutes]
                df = pd.read_sql(query, conn, params=params)
            
            conn.close()
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
                
                # Ensure we have OHLCV columns
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if col not in df.columns:
                        df[col] = df.get('close', 0)
                
                # Convert to float
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                
                logger.debug(f"📊 {symbol} {timeframe}: {len(df)} candles")
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting OHLCV data for {symbol} {timeframe}: {e}")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate technical indicators for signal generation"""
        try:
            if df.empty or len(df) < 20:
                return {}
            
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values
            
            indicators = {}
            
            # Trend indicators
            try:
                indicators['sma_20'] = ta.trend.sma_indicator(close, window=20)
                indicators['sma_50'] = ta.trend.sma_indicator(close, window=50) if len(close) >= 50 else indicators['sma_20']
                indicators['ema_12'] = ta.trend.ema_indicator(close, window=12)
                indicators['ema_26'] = ta.trend.ema_indicator(close, window=26)
            except:
                indicators['sma_20'] = close
                indicators['sma_50'] = close
                indicators['ema_12'] = close
                indicators['ema_26'] = close
            
            # MACD
            try:
                # MACD
                macd_line = ta.trend.macd(close, window_fast=12, window_slow=26)
                macdsignal = ta.trend.macd_signal(close, window_fast=12, window_slow=26, window_sign=9)
                macdhist = ta.trend.macd_diff(close, window_fast=12, window_slow=26, window_sign=9)
                indicators['macd'] = macd_line
                indicators['macd_signal'] = macdsignal
                indicators['macd_histogram'] = macdhist
            except:
                indicators['macd'] = np.zeros_like(close)
                indicators['macd_signal'] = np.zeros_like(close)
                indicators['macd_histogram'] = np.zeros_like(close)
            
            # RSI
            try:
                # RSI
                indicators['rsi'] = ta.momentum.rsi(close, window=14)
            except:
                indicators['rsi'] = np.full_like(close, 50)
            
            # Bollinger Bands
            try:
                bb_upper = ta.volatility.bollinger_hband(close, window=20, window_dev=2)
                bb_middle = ta.volatility.bollinger_mavg(close, window=20)
                bb_lower = ta.volatility.bollinger_lband(close, window=20, window_dev=2)
                indicators['bb_upper'] = bb_upper
                indicators['bb_middle'] = bb_middle
                indicators['bb_lower'] = bb_lower
            except:
                indicators['bb_upper'] = close * 1.02
                indicators['bb_middle'] = close
                indicators['bb_lower'] = close * 0.98
            
            # Stochastic
            try:
                slowk = ta.momentum.stoch(high, low, close, window=14, smooth_window=3)
                slowd = ta.momentum.stoch_signal(high, low, close, window=14, smooth_window=3)
                indicators['stoch_k'] = slowk
                indicators['stoch_d'] = slowd
            except:
                indicators['stoch_k'] = np.full_like(close, 50)
                indicators['stoch_d'] = np.full_like(close, 50)
            
            # Volume indicators
            try:
                indicators['volume_sma'] = ta.trend.sma_indicator(volume, window=20)
            except:
                indicators['volume_sma'] = volume
            
            # ADX (trend strength)
            try:
                indicators['adx'] = ta.trend.adx(high, low, close, window=14)
            except:
                indicators['adx'] = np.full_like(close, 25)
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return {}
    
    def generate_timeframe_signal(self, symbol: str, timeframe: str) -> Dict:
        """Generate signal for specific timeframe"""
        try:
            # Get OHLCV data
            df = self.get_ohlcv_data(symbol, timeframe)
            
            if df.empty or len(df) < 20:
                return {
                    'signal': 'HOLD',
                    'confidence': 0.0,
                    'indicators': {},
                    'reasoning': 'Insufficient data'
                }
            
            # Calculate technical indicators
            indicators = self.calculate_technical_indicators(df)
            
            if not indicators:
                return {
                    'signal': 'HOLD',
                    'confidence': 0.0,
                    'indicators': {},
                    'reasoning': 'Failed to calculate indicators'
                }
            
            # Get latest values
            current_price = df['close'].iloc[-1]
            
            # Initialize signal scoring
            signal_score = 0.0
            confidence_factors = []
            reasoning_parts = []
            
            # 1. Trend Analysis (40% weight)
            try:
                sma_20 = indicators['sma_20'][-1] if not np.isnan(indicators['sma_20'][-1]) else current_price
                sma_50 = indicators['sma_50'][-1] if not np.isnan(indicators['sma_50'][-1]) else current_price
                
                if current_price > sma_20 > sma_50:
                    signal_score += 0.4
                    reasoning_parts.append("Uptrend (price > SMA20 > SMA50)")
                    confidence_factors.append(0.8)
                elif current_price > sma_20:
                    signal_score += 0.2
                    reasoning_parts.append("Above SMA20")
                    confidence_factors.append(0.6)
                elif current_price < sma_20 < sma_50:
                    signal_score -= 0.4
                    reasoning_parts.append("Downtrend (price < SMA20 < SMA50)")
                    confidence_factors.append(0.8)
                elif current_price < sma_20:
                    signal_score -= 0.2
                    reasoning_parts.append("Below SMA20")
                    confidence_factors.append(0.6)
            except:
                pass
            
            # 2. MACD Analysis (25% weight)
            try:
                macd = indicators['macd'][-1]
                macd_signal = indicators['macd_signal'][-1]
                macd_hist = indicators['macd_histogram'][-1]
                
                if not (np.isnan(macd) or np.isnan(macd_signal) or np.isnan(macd_hist)):
                    if macd > macd_signal and macd_hist > 0:
                        signal_score += 0.25
                        reasoning_parts.append("MACD bullish")
                        confidence_factors.append(0.7)
                    elif macd < macd_signal and macd_hist < 0:
                        signal_score -= 0.25
                        reasoning_parts.append("MACD bearish")
                        confidence_factors.append(0.7)
            except:
                pass
            
            # 3. RSI Analysis (20% weight)
            try:
                rsi = indicators['rsi'][-1]
                
                if not np.isnan(rsi):
                    if rsi < 30:
                        signal_score += 0.15
                        reasoning_parts.append(f"RSI oversold ({rsi:.1f})")
                        confidence_factors.append(0.8)
                    elif rsi > 70:
                        signal_score -= 0.15
                        reasoning_parts.append(f"RSI overbought ({rsi:.1f})")
                        confidence_factors.append(0.8)
                    elif 45 <= rsi <= 55:
                        reasoning_parts.append(f"RSI neutral ({rsi:.1f})")
                        confidence_factors.append(0.5)
            except:
                pass
            
            # 4. Bollinger Bands Analysis (10% weight)
            try:
                bb_upper = indicators['bb_upper'][-1]
                bb_lower = indicators['bb_lower'][-1]
                
                if not (np.isnan(bb_upper) or np.isnan(bb_lower)):
                    if current_price <= bb_lower:
                        signal_score += 0.1
                        reasoning_parts.append("Price at BB lower")
                        confidence_factors.append(0.7)
                    elif current_price >= bb_upper:
                        signal_score -= 0.1
                        reasoning_parts.append("Price at BB upper")
                        confidence_factors.append(0.7)
            except:
                pass
            
            # 5. Volume Analysis (5% weight)
            try:
                current_volume = df['volume'].iloc[-1]
                volume_sma = indicators['volume_sma'][-1]
                
                if not np.isnan(volume_sma) and volume_sma > 0:
                    volume_ratio = current_volume / volume_sma
                    if volume_ratio > 1.5:
                        if signal_score > 0:
                            signal_score += 0.05  # High volume confirms bullish signal
                        else:
                            signal_score -= 0.05  # High volume confirms bearish signal
                        reasoning_parts.append(f"High volume ({volume_ratio:.1f}x avg)")
                        confidence_factors.append(0.6)
            except:
                pass
            
            # Determine final signal
            if signal_score >= 0.3:
                signal = 'BUY'
            elif signal_score <= -0.3:
                signal = 'SELL'
            else:
                signal = 'HOLD'
            
            # Calculate confidence
            confidence = min(0.95, max(0.1, np.mean(confidence_factors) if confidence_factors else 0.5))
            
            # Adjust confidence based on signal strength
            if abs(signal_score) > 0.5:
                confidence += 0.1
            elif abs(signal_score) < 0.1:
                confidence -= 0.1
            
            confidence = max(0.1, min(0.95, confidence))
            
            return {
                'signal': signal,
                'confidence': confidence,
                'signal_score': signal_score,
                'indicators': {
                    'rsi': indicators.get('rsi', [np.nan])[-1],
                    'macd': indicators.get('macd', [np.nan])[-1],
                    'macd_signal': indicators.get('macd_signal', [np.nan])[-1],
                    'sma_20': indicators.get('sma_20', [np.nan])[-1],
                    'current_price': current_price
                },
                'reasoning': '; '.join(reasoning_parts) if reasoning_parts else 'No clear signals'
            }
            
        except Exception as e:
            logger.error(f"Error generating {timeframe} signal for {symbol}: {e}")
            return {
                'signal': 'HOLD',
                'confidence': 0.0,
                'indicators': {},
                'reasoning': f'Error: {str(e)}'
            }
    
    def generate_consensus_signal(self, timeframe_signals: Dict[str, Dict]) -> Dict:
        """Generate consensus signal from multiple timeframes"""
        try:
            total_weight = 0.0
            weighted_score = 0.0
            signal_counts = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
            confidence_sum = 0.0
            
            for timeframe, signal_data in timeframe_signals.items():
                if timeframe not in self.timeframe_weights:
                    continue
                
                weight = self.timeframe_weights[timeframe]
                signal = signal_data.get('signal', 'HOLD')
                confidence = signal_data.get('confidence', 0.0)
                signal_score = signal_data.get('signal_score', 0.0)
                
                # Count signals
                signal_counts[signal] += 1
                
                # Calculate weighted score
                signal_value = 1.0 if signal == 'BUY' else (-1.0 if signal == 'SELL' else 0.0)
                weighted_score += signal_value * weight * confidence
                confidence_sum += confidence * weight
                total_weight += weight
            
            # Normalize
            if total_weight > 0:
                final_score = weighted_score / total_weight
                avg_confidence = confidence_sum / total_weight
            else:
                final_score = 0.0
                avg_confidence = 0.0
            
            # Determine consensus signal
            if final_score >= 0.3:
                consensus_signal = 'BUY'
            elif final_score <= -0.3:
                consensus_signal = 'SELL'
            else:
                consensus_signal = 'HOLD'
            
            # Calculate strength score (agreement between timeframes)
            total_signals = sum(signal_counts.values())
            if total_signals > 0:
                max_agreement = max(signal_counts.values()) / total_signals
                strength_score = max_agreement * avg_confidence
            else:
                strength_score = 0.0
            
            # Adjust confidence based on agreement
            if signal_counts[consensus_signal] >= 3:  # Strong agreement
                consensus_confidence = min(0.95, avg_confidence + 0.1)
            elif signal_counts[consensus_signal] >= 2:  # Moderate agreement
                consensus_confidence = avg_confidence
            else:  # Weak agreement
                consensus_confidence = max(0.1, avg_confidence - 0.2)
            
            return {
                'signal': consensus_signal,
                'confidence': consensus_confidence,
                'strength_score': strength_score,
                'final_score': final_score,
                'signal_distribution': signal_counts,
                'reasoning': f"{signal_counts[consensus_signal]}/{total_signals} timeframes agree"
            }
            
        except Exception as e:
            logger.error(f"Error generating consensus signal: {e}")
            return {
                'signal': 'HOLD',
                'confidence': 0.0,
                'strength_score': 0.0,
                'final_score': 0.0,
                'signal_distribution': {'BUY': 0, 'SELL': 0, 'HOLD': 0},
                'reasoning': f'Error: {str(e)}'
            }
    
    def analyze_multi_timeframe(self, symbol: str, timeframes: List[str] = None) -> Dict:
        """Analyze symbol across multiple timeframes"""
        try:
            if not timeframes:
                timeframes = ['1m', '15m', '4h', '1d']
            
            timeframe_signals = {}
            
            # Generate signals for each timeframe
            for timeframe in timeframes:
                if timeframe in self.timeframes:
                    signal_data = self.generate_timeframe_signal(symbol, timeframe)
                    timeframe_signals[timeframe] = signal_data
                    logger.debug(f"📊 {symbol} {timeframe}: {signal_data['signal']} ({signal_data['confidence']:.2f})")
            
            # Generate consensus signal
            consensus = self.generate_consensus_signal(timeframe_signals)
            
            return {
                'symbol': symbol,
                'timeframe_signals': timeframe_signals,
                'consensus_signal': consensus,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in multi-timeframe analysis for {symbol}: {e}")
            return {
                'symbol': symbol,
                'timeframe_signals': {},
                'consensus_signal': {
                    'signal': 'HOLD',
                    'confidence': 0.0,
                    'strength_score': 0.0,
                    'reasoning': f'Error: {str(e)}'
                },
                'timestamp': datetime.now().isoformat()
            }

# Global signal generator instance
mtf_generator = MultiTimeframeSignalGenerator()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "multi-timeframe-signals", "timestamp": datetime.now().isoformat()}

@app.post("/analyze_timeframes", response_model=TimeframeSignalResponse)
async def analyze_timeframes_endpoint(request: TimeframeSignalRequest):
    """Analyze symbol across multiple timeframes"""
    try:
        result = mtf_generator.analyze_multi_timeframe(
            request.symbol.upper(),
            request.timeframes
        )
        
        return TimeframeSignalResponse(
            symbol=result['symbol'],
            timeframe_signals=result['timeframe_signals'],
            consensus_signal=result['consensus_signal'],
            strength_score=result['consensus_signal'].get('strength_score', 0.0),
            timestamp=result['timestamp']
        )
        
    except Exception as e:
        logger.error(f"Error in timeframe analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/quick_analysis/{symbol}")
async def quick_analysis(symbol: str):
    """Quick analysis with default timeframes"""
    try:
        result = mtf_generator.analyze_multi_timeframe(symbol.upper())
        return result
        
    except Exception as e:
        logger.error(f"Error in quick analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/timeframe_data/{symbol}/{timeframe}")
async def get_timeframe_data(symbol: str, timeframe: str):
    """Get OHLCV data for specific timeframe"""
    try:
        df = mtf_generator.get_ohlcv_data(symbol.upper(), timeframe)
        
        if df.empty:
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'data': [],
                'count': 0
            }
        
        # Convert to records for JSON response
        data = []
        for idx, row in df.tail(50).iterrows():  # Last 50 candles
            data.append({
                'timestamp': idx.isoformat(),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume'])
            })
        
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'data': data,
            'count': len(data)
        }
        
    except Exception as e:
        logger.error(f"Error getting timeframe data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/batch_analysis")
async def batch_analysis(symbols: str = "BTC,ETH,ADA"):
    """Analyze multiple symbols"""
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(',')]
        results = {}
        
        for symbol in symbol_list:
            result = mtf_generator.analyze_multi_timeframe(symbol)
            results[symbol] = result
        
        return {
            'results': results,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in batch analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "multi_timeframe_service:app",
        host="0.0.0.0",
        port=8029,
        log_level="info"
    )
