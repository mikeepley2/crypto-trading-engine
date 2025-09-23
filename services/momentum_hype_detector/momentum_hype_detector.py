#!/usr/bin/env python3
"""
Momentum & Hype Coin Detection Service
Advanced momentum and viral coin detection with social sentiment analysis.

Features:
- Real-time social media sentiment monitoring
- Volume surge detection with statistical analysis
- Price momentum pattern recognition
- Viral coin identification via news/social buzz
- On-chain activity spike detection
- Quick exit signal generation when momentum reverses
- Integration with existing risk management systems

Strategy Logic:
1. Detect hype/momentum via multiple signals (social, volume, price, news)
2. Calculate momentum strength and sustainability scores
3. Generate BUY signals for strong momentum coins
4. Monitor for momentum reversal patterns
5. Generate quick SELL signals when momentum breaks
6. Coordinate with market selloff protection and other strategies
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
from textblob import TextBlob
import re
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [MOMENTUM_DETECTOR] %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class MomentumSignal:
    """Momentum/hype detection signal."""
    timestamp: datetime
    symbol: str
    momentum_type: str  # viral, volume_surge, price_momentum, social_buzz, breakout
    momentum_strength: float  # 0-1 scale
    sustainability_score: float  # Expected duration of momentum
    entry_confidence: float  # Confidence in entry signal
    exit_confidence: float  # Confidence in exit timing
    volume_surge_ratio: float  # Current volume / average volume
    price_momentum_score: float  # Price acceleration indicator
    social_sentiment_spike: float  # Social sentiment acceleration
    news_buzz_score: float  # News mention frequency
    risk_multiplier: float  # Risk adjustment for position sizing
    expected_duration: str  # minutes, hours, days
    entry_trigger: str  # What triggered the momentum signal
    exit_strategy: str  # How to exit the position
    stop_loss_percentage: float  # Recommended stop loss
    profit_target_percentage: float  # Target profit level
    viral_indicators: List[str]  # Specific viral/hype indicators
    caution_flags: List[str]  # Risk warnings

@dataclass
class QuickExitSignal:
    """Quick exit signal for momentum positions."""
    timestamp: datetime
    symbol: str
    exit_reason: str  # momentum_reversal, profit_target, stop_loss, volume_decline
    urgency: str  # low, medium, high, emergency
    confidence: float
    price_target: float
    expected_completion: str  # seconds, minutes
    exit_percentage: float  # What % of position to exit
    momentum_breakdown: Dict  # Detailed breakdown of momentum loss

class MomentumHypeDetector:
    """Advanced momentum and hype coin detection service."""
    
    def __init__(self):
        self.db_config = {
            'host': 'host.docker.internal',
            'user': 'news_collector',
            'password': '99Rules!',
            'database': 'crypto_prices'
        }
        
        # Momentum detection parameters (configurable)
        self.volume_surge_threshold = float(os.getenv('VOLUME_SURGE_THRESHOLD', '5.0'))  # 5x normal volume
        self.price_momentum_threshold = float(os.getenv('PRICE_MOMENTUM_THRESHOLD', '0.15'))  # 15% price move
        self.social_sentiment_threshold = float(os.getenv('SOCIAL_SENTIMENT_THRESHOLD', '0.7'))  # High sentiment
        self.news_buzz_threshold = float(os.getenv('NEWS_BUZZ_THRESHOLD', '3.0'))  # 3x normal mentions
        
        # Quick exit parameters
        self.momentum_reversal_threshold = float(os.getenv('MOMENTUM_REVERSAL_THRESHOLD', '0.3'))
        self.volume_decline_threshold = float(os.getenv('VOLUME_DECLINE_THRESHOLD', '0.5'))  # 50% volume drop
        self.price_reversal_threshold = float(os.getenv('PRICE_REVERSAL_THRESHOLD', '0.05'))  # 5% reversal
        
        # Position management parameters
        self.max_momentum_position_pct = float(os.getenv('MAX_MOMENTUM_POSITION_PCT', '0.10'))  # 10% max per momentum trade
        self.momentum_stop_loss_pct = float(os.getenv('MOMENTUM_STOP_LOSS_PCT', '0.08'))  # 8% stop loss
        self.momentum_profit_target_pct = float(os.getenv('MOMENTUM_PROFIT_TARGET_PCT', '0.25'))  # 25% profit target
        
        # Strategy integration URLs
        self.sentiment_service_url = os.getenv('SENTIMENT_SERVICE_URL', 'http://host.docker.internal:8014')
        self.market_selloff_url = os.getenv('MARKET_SELLOFF_URL', 'http://host.docker.internal:8028')
        self.trading_engine_url = os.getenv('TRADING_ENGINE_URL', 'http://host.docker.internal:8024')
        
        # Data caches
        self.momentum_cache = {}
        self.social_cache = {}
        self.volume_cache = {}
        self.last_analysis = {}
        
        # Viral coin keywords and patterns
        self.viral_keywords = [
            'moon', 'rocket', 'hodl', 'diamond hands', 'to the moon', 'pump',
            'viral', 'trending', 'explosion', 'massive', 'insane', 'crazy gains',
            'meme', 'doge', 'pepe', 'shib', 'floki', 'baby', 'inu', 'safe',
            'elon', 'tweet', 'mention', 'celebrity', 'influencer'
        ]
        
        # Risk warning patterns
        self.caution_patterns = [
            'rug pull', 'scam', 'dump', 'fake', 'ponzi', 'honeypot',
            'exit scam', 'developer sold', 'whale dump', 'manipulation'
        ]
        
        self.app = FastAPI(title="Momentum & Hype Detector", version="1.0.0")
        self.setup_routes()
    
    def setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {"status": "healthy", "service": "momentum-hype-detector", "timestamp": datetime.now().isoformat()}
        
        @self.app.get("/momentum/current")
        async def get_current_momentum():
            """Get current momentum opportunities."""
            try:
                signals = await self.detect_momentum_opportunities()
                return JSONResponse(content=[asdict(signal) for signal in signals])
            except Exception as e:
                logger.error(f"Error getting momentum opportunities: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/momentum/symbol/{symbol}")
        async def get_symbol_momentum(symbol: str):
            """Get momentum analysis for specific symbol."""
            try:
                signal = await self.analyze_symbol_momentum(symbol.upper())
                if signal:
                    signal_dict = asdict(signal)
                    signal_dict['timestamp'] = signal.timestamp.isoformat()
                    return JSONResponse(content=signal_dict)
                else:
                    return JSONResponse(content={"error": f"No momentum data for {symbol}"})
            except Exception as e:
                logger.error(f"Error analyzing {symbol} momentum: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/exit/current")
        async def get_exit_signals():
            """Get current quick exit signals."""
            try:
                signals = await self.detect_exit_signals()
                return JSONResponse(content=[asdict(signal) for signal in signals])
            except Exception as e:
                logger.error(f"Error getting exit signals: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/viral/trending")
        async def get_trending_coins():
            """Get currently trending/viral coins."""
            try:
                trending = await self.detect_viral_coins()
                return JSONResponse(content=trending)
            except Exception as e:
                logger.error(f"Error getting trending coins: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    async def get_recent_price_data(self, symbols: List[str] = None, hours: int = 96) -> pd.DataFrame:
        """Get recent price and volume data for analysis."""
        try:
            conn = mysql.connector.connect(**self.db_config)
            
            if symbols:
                symbol_filter = f"AND symbol IN ({','.join(['%s'] * len(symbols))})"
                params = [hours] + symbols  # hours first, then symbols
            else:
                symbol_filter = ""
                params = [hours]
            
            query = f"""
            SELECT symbol, timestamp_iso, current_price, volume_24h, market_cap,
                   price_change_percentage_24h
            FROM ml_features_materialized
            WHERE timestamp_iso >= DATE_SUB(NOW(), INTERVAL %s HOUR) {symbol_filter}
            ORDER BY symbol, timestamp_iso DESC
            """
            
            df = pd.read_sql(query, conn, params=params)
            conn.close()
            
            if df.empty:
                logger.warning("No price data retrieved for momentum analysis")
                return pd.DataFrame()
            
            df['timestamp'] = pd.to_datetime(df['timestamp_iso'])
            return df.sort_values(['symbol', 'timestamp'])
            
        except Exception as e:
            logger.error(f"Error getting price data: {e}")
            return pd.DataFrame()
    
    async def analyze_volume_surge(self, symbol: str, df: pd.DataFrame) -> Dict:
        """Analyze volume surge patterns for momentum detection."""
        try:
            symbol_data = df[df['symbol'] == symbol].copy()
            if len(symbol_data) < 10:
                return {"volume_surge_ratio": 1.0, "surge_detected": False}
            
            symbol_data = symbol_data.sort_values('timestamp')
            
            # Calculate recent vs baseline volume
            recent_volume = symbol_data['volume_24h'].tail(3).mean()  # Last 3 data points
            baseline_volume = symbol_data['volume_24h'].iloc[:-3].mean()  # Earlier data
            
            volume_surge_ratio = recent_volume / baseline_volume if baseline_volume > 0 else 1.0
            
            # Check for sustained volume increase
            volume_trend = symbol_data['volume_24h'].tail(6).pct_change().mean()
            
            analysis = {
                "volume_surge_ratio": float(volume_surge_ratio),
                "volume_trend": float(volume_trend),
                "recent_volume": float(recent_volume),
                "baseline_volume": float(baseline_volume),
                "surge_detected": volume_surge_ratio > self.volume_surge_threshold,
                "sustained_surge": volume_trend > 0.1 and volume_surge_ratio > 2.0
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing volume surge for {symbol}: {e}")
            return {"volume_surge_ratio": 1.0, "surge_detected": False}
    
    async def analyze_price_momentum(self, symbol: str, df: pd.DataFrame) -> Dict:
        """Analyze price momentum patterns."""
        try:
            symbol_data = df[df['symbol'] == symbol].copy()
            if len(symbol_data) < 5:
                return {"momentum_score": 0.0, "momentum_detected": False}
            
            symbol_data = symbol_data.sort_values('timestamp')
            
            # Calculate various momentum indicators
            prices = symbol_data['current_price'].values
            
            # Price acceleration (second derivative)
            price_changes = np.diff(prices)
            price_acceleration = np.diff(price_changes)
            avg_acceleration = np.mean(price_acceleration[-3:]) if len(price_acceleration) >= 3 else 0
            
            # Recent price momentum
            recent_change = (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 else 0
            
            # Momentum consistency (how consistent the upward movement is)
            recent_changes = price_changes[-5:] if len(price_changes) >= 5 else price_changes
            positive_changes = np.sum(recent_changes > 0) / len(recent_changes)
            
            # Calculate momentum score (0-1)
            momentum_score = min(1.0, max(0.0, (
                abs(recent_change) * 2 +  # Recent price change (weighted heavily)
                positive_changes +  # Consistency of direction
                min(1.0, abs(avg_acceleration) * 100)  # Acceleration component
            ) / 4))
            
            analysis = {
                "momentum_score": float(momentum_score),
                "recent_change": float(recent_change),
                "price_acceleration": float(avg_acceleration),
                "positive_change_ratio": float(positive_changes),
                "momentum_detected": momentum_score > 0.6 and abs(recent_change) > self.price_momentum_threshold,
                "momentum_direction": "bullish" if recent_change > 0 else "bearish",
                "momentum_strength": "strong" if momentum_score > 0.8 else "moderate" if momentum_score > 0.6 else "weak"
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing price momentum for {symbol}: {e}")
            return {"momentum_score": 0.0, "momentum_detected": False}
    
    async def analyze_social_sentiment_spike(self, symbol: str) -> Dict:
        """Analyze social sentiment spikes and viral indicators."""
        try:
            # Temporarily skip sentiment service call due to hanging endpoint
            # TODO: Fix sentiment service /sentiment/{symbol} endpoint
            logger.warning(f"Skipping sentiment service call for {symbol} - endpoint hanging")
            return {
                "current_sentiment": 0.0,
                "sentiment_confidence": 0.0,
                "sentiment_spike": 0.0,
                "viral_keyword_count": 0,
                "caution_flag_count": 0,
                "social_buzz_detected": False,
                "viral_indicators": [],
                "caution_flags": []
            }
            
            # Original code (temporarily disabled):
            # response = requests.get(f"{self.sentiment_service_url}/sentiment/{symbol}", timeout=5)
            if False:  # response.status_code == 200:
                sentiment_data = response.json()
                
                current_sentiment = sentiment_data.get('sentiment_score', 0.0)
                sentiment_confidence = sentiment_data.get('confidence', 0.0)
                
                # Check for viral keywords in recent sentiment analysis
                sentiment_text = sentiment_data.get('analysis_text', '').lower()
                viral_keyword_count = sum(1 for keyword in self.viral_keywords if keyword in sentiment_text)
                
                caution_flag_count = sum(1 for pattern in self.caution_patterns if pattern in sentiment_text)
                
                # Calculate sentiment spike score
                baseline_sentiment = 0.0  # TODO: Get historical baseline
                sentiment_spike = max(0.0, current_sentiment - baseline_sentiment)
                
                analysis = {
                    "current_sentiment": float(current_sentiment),
                    "sentiment_confidence": float(sentiment_confidence),
                    "sentiment_spike": float(sentiment_spike),
                    "viral_keyword_count": viral_keyword_count,
                    "caution_flag_count": caution_flag_count,
                    "social_buzz_detected": current_sentiment > self.social_sentiment_threshold and viral_keyword_count > 0,
                    "viral_indicators": [kw for kw in self.viral_keywords if kw in sentiment_text],
                    "caution_flags": [cf for cf in self.caution_patterns if cf in sentiment_text]
                }
                
                return analysis
            else:
                return {
                    "current_sentiment": 0.0,
                    "social_buzz_detected": False,
                    "viral_indicators": [],
                    "caution_flags": []
                }
                
        except Exception as e:
            logger.warning(f"Error analyzing social sentiment for {symbol}: {e}")
            return {
                "current_sentiment": 0.0,
                "social_buzz_detected": False,
                "viral_indicators": [],
                "caution_flags": []
            }
    
    async def analyze_symbol_momentum(self, symbol: str) -> Optional[MomentumSignal]:
        """Comprehensive momentum analysis for a specific symbol."""
        try:
            logger.info(f"[DEBUG] Starting momentum analysis for {symbol}")
            
            # Get recent data
            logger.info(f"[DEBUG] Getting recent price data for {symbol}")
            df = await self.get_recent_price_data([symbol], hours=96)
            if df.empty:
                logger.info(f"[DEBUG] No price data found for {symbol}")
                return None
            logger.info(f"[DEBUG] Got {len(df)} price records for {symbol}")
            
            # Analyze different momentum components
            logger.info(f"[DEBUG] Starting volume analysis for {symbol}")
            volume_analysis = await self.analyze_volume_surge(symbol, df)
            logger.info(f"[DEBUG] Volume analysis complete for {symbol}")
            
            logger.info(f"[DEBUG] Starting price analysis for {symbol}")
            price_analysis = await self.analyze_price_momentum(symbol, df)
            logger.info(f"[DEBUG] Price analysis complete for {symbol}")
            
            logger.info(f"[DEBUG] Starting social analysis for {symbol}")
            social_analysis = await self.analyze_social_sentiment_spike(symbol)
            logger.info(f"[DEBUG] Social analysis complete for {symbol}")
            
            # Calculate overall momentum strength
            momentum_components = [
                volume_analysis.get('volume_surge_ratio', 1.0) / 5.0,  # Normalize to 0-1
                price_analysis.get('momentum_score', 0.0),
                social_analysis.get('current_sentiment', 0.0),
                min(1.0, social_analysis.get('viral_keyword_count', 0) / 3.0)
            ]
            
            momentum_strength = np.mean(momentum_components)
            
            # Determine momentum type
            momentum_types = []
            if volume_analysis.get('surge_detected', False):
                momentum_types.append('volume_surge')
            if price_analysis.get('momentum_detected', False):
                momentum_types.append('price_momentum')
            if social_analysis.get('social_buzz_detected', False):
                momentum_types.append('social_buzz')
            if social_analysis.get('viral_keyword_count', 0) > 2:
                momentum_types.append('viral')
            
            if not momentum_types:
                return None  # No momentum detected
            
            momentum_type = ','.join(momentum_types)
            
            # Calculate sustainability score
            sustainability_factors = [
                min(1.0, volume_analysis.get('volume_surge_ratio', 1.0) / 10.0),  # Higher volume = more sustainable
                price_analysis.get('positive_change_ratio', 0.5),  # Consistency
                min(1.0, social_analysis.get('sentiment_confidence', 0.0)),  # Sentiment confidence
                max(0.0, 1.0 - social_analysis.get('caution_flag_count', 0) / 3.0)  # Reduce for caution flags
            ]
            
            sustainability_score = np.mean(sustainability_factors)
            
            # Calculate entry and exit confidence
            entry_confidence = min(0.95, momentum_strength * 1.2)
            exit_confidence = sustainability_score * 0.8  # Lower confidence = quicker exit
            
            # Risk multiplier based on caution flags and volatility
            caution_flags = social_analysis.get('caution_flags', [])
            risk_multiplier = 1.0 + len(caution_flags) * 0.5  # Higher risk for caution flags
            
            # Expected duration based on momentum type and strength
            if 'viral' in momentum_type and momentum_strength > 0.8:
                expected_duration = 'hours'
            elif momentum_strength > 0.7:
                expected_duration = 'hours'
            else:
                expected_duration = 'minutes'
            
            # Create momentum signal
            signal = MomentumSignal(
                timestamp=datetime.now(),
                symbol=symbol,
                momentum_type=momentum_type,
                momentum_strength=momentum_strength,
                sustainability_score=sustainability_score,
                entry_confidence=entry_confidence,
                exit_confidence=exit_confidence,
                volume_surge_ratio=volume_analysis.get('volume_surge_ratio', 1.0),
                price_momentum_score=price_analysis.get('momentum_score', 0.0),
                social_sentiment_spike=social_analysis.get('current_sentiment', 0.0),
                news_buzz_score=1.0,  # TODO: Implement news buzz detection
                risk_multiplier=risk_multiplier,
                expected_duration=expected_duration,
                entry_trigger=f"Momentum detected: {momentum_type}",
                exit_strategy="Quick exit on momentum reversal or profit target",
                stop_loss_percentage=self.momentum_stop_loss_pct,
                profit_target_percentage=self.momentum_profit_target_pct,
                viral_indicators=social_analysis.get('viral_indicators', []),
                caution_flags=caution_flags
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error analyzing momentum for {symbol}: {e}")
            return None
    
    async def detect_momentum_opportunities(self) -> List[MomentumSignal]:
        """Detect current momentum opportunities across all tracked symbols."""
        try:
            # Get recent data for all symbols
            df = await self.get_recent_price_data(hours=96)
            if df.empty:
                return []
            
            symbols = df['symbol'].unique()
            momentum_signals = []
            
            for symbol in symbols:
                signal = await self.analyze_symbol_momentum(symbol)
                if signal and signal.momentum_strength > 0.6:  # Only strong momentum
                    momentum_signals.append(signal)
                    logger.info(f"🚀 MOMENTUM DETECTED: {symbol} - {signal.momentum_type} ({signal.momentum_strength:.2f} strength)")
            
            # Sort by momentum strength
            momentum_signals.sort(key=lambda x: x.momentum_strength, reverse=True)
            
            return momentum_signals[:10]  # Return top 10 opportunities
            
        except Exception as e:
            logger.error(f"Error detecting momentum opportunities: {e}")
            return []
    
    async def detect_exit_signals(self) -> List[QuickExitSignal]:
        """Detect quick exit signals for current momentum positions."""
        try:
            # Get current portfolio positions
            response = requests.get(f"{self.trading_engine_url}/portfolio", timeout=10)
            if response.status_code != 200:
                return []
            
            portfolio = response.json()
            positions = portfolio.get('positions', {})
            
            exit_signals = []
            
            for symbol, position_data in positions.items():
                position_value = position_data.get('value_usd', 0)
                if position_value < 25:  # Skip small positions
                    continue
                
                # Analyze current momentum for exit signals
                current_momentum = await self.analyze_symbol_momentum(symbol)
                if not current_momentum:
                    continue
                
                # Check for momentum breakdown
                exit_signal = None
                
                # 1. Momentum reversal (strength dropped significantly)
                if current_momentum.momentum_strength < 0.3:
                    exit_signal = QuickExitSignal(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        exit_reason="momentum_reversal",
                        urgency="high",
                        confidence=0.8,
                        price_target=position_data.get('current_price', 0) * 0.95,
                        expected_completion="minutes",
                        exit_percentage=0.8,  # Exit 80% of position
                        momentum_breakdown={
                            "momentum_strength": current_momentum.momentum_strength,
                            "volume_decline": current_momentum.volume_surge_ratio < 2.0,
                            "social_sentiment_drop": current_momentum.social_sentiment_spike < 0.3
                        }
                    )
                
                # 2. Profit target reached
                elif position_value > 0 and current_momentum.momentum_strength > 0.7:
                    # Check if we've reached profit target (simplified)
                    exit_signal = QuickExitSignal(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        exit_reason="profit_target",
                        urgency="medium",
                        confidence=0.6,
                        price_target=position_data.get('current_price', 0) * 1.05,
                        expected_completion="minutes",
                        exit_percentage=0.5,  # Take profits on 50%
                        momentum_breakdown={
                            "momentum_maintained": True,
                            "profit_protection": True
                        }
                    )
                
                if exit_signal:
                    exit_signals.append(exit_signal)
                    logger.warning(f"⚡ QUICK EXIT: {symbol} - {exit_signal.exit_reason} ({exit_signal.urgency} urgency)")
            
            return exit_signals
            
        except Exception as e:
            logger.error(f"Error detecting exit signals: {e}")
            return []
    
    async def detect_viral_coins(self) -> Dict:
        """Detect currently viral/trending coins."""
        try:
            momentum_signals = await self.detect_momentum_opportunities()
            
            viral_coins = []
            trending_analysis = {
                "timestamp": datetime.now().isoformat(),
                "total_opportunities": len(momentum_signals),
                "viral_count": 0,
                "trending_coins": []
            }
            
            for signal in momentum_signals:
                if 'viral' in signal.momentum_type or len(signal.viral_indicators) > 2:
                    viral_info = {
                        "symbol": signal.symbol,
                        "momentum_strength": signal.momentum_strength,
                        "viral_indicators": signal.viral_indicators,
                        "volume_surge": signal.volume_surge_ratio,
                        "social_sentiment": signal.social_sentiment_spike,
                        "caution_flags": signal.caution_flags,
                        "risk_level": "high" if signal.risk_multiplier > 1.5 else "medium"
                    }
                    viral_coins.append(viral_info)
            
            trending_analysis["viral_count"] = len(viral_coins)
            trending_analysis["trending_coins"] = viral_coins
            
            return trending_analysis
            
        except Exception as e:
            logger.error(f"Error detecting viral coins: {e}")
            return {"error": str(e)}

async def main():
    """Main function to run the service."""
    detector = MomentumHypeDetector()
    
    # Start the FastAPI server
    config = uvicorn.Config(
        detector.app,
        host="0.0.0.0",
        port=8029,
        log_level="info"
    )
    server = uvicorn.Server(config)
    
    logger.info("🚀 Momentum & Hype Detector starting on port 8029...")
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())
