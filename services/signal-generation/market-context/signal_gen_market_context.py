#!/usr/bin/env python3
"""
Signal Generation Market Context Service
Handles market sentiment, regime detection, and momentum analysis

This microservice extracts market context functionality from enhanced_signal_generator.py
Responsibilities:
- Market sentiment analysis and trend detection
- Market regime detection (bull/bear/sideways)
- Momentum and viral coin detection
- Social media hype analysis
- Market selloff and recovery detection
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

class MarketContextRequest(BaseModel):
    """Request model for market context analysis"""
    symbol: str
    analysis_type: str = "comprehensive"  # sentiment, momentum, regime, comprehensive
    lookback_minutes: int = 60
    timestamp: Optional[str] = None

class SentimentData(BaseModel):
    """Sentiment analysis data"""
    sentiment_score: float
    sentiment_trend: str
    volume_score: float
    confidence: float

class MomentumData(BaseModel):
    """Momentum analysis data"""
    momentum_detected: bool
    momentum_strength: float
    momentum_type: str
    viral_indicators: List[str]
    caution_flags: List[str]
    risk_multiplier: float
    hype_rating: float

class MarketRegimeData(BaseModel):
    """Market regime data"""
    regime: str
    strength: float
    trend_direction: str
    volatility_level: str

class MarketContextResponse(BaseModel):
    """Response model for market context analysis"""
    symbol: str
    sentiment_data: SentimentData
    momentum_data: MomentumData
    regime_data: MarketRegimeData
    market_conditions: Dict[str, Any]
    timestamp: str

class SignalGenMarketContext:
    """Market Context Analysis Service for signal generation"""
    
    def __init__(self):
        self.app = FastAPI(
            title="Signal Generation Market Context",
            description="Market sentiment, momentum, and regime analysis for trading signals",
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
        self.sentiment_service_url = os.getenv('SENTIMENT_SERVICE_URL', 'http://host.docker.internal:8032')
        self.momentum_detector_url = os.getenv('MOMENTUM_DETECTOR_URL', 'http://host.docker.internal:8029')
        self.market_selloff_url = os.getenv('MARKET_SELLOFF_URL', 'http://host.docker.internal:8028')
        
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
                'database': os.getenv('DATABASE_NAME', 'crypto_prices'),
                'pool_name': 'market_context_pool',
                'pool_size': 5,
                'pool_reset_session': True,
                'autocommit': True
            }
            
            self.db_pool = pooling.MySQLConnectionPool(**db_config)
            logger.info("✅ Database connection pool initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup database: {e}")
            self.db_pool = None
    
    async def get_sentiment_data(self, symbol: str, lookback_minutes: int = 60) -> SentimentData:
        """Get sentiment data from sentiment analysis service"""
        try:
            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                payload = {
                    "symbol": symbol,
                    "lookback_minutes": lookback_minutes
                }
                
                async with session.post(
                    f"{self.sentiment_service_url}/analyze_sentiment",
                    json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.debug(f"📊 {symbol} sentiment: {data.get('sentiment_score', 0.0):.3f}")
                        
                        return SentimentData(
                            sentiment_score=data.get('sentiment_score', 0.0),
                            sentiment_trend=data.get('sentiment_trend', 'STABLE'),
                            volume_score=data.get('volume_score', 0.0),
                            confidence=data.get('confidence', 0.0)
                        )
                    else:
                        logger.warning(f"Sentiment service unavailable for {symbol}: {response.status}")
                        
        except Exception as e:
            logger.warning(f"Error getting sentiment for {symbol}: {e}")
        
        # Return default sentiment data
        return SentimentData(
            sentiment_score=0.0,
            sentiment_trend='STABLE',
            volume_score=0.0,
            confidence=0.0
        )
    
    async def get_momentum_data(self, symbol: str) -> MomentumData:
        """Get momentum and viral coin detection data"""
        try:
            # Try external momentum detector first
            external_momentum = None
            try:
                timeout = aiohttp.ClientTimeout(total=10)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(
                        f"{self.momentum_detector_url}/momentum/symbol/{symbol}"
                    ) as response:
                        if response.status == 200:
                            external_momentum = await response.json()
                            if not external_momentum.get('error'):
                                logger.debug(f"🚀 {symbol} external momentum: {external_momentum.get('momentum_strength', 0):.2f}")
            except Exception as e:
                logger.debug(f"External momentum detector unavailable for {symbol}: {e}")
            
            # Get social hype metrics
            hype_data = await self.get_social_hype_metrics(symbol)
            hype_rating = hype_data.get('hype_rating', 0.0)
            
            # Calculate social momentum indicators
            combined_strength = hype_data.get('combined_strength', {})
            recent_mentions = combined_strength.get('1h', {}).get('mentions', 0)
            recent_sentiment = combined_strength.get('1h', {}).get('avg_sentiment', 0.0)
            daily_mentions = combined_strength.get('24h', {}).get('mentions', 0)
            
            # Social momentum calculation
            social_momentum_strength = 0.0
            if daily_mentions > 0:
                mention_velocity = min((recent_mentions * 24) / max(daily_mentions, 1), 5.0)
                sentiment_boost = max(recent_sentiment, 0) * 2
                social_momentum_strength = (hype_rating / 10.0) * (1 + mention_velocity) * (1 + sentiment_boost)
            
            # Determine viral indicators
            viral_indicators = []
            if recent_mentions > 10:
                viral_indicators.append("high_mention_volume")
            if recent_sentiment > 0.3:
                viral_indicators.append("positive_sentiment_spike")
            if daily_mentions > 0 and (recent_mentions * 24) / daily_mentions > 3:
                viral_indicators.append("accelerating_discussion")
            if hype_rating > 20:
                viral_indicators.append("strong_social_momentum")
            
            # Caution flags
            caution_flags = []
            if recent_sentiment < -0.2:
                caution_flags.append("negative_sentiment")
            if recent_mentions > 20 and recent_sentiment < -0.1:
                caution_flags.append("fud_campaign")
            
            # If external momentum available, enhance with social data
            if external_momentum:
                base_strength = external_momentum.get('momentum_strength', 0.0)
                social_boost = min(social_momentum_strength * 0.3, 1.0)
                final_strength = base_strength * (1 + social_boost)
                
                external_viral = external_momentum.get('viral_indicators', [])
                combined_viral = list(set(external_viral + viral_indicators))
                
                external_caution = external_momentum.get('caution_flags', [])
                combined_caution = list(set(external_caution + caution_flags))
                
                return MomentumData(
                    momentum_detected=True,
                    momentum_strength=final_strength,
                    momentum_type=external_momentum.get('momentum_type', 'none'),
                    viral_indicators=combined_viral,
                    caution_flags=combined_caution,
                    risk_multiplier=external_momentum.get('risk_multiplier', 1.0),
                    hype_rating=hype_rating
                )
            
            # Use social momentum analysis only
            momentum_detected = social_momentum_strength > 1.0 or len(viral_indicators) >= 2
            momentum_type = 'none'
            
            if social_momentum_strength > 5.0:
                momentum_type = 'viral_breakout'
            elif social_momentum_strength > 2.0:
                momentum_type = 'social_trend'
            elif len(viral_indicators) >= 2:
                momentum_type = 'social_buzz'
            
            risk_multiplier = 1.0
            if len(caution_flags) > 0:
                risk_multiplier += len(caution_flags) * 0.2
            if recent_sentiment < -0.3:
                risk_multiplier += 0.5
            
            return MomentumData(
                momentum_detected=momentum_detected,
                momentum_strength=social_momentum_strength,
                momentum_type=momentum_type,
                viral_indicators=viral_indicators,
                caution_flags=caution_flags,
                risk_multiplier=risk_multiplier,
                hype_rating=hype_rating
            )
            
        except Exception as e:
            logger.error(f"❌ Error getting momentum data for {symbol}: {e}")
            return MomentumData(
                momentum_detected=False,
                momentum_strength=0.0,
                momentum_type='none',
                viral_indicators=[],
                caution_flags=[],
                risk_multiplier=1.0,
                hype_rating=0.0
            )
    
    async def get_social_hype_metrics(self, symbol: str) -> Dict[str, Any]:
        """Get social hype metrics from social collectors"""
        hype_data = {
            'reddit_metrics': {},
            'twitter_metrics': {},
            'combined_strength': {},
            'hype_rating': 0.0
        }
        
        try:
            # Try Reddit hype metrics
            timeout = aiohttp.ClientTimeout(total=3)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                try:
                    async with session.get(f"http://localhost:8033/hype_metrics/{symbol}") as response:
                        if response.status == 200:
                            data = await response.json()
                            hype_data['reddit_metrics'] = data.get(symbol, {})
                except Exception as e:
                    logger.debug(f"Could not get Reddit hype metrics for {symbol}: {e}")
                
                # Try Twitter hype metrics
                try:
                    async with session.get(f"http://localhost:8034/hype_metrics/{symbol}") as response:
                        if response.status == 200:
                            data = await response.json()
                            hype_data['twitter_metrics'] = data.get(symbol, {})
                except Exception as e:
                    logger.debug(f"Could not get Twitter hype metrics for {symbol}: {e}")
        
        except Exception as e:
            logger.debug(f"Error setting up session for hype metrics: {e}")
        
        # Calculate combined sentiment strength
        windows = ['1h', '4h', '24h', '1w']
        for window in windows:
            reddit_strength = hype_data['reddit_metrics'].get(window, {}).get('sentiment_strength', 0.0)
            twitter_strength = hype_data['twitter_metrics'].get(window, {}).get('sentiment_strength', 0.0)
            
            reddit_mentions = hype_data['reddit_metrics'].get(window, {}).get('mentions', 0)
            twitter_mentions = hype_data['twitter_metrics'].get(window, {}).get('mentions', 0)
            total_mentions = reddit_mentions + twitter_mentions
            
            if reddit_mentions > 0 and twitter_mentions > 0:
                reddit_sentiment = hype_data['reddit_metrics'].get(window, {}).get('avg_sentiment', 0.0)
                twitter_sentiment = hype_data['twitter_metrics'].get(window, {}).get('avg_sentiment', 0.0)
                combined_sentiment = (reddit_sentiment * reddit_mentions + twitter_sentiment * twitter_mentions) / total_mentions
                combined_strength = combined_sentiment * total_mentions
            else:
                combined_strength = reddit_strength + twitter_strength
                combined_sentiment = (reddit_strength + twitter_strength) / max(total_mentions, 1)
            
            hype_data['combined_strength'][window] = {
                'sentiment_strength': combined_strength,
                'mentions': total_mentions,
                'avg_sentiment': combined_sentiment,
                'reddit_mentions': reddit_mentions,
                'twitter_mentions': twitter_mentions
            }
        
        # Calculate overall hype rating
        recent_strength = hype_data['combined_strength'].get('1h', {}).get('sentiment_strength', 0.0)
        medium_strength = hype_data['combined_strength'].get('4h', {}).get('sentiment_strength', 0.0)
        daily_strength = hype_data['combined_strength'].get('24h', {}).get('sentiment_strength', 0.0)
        
        hype_data['hype_rating'] = (recent_strength * 0.5 + medium_strength * 0.3 + daily_strength * 0.2)
        
        return hype_data
    
    async def get_market_regime(self) -> MarketRegimeData:
        """Get current market regime (bull/bear/sideways)"""
        try:
            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(f"{self.sentiment_service_url}/regime") as response:
                    if response.status == 200:
                        regime_data = await response.json()
                        return MarketRegimeData(
                            regime=regime_data.get('regime', 'SIDEWAYS'),
                            strength=regime_data.get('strength', 0.5),
                            trend_direction=regime_data.get('trend_direction', 'NEUTRAL'),
                            volatility_level=regime_data.get('volatility_level', 'MEDIUM')
                        )
        except Exception as e:
            logger.warning(f"Error getting market regime: {e}")
        
        return MarketRegimeData(
            regime='SIDEWAYS',
            strength=0.5,
            trend_direction='NEUTRAL',
            volatility_level='MEDIUM'
        )
    
    async def get_market_conditions(self) -> Dict[str, Any]:
        """Get comprehensive market conditions"""
        try:
            market_conditions = {}
            
            # Get selloff status
            try:
                timeout = aiohttp.ClientTimeout(total=10)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(f"{self.market_selloff_url}/selloff/current") as response:
                        if response.status == 200:
                            selloff_data = await response.json()
                            market_conditions['selloff'] = selloff_data
                        else:
                            market_conditions['selloff'] = {
                                'severity': 'mild',
                                'confidence': 0.0,
                                'suggested_cash_allocation': 0.05,
                                'emergency_liquidation': False
                            }
            except Exception as e:
                logger.debug(f"Could not get selloff data: {e}")
                market_conditions['selloff'] = {
                    'severity': 'mild',
                    'confidence': 0.0,
                    'suggested_cash_allocation': 0.05,
                    'emergency_liquidation': False
                }
            
            # Get recovery status
            try:
                timeout = aiohttp.ClientTimeout(total=10)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(f"{self.market_selloff_url}/recovery/current") as response:
                        if response.status == 200:
                            recovery_data = await response.json()
                            market_conditions['recovery'] = recovery_data
                        else:
                            market_conditions['recovery'] = {
                                'recovery_stage': 'early',
                                'confidence': 0.0,
                                'suggested_re_entry': 0.0
                            }
            except Exception as e:
                logger.debug(f"Could not get recovery data: {e}")
                market_conditions['recovery'] = {
                    'recovery_stage': 'early',
                    'confidence': 0.0,
                    'suggested_re_entry': 0.0
                }
            
            return market_conditions
            
        except Exception as e:
            logger.error(f"❌ Error getting market conditions: {e}")
            return {
                'selloff': {'severity': 'mild', 'confidence': 0.0},
                'recovery': {'recovery_stage': 'early', 'confidence': 0.0}
            }
    
    async def analyze_market_context(self, request: MarketContextRequest) -> MarketContextResponse:
        """Main market context analysis endpoint"""
        try:
            # Get all market context data
            if request.analysis_type in ["comprehensive", "sentiment"]:
                sentiment_data = await self.get_sentiment_data(request.symbol, request.lookback_minutes)
            else:
                sentiment_data = SentimentData(sentiment_score=0.0, sentiment_trend='STABLE', volume_score=0.0, confidence=0.0)
            
            if request.analysis_type in ["comprehensive", "momentum"]:
                momentum_data = await self.get_momentum_data(request.symbol)
            else:
                momentum_data = MomentumData(momentum_detected=False, momentum_strength=0.0, momentum_type='none', 
                                           viral_indicators=[], caution_flags=[], risk_multiplier=1.0, hype_rating=0.0)
            
            if request.analysis_type in ["comprehensive", "regime"]:
                regime_data = await self.get_market_regime()
            else:
                regime_data = MarketRegimeData(regime='SIDEWAYS', strength=0.5, trend_direction='NEUTRAL', volatility_level='MEDIUM')
            
            market_conditions = await self.get_market_conditions()
            
            logger.info(f"📊 Market context for {request.symbol}: "
                       f"sentiment={sentiment_data.sentiment_score:.3f}, "
                       f"momentum={momentum_data.momentum_strength:.3f}, "
                       f"regime={regime_data.regime}")
            
            return MarketContextResponse(
                symbol=request.symbol,
                sentiment_data=sentiment_data,
                momentum_data=momentum_data,
                regime_data=regime_data,
                market_conditions=market_conditions,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"❌ Error analyzing market context for {request.symbol}: {e}")
            raise HTTPException(status_code=500, detail=f"Market context analysis failed: {str(e)}")
    
    def setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            db_status = "connected" if self.db_pool else "disconnected"
            
            return {
                "status": "healthy",
                "service": "signal-gen-market-context",
                "database_status": db_status,
                "external_services": {
                    "sentiment_service": self.sentiment_service_url,
                    "momentum_detector": self.momentum_detector_url,
                    "market_selloff": self.market_selloff_url
                },
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.post("/analyze", response_model=MarketContextResponse)
        async def analyze_market_context_endpoint(request: MarketContextRequest):
            """Market context analysis endpoint"""
            return await self.analyze_market_context(request)
        
        @self.app.post("/batch_analyze")
        async def batch_analyze_endpoint(requests: List[MarketContextRequest]):
            """Batch market context analysis endpoint"""
            results = []
            for request in requests:
                try:
                    result = await self.analyze_market_context(request)
                    results.append(result)
                except Exception as e:
                    logger.error(f"❌ Error analyzing market context for {request.symbol}: {e}")
                    continue
            
            return {
                "results": results,
                "successful": len(results),
                "total": len(requests),
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.get("/sentiment/{symbol}")
        async def get_sentiment_endpoint(symbol: str, lookback_minutes: int = 60):
            """Get sentiment data for a specific symbol"""
            sentiment_data = await self.get_sentiment_data(symbol, lookback_minutes)
            return sentiment_data
        
        @self.app.get("/momentum/{symbol}")
        async def get_momentum_endpoint(symbol: str):
            """Get momentum data for a specific symbol"""
            momentum_data = await self.get_momentum_data(symbol)
            return momentum_data
        
        @self.app.get("/regime")
        async def get_regime_endpoint():
            """Get current market regime"""
            regime_data = await self.get_market_regime()
            return regime_data
        
        @self.app.get("/market_conditions")
        async def get_market_conditions_endpoint():
            """Get comprehensive market conditions"""
            market_conditions = await self.get_market_conditions()
            return market_conditions
        
        @self.app.get("/status")
        async def get_status():
            """Detailed status information"""
            return {
                "service": "signal-gen-market-context",
                "version": "1.0.0",
                "database_connected": bool(self.db_pool),
                "analysis_types": ["comprehensive", "sentiment", "momentum", "regime"],
                "external_services_configured": {
                    "sentiment_service": bool(self.sentiment_service_url),
                    "momentum_detector": bool(self.momentum_detector_url),
                    "market_selloff": bool(self.market_selloff_url)
                },
                "timestamp": datetime.now().isoformat()
            }

def main():
    """Main function to run the Market Context service"""
    try:
        logger.info("🚀 Starting Signal Generation Market Context...")
        
        market_context = SignalGenMarketContext()
        
        # Get port from environment or use default
        port = int(os.getenv('MARKET_CONTEXT_PORT', 8053))
        
        logger.info(f"📊 Market Context service starting on port {port}")
        
        # Run the FastAPI application
        uvicorn.run(
            market_context.app,
            host="0.0.0.0",
            port=port,
            log_level="info"
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to start Market Context: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
