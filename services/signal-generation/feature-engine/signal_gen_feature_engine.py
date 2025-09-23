#!/usr/bin/env python3
"""
Signal Generation Feature Engineering Service
Handles feature engineering and technical analysis calculations

This microservice extracts the feature engineering functionality from enhanced_signal_generator.py
Responsibilities:
- Engineer missing features from existing data (71+ features)
- Calculate technical indicators (moving averages, volatility, momentum)
- Time-based feature engineering
- Market correlation features
- Volume analysis features
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FeatureEngineeringRequest(BaseModel):
    """Request model for feature engineering"""
    symbol: str
    raw_data: Dict[str, Any]
    engineering_type: str = "comprehensive"  # minimal, standard, comprehensive
    timestamp: Optional[str] = None

class FeatureEngineeringResponse(BaseModel):
    """Response model for engineered features"""
    symbol: str
    engineered_features: Dict[str, float]
    features_count: int
    engineering_type: str
    timestamp: str

class SignalGenFeatureEngine:
    """Feature Engineering Service for signal generation"""
    
    def __init__(self):
        self.app = FastAPI(
            title="Signal Generation Feature Engine",
            description="Feature engineering and technical analysis for trading signals",
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
                'pool_name': 'feature_engine_pool',
                'pool_size': 5,
                'pool_reset_session': True,
                'autocommit': True
            }
            
            self.db_pool = pooling.MySQLConnectionPool(**db_config)
            logger.info("✅ Database connection pool initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup database: {e}")
            self.db_pool = None
    
    def engineer_time_features(self, timestamp: str) -> Dict[str, float]:
        """Engineer time-based features (17 features)"""
        try:
            dt = pd.to_datetime(timestamp)
            
            features = {
                'hour': float(dt.hour),
                'day_of_week': float(dt.dayofweek),
                'month': float(dt.month),
                'quarter': float(dt.quarter),
                'is_weekend': float(dt.dayofweek >= 5),
                'is_month_start': float(dt.day <= 3),
                'is_month_end': float(dt.day >= 28),
                'is_quarter_end': float(dt.month in [3, 6, 9, 12]),
                'is_us_market_hours': float(9 <= dt.hour <= 16),
                'is_asian_session': float(dt.hour >= 22 or dt.hour <= 8),
                'is_european_session': float(8 <= dt.hour <= 16),
                
                # Cyclical encoding
                'hour_sin': float(np.sin(2 * np.pi * dt.hour / 24)),
                'hour_cos': float(np.cos(2 * np.pi * dt.hour / 24)),
                'day_sin': float(np.sin(2 * np.pi * dt.dayofweek / 7)),
                'day_cos': float(np.cos(2 * np.pi * dt.dayofweek / 7)),
                'month_sin': float(np.sin(2 * np.pi * dt.month / 12)),
                'month_cos': float(np.cos(2 * np.pi * dt.month / 12))
            }
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Error engineering time features: {e}")
            return {}
    
    def engineer_technical_features(self, raw_data: Dict[str, Any]) -> Dict[str, float]:
        """Engineer technical analysis features"""
        try:
            features = {}
            
            # Extract price data
            price = float(raw_data.get('current_price', 0))
            if price <= 0:
                return features
            
            # Price change features
            price_change_24h = float(raw_data.get('price_change_percentage_24h', 0))
            price_change_7d = float(raw_data.get('price_change_percentage_7d', 0))
            
            features.update({
                'price_momentum_24h': price_change_24h,
                'price_momentum_7d': price_change_7d,
                'price_momentum_ratio': price_change_24h / max(abs(price_change_7d), 0.1),
                'price_acceleration': price_change_24h - price_change_7d / 7,  # Daily acceleration
            })
            
            # Volume features
            volume_24h = float(raw_data.get('volume_24h', 0))
            if volume_24h > 0:
                market_cap = float(raw_data.get('market_cap', 0))
                if market_cap > 0:
                    features['volume_to_mcap_ratio'] = volume_24h / market_cap
                else:
                    features['volume_to_mcap_ratio'] = 0.0
            else:
                features['volume_to_mcap_ratio'] = 0.0
            
            # Technical indicator proxies (simplified since we don't have historical data)
            features.update({
                'rsi_proxy': min(100, max(0, 50 + price_change_24h * 2)),  # Rough RSI approximation
                'momentum_strength': abs(price_change_24h) / max(abs(price_change_7d), 1),
                'volatility_proxy': abs(price_change_24h) / 10,  # Rough volatility measure
            })
            
            # Price position features
            high_24h = float(raw_data.get('high_24h', price))
            low_24h = float(raw_data.get('low_24h', price))
            
            if high_24h > low_24h:
                features['price_position_24h'] = (price - low_24h) / (high_24h - low_24h)
                features['near_high_24h'] = float(features['price_position_24h'] > 0.9)
                features['near_low_24h'] = float(features['price_position_24h'] < 0.1)
            else:
                features.update({
                    'price_position_24h': 0.5,
                    'near_high_24h': 0.0,
                    'near_low_24h': 0.0
                })
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Error engineering technical features: {e}")
            return {}
    
    def engineer_market_features(self, raw_data: Dict[str, Any]) -> Dict[str, float]:
        """Engineer market-related features"""
        try:
            features = {}
            
            # Market cap features
            market_cap = float(raw_data.get('market_cap', 0))
            if market_cap > 0:
                features.update({
                    'market_cap_log': float(np.log10(market_cap)),
                    'is_large_cap': float(market_cap > 10e9),  # > $10B
                    'is_mid_cap': float(1e9 <= market_cap <= 10e9),  # $1B - $10B
                    'is_small_cap': float(market_cap < 1e9),  # < $1B
                })
            else:
                features.update({
                    'market_cap_log': 0.0,
                    'is_large_cap': 0.0,
                    'is_mid_cap': 0.0,
                    'is_small_cap': 1.0,  # Default to small cap if unknown
                })
            
            # Rank features
            market_cap_rank = int(raw_data.get('market_cap_rank', 1000))
            features.update({
                'market_cap_rank': float(market_cap_rank),
                'is_top_10': float(market_cap_rank <= 10),
                'is_top_50': float(market_cap_rank <= 50),
                'is_top_100': float(market_cap_rank <= 100),
                'rank_score': float(max(0, 1 - market_cap_rank / 1000))  # Normalized rank score
            })
            
            # Supply features
            circulating_supply = float(raw_data.get('circulating_supply', 0))
            total_supply = float(raw_data.get('total_supply', 0))
            max_supply = float(raw_data.get('max_supply', 0))
            
            if circulating_supply > 0 and total_supply > 0:
                features['supply_ratio'] = circulating_supply / total_supply
            else:
                features['supply_ratio'] = 1.0
            
            if max_supply and max_supply > 0 and circulating_supply > 0:
                features['inflation_potential'] = (max_supply - circulating_supply) / max_supply
            else:
                features['inflation_potential'] = 0.0
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Error engineering market features: {e}")
            return {}
    
    def engineer_comprehensive_features(self, symbol: str, raw_data: Dict[str, Any], timestamp: str) -> Dict[str, float]:
        """Engineer comprehensive feature set for ML model"""
        try:
            all_features = {}
            
            # Time-based features
            time_features = self.engineer_time_features(timestamp)
            all_features.update(time_features)
            
            # Technical features
            technical_features = self.engineer_technical_features(raw_data)
            all_features.update(technical_features)
            
            # Market features
            market_features = self.engineer_market_features(raw_data)
            all_features.update(market_features)
            
            # Symbol-specific features
            symbol_features = self.engineer_symbol_features(symbol)
            all_features.update(symbol_features)
            
            # Fill any NaN or None values with 0
            for key, value in all_features.items():
                if pd.isna(value) or value is None:
                    all_features[key] = 0.0
                else:
                    all_features[key] = float(value)
            
            logger.info(f"🔧 Engineered {len(all_features)} features for {symbol}")
            return all_features
            
        except Exception as e:
            logger.error(f"❌ Error engineering comprehensive features for {symbol}: {e}")
            return {}
    
    def engineer_symbol_features(self, symbol: str) -> Dict[str, float]:
        """Engineer symbol-specific features"""
        try:
            features = {}
            
            # Symbol category features (simplified categorization)
            major_symbols = ['BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'SOL', 'DOGE', 'DOT', 'AVAX', 'SHIB']
            defi_symbols = ['UNI', 'LINK', 'AAVE', 'MKR', 'COMP', 'SUSHI', 'YFI', 'CRV', '1INCH', 'BAL']
            layer1_symbols = ['ETH', 'ADA', 'SOL', 'DOT', 'AVAX', 'ALGO', 'ATOM', 'NEAR', 'FTM', 'LUNA']
            
            features.update({
                'is_major_symbol': float(symbol in major_symbols),
                'is_defi_symbol': float(symbol in defi_symbols),
                'is_layer1_symbol': float(symbol in layer1_symbols),
                'is_btc': float(symbol == 'BTC'),
                'is_eth': float(symbol == 'ETH'),
            })
            
            # Symbol string features (simple encoding)
            features.update({
                'symbol_length': float(len(symbol)),
                'has_digit': float(any(c.isdigit() for c in symbol)),
                'starts_with_vowel': float(symbol[0].upper() in 'AEIOU') if symbol else 0.0,
            })
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Error engineering symbol features for {symbol}: {e}")
            return {}
    
    async def engineer_features(self, request: FeatureEngineeringRequest) -> FeatureEngineeringResponse:
        """Main feature engineering endpoint"""
        try:
            timestamp = request.timestamp or datetime.now().isoformat()
            
            if request.engineering_type == "comprehensive":
                engineered_features = self.engineer_comprehensive_features(
                    request.symbol, 
                    request.raw_data, 
                    timestamp
                )
            elif request.engineering_type == "technical":
                engineered_features = self.engineer_technical_features(request.raw_data)
            elif request.engineering_type == "time":
                engineered_features = self.engineer_time_features(timestamp)
            else:
                # Default to comprehensive
                engineered_features = self.engineer_comprehensive_features(
                    request.symbol, 
                    request.raw_data, 
                    timestamp
                )
            
            return FeatureEngineeringResponse(
                symbol=request.symbol,
                engineered_features=engineered_features,
                features_count=len(engineered_features),
                engineering_type=request.engineering_type,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"❌ Error in feature engineering for {request.symbol}: {e}")
            raise HTTPException(status_code=500, detail=f"Feature engineering failed: {str(e)}")
    
    def setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            db_status = "connected" if self.db_pool else "disconnected"
            
            return {
                "status": "healthy",
                "service": "signal-gen-feature-engine",
                "database_status": db_status,
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.post("/engineer", response_model=FeatureEngineeringResponse)
        async def engineer_features_endpoint(request: FeatureEngineeringRequest):
            """Feature engineering endpoint"""
            return await self.engineer_features(request)
        
        @self.app.post("/batch_engineer")
        async def batch_engineer_endpoint(requests: List[FeatureEngineeringRequest]):
            """Batch feature engineering endpoint"""
            results = []
            for request in requests:
                try:
                    result = await self.engineer_features(request)
                    results.append(result)
                except Exception as e:
                    logger.error(f"❌ Error engineering features for {request.symbol}: {e}")
                    continue
            
            return {
                "results": results,
                "successful": len(results),
                "total": len(requests),
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.get("/feature_info")
        async def feature_info():
            """Get information about available feature types"""
            return {
                "service": "signal-gen-feature-engine",
                "available_types": [
                    "comprehensive",
                    "technical", 
                    "time",
                    "market",
                    "symbol"
                ],
                "feature_categories": {
                    "time_features": 17,
                    "technical_features": 15,
                    "market_features": 12,
                    "symbol_features": 8
                },
                "total_features": "50+",
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.get("/status")
        async def get_status():
            """Detailed status information"""
            return {
                "service": "signal-gen-feature-engine",
                "version": "1.0.0",
                "database_connected": bool(self.db_pool),
                "available_engineering_types": ["comprehensive", "technical", "time", "market", "symbol"],
                "timestamp": datetime.now().isoformat()
            }

def main():
    """Main function to run the Feature Engine service"""
    try:
        logger.info("🚀 Starting Signal Generation Feature Engine...")
        
        feature_engine = SignalGenFeatureEngine()
        
        # Get port from environment or use default
        port = int(os.getenv('FEATURE_ENGINE_PORT', 8052))
        
        logger.info(f"🔧 Feature Engine service starting on port {port}")
        
        # Run the FastAPI application
        uvicorn.run(
            feature_engine.app,
            host="0.0.0.0",
            port=port,
            log_level="info"
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to start Feature Engine: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()