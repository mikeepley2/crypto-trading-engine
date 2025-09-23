#!/usr/bin/env python3
"""
Signal Generation Orchestrator Serv        self.service_urls = {
            'ml_engine': f"http://signal-gen-ml-engine.crypto-trading.svc.cluster.local:8051",
            'feature_engine': f"http://signal-gen-feature-engine.crypto-trading.svc.cluster.local:8052", 
            'market_context': f"http://signal-gen-market-context.crypto-trading.svc.cluster.local:8053",
            'portfolio': f"http://signal-gen-portfolio.crypto-trading.svc.cluster.local:8054",
            'risk_mgmt': f"http://signal-gen-risk-mgmt.crypto-trading.svc.cluster.local:8055",
            'analytics': f"http://signal-gen-analytics.crypto-trading.svc.cluster.local:8056"
        }rdinates all signal generation microservices and maintains API compatibility
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
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
import uvicorn
from pydantic import BaseModel
import mysql.connector
from mysql.connector import pooling

# Setup basic logging for Kubernetes
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SignalGenerationRequest(BaseModel):
    """Request model for signal generation"""
    symbol: str
    analysis_type: str = "comprehensive"  # minimal, standard, comprehensive
    timestamp: Optional[str] = None

class SignalGenerationResponse(BaseModel):
    """Response model for signal generation - maintains compatibility"""
    symbol: str
    signal: str  # BUY, SELL, HOLD
    signal_type: Optional[str] = None  # Alias for signal for backward compatibility
    confidence: float
    signal_strength: float
    recommended_action: str
    position_size: Optional[float] = None
    risk_assessment: Dict[str, Any]
    analysis_details: Dict[str, Any]
    timestamp: str

class SignalGenOrchestrator:
    """Signal Generation Orchestrator - coordinates all microservices"""
    
    def __init__(self):
        self.app = FastAPI(title="Signal Generation Orchestrator", version="1.0.0")
        
        # Setup Prometheus metrics
        self.instrumentator = Instrumentator()
        self.instrumentator.instrument(self.app).expose(self.app)
        
        self.db_pool = None
        self.session = None
        
        # Service endpoints
        self.services = {
            'ml_engine': f"http://signal-gen-ml-engine.crypto-trading.svc.cluster.local:8051",
            'feature_engine': f"http://signal-gen-feature-engine.crypto-trading.svc.cluster.local:8052", 
            'market_context': f"http://signal-gen-market-context.crypto-trading.svc.cluster.local:8053",
            'portfolio': f"http://signal-gen-portfolio.crypto-trading.svc.cluster.local:8054",
            'risk_mgmt': f"http://signal-gen-risk-mgmt.crypto-trading.svc.cluster.local:8055",
            'analytics': f"http://signal-gen-analytics.crypto-trading.svc.cluster.local:8056"
        }
        
        self.setup_routes()
        self.setup_database()
        self.setup_http_session()
    
    def setup_database(self):
        """Setup database connection pool"""
        try:
            db_config = {
                'host': os.getenv('MYSQL_HOST', 'host.docker.internal'),  # Use proper host for K8s
                'port': int(os.getenv('MYSQL_PORT', 3306)),
                'user': os.getenv('MYSQL_USER', 'news_collector'),
                'password': os.getenv('MYSQL_PASSWORD', '99Rules!'),
                'database': os.getenv('MYSQL_DATABASE', 'crypto_prices'),
                'pool_name': 'orchestrator_pool',
                'pool_size': 5,
                'pool_reset_session': True,
                'autocommit': True
            }
            
            self.db_pool = pooling.MySQLConnectionPool(**db_config)
            logger.info("Database connection pool initialized")
            
        except Exception as e:
            logger.error(f"Failed to setup database: {e}")
            self.db_pool = None
    
    def setup_http_session(self):
        """Setup HTTP session for service communication"""
        # Don't initialize session here - create it per request to avoid async context issues
        self.session = None
        logger.info("HTTP session setup configured")
    
    async def get_session(self):
        """Get or create HTTP session"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    async def call_service(self, service_name: str, endpoint: str, data: Dict = None) -> Dict:
        """Make HTTP call to a microservice"""
        import time
        start_time = time.time()
        
        try:
            url = f"{self.services[service_name]}{endpoint}"
            
            # Create session per request to avoid async context issues
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                if data:
                    async with session.post(url, json=data) as response:
                        duration_ms = (time.time() - start_time) * 1000
                        success = response.status == 200
                        
                        logger.info(f"Service call to {service_name}{endpoint} completed in {duration_ms:.2f}ms with status {response.status}")
                        
                        if success:
                            return await response.json()
                        else:
                            response_text = await response.text()
                            logger.error(f"Service call failed", extra={
                                "service": service_name,
                                "endpoint": endpoint,
                                "status_code": response.status,
                                "response": response_text
                            })
                            return {}
                else:
                    async with session.get(url) as response:
                        duration_ms = (time.time() - start_time) * 1000
                        success = response.status == 200
                        
                        logger.info(f"Service call to {service_name}{endpoint} completed in {duration_ms:.2f}ms with status {response.status}")
                        
                        if success:
                            return await response.json()
                        else:
                            logger.error(f"Service {service_name} returned {response.status}")
                            response_text = await response.text()
                            logger.error(f"Response: {response_text}")
                            return {}
                        
        except Exception as e:
            logger.error(f"Error calling {service_name}{endpoint}: {e}")
            return {}
    
    async def check_service_health(self, service_name: str) -> bool:
        """Check if a service is healthy"""
        try:
            result = await self.call_service(service_name, "/health")
            return result.get('status') == 'healthy'
        except:
            return False
    
    async def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        try:
            if not self.db_pool:
                logger.warning("Database not available, using mock price")
                return 50000.0  # Mock price for testing
            
            connection = self.db_pool.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            query = """
            SELECT close FROM crypto_prices 
            WHERE symbol = %s 
            ORDER BY timestamp DESC 
            LIMIT 1
            """
            
            cursor.execute(query, (symbol,))
            result = cursor.fetchone()
            
            cursor.close()
            connection.close()
            
            return float(result['close']) if result else None
            
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return None  # No fallback - require real data
    
    async def generate_features(self, symbol: str) -> Dict[str, float]:
        """Generate features using Feature Engine microservice"""
        try:
            # Prepare raw data for feature engineering
            raw_data = {
                "symbol": symbol,
                "current_price": await self.get_current_price(symbol),
                "timestamp": datetime.now().isoformat(),
                "volume": 1000000,  # Default volume
                "market_cap": 1000000000  # Default market cap
            }
            
            request_data = {
                "symbol": symbol,
                "raw_data": raw_data,
                "engineering_type": "comprehensive"
            }
            
            result = await self.call_service('feature_engine', '/engineer', request_data)
            if result and result.get('engineered_features'):
                return result.get('engineered_features', {})
            else:
                logger.warning(f"Feature engineering failed for {symbol}")
                return {}
            
        except Exception as e:
            logger.error(f"Error generating features for {symbol}: {e}")
            return {}
    
    async def get_ml_prediction(self, symbol: str, features: Dict[str, float]) -> Dict[str, Any]:
        """Get ML prediction using ML Engine microservice - NO FALLBACKS"""
        request_data = {
            "symbol": symbol,
            "features": features,
            "model_version": "latest"
        }
        
        result = await self.call_service('ml_engine', '/predict', request_data)
        if not result:
            raise Exception(f"ML Engine prediction failed for {symbol} - service must be operational")
        
        logger.info(f"✅ ML prediction received for {symbol}")
        return result
    
    async def get_market_context(self, symbol: str) -> Dict[str, Any]:
        """Get market context using Market Context microservice - NO FALLBACKS"""
        request_data = {
            "symbol": symbol
        }
        
        result = await self.call_service('market_context', '/analyze', request_data)
        if not result:
            raise Exception(f"Market Context service failed for {symbol} - service must be operational")
        
        return result
    
    async def assess_risk(self, symbol: str, signal_type: str, signal_strength: float, confidence: float, current_price: float) -> Dict[str, Any]:
        """Assess risk using Risk Management service"""
        try:
            request_data = {
                "symbol": symbol,
                "signal_type": signal_type,
                "confidence": confidence,
                "current_price": current_price,
                "position_size": None,
                "timestamp": None
            }
            
            result = await self.call_service('risk_mgmt', '/assess', request_data)
            return result
            
        except Exception as e:
            logger.error(f"Error assessing risk for {symbol}: {e}")
            return {}
    
    async def calculate_position_size(self, symbol: str, signal_strength: float, confidence: float, current_price: float) -> Dict[str, Any]:
        """Calculate position size using Portfolio service"""
        try:
            request_data = {
                "symbol": symbol,
                "confidence": confidence,
                "total_portfolio_value": 100000.0,  # Mock portfolio value
                "cash_balance": 20000.0,  # Mock cash balance (20% cash)
                "win_rate_override": None
            }
            
            result = await self.call_service('portfolio', '/kelly_sizing', request_data)
            return result
            
        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {e}")
            return {}
    
    def determine_signal(self, ml_prediction: Dict[str, Any], market_context: Dict[str, Any], risk_assessment: Dict[str, Any]) -> Tuple[str, float]:
        """Determine final signal based on all analysis"""
        try:
            # Get ML prediction - handle new ML Engine format
            if ml_prediction:
                ml_signal_type = ml_prediction.get('signal_type', 'HOLD')
                ml_confidence = ml_prediction.get('confidence', 0.5)
                prediction_probability = ml_prediction.get('prediction_probability', 0.5)
                
                # Map ML signal types to trading signals
                signal_mapping = {
                    'BUY': 'BUY',
                    'SELL': 'SELL', 
                    'HOLD': 'HOLD',
                    'STRONG_BUY': 'BUY',
                    'STRONG_SELL': 'SELL'
                }
                base_signal = signal_mapping.get(ml_signal_type, 'HOLD')
                base_confidence = ml_confidence
            else:
                base_signal = "HOLD"
                base_confidence = 0.1
            
            # Get market context - handle new Market Context format
            if market_context:
                sentiment_data = market_context.get('sentiment_data', {})
                regime_data = market_context.get('regime_data', {})
                market_sentiment = sentiment_data.get('sentiment_trend', 'STABLE')
                trend_direction = regime_data.get('trend_direction', 'NEUTRAL')
                context_score = sentiment_data.get('confidence', 0.5)
            else:
                market_sentiment = 'STABLE'
                trend_direction = 'NEUTRAL'
                context_score = 0.5
            
            # Get risk assessment - handle risk management format
            if risk_assessment:
                trade_approved = not risk_assessment.get('should_block_signal', False)
                risk_score = risk_assessment.get('risk_multiplier', 1.0)
            else:
                trade_approved = True
                risk_score = 1.0
            
            # Apply market context adjustments
            if market_sentiment == 'BULLISH' and base_signal == "BUY":
                base_confidence *= 1.1
            elif market_sentiment == 'BEARISH' and base_signal == "SELL":
                base_confidence *= 1.1
            elif market_sentiment == 'STABLE':
                base_confidence *= 0.9
            
            # Apply risk adjustments
            if not trade_approved:
                base_signal = "HOLD"
                base_confidence = 0.1
            else:
                base_confidence = min(base_confidence * risk_score, 1.0)
            
            # Ensure confidence is within valid range
            final_confidence = max(0.1, min(base_confidence, 1.0))
            
            return base_signal, final_confidence
            
        except Exception as e:
            logger.error(f"Error determining signal: {e}")
            return "HOLD", 0.1
    
    async def save_signal_to_database(self, signal_response: SignalGenerationResponse, current_price: float):
        """Save the generated signal to the trading_signals database for the signal bridge to pick up"""
        import time
        start_time = time.time()
        
        try:
            if not self.db_pool:
                logger.warning("Database not available, signal not saved to database")
                return
            
            connection = self.db_pool.get_connection()
            cursor = connection.cursor()
            
            # Map our signal types to database enum values
            signal_type_mapping = {
                "BUY": "BUY",
                "SELL": "SELL", 
                "STRONG_BUY": "STRONG_BUY",
                "STRONG_SELL": "STRONG_SELL",
                "HOLD": "HOLD"
            }
            db_signal_type = signal_type_mapping.get(signal_response.signal, "HOLD")
            
            # Extract analysis details
            ml_prediction = signal_response.analysis_details.get('ml_prediction', {})
            market_context = signal_response.analysis_details.get('market_context', {})
            risk_assessment = signal_response.risk_assessment
            
            # Insert signal into trading_signals table with comprehensive data
            insert_query = """
            INSERT INTO trading_signals (
                timestamp, symbol, price, signal_type, model, confidence, 
                threshold, regime, model_version, features_used, xgboost_confidence,
                data_source, is_mock, created_at, sentiment_score, features,
                prediction, llm_analysis, llm_confidence, processed
            ) VALUES (
                NOW(), %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, 0, NOW(), %s, %s,
                %s, %s, %s, 0
            )
            """
            
            # Prepare data values
            sentiment_data = market_context.get('sentiment_data', {})
            features_json = json.dumps(signal_response.analysis_details.get('features', {}))
            llm_analysis_json = json.dumps({
                'risk_assessment': risk_assessment,
                'market_context': market_context,
                'signal_strength': signal_response.signal_strength,
                'recommended_action': signal_response.recommended_action
            })
            
            cursor.execute(insert_query, (
                signal_response.symbol,                                    # symbol
                current_price,                                             # price
                db_signal_type,                                           # signal_type
                'orchestrator',                                           # model
                signal_response.confidence,                               # confidence
                0.75,                                                     # threshold (default for bridge)
                'sideways',                                              # regime (from market context)
                'v1.0',                                                  # model_version
                signal_response.analysis_details.get('features_count', 0), # features_used
                ml_prediction.get('confidence', signal_response.confidence), # xgboost_confidence
                'signal_orchestrator',                                    # data_source
                sentiment_data.get('sentiment_score', 0.0),              # sentiment_score
                features_json,                                           # features (JSON)
                ml_prediction.get('prediction_probability', 0.5),        # prediction
                llm_analysis_json,                                       # llm_analysis (JSON)
                signal_response.confidence                               # llm_confidence
            ))
            
            connection.commit()
            affected_rows = cursor.rowcount
            cursor.close()
            connection.close()
            
            duration_ms = (time.time() - start_time) * 1000
            
            logger.info(f"Database INSERT completed in {duration_ms:.2f}ms - affected rows: {affected_rows}")
            
            logger.info(f"✅ Signal saved to database: {signal_response.symbol} {signal_response.signal} (confidence: {signal_response.confidence:.3f})")
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            
            logger.error(f"Database INSERT failed in {duration_ms:.2f}ms - error: {str(e)}")
            
            logger.error(f"Error saving signal to database: {e}")
    
    async def generate_signal(self, request: SignalGenerationRequest) -> SignalGenerationResponse:
        """Generate comprehensive trading signal"""
        start_time = datetime.now()
        
        try:
            symbol = request.symbol
            logger.info(f"Generating signal for {symbol}")
            
            # Step 1: Get current price
            current_price = await self.get_current_price(symbol)
            if current_price <= 0:
                raise HTTPException(status_code=404, detail=f"No price data available for {symbol}")
            
            # Step 2: Generate features
            features = await self.generate_features(symbol)
            if not features:
                logger.warning(f"No features generated for {symbol}")
                features = {}
            
            # Step 3: Get ML prediction
            ml_prediction = await self.get_ml_prediction(symbol, features)
            
            # Step 4: Get market context
            market_context = await self.get_market_context(symbol)
            
            # Step 5: Determine preliminary signal
            signal, confidence = self.determine_signal(ml_prediction, market_context, {})
            
            # Step 6: Assess risk
            risk_assessment = await self.assess_risk(symbol, signal, confidence, confidence, current_price)
            
            # Step 7: Re-evaluate signal with risk assessment
            final_signal, final_confidence = self.determine_signal(ml_prediction, market_context, risk_assessment)
            
            # Step 8: Calculate position size if signal is not HOLD
            position_size_info = {}
            if final_signal != "HOLD":
                position_size_info = await self.calculate_position_size(symbol, final_confidence, final_confidence, current_price)
            
            # Step 9: Determine recommended action
            if final_signal == "BUY" and final_confidence >= 0.75:
                recommended_action = "STRONG_BUY"
            elif final_signal == "BUY" and final_confidence >= 0.6:
                recommended_action = "BUY"
            elif final_signal == "SELL" and final_confidence >= 0.75:
                recommended_action = "STRONG_SELL"
            elif final_signal == "SELL" and final_confidence >= 0.6:
                recommended_action = "SELL"
            else:
                recommended_action = "HOLD"
            
            # Compile analysis details
            analysis_details = {
                "ml_prediction": ml_prediction,
                "market_context": market_context,
                "features_count": len(features),
                "current_price": current_price,
                "processing_time_seconds": (datetime.now() - start_time).total_seconds()
            }
            
            # Get position size
            position_size = position_size_info.get('recommended_size', 0.0) if position_size_info else 0.0
            
            # Create response object
            response = SignalGenerationResponse(
                symbol=symbol,
                signal=final_signal,
                signal_type=final_signal,  # For backward compatibility
                confidence=final_confidence,
                signal_strength=final_confidence,  # For compatibility
                recommended_action=recommended_action,
                position_size=position_size,
                risk_assessment=risk_assessment,
                analysis_details=analysis_details,
                timestamp=datetime.now().isoformat()
            )
            
            # Step 10: Save signal to database for signal bridge to pick up
            await self.save_signal_to_database(response, current_price)
            
            logger.info(f"Signal generated for {symbol}: {final_signal} (confidence: {final_confidence:.3f}, action: {recommended_action})")
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating signal for {request.symbol}: {e}")
            raise HTTPException(status_code=500, detail=f"Signal generation failed: {str(e)}")
    
    def setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            db_status = "connected" if self.db_pool else "disconnected"
            session_status = "active" if self.session else "inactive"
            
            # Check service health
            service_health = {}
            for service_name in self.services.keys():
                service_health[service_name] = await self.check_service_health(service_name)
            
            return {
                "status": "healthy",
                "service": "signal-gen-orchestrator",
                "database_status": db_status,
                "session_status": session_status,
                "microservices_health": service_health,
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.post("/generate_signal", response_model=SignalGenerationResponse)
        async def generate_signal_endpoint(request: SignalGenerationRequest):
            """Generate comprehensive trading signal"""
            import uuid
            import time
            
            request_id = str(uuid.uuid4())
            start_time = time.time()
            
            # Log request
            logger.info(f"Request {request_id}: POST /generate_signal - symbol: {request.symbol}, analysis_type: {request.analysis_type}")
            
            try:
                result = await self.generate_signal(request)
                duration_ms = (time.time() - start_time) * 1000
                
                # Log successful response
                logger.info(f"Request {request_id}: 200 response in {duration_ms:.2f}ms - signal: {result.signal}, confidence: {result.confidence}")
                
                return result
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                
                # Log error response
                logger.error(f"Request {request_id}: 500 response in {duration_ms:.2f}ms - error: {str(e)}")
                
                raise e
        
        # Legacy compatibility endpoints
        @self.app.post("/analyze_symbol")
        async def analyze_symbol_endpoint(request: SignalGenerationRequest):
            """Legacy compatibility endpoint"""
            return await self.generate_signal(request)
        
        @self.app.get("/service_status")
        async def service_status():
            """Get status of all microservices"""
            status = {}
            for service_name in self.services.keys():
                try:
                    health = await self.call_service(service_name, "/health")
                    status[service_name] = {
                        "status": health.get('status', 'unknown'),
                        "endpoint": self.services[service_name],
                        "last_check": datetime.now().isoformat()
                    }
                except Exception as e:
                    status[service_name] = {
                        "status": "error",
                        "error": str(e),
                        "endpoint": self.services[service_name],
                        "last_check": datetime.now().isoformat()
                    }
            
            return status
        
        @self.app.get("/service_endpoints")
        async def service_endpoints():
            """Get all service endpoints"""
            return {
                "services": self.services,
                "orchestrator": "http://localhost:8025"
            }
        
        @self.app.get("/signals/latest")
        async def get_latest_signals():
            """Get latest signals from the database"""
            try:
                if not self.db_pool:
                    return {"error": "Database not available"}
                
                connection = self.db_pool.get_connection()
                cursor = connection.cursor(dictionary=True)
                
                query = """
                SELECT symbol, signal_type, confidence, timestamp, model
                FROM trading_signals 
                ORDER BY timestamp DESC 
                LIMIT 10
                """
                
                cursor.execute(query)
                results = cursor.fetchall()
                
                cursor.close()
                connection.close()
                
                return {
                    "latest_signals": results,
                    "count": len(results),
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Error fetching latest signals: {e}")
                return {"error": str(e)}
        
        @self.app.post("/start_session")
        async def start_session():
            """Start a signal generation session for multiple symbols"""
            return await generate_signals_batch()
        
        @self.app.post("/generate_signals")
        async def generate_signals():
            """Generate signals for multiple crypto symbols - main endpoint"""
            return await generate_signals_batch()
        
        async def generate_signals_batch():
            """Internal function to generate signals for multiple symbols"""
            try:
                # Generate signals for top crypto symbols
                symbols = ["BTC", "ETH", "SOL", "ADA", "DOT"]
                results = []
                
                for symbol in symbols:
                    try:
                        request = SignalGenerationRequest(symbol=symbol, analysis_type="comprehensive")
                        signal_result = await self.generate_signal(request)
                        results.append({
                            "symbol": symbol,
                            "signal": signal_result.signal,
                            "confidence": signal_result.confidence,
                            "status": "success"
                        })
                        logger.info(f"Generated signal for {symbol}: {signal_result.signal} (confidence: {signal_result.confidence:.3f})")
                    except Exception as e:
                        results.append({
                            "symbol": symbol,
                            "error": str(e),
                            "status": "failed"
                        })
                        logger.error(f"Failed to generate signal for {symbol}: {e}")
                
                return {
                    "session_started": True,
                    "signals_generated": len([r for r in results if r.get("status") == "success"]),
                    "signals_failed": len([r for r in results if r.get("status") == "failed"]),
                    "results": results,
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Error starting signal generation session: {e}")
                return {"error": str(e), "session_started": False}
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()

def main():
    """Main function to run the Orchestrator service"""
    try:
        logger.info("Starting Signal Generation Orchestrator...")
        
        orchestrator = SignalGenOrchestrator()
        
        # Run the FastAPI application
        uvicorn.run(
            orchestrator.app,
            host="0.0.0.0",
            port=8025,
            log_level="info"
        )
        
    except Exception as e:
        logger.error(f"Failed to start Orchestrator: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()