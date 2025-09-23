#!/usr/bin/env python3
"""
Signal Generation Analytics Service
Handles strategy tracking, performance metrics, and health monitoring

This microservice extracts analytics functionality from enhanced_signal_generator.py
Responsibilities:
- Strategy performance tracking and analysis
- Signal generation metrics and statistics
- Health monitoring of signal generation components
- A/B testing and strategy comparison
- Reporting and dashboard data preparation
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
from collections import defaultdict, deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SignalMetrics(BaseModel):
    """Signal generation metrics"""
    total_signals: int
    buy_signals: int
    sell_signals: int
    hold_signals: int
    avg_confidence: float
    success_rate: float
    signals_per_hour: float

class StrategyPerformance(BaseModel):
    """Strategy performance metrics"""
    strategy_name: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_return: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    profit_factor: float

class ServiceHealth(BaseModel):
    """Service health metrics"""
    service_name: str
    status: str
    response_time_ms: float
    last_updated: str
    error_count: int
    uptime_percentage: float

class AnalyticsRequest(BaseModel):
    """Request for analytics data"""
    time_range: str  # 1h, 4h, 24h, 7d, 30d
    include_strategies: Optional[List[str]] = None
    include_services: Optional[List[str]] = None

class AnalyticsResponse(BaseModel):
    """Analytics response"""
    time_range: str
    signal_metrics: SignalMetrics
    strategy_performance: List[StrategyPerformance]
    service_health: List[ServiceHealth]
    market_conditions: Dict[str, Any]
    recommendations: List[str]
    timestamp: str

class SignalGenAnalytics:
    """Analytics Service for signal generation"""
    
    def __init__(self):
        self.app = FastAPI(
            title="Signal Generation Analytics Service",
            description="Analytics and monitoring for trading signal generation",
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
        
        # Service URLs for health monitoring
        self.service_urls = {
            'ml-engine': os.getenv('ML_ENGINE_URL', 'http://host.docker.internal:8051'),
            'feature-engine': os.getenv('FEATURE_ENGINE_URL', 'http://host.docker.internal:8052'),
            'market-context': os.getenv('MARKET_CONTEXT_URL', 'http://host.docker.internal:8053'),
            'portfolio': os.getenv('PORTFOLIO_URL', 'http://host.docker.internal:8054'),
            'risk-mgmt': os.getenv('RISK_MGMT_URL', 'http://host.docker.internal:8055'),
            'orchestrator': os.getenv('ORCHESTRATOR_URL', 'http://host.docker.internal:8025')
        }
        
        # In-memory metrics storage (for demo purposes)
        self.signal_history = deque(maxlen=10000)
        self.service_health_history = defaultdict(lambda: deque(maxlen=1000))
        
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
                'pool_name': 'analytics_pool',
                'pool_size': 5,
                'pool_reset_session': True,
                'autocommit': True
            }
            
            self.db_pool = pooling.MySQLConnectionPool(**db_config)
            logger.info("✅ Database connection pool initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup database: {e}")
            self.db_pool = None
    
    async def record_signal(self, signal_data: Dict[str, Any]):
        """Record a signal for analytics tracking"""
        try:
            signal_record = {
                'timestamp': datetime.now().isoformat(),
                'symbol': signal_data.get('symbol'),
                'signal_type': signal_data.get('signal_type'),
                'confidence': signal_data.get('confidence'),
                'price': signal_data.get('price'),
                'strategy': signal_data.get('strategy', 'default')
            }
            
            self.signal_history.append(signal_record)
            
            # Also store in database if available
            if self.db_pool:
                await self.store_signal_in_db(signal_record)
            
            logger.debug(f"📊 Recorded signal: {signal_data.get('symbol')} {signal_data.get('signal_type')}")
            
        except Exception as e:
            logger.warning(f"Error recording signal: {e}")
    
    async def store_signal_in_db(self, signal_record: Dict[str, Any]):
        """Store signal record in database"""
        try:
            connection = self.db_pool.get_connection()
            cursor = connection.cursor()
            
            query = """
            INSERT INTO signal_analytics 
            (timestamp, symbol, signal_type, confidence, price, strategy)
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            
            cursor.execute(query, (
                signal_record['timestamp'],
                signal_record['symbol'],
                signal_record['signal_type'],
                signal_record['confidence'],
                signal_record['price'],
                signal_record['strategy']
            ))
            
            connection.commit()
            cursor.close()
            connection.close()
            
        except Exception as e:
            logger.warning(f"Error storing signal in database: {e}")
    
    async def check_service_health(self, service_name: str, url: str) -> ServiceHealth:
        """Check health of a signal generation service"""
        try:
            start_time = datetime.now()
            
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(f"{url}/health") as response:
                    end_time = datetime.now()
                    response_time = (end_time - start_time).total_seconds() * 1000
                    
                    if response.status == 200:
                        status = "healthy"
                        error_count = 0
                    else:
                        status = "unhealthy"
                        error_count = 1
                    
                    health_record = ServiceHealth(
                        service_name=service_name,
                        status=status,
                        response_time_ms=response_time,
                        last_updated=datetime.now().isoformat(),
                        error_count=error_count,
                        uptime_percentage=99.0 if status == "healthy" else 80.0
                    )
                    
                    # Store health record
                    self.service_health_history[service_name].append({
                        'timestamp': datetime.now().isoformat(),
                        'status': status,
                        'response_time': response_time,
                        'error_count': error_count
                    })
                    
                    return health_record
                    
        except Exception as e:
            logger.warning(f"Health check failed for {service_name}: {e}")
            return ServiceHealth(
                service_name=service_name,
                status="unhealthy",
                response_time_ms=10000,
                last_updated=datetime.now().isoformat(),
                error_count=1,
                uptime_percentage=0.0
            )
    
    def calculate_signal_metrics(self, time_range: str) -> SignalMetrics:
        """Calculate signal generation metrics for time range"""
        try:
            # Parse time range
            hours = self.parse_time_range(time_range)
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            # Filter signals by time range
            recent_signals = [
                s for s in self.signal_history
                if datetime.fromisoformat(s['timestamp']) >= cutoff_time
            ]
            
            if not recent_signals:
                return SignalMetrics(
                    total_signals=0,
                    buy_signals=0,
                    sell_signals=0,
                    hold_signals=0,
                    avg_confidence=0.0,
                    success_rate=0.0,
                    signals_per_hour=0.0
                )
            
            # Calculate metrics
            total_signals = len(recent_signals)
            buy_signals = sum(1 for s in recent_signals if s['signal_type'] == 'BUY')
            sell_signals = sum(1 for s in recent_signals if s['signal_type'] == 'SELL')
            hold_signals = sum(1 for s in recent_signals if s['signal_type'] == 'HOLD')
            
            avg_confidence = np.mean([s['confidence'] for s in recent_signals])
            
            # Simple success rate estimate (placeholder)
            success_rate = 0.65 if avg_confidence > 0.7 else 0.55
            
            signals_per_hour = total_signals / hours if hours > 0 else 0
            
            return SignalMetrics(
                total_signals=total_signals,
                buy_signals=buy_signals,
                sell_signals=sell_signals,
                hold_signals=hold_signals,
                avg_confidence=float(avg_confidence),
                success_rate=success_rate,
                signals_per_hour=signals_per_hour
            )
            
        except Exception as e:
            logger.warning(f"Error calculating signal metrics: {e}")
            return SignalMetrics(
                total_signals=0,
                buy_signals=0,
                sell_signals=0,
                hold_signals=0,
                avg_confidence=0.0,
                success_rate=0.0,
                signals_per_hour=0.0
            )
    
    def parse_time_range(self, time_range: str) -> int:
        """Parse time range string to hours"""
        try:
            if time_range == "1h":
                return 1
            elif time_range == "4h":
                return 4
            elif time_range == "24h":
                return 24
            elif time_range == "7d":
                return 168  # 7 * 24
            elif time_range == "30d":
                return 720  # 30 * 24
            else:
                return 24  # Default to 24h
        except:
            return 24
    
    async def get_strategy_performance(self, time_range: str) -> List[StrategyPerformance]:
        """Get strategy performance metrics"""
        try:
            # This would typically query the database for actual performance data
            # For now, return mock data
            
            strategies = [
                StrategyPerformance(
                    strategy_name="ml_momentum",
                    total_trades=45,
                    winning_trades=28,
                    losing_trades=17,
                    win_rate=62.2,
                    avg_return=1.8,
                    total_return=12.4,
                    sharpe_ratio=1.6,
                    max_drawdown=-8.2,
                    profit_factor=1.85
                ),
                StrategyPerformance(
                    strategy_name="sentiment_reversal",
                    total_trades=32,
                    winning_trades=19,
                    losing_trades=13,
                    win_rate=59.4,
                    avg_return=1.2,
                    total_return=8.1,
                    sharpe_ratio=1.3,
                    max_drawdown=-6.5,
                    profit_factor=1.62
                )
            ]
            
            return strategies
            
        except Exception as e:
            logger.warning(f"Error getting strategy performance: {e}")
            return []
    
    async def get_market_conditions_summary(self) -> Dict[str, Any]:
        """Get market conditions summary for analytics"""
        try:
            # Get data from market context service
            market_context_url = self.service_urls.get('market-context')
            if market_context_url:
                timeout = aiohttp.ClientTimeout(total=10)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(f"{market_context_url}/sentiment") as response:
                        if response.status == 200:
                            return await response.json()
            
            # Fallback data
            return {
                "market_regime": "SIDEWAYS",
                "volatility_index": 0.45,
                "sentiment_score": 0.52,
                "fear_greed_index": 48
            }
            
        except Exception as e:
            logger.warning(f"Error getting market conditions: {e}")
            return {}
    
    def generate_recommendations(self, signal_metrics: SignalMetrics, 
                               service_health: List[ServiceHealth]) -> List[str]:
        """Generate recommendations based on analytics"""
        recommendations = []
        
        try:
            # Signal quality recommendations
            if signal_metrics.avg_confidence < 0.6:
                recommendations.append("Signal confidence is low - consider tuning ML models")
            
            if signal_metrics.signals_per_hour > 10:
                recommendations.append("High signal frequency - review signal filters")
            elif signal_metrics.signals_per_hour < 0.5:
                recommendations.append("Low signal frequency - check market data feeds")
            
            # Service health recommendations
            unhealthy_services = [s for s in service_health if s.status != "healthy"]
            if unhealthy_services:
                service_names = [s.service_name for s in unhealthy_services]
                recommendations.append(f"Services need attention: {', '.join(service_names)}")
            
            # Performance recommendations
            buy_sell_ratio = signal_metrics.buy_signals / max(signal_metrics.sell_signals, 1)
            if buy_sell_ratio > 3:
                recommendations.append("High BUY signal bias - review market conditions")
            elif buy_sell_ratio < 0.5:
                recommendations.append("High SELL signal bias - check risk management")
            
            if not recommendations:
                recommendations.append("System operating normally - no issues detected")
            
            return recommendations
            
        except Exception as e:
            logger.warning(f"Error generating recommendations: {e}")
            return ["Analytics engine error - manual review recommended"]
    
    async def get_analytics(self, request: AnalyticsRequest) -> AnalyticsResponse:
        """Main analytics endpoint"""
        try:
            # Calculate signal metrics
            signal_metrics = self.calculate_signal_metrics(request.time_range)
            
            # Get strategy performance
            strategy_performance = await self.get_strategy_performance(request.time_range)
            
            # Check service health
            service_health = []
            for service_name, url in self.service_urls.items():
                if not request.include_services or service_name in request.include_services:
                    health = await self.check_service_health(service_name, url)
                    service_health.append(health)
            
            # Get market conditions
            market_conditions = await self.get_market_conditions_summary()
            
            # Generate recommendations
            recommendations = self.generate_recommendations(signal_metrics, service_health)
            
            logger.info(f"📊 Analytics generated for {request.time_range}: "
                       f"{signal_metrics.total_signals} signals, "
                       f"{len([s for s in service_health if s.status == 'healthy'])}/{len(service_health)} services healthy")
            
            return AnalyticsResponse(
                time_range=request.time_range,
                signal_metrics=signal_metrics,
                strategy_performance=strategy_performance,
                service_health=service_health,
                market_conditions=market_conditions,
                recommendations=recommendations,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"❌ Error generating analytics: {e}")
            raise HTTPException(status_code=500, detail=f"Analytics generation failed: {str(e)}")
    
    def setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            db_status = "connected" if self.db_pool else "disconnected"
            
            return {
                "status": "healthy",
                "service": "signal-gen-analytics",
                "database_status": db_status,
                "signal_history_size": len(self.signal_history),
                "monitored_services": len(self.service_urls),
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.post("/analytics", response_model=AnalyticsResponse)
        async def get_analytics_endpoint(request: AnalyticsRequest):
            """Get analytics data"""
            return await self.get_analytics(request)
        
        @self.app.post("/record_signal")
        async def record_signal_endpoint(signal_data: Dict[str, Any]):
            """Record a signal for analytics tracking"""
            await self.record_signal(signal_data)
            return {"status": "recorded", "timestamp": datetime.now().isoformat()}
        
        @self.app.get("/signal_metrics")
        async def get_signal_metrics_endpoint(time_range: str = "24h"):
            """Get signal metrics for time range"""
            return self.calculate_signal_metrics(time_range)
        
        @self.app.get("/service_health")
        async def get_service_health_endpoint():
            """Get health status of all services"""
            service_health = []
            for service_name, url in self.service_urls.items():
                health = await self.check_service_health(service_name, url)
                service_health.append(health)
            
            return {
                "services": service_health,
                "healthy_count": len([s for s in service_health if s.status == "healthy"]),
                "total_count": len(service_health),
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.get("/strategy_performance")
        async def get_strategy_performance_endpoint(time_range: str = "24h"):
            """Get strategy performance metrics"""
            return await self.get_strategy_performance(time_range)
        
        @self.app.get("/dashboard_data")
        async def get_dashboard_data():
            """Get comprehensive dashboard data"""
            request = AnalyticsRequest(time_range="24h")
            analytics = await self.get_analytics(request)
            
            return {
                "overview": {
                    "total_signals": analytics.signal_metrics.total_signals,
                    "avg_confidence": analytics.signal_metrics.avg_confidence,
                    "healthy_services": len([s for s in analytics.service_health if s.status == "healthy"]),
                    "total_services": len(analytics.service_health)
                },
                "signal_breakdown": {
                    "buy": analytics.signal_metrics.buy_signals,
                    "sell": analytics.signal_metrics.sell_signals,
                    "hold": analytics.signal_metrics.hold_signals
                },
                "performance": analytics.strategy_performance,
                "alerts": [r for r in analytics.recommendations if "error" in r.lower() or "attention" in r.lower()],
                "timestamp": analytics.timestamp
            }
        
        @self.app.get("/status")
        async def get_status():
            """Detailed status information"""
            return {
                "service": "signal-gen-analytics",
                "version": "1.0.0",
                "database_connected": bool(self.db_pool),
                "trading_engine_url": self.trading_engine_url,
                "monitored_services": list(self.service_urls.keys()),
                "signal_history_size": len(self.signal_history),
                "features": [
                    "signal_tracking",
                    "performance_analysis",
                    "service_health_monitoring",
                    "strategy_comparison",
                    "recommendation_engine",
                    "dashboard_data"
                ],
                "timestamp": datetime.now().isoformat()
            }

def main():
    """Main function to run the Analytics service"""
    try:
        logger.info("🚀 Starting Signal Generation Analytics Service...")
        
        analytics_service = SignalGenAnalytics()
        
        # Get port from environment or use default
        port = int(os.getenv('ANALYTICS_PORT', 8056))
        
        logger.info(f"📊 Analytics service starting on port {port}")
        
        # Run the FastAPI application
        uvicorn.run(
            analytics_service.app,
            host="0.0.0.0",
            port=port,
            log_level="info"
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to start Analytics service: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
