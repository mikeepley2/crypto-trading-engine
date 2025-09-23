#!/usr/bin/env python3
"""
Standardized Async Health Check Framework
========================================

A reusable framework for implementing comprehensive health, status, and metrics
endpoints in FastAPI services. Designed to remain responsive even when services
are under heavy load by using async operations and separate thread pools.

Features:
- Non-blocking async health checks
- Database connection monitoring
- System resource metrics
- Service-specific metrics extension
- Standard response formats
- Background health monitoring
- Graceful error handling
- Performance metrics

Usage:
    from backend.shared.health_framework import HealthFramework
    
    app = FastAPI()
    health = HealthFramework(app, service_name="my-service")
    
    # Add custom checks
    @health.add_custom_check("database")
    async def check_database():
        # Your database check logic
        return {"status": "healthy", "response_time": 25}
"""

import asyncio
import logging
import time
import traceback
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
import psutil
import json
import os

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
import mysql.connector
from mysql.connector import pooling
import redis


logger = logging.getLogger(__name__)


class HealthStatus:
    """Health status constants"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class HealthMetrics:
    """Container for health metrics"""
    
    def __init__(self):
        self.start_time = datetime.now(timezone.utc)
        self.request_count = 0
        self.error_count = 0
        self.last_error_time = None
        self.last_error_message = None
        self.response_times = []
        self.custom_metrics = {}
    
    def record_request(self, response_time: float, error: bool = False, error_msg: str = None):
        """Record a request metric"""
        self.request_count += 1
        self.response_times.append(response_time)
        
        # Keep only last 100 response times for memory efficiency
        if len(self.response_times) > 100:
            self.response_times = self.response_times[-100:]
        
        if error:
            self.error_count += 1
            self.last_error_time = datetime.now(timezone.utc)
            self.last_error_message = error_msg
    
    def get_avg_response_time(self) -> float:
        """Get average response time"""
        return sum(self.response_times) / len(self.response_times) if self.response_times else 0.0
    
    def get_error_rate(self) -> float:
        """Get error rate as percentage"""
        return (self.error_count / max(self.request_count, 1)) * 100
    
    def get_uptime(self) -> float:
        """Get uptime in seconds"""
        return (datetime.now(timezone.utc) - self.start_time).total_seconds()


class AsyncHealthCheck:
    """Async wrapper for health check functions"""
    
    def __init__(self, check_func: Callable, name: str, timeout: float = 5.0):
        self.check_func = check_func
        self.name = name
        self.timeout = timeout
        self.last_result = None
        self.last_check_time = None
    
    async def execute(self) -> Dict[str, Any]:
        """Execute the health check with timeout"""
        start_time = time.time()
        
        try:
            # If check_func is already async, await it directly
            if asyncio.iscoroutinefunction(self.check_func):
                result = await asyncio.wait_for(self.check_func(), timeout=self.timeout)
            else:
                # Run sync function in thread pool
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, self.check_func
                )
            
            response_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Ensure result is a dict
            if not isinstance(result, dict):
                result = {"status": result}
            
            result.update({
                "check_name": self.name,
                "response_time_ms": round(response_time, 2),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            self.last_result = result
            self.last_check_time = datetime.now(timezone.utc)
            
            return result
            
        except asyncio.TimeoutError:
            error_result = {
                "check_name": self.name,
                "status": HealthStatus.UNHEALTHY,
                "error": "Timeout",
                "timeout_seconds": self.timeout,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            self.last_result = error_result
            return error_result
            
        except Exception as e:
            error_result = {
                "check_name": self.name,
                "status": HealthStatus.UNHEALTHY,
                "error": str(e),
                "traceback": traceback.format_exc() if logger.isEnabledFor(logging.DEBUG) else None,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            self.last_result = error_result
            logger.error(f"Health check '{self.name}' failed: {e}")
            return error_result


class HealthFramework:
    """Main health framework class"""
    
    def __init__(self, app: FastAPI, service_name: str, service_version: str = "1.0.0"):
        self.app = app
        self.service_name = service_name
        self.service_version = service_version
        self.metrics = HealthMetrics()
        self.health_checks: Dict[str, AsyncHealthCheck] = {}
        self.background_monitoring = False
        self.executor = ThreadPoolExecutor(max_workers=5, thread_name_prefix=f"{service_name}-health")
        
        # Default database configs (can be overridden)
        self.db_configs = {
            "crypto_prices": {
                'host': 'host.docker.internal' if os.path.exists('/.dockerenv') else 'localhost',
                'user': 'news_collector',
                'password': '99Rules!',
                'database': 'crypto_prices'
            },
            "crypto_transactions": {
                'host': 'host.docker.internal' if os.path.exists('/.dockerenv') else 'localhost',
                'user': 'news_collector',
                'password': '99Rules!',
                'database': 'crypto_transactions'
            }
        }
        
        self.redis_config = {
            'host': 'host.docker.internal' if os.path.exists('/.dockerenv') else 'localhost',
            'port': 6379,
            'decode_responses': True
        }
        
        # Setup default endpoints
        self._setup_default_endpoints()
        self._setup_default_checks()
    
    def _setup_default_endpoints(self):
        """Setup standard health endpoints"""
        
        @self.app.get("/health")
        async def health_endpoint():
            """Primary health check endpoint - should be fast and lightweight"""
            start_time = time.time()
            
            try:
                # Run only critical health checks
                critical_checks = {
                    name: check for name, check in self.health_checks.items()
                    if name in ['system_basic', 'primary_database']
                }
                
                if critical_checks:
                    results = await asyncio.gather(
                        *[check.execute() for check in critical_checks.values()],
                        return_exceptions=True
                    )
                    
                    # Determine overall health
                    overall_status = HealthStatus.HEALTHY
                    for result in results:
                        if isinstance(result, dict) and result.get('status') == HealthStatus.UNHEALTHY:
                            overall_status = HealthStatus.UNHEALTHY
                            break
                        elif isinstance(result, dict) and result.get('status') == HealthStatus.DEGRADED:
                            overall_status = HealthStatus.DEGRADED
                else:
                    results = []
                    overall_status = HealthStatus.HEALTHY
                
                response_time = (time.time() - start_time) * 1000
                self.metrics.record_request(response_time, overall_status == HealthStatus.UNHEALTHY)
                
                response_data = {
                    "status": overall_status,
                    "service": self.service_name,
                    "version": self.service_version,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "uptime_seconds": int(self.metrics.get_uptime()),
                    "response_time_ms": round(response_time, 2)
                }
                
                status_code = 200 if overall_status == HealthStatus.HEALTHY else 503
                return JSONResponse(status_code=status_code, content=response_data)
                
            except Exception as e:
                error_response = {
                    "status": HealthStatus.UNHEALTHY,
                    "service": self.service_name,
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                self.metrics.record_request((time.time() - start_time) * 1000, True, str(e))
                return JSONResponse(status_code=503, content=error_response)
        
        @self.app.get("/status")
        async def status_endpoint():
            """Comprehensive status endpoint with detailed information"""
            start_time = time.time()
            
            try:
                # Run all health checks
                if self.health_checks:
                    check_results = await asyncio.gather(
                        *[check.execute() for check in self.health_checks.values()],
                        return_exceptions=True
                    )
                    
                    # Process results
                    checks_status = {}
                    overall_status = HealthStatus.HEALTHY
                    
                    for i, result in enumerate(check_results):
                        check_name = list(self.health_checks.keys())[i]
                        
                        if isinstance(result, Exception):
                            checks_status[check_name] = {
                                "status": HealthStatus.UNHEALTHY,
                                "error": str(result)
                            }
                            overall_status = HealthStatus.UNHEALTHY
                        else:
                            checks_status[check_name] = result
                            if result.get('status') == HealthStatus.UNHEALTHY:
                                overall_status = HealthStatus.UNHEALTHY
                            elif result.get('status') == HealthStatus.DEGRADED and overall_status == HealthStatus.HEALTHY:
                                overall_status = HealthStatus.DEGRADED
                else:
                    checks_status = {}
                    overall_status = HealthStatus.HEALTHY
                
                response_time = (time.time() - start_time) * 1000
                
                response_data = {
                    "service": self.service_name,
                    "version": self.service_version,
                    "status": overall_status,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "uptime_seconds": int(self.metrics.get_uptime()),
                    "response_time_ms": round(response_time, 2),
                    "checks": checks_status,
                    "metrics": {
                        "total_requests": self.metrics.request_count,
                        "error_count": self.metrics.error_count,
                        "error_rate_percent": round(self.metrics.get_error_rate(), 2),
                        "avg_response_time_ms": round(self.metrics.get_avg_response_time(), 2),
                        "last_error": {
                            "time": self.metrics.last_error_time.isoformat() if self.metrics.last_error_time else None,
                            "message": self.metrics.last_error_message
                        }
                    }
                }
                
                status_code = 200 if overall_status != HealthStatus.UNHEALTHY else 503
                return JSONResponse(status_code=status_code, content=response_data)
                
            except Exception as e:
                logger.error(f"Status endpoint error: {e}")
                error_response = {
                    "service": self.service_name,
                    "status": HealthStatus.UNHEALTHY,
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                return JSONResponse(status_code=503, content=error_response)
        
        @self.app.get("/metrics")
        async def metrics_endpoint():
            """Detailed metrics endpoint"""
            try:
                # Get system metrics
                system_metrics = await self._get_system_metrics()
                
                response_data = {
                    "service": self.service_name,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "system": system_metrics,
                    "service_metrics": {
                        "uptime_seconds": int(self.metrics.get_uptime()),
                        "total_requests": self.metrics.request_count,
                        "error_count": self.metrics.error_count,
                        "error_rate_percent": round(self.metrics.get_error_rate(), 2),
                        "avg_response_time_ms": round(self.metrics.get_avg_response_time(), 2),
                        "custom_metrics": self.metrics.custom_metrics
                    }
                }
                
                return JSONResponse(content=response_data)
                
            except Exception as e:
                logger.error(f"Metrics endpoint error: {e}")
                return JSONResponse(
                    status_code=500, 
                    content={"error": str(e), "timestamp": datetime.now(timezone.utc).isoformat()}
                )
        
        @self.app.get("/ready")
        async def readiness_endpoint():
            """Kubernetes-style readiness probe"""
            start_time = time.time()
            
            try:
                # Check if service is ready to accept traffic
                # This should include checks for essential dependencies
                essential_checks = {
                    name: check for name, check in self.health_checks.items()
                    if name in ['primary_database', 'critical_dependencies']
                }
                
                if essential_checks:
                    results = await asyncio.gather(
                        *[check.execute() for check in essential_checks.values()],
                        return_exceptions=True
                    )
                    
                    ready = True
                    for result in results:
                        if isinstance(result, dict) and result.get('status') == HealthStatus.UNHEALTHY:
                            ready = False
                            break
                        elif isinstance(result, Exception):
                            ready = False
                            break
                else:
                    ready = True
                
                response_time = (time.time() - start_time) * 1000
                
                response_data = {
                    "ready": ready,
                    "service": self.service_name,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "response_time_ms": round(response_time, 2)
                }
                
                status_code = 200 if ready else 503
                return JSONResponse(status_code=status_code, content=response_data)
                
            except Exception as e:
                return JSONResponse(
                    status_code=503,
                    content={
                        "ready": False,
                        "service": self.service_name,
                        "error": str(e),
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                )
    
    def _setup_default_checks(self):
        """Setup default health checks that most services need"""
        
        @self.add_custom_check("system_basic")
        async def system_basic_check():
            """Basic system health check"""
            try:
                # Check basic system resources
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                
                status = HealthStatus.HEALTHY
                if cpu_percent > 90:
                    status = HealthStatus.DEGRADED
                if memory.percent > 90:
                    status = HealthStatus.UNHEALTHY
                
                return {
                    "status": status,
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "available_memory_gb": round(memory.available / (1024**3), 2)
                }
            except Exception as e:
                return {
                    "status": HealthStatus.UNHEALTHY,
                    "error": str(e)
                }
    
    async def _get_system_metrics(self) -> Dict[str, Any]:
        """Get detailed system metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                "cpu_percent": cpu_percent,
                "memory": {
                    "total_gb": round(memory.total / (1024**3), 2),
                    "available_gb": round(memory.available / (1024**3), 2),
                    "used_gb": round(memory.used / (1024**3), 2),
                    "percent": memory.percent
                },
                "disk": {
                    "total_gb": round(disk.total / (1024**3), 2),
                    "free_gb": round(disk.free / (1024**3), 2),
                    "used_gb": round(disk.used / (1024**3), 2),
                    "percent": round((disk.used / disk.total) * 100, 2)
                }
            }
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            return {"error": str(e)}
    
    def add_custom_check(self, name: str, timeout: float = 5.0):
        """Decorator to add a custom health check"""
        def decorator(func: Callable):
            self.health_checks[name] = AsyncHealthCheck(func, name, timeout)
            return func
        return decorator
    
    def add_database_check(self, name: str, database: str, timeout: float = 5.0):
        """Add a database connectivity check"""
        
        async def database_check():
            try:
                config = self.db_configs.get(database, {})
                if not config:
                    return {
                        "status": HealthStatus.UNHEALTHY,
                        "error": f"No configuration found for database: {database}"
                    }
                
                # Use thread pool for blocking database operation
                loop = asyncio.get_event_loop()
                
                def check_connection():
                    try:
                        conn = mysql.connector.connect(**config, connection_timeout=3)
                        cursor = conn.cursor()
                        cursor.execute("SELECT 1")
                        cursor.fetchone()
                        cursor.close()
                        conn.close()
                        return True
                    except Exception as e:
                        raise e
                
                await loop.run_in_executor(self.executor, check_connection)
                
                return {
                    "status": HealthStatus.HEALTHY,
                    "database": database,
                    "host": config.get('host', 'unknown')
                }
                
            except Exception as e:
                return {
                    "status": HealthStatus.UNHEALTHY,
                    "database": database,
                    "error": str(e)
                }
        
        self.health_checks[name] = AsyncHealthCheck(database_check, name, timeout)
    
    def add_redis_check(self, name: str = "redis", timeout: float = 5.0):
        """Add Redis connectivity check"""
        
        async def redis_check():
            try:
                # Use thread pool for blocking Redis operation
                loop = asyncio.get_event_loop()
                
                def check_redis():
                    client = redis.Redis(**self.redis_config, socket_timeout=3)
                    client.ping()
                    client.close()
                    return True
                
                await loop.run_in_executor(self.executor, check_redis)
                
                return {
                    "status": HealthStatus.HEALTHY,
                    "host": self.redis_config.get('host', 'unknown'),
                    "port": self.redis_config.get('port', 6379)
                }
                
            except Exception as e:
                return {
                    "status": HealthStatus.UNHEALTHY,
                    "error": str(e)
                }
        
        self.health_checks[name] = AsyncHealthCheck(redis_check, name, timeout)
    
    def add_http_dependency_check(self, name: str, url: str, timeout: float = 5.0):
        """Add HTTP dependency check"""
        
        async def http_check():
            try:
                import aiohttp
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                        return {
                            "status": HealthStatus.HEALTHY if response.status == 200 else HealthStatus.DEGRADED,
                            "url": url,
                            "status_code": response.status,
                            "response_time_ms": response.headers.get('x-response-time', 'unknown')
                        }
                        
            except Exception as e:
                return {
                    "status": HealthStatus.UNHEALTHY,
                    "url": url,
                    "error": str(e)
                }
        
        self.health_checks[name] = AsyncHealthCheck(http_check, name, timeout)
    
    def set_custom_metric(self, key: str, value: Any):
        """Set a custom metric value"""
        self.metrics.custom_metrics[key] = {
            "value": value,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def increment_custom_metric(self, key: str, amount: Union[int, float] = 1):
        """Increment a custom metric"""
        current = self.metrics.custom_metrics.get(key, {}).get("value", 0)
        self.set_custom_metric(key, current + amount)
    
    def update_database_config(self, database: str, config: Dict[str, Any]):
        """Update database configuration"""
        self.db_configs[database] = config
    
    def update_redis_config(self, config: Dict[str, Any]):
        """Update Redis configuration"""
        self.redis_config.update(config)
