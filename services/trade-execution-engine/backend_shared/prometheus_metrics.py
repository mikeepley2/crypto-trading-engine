#!/usr/bin/env python3
"""
Prometheus Metrics Module for Crypto Services
==============================================

Provides a standardized way to export Prometheus metrics from all crypto services.
This module wraps the prometheus_client library and provides service-specific metrics.

Usage:
    from prometheus_metrics import CryptoServiceMetrics
    
    metrics = CryptoServiceMetrics("crypto-prices", "1.1.0")
    metrics.increment_api_calls()
    metrics.set_cache_entries(150)
    
    # Add to FastAPI app
    app.get("/metrics")(metrics.get_prometheus_metrics)
"""

import time
import logging
from typing import Dict, Any, Optional
from prometheus_client import Counter, Gauge, Histogram, Info, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response

logger = logging.getLogger(__name__)

class CryptoServiceMetrics:
    """
    Prometheus metrics collector for crypto services.
    Provides standardized metrics across all microservices.
    """
    
    def __init__(self, service_name: str, version: str):
        self.service_name = service_name
        self.version = version
        self.start_time = time.time()
        
        # Service info metric
        self.service_info = Info(
            'crypto_service_info',
            'Information about the crypto service',
            ['service_name', 'version']
        )
        self.service_info.labels(service_name=service_name, version=version).info({
            'service_name': service_name,
            'version': version,
            'start_time': str(self.start_time)
        })
        
        # Standard metrics for all services
        self.uptime_seconds = Gauge(
            'crypto_service_uptime_seconds',
            'Time in seconds since the service started',
            ['service_name']
        )
        
        self.api_calls_total = Counter(
            'crypto_service_api_calls_total',
            'Total number of API calls made',
            ['service_name', 'endpoint', 'status']
        )
        
        self.cache_entries = Gauge(
            'crypto_service_cache_entries',
            'Number of entries in the service cache',
            ['service_name']
        )
        
        self.mysql_connected = Gauge(
            'crypto_service_mysql_connected',
            'Whether MySQL is connected (1=connected, 0=disconnected)',
            ['service_name']
        )
        
        self.supported_coins = Gauge(
            'crypto_service_supported_coins',
            'Number of supported cryptocurrency coins',
            ['service_name']
        )
        
        self.request_duration = Histogram(
            'crypto_service_request_duration_seconds',
            'Time spent processing requests',
            ['service_name', 'endpoint']
        )
        
        self.errors_total = Counter(
            'crypto_service_errors_total',
            'Total number of errors',
            ['service_name', 'error_type']
        )
        
        # Service-specific metrics (will be added per service type)
        self.custom_gauges: Dict[str, Gauge] = {}
        self.custom_counters: Dict[str, Counter] = {}
        
    def get_uptime(self) -> float:
        """Get current uptime in seconds."""
        return time.time() - self.start_time
    
    def update_uptime(self):
        """Update the uptime metric."""
        self.uptime_seconds.labels(service_name=self.service_name).set(self.get_uptime())
    
    def increment_api_calls(self, endpoint: str = "unknown", status: str = "success"):
        """Increment API call counter."""
        self.api_calls_total.labels(
            service_name=self.service_name,
            endpoint=endpoint,
            status=status
        ).inc()
    
    def set_cache_entries(self, count: int):
        """Set current cache entry count."""
        self.cache_entries.labels(service_name=self.service_name).set(count)
    
    def set_mysql_connected(self, connected: bool):
        """Set MySQL connection status."""
        self.mysql_connected.labels(service_name=self.service_name).set(1 if connected else 0)
    
    def set_supported_coins(self, count: int):
        """Set number of supported coins."""
        self.supported_coins.labels(service_name=self.service_name).set(count)
    
    def time_request(self, endpoint: str):
        """Context manager to time requests."""
        return self.request_duration.labels(service_name=self.service_name, endpoint=endpoint).time()
    
    def increment_errors(self, error_type: str = "unknown"):
        """Increment error counter."""
        self.errors_total.labels(service_name=self.service_name, error_type=error_type).inc()
    
    def add_custom_gauge(self, name: str, description: str, labels: list = None):
        """Add a custom gauge metric."""
        if labels is None:
            labels = ['service_name']
        else:
            labels = ['service_name'] + labels
            
        self.custom_gauges[name] = Gauge(
            f'crypto_service_{name}',
            description,
            labels
        )
    
    def add_custom_counter(self, name: str, description: str, labels: list = None):
        """Add a custom counter metric."""
        if labels is None:
            labels = ['service_name']
        else:
            labels = ['service_name'] + labels
            
        self.custom_counters[name] = Counter(
            f'crypto_service_{name}',
            description,
            labels
        )
    
    def set_custom_gauge(self, name: str, value: float, **label_values):
        """Set value for a custom gauge."""
        if name in self.custom_gauges:
            labels = {'service_name': self.service_name, **label_values}
            self.custom_gauges[name].labels(**labels).set(value)
    
    def increment_custom_counter(self, name: str, **label_values):
        """Increment a custom counter."""
        if name in self.custom_counters:
            labels = {'service_name': self.service_name, **label_values}
            self.custom_counters[name].labels(**labels).inc()
    
    async def get_prometheus_metrics(self):
        """
        FastAPI endpoint handler that returns Prometheus metrics.
        Use this as: app.get("/metrics")(metrics.get_prometheus_metrics)
        """
        # Update dynamic metrics
        self.update_uptime()
        
        # Generate Prometheus format metrics
        metrics_output = generate_latest()
        return Response(content=metrics_output, media_type=CONTENT_TYPE_LATEST)


class CryptoPricesMetrics(CryptoServiceMetrics):
    """Extended metrics for crypto prices service."""
    
    def __init__(self, service_name: str = "crypto-prices", version: str = "1.1.0"):
        super().__init__(service_name, version)
        
        # Crypto prices specific metrics
        self.add_custom_gauge('coinbase_supported_coins', 'Number of Coinbase supported coins')
        self.add_custom_gauge('cache_ttl_seconds', 'Cache TTL in seconds')
        self.add_custom_gauge('api_calls_today', 'API calls made today')
        self.add_custom_counter('price_requests_total', 'Total price requests', ['coin_id', 'source'])
        self.add_custom_gauge('last_api_call_timestamp', 'Timestamp of last API call')


class CryptoSentimentMetrics(CryptoServiceMetrics):
    """Extended metrics for sentiment analysis services."""
    
    def __init__(self, service_name: str = "crypto-sentiment", version: str = "1.0.0"):
        super().__init__(service_name, version)
        
        # Sentiment analysis specific metrics
        self.add_custom_counter('sentiment_analysis_total', 'Total sentiment analyses', ['sentiment', 'model'])
        self.add_custom_gauge('model_accuracy', 'Model accuracy percentage', ['model'])
        self.add_custom_gauge('processing_time_ms', 'Average processing time in milliseconds', ['model'])
        self.add_custom_counter('texts_processed_total', 'Total texts processed')


class CryptoNewsMetrics(CryptoServiceMetrics):
    """Extended metrics for news collection services."""
    
    def __init__(self, service_name: str = "crypto-news", version: str = "1.0.0"):
        super().__init__(service_name, version)
        
        # News collection specific metrics
        self.add_custom_counter('articles_collected_total', 'Total articles collected', ['source'])
        self.add_custom_gauge('sources_active', 'Number of active news sources')
        self.add_custom_gauge('last_collection_timestamp', 'Timestamp of last collection')
        self.add_custom_counter('collection_errors_total', 'Collection errors', ['source', 'error_type'])


class TechnicalIndicatorsMetrics(CryptoServiceMetrics):
    """Extended metrics for technical indicators service."""
    
    def __init__(self, service_name: str = "technical-indicators", version: str = "1.0.0"):
        super().__init__(service_name, version)
        
        # Technical indicators specific metrics
        self.add_custom_counter('indicators_calculated_total', 'Total indicators calculated', ['indicator_type', 'coin'])
        self.add_custom_gauge('calculation_time_ms', 'Calculation time in milliseconds', ['indicator_type'])
        self.add_custom_gauge('data_freshness_seconds', 'Age of source data in seconds', ['coin'])
        self.add_custom_gauge('indicators_available', 'Number of available indicators')


# Global metrics instances (to be used by services)
_metrics_instances = {}

def get_service_metrics(service_type: str, service_name: str = None, version: str = None) -> CryptoServiceMetrics:
    """
    Factory function to get the appropriate metrics instance for a service.
    
    Args:
        service_type: Type of service ('prices', 'sentiment', 'news', 'indicators', 'generic')
        service_name: Override service name (optional)
        version: Override version (optional)
    
    Returns:
        Appropriate metrics instance
    """
    key = f"{service_type}_{service_name or service_type}"
    
    if key not in _metrics_instances:
        if service_type == 'prices':
            _metrics_instances[key] = CryptoPricesMetrics(
                service_name or "crypto-prices", 
                version or "1.1.0"
            )
        elif service_type == 'sentiment':
            _metrics_instances[key] = CryptoSentimentMetrics(
                service_name or "crypto-sentiment",
                version or "1.0.0"
            )
        elif service_type == 'news':
            _metrics_instances[key] = CryptoNewsMetrics(
                service_name or "crypto-news",
                version or "1.0.0"
            )
        elif service_type == 'indicators':
            _metrics_instances[key] = TechnicalIndicatorsMetrics(
                service_name or "technical-indicators",
                version or "1.0.0"
            )
        else:
            _metrics_instances[key] = CryptoServiceMetrics(
                service_name or service_type,
                version or "1.0.0"
            )
    
    return _metrics_instances[key]
