#!/usr/bin/env python3
"""
Shared logging configuration for signal generation microservices
Provides Loki-compatible JSON structured logging
"""

import logging
import json
import sys
from datetime import datetime, timezone
from typing import Dict, Any, Optional

class LokiJSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for Loki compatibility
    Outputs structured logs that Loki can easily parse and index
    """
    
    def __init__(self, service_name: str, version: str = "1.0.0"):
        super().__init__()
        self.service_name = service_name
        self.version = version
        
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON suitable for Loki"""
        
        # Base log structure
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "service": self.service_name,
            "version": self.version,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
            
        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 'filename',
                          'module', 'exc_info', 'exc_text', 'stack_info', 'lineno', 'funcName',
                          'created', 'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'getMessage']:
                log_entry[key] = value
                
        return json.dumps(log_entry, default=str)

def setup_logging(service_name: str, log_level: str = "INFO") -> logging.Logger:
    """
    Setup Loki-compatible logging for a service
    
    Args:
        service_name: Name of the service (used in log entries)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured logger instance
    """
    
    # Create logger
    logger = logging.getLogger(service_name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Create console handler with JSON formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(LokiJSONFormatter(service_name))
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger

def log_request_info(logger: logging.Logger, request_id: str, endpoint: str, 
                    method: str = "POST", extra_data: Optional[Dict[str, Any]] = None):
    """Log incoming request information"""
    log_data = {
        "request_id": request_id,
        "endpoint": endpoint,
        "method": method,
        "event_type": "request_received"
    }
    if extra_data:
        log_data.update(extra_data)
        
    logger.info("Request received", extra=log_data)

def log_response_info(logger: logging.Logger, request_id: str, endpoint: str,
                     status_code: int, duration_ms: float, 
                     extra_data: Optional[Dict[str, Any]] = None):
    """Log response information"""
    log_data = {
        "request_id": request_id,
        "endpoint": endpoint,
        "status_code": status_code,
        "duration_ms": round(duration_ms, 2),
        "event_type": "request_completed"
    }
    if extra_data:
        log_data.update(extra_data)
        
    logger.info("Request completed", extra=log_data)

def log_service_call(logger: logging.Logger, service_name: str, endpoint: str,
                    duration_ms: float, success: bool,
                    extra_data: Optional[Dict[str, Any]] = None):
    """Log service-to-service calls"""
    log_data = {
        "target_service": service_name,
        "target_endpoint": endpoint,
        "duration_ms": round(duration_ms, 2),
        "success": success,
        "event_type": "service_call"
    }
    if extra_data:
        log_data.update(extra_data)
        
    level = "info" if success else "error"
    getattr(logger, level)("Service call completed", extra=log_data)

def log_database_operation(logger: logging.Logger, operation: str, table: str,
                          duration_ms: float, success: bool, affected_rows: int = 0,
                          extra_data: Optional[Dict[str, Any]] = None):
    """Log database operations"""
    log_data = {
        "db_operation": operation,
        "db_table": table,
        "duration_ms": round(duration_ms, 2),
        "success": success,
        "affected_rows": affected_rows,
        "event_type": "database_operation"
    }
    if extra_data:
        log_data.update(extra_data)
        
    level = "info" if success else "error"
    getattr(logger, level)("Database operation completed", extra=log_data)

def log_model_operation(logger: logging.Logger, operation: str, model_name: str,
                       duration_ms: float, success: bool,
                       extra_data: Optional[Dict[str, Any]] = None):
    """Log ML model operations"""
    log_data = {
        "model_operation": operation,
        "model_name": model_name,
        "duration_ms": round(duration_ms, 2),
        "success": success,
        "event_type": "model_operation"
    }
    if extra_data:
        log_data.update(extra_data)
        
    level = "info" if success else "error"
    getattr(logger, level)("Model operation completed", extra=log_data)

# Health check logging
def log_health_check(logger: logging.Logger, service_name: str, status: str,
                    checks: Dict[str, bool], extra_data: Optional[Dict[str, Any]] = None):
    """Log health check results"""
    log_data = {
        "service": service_name,
        "health_status": status,
        "health_checks": checks,
        "event_type": "health_check"
    }
    if extra_data:
        log_data.update(extra_data)
        
    logger.info("Health check performed", extra=log_data)