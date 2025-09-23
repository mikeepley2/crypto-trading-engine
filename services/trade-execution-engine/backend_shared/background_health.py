#!/usr/bin/env python3
"""
Background Service Health Status Framework
==========================================

Framework for services that don't have HTTP endpoints to report their health status.
Uses file-based status reporting that can be monitored by the unified monitoring dashboard.

Features:
- JSON status file management
- Health status reporting
- Heartbeat mechanism
- Error logging and recovery
- Resource usage monitoring
- Performance metrics

Usage:
    from backend.shared.background_health import BackgroundHealthReporter
    
    reporter = BackgroundHealthReporter("sentiment-aggregator", "/tmp/status")
    reporter.start_monitoring()
    
    # Update status
    reporter.update_status("processing", {"processed": 100, "errors": 0})
    
    # Report error
    reporter.report_error("Database connection failed")
    
    # Stop monitoring
    reporter.stop_monitoring()
"""

import json
import logging
import os
import psutil
import threading
import time
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict


logger = logging.getLogger(__name__)


@dataclass
class HealthReport:
    """Health report data structure"""
    service_name: str
    status: str  # healthy, degraded, unhealthy, starting, stopping
    timestamp: str
    uptime_seconds: int
    process_id: int
    version: str
    last_activity: str
    errors: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    system_resources: Dict[str, Any]
    custom_data: Dict[str, Any]


class BackgroundHealthReporter:
    """Background service health status reporter"""
    
    def __init__(self, service_name: str, status_dir: str = "/tmp/service_status", version: str = "1.0.0"):
        self.service_name = service_name
        self.status_dir = Path(status_dir)
        self.version = version
        self.status_file = self.status_dir / f"{service_name}_status.json"
        self.heartbeat_file = self.status_dir / f"{service_name}_heartbeat.json"
        
        # Create status directory if it doesn't exist
        self.status_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize state
        self.start_time = datetime.now(timezone.utc)
        self.last_activity = self.start_time
        self.errors: List[Dict[str, Any]] = []
        self.metrics: Dict[str, Any] = {}
        self.custom_data: Dict[str, Any] = {}
        self.status = "starting"
        
        # Monitoring thread
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_active = False
        self.heartbeat_interval = 30  # seconds
        
        logger.info(f"Initialized health reporter for {service_name}")
    
    def start_monitoring(self):
        """Start the background health monitoring"""
        if self.monitoring_active:
            logger.warning(f"Monitoring already active for {self.service_name}")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            name=f"{self.service_name}-health-monitor",
            daemon=True
        )
        self.monitoring_thread.start()
        
        self.status = "healthy"
        self.update_status_file()
        logger.info(f"Started health monitoring for {self.service_name}")
    
    def stop_monitoring(self):
        """Stop the background health monitoring"""
        self.monitoring_active = False
        self.status = "stopping"
        self.update_status_file()
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        
        logger.info(f"Stopped health monitoring for {self.service_name}")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                self._update_system_metrics()
                self.update_status_file()
                self._write_heartbeat()
                time.sleep(self.heartbeat_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop for {self.service_name}: {e}")
                time.sleep(5)  # Short delay before retrying
    
    def _update_system_metrics(self):
        """Update system resource metrics"""
        try:
            process = psutil.Process(os.getpid())
            
            self.metrics.update({
                "cpu_percent": process.cpu_percent(),
                "memory_mb": process.memory_info().rss / 1024 / 1024,
                "threads": process.num_threads(),
                "open_files": len(process.open_files()),
                "connections": len(process.connections()),
                "uptime_seconds": (datetime.now(timezone.utc) - self.start_time).total_seconds()
            })
            
            # System-wide metrics
            system_cpu = psutil.cpu_percent()
            system_memory = psutil.virtual_memory()
            
            self.metrics["system"] = {
                "cpu_percent": system_cpu,
                "memory_percent": system_memory.percent,
                "memory_available_gb": system_memory.available / (1024**3)
            }
            
        except Exception as e:
            logger.error(f"Failed to update system metrics: {e}")
    
    def update_status(self, status: str = None, custom_data: Dict[str, Any] = None):
        """Update service status and custom data"""
        if status:
            self.status = status
        
        if custom_data:
            self.custom_data.update(custom_data)
        
        self.last_activity = datetime.now(timezone.utc)
        
        if not self.monitoring_active:
            self.update_status_file()
    
    def report_error(self, error_message: str, error_type: str = "error", details: Dict[str, Any] = None):
        """Report an error"""
        error_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": error_type,
            "message": error_message,
            "details": details or {}
        }
        
        self.errors.append(error_entry)
        
        # Keep only last 50 errors to prevent unbounded growth
        if len(self.errors) > 50:
            self.errors = self.errors[-50:]
        
        # Update status based on error severity
        if error_type in ["critical", "fatal"]:
            self.status = "unhealthy"
        elif error_type == "warning" and self.status == "healthy":
            self.status = "degraded"
        
        logger.error(f"Service {self.service_name} reported {error_type}: {error_message}")
        
        if not self.monitoring_active:
            self.update_status_file()
    
    def clear_errors(self):
        """Clear all errors and reset status to healthy"""
        self.errors.clear()
        self.status = "healthy"
        self.update_status_file()
        logger.info(f"Cleared errors for service {self.service_name}")
    
    def set_metric(self, key: str, value: Any):
        """Set a custom metric"""
        self.metrics[key] = value
    
    def increment_metric(self, key: str, amount: float = 1.0):
        """Increment a custom metric"""
        current = self.metrics.get(key, 0)
        self.metrics[key] = current + amount
    
    def update_status_file(self):
        """Update the status file"""
        try:
            # Get system resource info
            system_resources = {}
            try:
                process = psutil.Process(os.getpid())
                cpu_percent = process.cpu_percent()
                memory_info = process.memory_info()
                
                system_resources = {
                    "cpu_percent": cpu_percent,
                    "memory_mb": memory_info.rss / 1024 / 1024,
                    "memory_vms_mb": memory_info.vms / 1024 / 1024,
                    "threads": process.num_threads(),
                    "open_files": len(process.open_files()),
                    "process_id": os.getpid()
                }
            except Exception as e:
                logger.debug(f"Could not get system resources: {e}")
            
            # Create health report
            report = HealthReport(
                service_name=self.service_name,
                status=self.status,
                timestamp=datetime.now(timezone.utc).isoformat(),
                uptime_seconds=int((datetime.now(timezone.utc) - self.start_time).total_seconds()),
                process_id=os.getpid(),
                version=self.version,
                last_activity=self.last_activity.isoformat(),
                errors=self.errors[-10:],  # Only include last 10 errors in status file
                metrics=self.metrics,
                system_resources=system_resources,
                custom_data=self.custom_data
            )
            
            # Write status file atomically
            temp_file = self.status_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(asdict(report), f, indent=2, default=str)
            
            temp_file.replace(self.status_file)
            
        except Exception as e:
            logger.error(f"Failed to update status file for {self.service_name}: {e}")
    
    def _write_heartbeat(self):
        """Write heartbeat file"""
        try:
            heartbeat_data = {
                "service_name": self.service_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "process_id": os.getpid(),
                "status": self.status
            }
            
            temp_file = self.heartbeat_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(heartbeat_data, f, indent=2, default=str)
            
            temp_file.replace(self.heartbeat_file)
            
        except Exception as e:
            logger.error(f"Failed to write heartbeat for {self.service_name}: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status as dictionary"""
        return {
            "service_name": self.service_name,
            "status": self.status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": int((datetime.now(timezone.utc) - self.start_time).total_seconds()),
            "version": self.version,
            "last_activity": self.last_activity.isoformat(),
            "error_count": len(self.errors),
            "recent_errors": self.errors[-5:],
            "metrics": self.metrics,
            "custom_data": self.custom_data
        }


class BackgroundHealthReader:
    """Reader for background service health status files"""
    
    def __init__(self, status_dir: str = "/tmp/service_status"):
        self.status_dir = Path(status_dir)
    
    def read_service_status(self, service_name: str) -> Optional[Dict[str, Any]]:
        """Read status for a specific service"""
        status_file = self.status_dir / f"{service_name}_status.json"
        
        if not status_file.exists():
            return None
        
        try:
            with open(status_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read status for {service_name}: {e}")
            return None
    
    def read_heartbeat(self, service_name: str) -> Optional[Dict[str, Any]]:
        """Read heartbeat for a specific service"""
        heartbeat_file = self.status_dir / f"{service_name}_heartbeat.json"
        
        if not heartbeat_file.exists():
            return None
        
        try:
            with open(heartbeat_file, 'r') as f:
                heartbeat_data = json.load(f)
            
            # Check if heartbeat is recent (within last 2 minutes)
            heartbeat_time = datetime.fromisoformat(heartbeat_data['timestamp'].replace('Z', '+00:00'))
            time_diff = datetime.now(timezone.utc) - heartbeat_time
            
            heartbeat_data['age_seconds'] = time_diff.total_seconds()
            heartbeat_data['is_recent'] = time_diff.total_seconds() < 120  # 2 minutes
            
            return heartbeat_data
            
        except Exception as e:
            logger.error(f"Failed to read heartbeat for {service_name}: {e}")
            return None
    
    def list_services(self) -> List[str]:
        """List all services with status files"""
        if not self.status_dir.exists():
            return []
        
        services = []
        for file in self.status_dir.glob("*_status.json"):
            service_name = file.stem.replace("_status", "")
            services.append(service_name)
        
        return services
    
    def get_all_statuses(self) -> Dict[str, Dict[str, Any]]:
        """Get status for all services"""
        statuses = {}
        
        for service_name in self.list_services():
            status = self.read_service_status(service_name)
            heartbeat = self.read_heartbeat(service_name)
            
            if status:
                status['heartbeat'] = heartbeat
                statuses[service_name] = status
        
        return statuses


# Convenience functions for common usage patterns
def create_background_reporter(service_name: str, status_dir: str = None, version: str = "1.0.0") -> BackgroundHealthReporter:
    """Create and start a background health reporter"""
    if status_dir is None:
        # Use a default location based on the service name
        status_dir = f"/tmp/service_status" if os.name != 'nt' else os.path.join(os.getenv('TEMP', 'C:/temp'), 'service_status')
    
    reporter = BackgroundHealthReporter(service_name, status_dir, version)
    reporter.start_monitoring()
    return reporter


def read_background_status(service_name: str, status_dir: str = None) -> Optional[Dict[str, Any]]:
    """Read status for a background service"""
    if status_dir is None:
        status_dir = f"/tmp/service_status" if os.name != 'nt' else os.path.join(os.getenv('TEMP', 'C:/temp'), 'service_status')
    
    reader = BackgroundHealthReader(status_dir)
    return reader.read_service_status(service_name)
