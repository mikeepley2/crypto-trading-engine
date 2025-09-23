#!/usr/bin/env python3
"""
Hot-patch script to add multi-platform endpoints to running trading engine
This script dynamically adds multi-platform support without restarting the service
"""

import sys
import os
import logging
from fastapi import HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the multi-platform extension
try:
    from multi_platform_extension import get_multi_platform_extension, initialize_multi_platform_extension
    logger.info("Multi-platform extension imported successfully")
except Exception as e:
    logger.error(f"Failed to import multi-platform extension: {e}")
    sys.exit(1)

def add_multi_platform_endpoints(app):
    """Add multi-platform endpoints to existing FastAPI app"""
    
    logger.info("Adding multi-platform endpoints to existing trading engine...")
    
    # Initialize multi-platform extension
    mp_extension = initialize_multi_platform_extension()
    if not mp_extension:
        logger.error("Failed to initialize multi-platform extension")
        return False
    
    # Multi-platform request models
    class MultiPlatformTradeRequest(BaseModel):
        symbol: str
        action: str  # BUY/SELL
        size_usd: float
        platform: Optional[str] = None  # If None, uses best available platform
        order_type: str = "MARKET"
    
    class PlatformConfigRequest(BaseModel):
        config: Dict[str, Any]
    
    # Platform Status Endpoint
    @app.get("/platforms")
    async def get_platforms_status():
        """Get status of all trading platforms"""
        try:
            mp_ext = get_multi_platform_extension()
            if not mp_ext:
                return {
                    "error": "Multi-platform extension not available",
                    "fallback_mode": "coinbase_only",
                    "timestamp": "2025-08-26T19:00:00Z"
                }
            
            return mp_ext.get_platforms_status()
        except Exception as e:
            logger.error(f"Platform status check failed: {e}")
            raise HTTPException(status_code=500, detail=f"Platform status check failed: {e}")
    
    # Platform Portfolios Endpoint
    @app.get("/platforms/portfolios")
    async def get_platform_portfolios():
        """Get portfolio information from all active platforms"""
        try:
            mp_ext = get_multi_platform_extension()
            if not mp_ext:
                raise HTTPException(status_code=500, detail="Multi-platform extension not available")
            
            return mp_ext.get_platform_portfolios()
        except Exception as e:
            logger.error(f"Platform portfolios check failed: {e}")
            raise HTTPException(status_code=500, detail=f"Platform portfolios check failed: {e}")
    
    # Available Symbols Endpoint
    @app.get("/platforms/symbols")
    async def get_available_symbols():
        """Get available trading symbols from all platforms"""
        try:
            mp_ext = get_multi_platform_extension()
            if not mp_ext:
                raise HTTPException(status_code=500, detail="Multi-platform extension not available")
            
            return mp_ext.get_available_symbols()
        except Exception as e:
            logger.error(f"Symbols check failed: {e}")
            raise HTTPException(status_code=500, detail=f"Symbols check failed: {e}")
    
    # Multi-platform Trade Execution
    @app.post("/platforms/execute_trade")
    async def execute_multi_platform_trade(trade_request: MultiPlatformTradeRequest):
        """Execute trade on specified platform or best available platform"""
        try:
            mp_ext = get_multi_platform_extension()
            if not mp_ext:
                # Fallback to existing single-platform execution
                logger.info("Multi-platform not available, falling back to standard execution")
                # Use the existing execute_trade endpoint logic
                raise HTTPException(status_code=503, detail="Multi-platform trading not available, use /execute_trade instead")
            
            return mp_ext.execute_multi_platform_trade(
                symbol=trade_request.symbol,
                action=trade_request.action,
                size_usd=trade_request.size_usd,
                platform=trade_request.platform
            )
        except Exception as e:
            logger.error(f"Multi-platform trade execution failed: {e}")
            raise HTTPException(status_code=500, detail=f"Multi-platform trade execution failed: {e}")
    
    # Platform Configuration Management
    @app.get("/platforms/config")
    async def get_platform_config():
        """Get current multi-platform configuration"""
        try:
            mp_ext = get_multi_platform_extension()
            if not mp_ext:
                raise HTTPException(status_code=500, detail="Multi-platform extension not available")
            
            return mp_ext.get_platform_config()
        except Exception as e:
            logger.error(f"Platform config retrieval failed: {e}")
            raise HTTPException(status_code=500, detail=f"Platform config retrieval failed: {e}")
    
    @app.post("/platforms/config")
    async def update_platform_config(config_request: PlatformConfigRequest):
        """Update multi-platform configuration"""
        try:
            mp_ext = get_multi_platform_extension()
            if not mp_ext:
                raise HTTPException(status_code=500, detail="Multi-platform extension not available")
            
            return mp_ext.update_platform_config(config_request.config)
        except Exception as e:
            logger.error(f"Platform config update failed: {e}")
            raise HTTPException(status_code=500, detail=f"Platform config update failed: {e}")
    
    # Platform Health Check
    @app.get("/platforms/health")
    async def platforms_health_check():
        """Comprehensive health check for all platforms"""
        try:
            mp_ext = get_multi_platform_extension()
            if not mp_ext:
                return {
                    "status": "limited",
                    "message": "Multi-platform extension not available",
                    "available_features": ["coinbase_only"],
                    "timestamp": "2025-08-26T19:00:00Z"
                }
            
            platforms_status = mp_ext.get_platforms_status()
            
            # Determine overall health
            active_count = len(platforms_status.get('active_platforms', []))
            total_enabled = platforms_status.get('total_enabled', 0)
            
            if active_count == 0:
                health_status = "critical"
            elif active_count < total_enabled:
                health_status = "degraded"
            else:
                health_status = "healthy"
            
            return {
                "status": health_status,
                "active_platforms": active_count,
                "total_enabled": total_enabled,
                "platforms": platforms_status,
                "timestamp": platforms_status.get('timestamp')
            }
            
        except Exception as e:
            logger.error(f"Platform health check failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": "2025-08-26T19:00:00Z"
            }
    
    logger.info("Multi-platform endpoints added successfully")
    return True

# Main execution for hot-patching
if __name__ == "__main__":
    logger.info("Multi-platform hot-patch script starting...")
    
    # This script is designed to be imported and executed from within the running application
    # It adds the endpoints dynamically to the existing FastAPI app
    logger.info("Hot-patch script loaded - use add_multi_platform_endpoints(app) to apply")
