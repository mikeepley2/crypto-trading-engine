#!/usr/bin/env python3
"""
Enhanced Trade Execution Engine with Multi-Platform Support
This file adds multi-platform endpoints to the existing trade execution engine
"""

# Add imports for multi-platform support
import json
from typing import Dict, List, Optional, Any
from pydantic import BaseModel

# Multi-platform extension class and functions
class MultiPlatformStatus:
    """Simple multi-platform status provider"""
    
    @staticmethod
    def get_platforms_status():
        """Get status of all configured platforms"""
        try:
            # Load configuration
            config_path = "/app/trading_platforms.json"
            config = {}
            
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
            
            platforms = config.get('platforms', {})
            live_trading_enabled = config.get('enable_live_trading', False)
            
            platforms_status = {}
            active_platforms = []
            
            for platform_name, platform_config in platforms.items():
                is_enabled = platform_config.get('enabled', False)
                has_credentials = bool(platform_config.get('api_key') or platform_config.get('private_key'))
                
                status = {
                    "enabled": is_enabled,
                    "has_credentials": has_credentials,
                    "status": "active" if is_enabled and has_credentials else "disabled"
                }
                
                # For Coinbase, we can check if it's actually working
                if platform_name == "coinbase" and is_enabled:
                    try:
                        # Use the existing trading engine's Coinbase API to test connection
                        if trading_engine and trading_engine.exec_mode.execution_mode == "live":
                            balance = trading_engine.coinbase_api.get_account_balance('USD')
                            status["connected"] = True
                            status["usd_balance"] = balance
                            active_platforms.append(platform_name)
                        else:
                            status["connected"] = False
                            status["note"] = "Mock mode"
                    except Exception as e:
                        status["connected"] = False
                        status["error"] = str(e)
                elif is_enabled:
                    status["note"] = "Ready for activation"
                
                platforms_status[platform_name] = status
            
            return {
                "platforms": platforms_status,
                "active_platforms": active_platforms,
                "total_enabled": sum(1 for p in platforms.values() if p.get('enabled', False)),
                "live_trading_enabled": live_trading_enabled,
                "current_mode": trading_engine.exec_mode.execution_mode if trading_engine else "unknown",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get platforms status: {e}")
            return {
                "error": f"Failed to get platforms status: {e}",
                "fallback_mode": "coinbase_only",
                "timestamp": datetime.now().isoformat()
            }
    
    @staticmethod
    def get_platform_config():
        """Get current platform configuration"""
        try:
            config_path = "/app/trading_platforms.json"
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    return json.load(f)
            else:
                return {
                    "error": "Configuration file not found",
                    "default_config": {
                        "enable_live_trading": True,
                        "platforms": {
                            "coinbase": {"enabled": True},
                            "binance_us": {"enabled": False},
                            "kucoin": {"enabled": False}
                        }
                    }
                }
        except Exception as e:
            return {"error": f"Failed to load configuration: {e}"}

# Request models for multi-platform endpoints
class MultiPlatformTradeRequest(BaseModel):
    symbol: str
    action: str  # BUY/SELL
    size_usd: float
    platform: Optional[str] = None  # If None, uses default platform (Coinbase)
    order_type: str = "MARKET"

class PlatformConfigRequest(BaseModel):
    config: Dict[str, Any]

# Add multi-platform endpoints to existing app

@app.get("/platforms")
async def get_platforms_status():
    """Get status of all trading platforms"""
    return MultiPlatformStatus.get_platforms_status()

@app.get("/platforms/health")
async def platforms_health_check():
    """Comprehensive health check for all platforms"""
    try:
        platforms_status = MultiPlatformStatus.get_platforms_status()
        
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
            "platforms": platforms_status.get('platforms', {}),
            "trading_engine_status": "running" if trading_engine else "not_initialized",
            "current_mode": platforms_status.get('current_mode', 'unknown'),
            "timestamp": platforms_status.get('timestamp')
        }
        
    except Exception as e:
        logger.error(f"Platform health check failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/platforms/portfolios")
async def get_platform_portfolios():
    """Get portfolio information from all active platforms"""
    try:
        platforms_status = MultiPlatformStatus.get_platforms_status()
        portfolios = {}
        
        # For now, we only support Coinbase through the existing engine
        for platform_name in platforms_status.get('active_platforms', []):
            if platform_name == "coinbase" and trading_engine:
                try:
                    portfolio = trading_engine.get_portfolio_status()
                    portfolios[platform_name] = portfolio
                except Exception as e:
                    portfolios[platform_name] = {
                        "error": f"Failed to get portfolio: {e}",
                        "timestamp": datetime.now().isoformat()
                    }
            else:
                portfolios[platform_name] = {
                    "status": "not_implemented",
                    "message": f"Portfolio access for {platform_name} not yet implemented",
                    "timestamp": datetime.now().isoformat()
                }
        
        return {
            "portfolios": portfolios,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Platform portfolios check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Platform portfolios check failed: {e}")

@app.get("/platforms/config")
async def get_platform_config():
    """Get current multi-platform configuration"""
    return MultiPlatformStatus.get_platform_config()

@app.post("/platforms/config")
async def update_platform_config(config_request: PlatformConfigRequest):
    """Update multi-platform configuration"""
    try:
        # Validate configuration
        config = config_request.config
        if 'platforms' not in config:
            raise HTTPException(status_code=400, detail="Configuration must include 'platforms' section")
        
        # Save configuration
        config_path = "/app/trading_platforms.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info("Multi-platform configuration updated")
        
        return {
            "status": "success",
            "message": "Platform configuration updated successfully",
            "note": "Restart may be required for some changes to take effect",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to update platform configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Configuration update failed: {e}")

@app.post("/platforms/execute_trade")
async def execute_multi_platform_trade(trade_request: MultiPlatformTradeRequest):
    """Execute trade on specified platform or default platform"""
    try:
        # For now, route all trades through the existing Coinbase engine
        # Future implementations can add routing logic for other platforms
        
        if trade_request.platform and trade_request.platform.lower() not in ["coinbase", "default", None]:
            raise HTTPException(
                status_code=400, 
                detail=f"Platform {trade_request.platform} not yet implemented. Currently only Coinbase is supported."
            )
        
        if not trading_engine:
            raise HTTPException(status_code=500, detail="Trading engine not initialized")
        
        # Convert to standard trade request and execute through existing engine
        standard_trade_request = TradeRequest(
            symbol=trade_request.symbol,
            action=trade_request.action,
            size_usd=trade_request.size_usd,
            order_type=trade_request.order_type
        )
        
        result = trading_engine.execute_trade(standard_trade_request)
        
        # Add platform information to result
        result['platform'] = 'coinbase'
        result['multi_platform_enabled'] = True
        
        return result
        
    except Exception as e:
        logger.error(f"Multi-platform trade execution failed: {e}")
        raise HTTPException(status_code=500, detail=f"Multi-platform trade execution failed: {e}")

@app.get("/platforms/symbols")
async def get_available_symbols():
    """Get available trading symbols from all platforms"""
    try:
        # For now, return a basic set of supported symbols
        # Future implementations can query each platform's API
        
        symbols_by_platform = {
            "coinbase": {
                "major_cryptos": ["BTC", "ETH", "ADA", "SOL", "LINK", "AVAX", "DOT", "UNI", "XRP", "DOGE"],
                "note": "This is a subset of available symbols. Coinbase supports many more pairs.",
                "status": "active" if trading_engine else "inactive"
            },
            "binance_us": {
                "status": "disabled",
                "note": "Not yet implemented - awaiting API credentials"
            },
            "kucoin": {
                "status": "disabled", 
                "note": "Not yet implemented - awaiting API credentials"
            }
        }
        
        return {
            "symbols_by_platform": symbols_by_platform,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Symbols check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Symbols check failed: {e}")

logger.info("Multi-platform endpoints successfully loaded")
