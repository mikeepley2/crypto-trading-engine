#!/usr/bin/env python3
"""
Automated Signal Bridge Service
Continuously monitors enhanced signals and converts them to trade recommendations
Enhanced with Symbol Standardization for consistent symbol handling
"""

import os
import time
import logging
import mysql.connector
import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import threading
from decimal import Decimal
from fastapi import FastAPI
import uvicorn
from symbol_utils import SymbolStandardizer
# from dynamic_position_manager import DynamicPositionManager
# from symbol_utils import SymbolStandardizer, normalize_symbol, to_coinbase_format

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [BRIDGE] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import asset filter for Coinbase support
try:
    # Try direct import first (for Docker container)
    from coinbase_asset_filter import is_asset_supported, get_supported_assets
    
    logger.info("✅ Successfully imported database-driven asset filtering")
    
except ImportError:
    try:
        # Try importing from shared directory (for local development)
        import sys
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        shared_path = os.path.join(current_dir, '..', 'shared')
        sys.path.insert(0, shared_path)
        
        from coinbase_asset_filter import is_asset_supported, get_supported_assets
        
        logger.info("✅ Successfully imported database-driven asset filtering from shared path")
        
    except ImportError as e:
        logger.error(f"❌ CRITICAL: Asset filter import failed: {e}")
        raise Exception("Asset filter module required - no fallbacks allowed")

# Global health status tracking
health_status = {
    "status": "starting",
    "last_signal_processing": None,
    "signals_processed_today": 0,
    "trades_executed_today": 0,
    "database_connected": False,
    "trading_engine_connected": False,
    "last_error": None,
    "start_time": datetime.now(),
    "total_signals_processed": 0,
    "total_trades_executed": 0,
    "service_name": "automated-signal-bridge",
    "processing_cycle_seconds": 300,
    "last_successful_trade": None,
    "last_consolidation": None,
    "consolidations_performed_today": 0,
    "total_consolidations_performed": 0,
    "consolidation_interval_hours": 12
}

# FastAPI app for health endpoints
app = FastAPI(title="Automated Signal Bridge Health API")

# Global bridge instance for API access
bridge_instance = None

@app.get("/health")
async def health_check():
    """Standard health check endpoint"""
    is_healthy = (
        health_status["database_connected"] and 
        health_status["trading_engine_connected"] and
        health_status["status"] not in ["error", "critical"]
    )
    
    return {
        "status": "healthy" if is_healthy else "unhealthy",
        "service": health_status["service_name"],
        "timestamp": datetime.now().isoformat()
    }

@app.get("/status")
async def get_status():
    """Detailed status information"""
    uptime = datetime.now() - health_status["start_time"]
    
    next_processing = None
    if health_status["last_signal_processing"]:
        last_proc = datetime.fromisoformat(health_status["last_signal_processing"].replace('Z', '+00:00'))
        next_proc_time = last_proc + timedelta(seconds=health_status["processing_cycle_seconds"])
        time_until_next = next_proc_time - datetime.now()
        if time_until_next.total_seconds() > 0:
            next_processing = f"{int(time_until_next.total_seconds() // 60)}m {int(time_until_next.total_seconds() % 60)}s"
        else:
            next_processing = "due now"
    
    return {
        **health_status,
        "uptime_seconds": uptime.total_seconds(),
        "uptime_human": str(uptime).split('.')[0],
        "next_processing_in": next_processing,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/metrics")
async def get_metrics():
    """Service metrics for monitoring"""
    uptime = datetime.now() - health_status["start_time"]
    
    return {
        "signals_processed_today": health_status["signals_processed_today"],
        "trades_executed_today": health_status["trades_executed_today"],
        "consolidations_performed_today": health_status["consolidations_performed_today"],
        "total_signals_processed": health_status["total_signals_processed"],
        "total_trades_executed": health_status["total_trades_executed"],
        "total_consolidations_performed": health_status["total_consolidations_performed"],
        "last_processing": health_status["last_signal_processing"],
        "last_successful_trade": health_status["last_successful_trade"],
        "last_consolidation": health_status["last_consolidation"],
        "uptime_seconds": uptime.total_seconds(),
        "processing_cycle_seconds": health_status["processing_cycle_seconds"],
        "consolidation_interval_hours": health_status["consolidation_interval_hours"],
        "error_count": 1 if health_status["last_error"] else 0,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/position-management/config")
async def get_position_management_config():
    """Get current dynamic position management configuration"""
    global bridge_instance
    if bridge_instance and bridge_instance.position_manager:
        return {
            "config": bridge_instance.position_manager.get_configuration(),
            "timestamp": datetime.now().isoformat()
        }
    return {"error": "Position manager not initialized"}

@app.post("/position-management/config")
async def update_position_management_config(config_updates: dict):
    """Update dynamic position management configuration"""
    global bridge_instance
    if bridge_instance and bridge_instance.position_manager:
        success = bridge_instance.position_manager.update_configuration(config_updates)
        if success:
            return {
                "status": "success",
                "message": f"Updated {len(config_updates)} parameters",
                "updated_config": bridge_instance.position_manager.get_configuration(),
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {"error": "Failed to update configuration"}
    return {"error": "Position manager not initialized"}

@app.post("/intelligence/signals")
async def receive_intelligence_signals(request: dict):
    """Receive and process real-time intelligence signals"""
    try:
        signals = request.get('signals', [])
        source = request.get('source', 'unknown')
        
        logger.info(f"📡 Received {len(signals)} intelligence signals from {source}")
        
        processed_count = 0
        for signal in signals:
            # Convert intelligence signal to trading signal format
            symbol = signal.get('symbol', '').upper()
            confidence = signal.get('confidence', 0.5)
            direction = signal.get('direction', 0)
            metadata = signal.get('metadata', {})
            
            if symbol and confidence > 0.4:  # Minimum confidence threshold
                # Create trading signal entry
                trading_signal = {
                    'symbol': symbol,
                    'confidence': confidence,
                    'direction': direction,
                    'source': 'intelligence_event',
                    'reasoning': metadata.get('reasoning', 'Intelligence event detected'),
                    'intelligence_source': metadata.get('intelligence_source', source),
                    'priority': metadata.get('priority', 'medium'),
                    'time_sensitivity': metadata.get('time_sensitivity', 60)
                }
                
                # Save to trading signals table for processing
                try:
                    # Use bridge instance database config
                    global bridge_instance
                    if bridge_instance:
                        conn = mysql.connector.connect(**bridge_instance.signals_db_config)
                        cursor = conn.cursor()
                        
                        cursor.execute("""
                            INSERT INTO trading_signals 
                            (symbol, confidence, source, metadata, created_at)
                            VALUES (%s, %s, %s, %s, NOW())
                        """, (
                            symbol,
                            confidence,
                            'intelligence_event',
                            json.dumps(trading_signal)
                        ))
                        
                        conn.commit()
                        cursor.close()
                        conn.close()
                        
                        processed_count += 1
                        logger.info(f"📈 Processed intelligence signal: {symbol} (confidence: {confidence:.2f})")
                    else:
                        logger.error("Bridge instance not available for intelligence signal processing")
                        
                except Exception as e:
                    logger.error(f"❌ Error saving intelligence signal for {symbol}: {e}")
        
        return {
            'status': 'success',
            'received_signals': len(signals),
            'processed_signals': processed_count,
            'message': f'Processed {processed_count}/{len(signals)} intelligence signals'
        }
        
    except Exception as e:
        logger.error(f"❌ Error processing intelligence signals: {e}")
        return {"error": str(e)}

@app.get("/position-management/status")
async def get_position_management_status():
    """Get current position management status and recent activity"""
    global bridge_instance
    if bridge_instance and bridge_instance.position_manager:
        config = bridge_instance.position_manager.get_configuration()
        portfolio = bridge_instance.position_manager.get_current_portfolio()
        
        # Get recent rebalancing activity
        try:
            conn = mysql.connector.connect(**bridge_instance.trades_db_config)
            cursor = conn.cursor(dictionary=True)
            
            # Get recent rebalancing trades
            cursor.execute("""
                SELECT symbol, action, confidence, reasoning, created_at
                FROM trade_recommendations
                WHERE reasoning LIKE '%Rebalancing%'
                AND created_at >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
                ORDER BY created_at DESC
                LIMIT 10
            """)
            recent_rebalancing = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
        except Exception as e:
            recent_rebalancing = []
            logger.error(f"Error fetching rebalancing history: {e}")
        
        return {
            "enabled": config.get('enable_dynamic_rebalancing', False),
            "strong_signal_threshold": config.get('strong_signal_threshold', 0.85),
            "max_position_multiplier": config.get('max_position_multiplier', 2.0),
            "portfolio_value": portfolio.get('total_portfolio_value', 0),
            "available_cash": portfolio.get('available_balance', 0),
            "position_count": len(portfolio.get('positions', [])),
            "recent_rebalancing": recent_rebalancing,
            "timestamp": datetime.now().isoformat()
        }
    return {"error": "Position manager not initialized"}

@app.get("/consolidation/status")
async def get_consolidation_status():
    """Get portfolio consolidation status and schedule"""
    global bridge_instance
    if bridge_instance:
        current_time = time.time()
        time_since_last = current_time - bridge_instance.last_consolidation
        time_until_next = bridge_instance.consolidation_interval - time_since_last
        
        return {
            "consolidation_enabled": True,
            "interval_hours": bridge_instance.consolidation_interval / 3600,
            "min_position_value_usd": bridge_instance.min_consolidation_value,
            "last_consolidation": health_status["last_consolidation"],
            "next_consolidation_in_seconds": max(0, time_until_next),
            "next_consolidation_in_hours": max(0, time_until_next / 3600),
            "consolidations_today": health_status["consolidations_performed_today"],
            "total_consolidations": health_status["total_consolidations_performed"],
            "timestamp": datetime.now().isoformat()
        }
    return {"error": "Bridge instance not initialized"}

@app.post("/consolidation/trigger")
async def trigger_consolidation():
    """Manually trigger portfolio consolidation"""
    global bridge_instance
    if bridge_instance:
        try:
            # Reset consolidation timer to force immediate check
            bridge_instance.last_consolidation = 0
            
            # Get consolidation signals
            consolidation_signals = bridge_instance.position_manager.consolidate_small_positions(
                min_position_value_usd=bridge_instance.min_consolidation_value
            )
            
            if not consolidation_signals:
                return {
                    "status": "success",
                    "message": "No small positions found for consolidation",
                    "signals_generated": 0,
                    "timestamp": datetime.now().isoformat()
                }
            
            # Process consolidation signals immediately
            processed_signals = 0
            for signal in consolidation_signals:
                try:
                    # Create trade recommendation for consolidation sell
                    recommendation = {
                        'symbol': signal['symbol'],
                        'action': 'sell',
                        'confidence': signal['confidence'],
                        'reasoning': signal['reasoning'],
                        'amount_usd': signal['amount_usd'],
                        'created_at': datetime.now()
                    }
                    
                    # Execute the sell trade immediately for consolidation
                    success = bridge_instance.execute_trade(recommendation)
                    if success:
                        processed_signals += 1
                        logger.info(f"[CONSOLIDATION] Successfully sold {signal['symbol']} for ${signal['amount_usd']:.2f}")
                    else:
                        logger.warning(f"[CONSOLIDATION] Failed to sell {signal['symbol']}")
                        
                except Exception as e:
                    logger.error(f"[CONSOLIDATION] Error processing signal for {signal['symbol']}: {e}")
            
            return {
                "status": "success", 
                "message": f"Generated {len(consolidation_signals)} consolidation signals, processed {processed_signals}",
                "signals_generated": len(consolidation_signals),
                "signals_processed": processed_signals,
                "signals": consolidation_signals,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error triggering consolidation: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    return {"error": "Bridge instance not initialized"}

@app.get("/recommendations")
async def get_recommendations():
    """Get fresh trade recommendations for the trade orchestrator"""
    try:
        # Connect to crypto_transactions database for trade recommendations
        trades_db_config = {
            'host': os.environ.get('DATABASE_HOST', '192.168.230.163'),
            'user': os.environ.get('DATABASE_USER', 'news_collector'),
            'password': os.environ.get('DATABASE_PASSWORD', '99Rules!'),
            'database': 'crypto_transactions',
            'port': int(os.environ.get('DATABASE_PORT', 3306))
        }
        
        conn = mysql.connector.connect(**trades_db_config)
        cursor = conn.cursor(dictionary=True)
        
        # Get recommendations from the last 4 hours that haven't been executed
        # TIMEZONE FIX: Use UTC_TIMESTAMP() for consistent timezone handling
        query = """
        SELECT id, symbol, action, confidence, reasoning, amount_usd, 
               created_at, source, signal_id, model_confidence, 
               risk_score, signal_strength, market_sentiment
        FROM trade_recommendations 
        WHERE created_at >= DATE_SUB(UTC_TIMESTAMP(), INTERVAL 4 HOUR)
        AND status = 'pending'
        ORDER BY created_at DESC, confidence DESC
        """
        
        cursor.execute(query)
        recommendations = cursor.fetchall()
        
        # Convert datetime objects to strings for JSON serialization
        for rec in recommendations:
            if rec['created_at']:
                rec['created_at'] = rec['created_at'].isoformat()
            # Convert Decimal fields to float for JSON serialization
            for key, value in rec.items():
                if isinstance(value, Decimal):
                    rec[key] = float(value)
            # No need to parse ml_features since it doesn't exist in this table
        
        cursor.close()
        conn.close()
        
        logger.info(f"📊 Served {len(recommendations)} recommendations to trade orchestrator")
        
        return {
            "recommendations": recommendations,
            "count": len(recommendations),
            "timestamp": datetime.now().isoformat(),
            "service": "signal-bridge-recommendations"
        }
        
    except Exception as e:
        logger.error(f"Error fetching recommendations: {e}")
        return {
            "recommendations": [],
            "count": 0,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "service": "signal-bridge-recommendations"
        }

def start_health_server():
    """Start the FastAPI health server in a background thread"""
    def run_server():
        try:
            uvicorn.run(app, host="0.0.0.0", port=8022, log_level="warning")
        except Exception as e:
            logger.error(f"Health server error: {e}")
    
    health_thread = threading.Thread(target=run_server, daemon=True)
    health_thread.start()
    logger.info("🏥 Health server started on port 8022")

class SimplePositionManager:
    """Simple position manager with basic functionality"""
    def __init__(self, trades_db_config, signals_db_config, trading_engine_url):
        self.trades_db_config = trades_db_config
        self.signals_db_config = signals_db_config
        self.trading_engine_url = trading_engine_url
        self.config = {
            'strong_signal_threshold': 0.85,
            'max_position_multiplier': 2.0,
            'enable_dynamic_rebalancing': True,
            'max_position_size_usd': 500.0
        }
    
    def get_configuration(self):
        return self.config
    
    def update_configuration(self, updates):
        self.config.update(updates)
        return True
    
    def get_current_portfolio(self):
        try:
            response = requests.get(f"{self.trading_engine_url}/portfolio", timeout=10)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return {'total_portfolio_value': 0, 'available_balance': 0, 'positions': []}
    
    def consolidate_small_positions(self, min_position_value_usd=10.0):
        """Identify small positions and create sell signals for consolidation"""
        try:
            portfolio = self.get_current_portfolio()
            positions = portfolio.get('positions', {})
            consolidation_signals = []
            
            logger.info(f"[CONSOLIDATION] Checking positions against threshold ${min_position_value_usd}")
            logger.info(f"[CONSOLIDATION] Positions type: {type(positions)}, count: {len(positions)}")
            
            # Handle both dict and list format for positions
            if isinstance(positions, list):
                logger.info(f"[CONSOLIDATION] Processing {len(positions)} positions in list format")
                # Convert list format to dict format for consistent processing
                for pos in positions:
                    if isinstance(pos, dict) and 'currency' in pos:
                        symbol = pos['currency']
                        value_usd = pos.get('value_usd', 0)
                        balance = pos.get('available_balance', 0)
                        
                        logger.debug(f"[CONSOLIDATION] Checking {symbol}: value=${value_usd:.6f}, balance={balance:.6f}")
                        
                        # Check if position is small enough to consolidate but large enough for Coinbase to process
                        min_trade_size = float(os.environ.get('MIN_TRADE_SIZE_USD', '5.00'))
                        min_coinbase_amount = float(os.environ.get('MIN_COINBASE_TRADE_AMOUNT', '1.00'))  # Minimum for Coinbase
                        
                        if (min_coinbase_amount <= value_usd < min_position_value_usd and 
                            balance > 0 and 
                            symbol not in ['USD', 'USDC', 'USDT']):
                            
                            # DATABASE-DRIVEN ASSET FILTERING: Check if asset is supported for trading
                            if not is_asset_supported(symbol):
                                logger.info(f"[CONSOLIDATION] Skipping {symbol}: not supported by Coinbase Advanced Trade")
                                continue
                            
                            logger.info(f"[CONSOLIDATION] Small position found: {symbol} = ${value_usd:.2f}")
                            
                            # Create sell signal for small position
                            sell_signal = {
                                'symbol': symbol,
                                'signal_type': 'sell',  # Match expected field name
                                'action': 'sell',
                                'amount_usd': value_usd,
                                'balance': balance,
                                'reasoning': f'Portfolio consolidation: position ${value_usd:.2f} < ${min_position_value_usd}',
                                'confidence': 0.9,  # High confidence for consolidation
                                'consolidation_value': value_usd,  # Expected field for value tracking
                                'consolidation': True
                            }
                            consolidation_signals.append(sell_signal)
                        elif 0 < value_usd < min_coinbase_amount:
                            # Log tiny positions that are being skipped
                            logger.debug(f"[CONSOLIDATION] Skipping {symbol}: ${value_usd:.2f} too small for Coinbase (min ${min_coinbase_amount:.2f})")
                            
            elif isinstance(positions, dict):
                logger.info(f"[CONSOLIDATION] Processing {len(positions)} positions in dict format")
                # Handle dict format positions
                for symbol, position_data in positions.items():
                    if isinstance(position_data, dict):
                        value_usd = position_data.get('value_usd', 0)
                        balance = position_data.get('balance', 0)
                        
                        logger.debug(f"[CONSOLIDATION] Checking {symbol}: value=${value_usd:.6f}, balance={balance:.6f}")
                        
                        # Check if position is small enough to consolidate but large enough for Coinbase to process
                        min_trade_size = float(os.environ.get('MIN_TRADE_SIZE_USD', '5.00'))
                        min_coinbase_amount = float(os.environ.get('MIN_COINBASE_TRADE_AMOUNT', '1.00'))  # Minimum for Coinbase
                        
                        if (min_coinbase_amount <= value_usd < min_position_value_usd and 
                            balance > 0 and 
                            symbol not in ['USD', 'USDC', 'USDT']):
                            
                            # DATABASE-DRIVEN ASSET FILTERING: Check if asset is supported for trading
                            if not is_asset_supported(symbol):
                                logger.info(f"[CONSOLIDATION] Skipping {symbol}: not supported by Coinbase Advanced Trade")
                                continue
                            
                            logger.info(f"[CONSOLIDATION] Small position found: {symbol} = ${value_usd:.2f}")
                            
                            # Create sell signal for small position
                            sell_signal = {
                                'symbol': symbol,
                                'signal_type': 'sell',  # Match expected field name
                                'action': 'sell',
                                'amount_usd': value_usd,
                                'balance': balance,
                                'reasoning': f'Portfolio consolidation: position ${value_usd:.2f} < ${min_position_value_usd}',
                                'confidence': 0.9,  # High confidence for consolidation
                                'consolidation_value': value_usd,  # Expected field for value tracking
                                'consolidation': True
                            }
                            consolidation_signals.append(sell_signal)
                        elif 0 < value_usd < min_coinbase_amount:
                            # Log tiny positions that are being skipped
                            logger.debug(f"[CONSOLIDATION] Skipping {symbol}: ${value_usd:.2f} too small for Coinbase (min ${min_coinbase_amount:.2f})")
            else:
                logger.error(f"[CONSOLIDATION] Unexpected positions format: {type(positions)}")
                return []
            
            if consolidation_signals:
                logger.info(f"[CONSOLIDATION] Generated {len(consolidation_signals)} consolidation sell signals")
                
                # Log total value being consolidated
                total_value = sum(signal['amount_usd'] for signal in consolidation_signals)
                logger.info(f"[CONSOLIDATION] Total value to consolidate: ${total_value:.2f}")
            else:
                logger.info(f"[CONSOLIDATION] No positions found under ${min_position_value_usd}")
                
            return consolidation_signals
            
        except Exception as e:
            logger.error(f"[CONSOLIDATION] Error in consolidate_small_positions: {e}")
            return []
    
    def calculate_dynamic_position_size(self, confidence: float, base_position_size: float) -> float:
        """Calculate dynamic position size based on confidence and configuration"""
        try:
            # Get configuration values
            strong_signal_threshold = self.config.get('strong_signal_threshold', 0.85)
            max_position_multiplier = self.config.get('max_position_multiplier', 2.0)
            
            # For high confidence signals, increase position size
            if confidence >= strong_signal_threshold:
                # Scale multiplier based on how much higher than threshold
                confidence_excess = confidence - strong_signal_threshold
                max_excess = 1.0 - strong_signal_threshold  # Maximum possible excess
                if max_excess > 0:
                    multiplier = 1.0 + (confidence_excess / max_excess) * (max_position_multiplier - 1.0)
                else:
                    multiplier = max_position_multiplier
                
                # Apply multiplier but cap at max
                dynamic_size = min(base_position_size * multiplier, base_position_size * max_position_multiplier)
                return dynamic_size
            else:
                # For lower confidence, keep base size
                return base_position_size
                
        except Exception as e:
            logger.error(f"Error calculating dynamic position size: {e}")
            return base_position_size
    
    def generate_rebalancing_signals(self, target_signal, additional_funding_needed):
        """Generate rebalancing signals to free up cash for a new position"""
        try:
            logger.info(f"[REBALANCE] Generating rebalancing signals for {target_signal['symbol']}, need ${additional_funding_needed:.2f}")
            
            # Get current portfolio positions
            portfolio = self.get_current_portfolio()
            positions = portfolio.get('positions', [])
            
            if not positions:
                logger.info("[REBALANCE] No positions available for rebalancing")
                return []
            
            rebalancing_signals = []
            funding_raised = 0.0
            
            # Sort positions by value (smallest first for partial liquidation strategy)
            if isinstance(positions, list):
                sorted_positions = sorted(
                    [pos for pos in positions if pos.get('currency') not in ['USD', 'USDC', 'USDT']],
                    key=lambda x: x.get('value_usd', 0)
                )
            else:
                sorted_positions = sorted(
                    [(symbol, data) for symbol, data in positions.items() if symbol not in ['USD', 'USDC', 'USDT']],
                    key=lambda x: x[1].get('value_usd', 0) if isinstance(x[1], dict) else 0
                )
            
            for position in sorted_positions:
                if funding_raised >= additional_funding_needed:
                    break
                
                # Handle both list and dict position formats
                if isinstance(position, dict):
                    symbol = position.get('currency', '')
                    value_usd = position.get('value_usd', 0)
                    balance = position.get('available_balance', 0)
                elif isinstance(position, tuple) and len(position) == 2:
                    symbol, position_data = position
                    value_usd = position_data.get('value_usd', 0) if isinstance(position_data, dict) else 0
                    balance = position_data.get('balance', 0) if isinstance(position_data, dict) else 0
                else:
                    continue
                
                if value_usd <= 0 or balance <= 0:
                    continue
                
                # Skip the target symbol we're trying to buy
                if symbol == target_signal.get('symbol'):
                    continue
                
                # Calculate how much to sell (partial or full liquidation)
                remaining_needed = additional_funding_needed - funding_raised
                
                if value_usd <= remaining_needed:
                    # Full liquidation
                    sell_amount_usd = value_usd
                    sell_percentage = 100.0
                else:
                    # Partial liquidation (sell just what we need + small buffer)
                    sell_amount_usd = remaining_needed * 1.1  # 10% buffer
                    sell_percentage = min(95.0, (sell_amount_usd / value_usd) * 100)  # Cap at 95%
                
                # SKIP REBALANCING TRADES BELOW MINIMUM THRESHOLD - ABSOLUTE CHECK
                ABSOLUTE_MIN_REBALANCE_SIZE = 5.00  # Hardcoded minimum to prevent any micro-trades
                min_rebalance_size = max(
                    float(os.environ.get('MIN_TRADE_SIZE_USD', '5.00')),
                    ABSOLUTE_MIN_REBALANCE_SIZE
                )
                if sell_amount_usd < min_rebalance_size:
                    logger.debug(f"[REBALANCE] Skipping {symbol} - trade size ${sell_amount_usd:.2f} below minimum ${min_rebalance_size:.2f}")
                    continue
                
                # Create rebalancing sell signal
                rebal_signal = {
                    'symbol': symbol,
                    'signal_type': 'sell',
                    'confidence': 0.85,  # High confidence for rebalancing
                    'reasoning': f'Portfolio rebalancing to fund {target_signal["symbol"]} position (need ${remaining_needed:.2f})',
                    'value_usd': sell_amount_usd,
                    'percentage': sell_percentage,
                    'is_rebalancing': True,
                    'rebalancing_for': target_signal.get('symbol'),
                    'created_at': datetime.now()
                }
                
                rebalancing_signals.append(rebal_signal)
                funding_raised += sell_amount_usd
                
                logger.info(f"[REBALANCE] Added {symbol} sell signal: ${sell_amount_usd:.2f} ({sell_percentage:.1f}%)")
            
            logger.info(f"[REBALANCE] Generated {len(rebalancing_signals)} signals, projected funding: ${funding_raised:.2f}")
            return rebalancing_signals
            
        except Exception as e:
            logger.error(f"[REBALANCE] Error generating rebalancing signals: {e}")
            return []

class AutomatedSignalBridge:
    def __init__(self):
        # Database host must be provided via environment - no IP fallbacks
        database_host = os.environ.get('DATABASE_HOST')
        if not database_host:
            raise Exception("DATABASE_HOST environment variable required - no IP fallbacks allowed")
            
        self.signals_db_config = {
            'host': database_host,
            'user': os.environ.get('DATABASE_USER', 'news_collector'),
            'password': os.environ.get('DATABASE_PASSWORD', '99Rules!'),
            'database': os.environ.get('SIGNALS_DATABASE', 'crypto_prices'),
            'port': int(os.environ.get('DATABASE_PORT', 3306))
        }
        
        self.trades_db_config = {
            'host': database_host,
            'user': os.environ.get('DATABASE_USER', 'news_collector'),
            'password': os.environ.get('DATABASE_PASSWORD', '99Rules!'),
            'database': os.environ.get('TRADES_DATABASE', 'crypto_transactions'),
            'port': int(os.environ.get('DATABASE_PORT', 3306))
        }
        
        # Trading Engine Configuration - no localhost fallbacks
        trade_execution_url = os.environ.get("TRADE_EXECUTION_URL")
        if not trade_execution_url:
            raise Exception("TRADE_EXECUTION_URL environment variable required - no localhost fallbacks allowed")
        self.trade_execution_url = trade_execution_url
        
        # Risk Management Service
        self.risk_management_url = os.environ.get("RISK_MANAGEMENT_URL", "http://localhost:8027")
        self.bridge_interval = int(os.environ.get('BRIDGE_INTERVAL_SECONDS', 30))
        
        # Initialize Dynamic Position Manager
        try:
            # Create a simple position manager proxy with basic config
            self.position_manager = SimplePositionManager(
                trades_db_config=self.trades_db_config,
                signals_db_config=self.signals_db_config,
                trading_engine_url=self.trade_execution_url
            )
        except Exception as e:
            logger.error(f"❌ CRITICAL: Could not initialize position manager: {e}")
            raise Exception("Position manager required - no fallbacks allowed")
        
        # Risk Management Configuration - Initialize from database
        self.config_cache = {}
        self.config_cache_ttl = 60  # Cache for 1 minute
        self.last_config_fetch = 0
        self.load_trade_configuration()
        
        self.processed_signals = set()
        
        # Add caching for portfolio status to reduce API calls
        self.portfolio_cache = {}
        self.portfolio_cache_ttl = 300  # Cache for 5 minutes
        self.last_portfolio_fetch = 0
        
        # Portfolio consolidation settings
        self.consolidation_interval = int(os.environ.get('CONSOLIDATION_INTERVAL_HOURS', 12)) * 3600  # Default 12 hours
        self.last_consolidation = 0
        self.min_consolidation_value = float(os.environ.get('MIN_POSITION_VALUE_USD', 15.0))  # Lowered from 25.0 to 15.0 to consolidate more small positions
        
        # Trade cooldown settings - EMERGENCY ANTI-CHURNING CONFIGURATION
        self.trade_cooldown_minutes = int(os.environ.get('TRADE_COOLDOWN_MINUTES', 240))  # 4 hours EMERGENCY - was 60 minutes
        self.contradictory_trade_cooldown_minutes = int(os.environ.get('CONTRADICTORY_TRADE_COOLDOWN_MINUTES', 360))  # 6 hours EMERGENCY - was 30 minutes
        
        # MINIMUM TRADE SIZE - CRITICAL FOR PREVENTING MICRO-TRADES
        self.ABSOLUTE_MIN_TRADE_SIZE = 5.00  # Hardcoded minimum to prevent any micro-trades
        self.min_trade_size = max(
            float(os.environ.get('MIN_TRADE_SIZE_USD', '5.00')),
            self.ABSOLUTE_MIN_TRADE_SIZE
        )
        logger.info(f"Minimum trade size enforced: ${self.min_trade_size:.2f}")
        
        # Update health status with consolidation interval
        health_status["consolidation_interval_hours"] = self.consolidation_interval / 3600
        
        logger.info(f"Initialized bridge with {self.bridge_interval}s monitoring interval")
        logger.info("Dynamic position management enabled")
        logger.info(f"Portfolio consolidation enabled: every {self.consolidation_interval/3600:.1f} hours, min position ${self.min_consolidation_value:.2f}")
    
    def load_trade_configuration(self):
        """Load trade configuration from database with caching"""
        current_time = time.time()
        
        # Use cache if valid
        if current_time - self.last_config_fetch < self.config_cache_ttl and self.config_cache:
            return self.config_cache
        
        try:
            conn = mysql.connector.connect(**self.trades_db_config)
            cursor = conn.cursor(dictionary=True)
            
            cursor.execute("SELECT parameter_name, parameter_value FROM trade_configuration")
            configs = cursor.fetchall()
            
            # Update cache
            self.config_cache = {}
            for config in configs:
                self.config_cache[config['parameter_name']] = float(config['parameter_value'])
            
            self.last_config_fetch = current_time
            
            cursor.close()
            conn.close()
            
            # Set instance variables for backward compatibility
            self.max_position_size_usd = self.config_cache.get('max_position_size_usd', 500.0)
            self.max_daily_trades = int(self.config_cache.get('max_daily_trades', 300.0))
            self.balance_utilization_percent = self.config_cache.get('balance_utilization_percent', 95.0)
            self.max_daily_loss_usd = self.config_cache.get('max_daily_loss_usd', 500.0)
            self.min_confidence_threshold = self.config_cache.get('min_confidence_threshold', 0.75)  # ANTI-CHURNING: 75% threshold prevents churning while allowing legitimate trades
            
            logger.info(f"Loaded trade configuration: max_position=${self.max_position_size_usd}, max_trades={self.max_daily_trades}")
            return self.config_cache
            
        except Exception as e:
            logger.warning(f"Error loading trade configuration, using defaults: {e}")
            # Fallback to environment variables and defaults
            self.max_position_size_usd = float(os.environ.get('MAX_POSITION_SIZE_USD', 500.0))
            self.max_daily_trades = int(os.environ.get('MAX_DAILY_TRADES', 300))
            self.balance_utilization_percent = float(os.environ.get('BALANCE_UTILIZATION_PERCENT', 95.0))
            self.max_daily_loss_usd = 500.0
            self.min_confidence_threshold = float(os.environ.get('CONFIDENCE_THRESHOLD', 0.75))  # ANTI-CHURNING: Read from environment with 75% default
            return {}
    
    def get_config_value(self, parameter_name: str, default_value: float = 0.0) -> float:
        """Get a configuration value with automatic cache refresh"""
        self.load_trade_configuration()  # This will use cache if valid
        return self.config_cache.get(parameter_name, default_value)
    
    def get_unprocessed_enhanced_signals(self) -> List[Dict]:
        """Get unprocessed high-confidence signals for processing - simplified approach to catch all valid signals"""
        try:
            conn = mysql.connector.connect(**self.trades_db_config)  # Use trades_db_config for trading_signals table
            cursor = conn.cursor(dictionary=True)
            
            # OPTIMIZED: Get high-confidence signals only (75% minimum to reduce noise)
            # CRYPTO-OPTIMIZED: 4-hour window instead of 24h - crypto markets move fast!
            max_age_time = datetime.now() - timedelta(hours=4)  # REDUCED from 24h to 4h for fresher signals
            min_confidence = self.get_config_value('min_confidence_threshold', 0.75)  # INCREASED from 0.10 to 0.75 (75%)
            
            # SIMPLIFIED: Get unprocessed signals without complex joins that filter out too much
            query = """
            SELECT id, symbol, signal_type, action, confidence, created_at, price,
                   COALESCE(processed, 0) as processed
            FROM trading_signals
            WHERE confidence >= %s 
            AND created_at >= %s
            AND signal_type IN ('BUY', 'SELL', 'HOLD', 'STRONG_BUY', 'STRONG_SELL', 'ML_SIGNAL')
            AND (processed IS NULL OR processed = 0)
            ORDER BY created_at DESC, confidence DESC
            LIMIT 100
            """
            
            cursor.execute(query, (min_confidence, max_age_time))
            signals = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            # Filter out already processed signals by checking if we already created recommendations
            unprocessed = []
            processed_symbols = set()
            
            for signal in signals:
                # Only process one signal per symbol to avoid spam (prioritize most recent/highest confidence)
                if signal['symbol'] not in processed_symbols:
                    if not self.signal_already_processed_recently(signal['symbol'], signal['signal_type'], signal['created_at']):
                        unprocessed.append(signal)
                        processed_symbols.add(signal['symbol'])
            
            logger.info(f"[BRIDGE] Found {len(unprocessed)} unprocessed signals from total {len(signals)} recent signals")
            return unprocessed
            
        except Exception as e:
            logger.error(f"Error getting enhanced signals: {e}")
            return []
    
    def signal_already_processed_recently(self, symbol: str, signal_type: str, signal_created_at: datetime) -> bool:
        """Check if we already created a recommendation for this symbol/action in the recent timeframe"""
        try:
            conn = mysql.connector.connect(**self.trades_db_config)
            cursor = conn.cursor()
            
            # Check if recommendation exists for this symbol/action in last 15 minutes
            recent_time = signal_created_at - timedelta(minutes=15)
            
            query = """
            SELECT COUNT(*) FROM trade_recommendations 
            WHERE symbol = %s AND action = %s AND created_at >= %s
            """
            
            cursor.execute(query, (symbol, signal_type, recent_time))
            count = cursor.fetchone()[0]
            
            cursor.close()
            conn.close()
            
            return count > 0
            
        except Exception as e:
            logger.error(f"Error checking if signal already processed: {e}")
            return False
    
    def signal_already_exists_as_recommendation(self, signal_id: int, symbol: str, signal_type: str) -> bool:
        """Check if this signal has already been converted to a recommendation"""
        try:
            conn = mysql.connector.connect(**self.trades_db_config)
            cursor = conn.cursor()
            
            # Check if recommendation exists for this signal in recent time
            recent_time = datetime.now() - timedelta(hours=1)
            
            query = """
            SELECT COUNT(*) FROM trade_recommendations 
            WHERE symbol = %s AND action = %s AND created_at >= %s
            """
            
            cursor.execute(query, (symbol, signal_type, recent_time))
            count = cursor.fetchone()[0]
            
            cursor.close()
            conn.close()
            
            return count > 0
            
        except Exception as e:
            logger.error(f"Error checking existing recommendations: {e}")
            return False
    
    def calculate_trade_amount(self, confidence: float, position_size_percent: float, signal_type: str = 'BUY', symbol: str = None) -> float:
        """Calculate appropriate trade amount based on actual portfolio size and signal confidence"""
        try:
            # Get actual portfolio value from trading engine (request fresh data for trading)
            portfolio_response = requests.get(f'{self.trade_execution_url}/portfolio?fresh=true', timeout=15)
            
            if portfolio_response.status_code == 200:
                portfolio_data = portfolio_response.json()
                
                # Calculate actual portfolio value (USD + crypto holdings)
                usd_balance = portfolio_data.get('usd_balance', 0)
                positions = portfolio_data.get('positions', [])
                crypto_value = sum(pos.get('value', 0) for pos in positions if pos.get('value', 0) > 0)
                actual_portfolio_value = usd_balance + crypto_value
                
                # For BUY orders, use available USD balance for realistic sizing
                # For SELL orders, check if we have the position to sell
                if signal_type in ['BUY', 'STRONG_BUY']:  # BUY order logic
                    # DYNAMIC SIZING FIX: Base trade amount on available USD, not total portfolio
                    fee_rate = 0.006  # Coinbase Advanced Trade fee rate (0.6%)
                    available_for_trade = usd_balance * 0.95  # Leave 5% buffer for fees and safety
                    
                    # Calculate ideal trade amount from portfolio percentage
                    ideal_amount = actual_portfolio_value * (position_size_percent / 100.0)
                    
                    # Use smaller of: ideal amount or available USD (minus fees)
                    max_trade_with_fees = available_for_trade / (1 + fee_rate)  # Account for fees upfront
                    trade_amount = min(ideal_amount, max_trade_with_fees)
                    
                    # Minimum trade size check
                    min_trade_size = 25.0  # Coinbase minimum
                    if trade_amount < min_trade_size:
                        if available_for_trade < min_trade_size:
                            logger.warning(f"[BUY_SIZING] Insufficient USD (${usd_balance:.2f}) for minimum trade (${min_trade_size})")
                            return 0.0  # Skip this trade
                        else:
                            trade_amount = min_trade_size  # Use minimum
                    
                    logger.info(f"[BUY_SIZING] USD: ${usd_balance:.2f}, Ideal: ${ideal_amount:.2f}, Final: ${trade_amount:.2f}")
                    
                    # Ensure we don't exceed available USD balance
                    if trade_amount > max_trade_with_fees:
                        logger.warning(f"[BUY_SIZING] Reduced trade from ${trade_amount:.2f} to ${max_trade_with_fees:.2f} (available USD limit)")
                else:  # SELL order logic 
                    # For SELL orders, find the actual position value for this symbol
                    position_value = 0
                    if symbol:
                        for pos in positions:
                            if pos.get('currency', '').upper() == symbol.upper():
                                position_value = pos.get('value_usd', pos.get('value', 0))
                                break
                    
                    # SMALL POSITION FIX: Skip positions below Coinbase minimums
                    min_sell_threshold = 5.0  # Coinbase minimum liquidation amount
                    
                    if position_value > 0:
                        if position_value < min_sell_threshold:
                            logger.warning(f"[SELL_SKIP] {symbol} position ${position_value:.2f} below minimum ${min_sell_threshold:.2f} - skipping")
                            return 0.0  # Skip this SELL recommendation
                        
                        # Use percentage of the actual position value
                        trade_amount = position_value * (position_size_percent / 100.0)
                        
                        # Ensure the sell amount meets minimum requirements
                        min_trade_size = 25.0  # Coinbase minimum trade size
                        if trade_amount < min_trade_size:
                            if position_value >= min_trade_size:
                                trade_amount = min_trade_size  # Sell minimum amount
                                logger.info(f"[SELL_AMOUNT] {symbol} adjusted to minimum: ${trade_amount:.2f}")
                            else:
                                logger.warning(f"[SELL_SKIP] {symbol} position ${position_value:.2f} too small for minimum trade ${min_trade_size:.2f}")
                                return 0.0  # Skip this trade
                        
                        logger.info(f"[SELL_AMOUNT] {symbol} position: ${position_value:.2f}, selling {position_size_percent:.1f}% = ${trade_amount:.2f}")
                    else:
                        # No position to sell - return 0
                        logger.warning(f"[SELL_AMOUNT] No {symbol} position found to sell")
                        return 0.0
                
                # Apply confidence-based adjustments
                if confidence >= 0.8:
                    # High confidence - allow full calculated amount
                    pass
                elif confidence >= 0.6:
                    # Medium confidence - reduce by 20%
                    trade_amount *= 0.8
                else:
                    # Lower confidence - reduce by 40%
                    trade_amount *= 0.6
                
                # REMOVED PROBLEMATIC MIN TRADE OVERRIDE - it was forcing $50 trades even with low USD balance
                # The dynamic sizing logic above already handles minimums appropriately
                
                # Ensure maximum position size limits (portfolio protection)
                max_trade = min(500.0, actual_portfolio_value * 0.15)  # Max 15% of portfolio
                trade_amount = min(trade_amount, max_trade)
                
                # Final safety check - don't exceed reasonable limits
                if signal_type in ['BUY', 'STRONG_BUY']:
                    # For BUY orders, respect the dynamic USD balance limits we calculated above
                    pass  # Already handled in BUY logic
                else:
                    # For SELL orders, ensure we don't exceed position value
                    if symbol and position_value > 0:
                        trade_amount = min(trade_amount, position_value)
                
                logger.info(f"[AMOUNT] Portfolio: ${actual_portfolio_value:.2f}, Trade: ${trade_amount:.2f} ({position_size_percent:.1f}%)")
                return round(trade_amount, 2)
                
            else:
                logger.error(f"[AMOUNT] ❌ CRITICAL: Cannot get portfolio data - aborting")
                raise Exception("Portfolio data required - no fallbacks allowed")
                
        except Exception as e:
            logger.error(f"[AMOUNT] ❌ CRITICAL: Error calculating trade amount: {e}")
            raise Exception("Trade amount calculation failed - no fallbacks allowed")
    
    def check_recent_trade_activity(self, symbol: str, action: str = None, hours_lookback: int = 2) -> Dict:
        """ENHANCED FAIL-SAFE anti-churn check - REJECTS trades by default if database unavailable
        This prevents excessive trading during connectivity issues
        """
        try:
            conn = mysql.connector.connect(**self.trades_db_config)
            cursor = conn.cursor(dictionary=True)
            
            lookback_time = datetime.now() - timedelta(hours=hours_lookback)
            
            cursor.execute("""
                SELECT action, size_usd, timestamp, status
                FROM trades 
                WHERE symbol = %s 
                AND timestamp >= %s
                AND status IN ('EXECUTED', 'PENDING')
                ORDER BY timestamp DESC
                LIMIT 10
            """, (symbol, lookback_time))
            
            recent_trades = cursor.fetchall()
            cursor.close()
            conn.close()
            
            # If no specific action provided, return all recent trades for same-symbol analysis
            if action is None:
                total_fee_impact = sum(float(trade['size_usd']) * 0.006 for trade in recent_trades)
                return {
                    'is_churning': len(recent_trades) >= 2,  # STRICTER: 2+ trades in period is churning
                    'recent_trades_count': len(recent_trades),
                    'recent_trades': recent_trades,
                    'estimated_fee_impact': total_fee_impact,
                    'database_available': True
                }
            
            # Look for recent trades in the opposite direction
            opposite_action = 'SELL' if action == 'BUY' else 'BUY'
            
            # Calculate anti-churning metrics
            opposite_trades = [t for t in recent_trades if t['action'] == opposite_action]
            same_direction_trades = [t for t in recent_trades if t['action'] == action]
            
            # Fee impact calculation
            total_fee_impact = sum(float(trade['size_usd']) * 0.006 for trade in recent_trades)
            
            # ENHANCED CHURNING DETECTION - More strict criteria
            is_churning = (
                len(opposite_trades) > 0 and 
                len(recent_trades) >= 2 and  # STRICTER: 2+ trades triggers churn protection
                (total_fee_impact > 1.0 or len(recent_trades) >= 3)  # Lower fee threshold OR 3+ trades
            )
            
            if is_churning:
                last_opposite = opposite_trades[0]
                minutes_since_opposite = (datetime.now() - last_opposite['timestamp']).total_seconds() / 60
                
                logger.warning(f"[ANTI_CHURN] {symbol} {action}: {len(opposite_trades)} {opposite_action} trades in last {hours_lookback}h, est. fees: ${total_fee_impact:.2f}")
                logger.warning(f"[ANTI_CHURN] Last {opposite_action}: {minutes_since_opposite:.1f} minutes ago")
            
            return {
                'is_churning': is_churning,
                'recent_trades_count': len(recent_trades),
                'opposite_trades_count': len(opposite_trades),
                'estimated_fee_impact': total_fee_impact,
                'minutes_since_opposite': (datetime.now() - opposite_trades[0]['timestamp']).total_seconds() / 60 if opposite_trades else None,
                'database_available': True
            }
            
        except Exception as e:
            logger.error(f"[ANTI_CHURN] ERROR: Database unavailable for {symbol}: {e}")
            # FAIL-SAFE: When database is unavailable, DEFAULT TO BLOCKING ALL TRADES
            return {
                'is_churning': True,  # FAIL-SAFE: Block trades when DB unavailable
                'recent_trades_count': 999,  # High count to trigger protection
                'opposite_trades_count': 999,
                'estimated_fee_impact': 999.0,
                'minutes_since_opposite': 0,  # Recent activity assumed
                'database_available': False,
                'error': str(e)
            }

    def convert_signal_to_recommendation(self, signal: Dict) -> Optional[Dict]:
        """Convert a trading signal to a trade recommendation with dynamic position management"""
        try:
            # SYMBOL STANDARDIZATION: Ensure consistent symbol format
            # standardized_symbol = normalize_symbol(signal['symbol'])  # Symbol utils not available
            standardized_symbol = signal['symbol']  # Use symbol as-is
            if standardized_symbol != signal['symbol']:
                logger.debug(f"[SYMBOL] Standardized signal symbol '{signal['symbol']}' -> '{standardized_symbol}'")
            
            # DATABASE-DRIVEN ASSET FILTERING: Check if asset is supported for trading
            symbol = standardized_symbol  # Use standardized symbol
            if not is_asset_supported(symbol):
                logger.info(f"[ASSET_FILTER] Rejecting signal for {symbol}: not supported by Coinbase Advanced Trade")
                return None
            
            # ANTI-CHURNING CHECK: Prevent rapid buy/sell cycles that waste fees
            signal_type = signal['signal_type']
            if signal_type in ['BUY', 'SELL', 'STRONG_BUY', 'STRONG_SELL']:
                action = 'BUY' if signal_type in ['BUY', 'STRONG_BUY'] else 'SELL'
                churn_check = self.check_recent_trade_activity(symbol, action, hours_lookback=2)
                
                # CRITICAL: Check database availability first
                if not churn_check.get('database_available', False):
                    logger.error(f"[SAFETY] ❌ Blocking {symbol} {action}: Database unavailable - FAIL-SAFE ENGAGED")
                    return None
                
                # ENHANCED ANTI-CHURN: Block same-symbol trades within 120 minutes regardless of direction
                same_symbol_check = self.check_recent_trade_activity(symbol, None, hours_lookback=2)  # Check any action in 2 hours
                if same_symbol_check['recent_trades_count'] > 0:
                    minutes_since_last = min([
                        (datetime.now() - trade['timestamp']).total_seconds() / 60 
                        for trade in same_symbol_check.get('recent_trades', [])
                    ])
                    if minutes_since_last < 120:  # 2 hour cooldown for same symbol (increased from 1 hour)
                        logger.warning(f"[ANTI_CHURN] ❌ Rejecting {symbol} {action} signal: same symbol traded {minutes_since_last:.1f} minutes ago")
                        logger.warning(f"[ANTI_CHURN] Same-symbol cooldown: wait {120 - minutes_since_last:.1f} more minutes")
                        return None
                
                if churn_check['is_churning']:
                    logger.warning(f"[ANTI_CHURN] ❌ Rejecting {symbol} {action} signal: would cause churning")
                    logger.warning(f"[ANTI_CHURN] Recent activity: {churn_check['recent_trades_count']} trades, ${churn_check['estimated_fee_impact']:.2f} fees")
                    return None
                elif churn_check['opposite_trades_count'] > 0:
                    logger.info(f"[ANTI_CHURN] ⚠️ Proceeding with {symbol} {action} despite {churn_check['opposite_trades_count']} recent opposite trades")
            
            # ENHANCED CONFIDENCE FILTERING: Only process high-confidence actionable signals
            confidence = float(signal['confidence'])
            
            # Extract entry price from signal
            entry_price = float(signal.get('price', 0))  # Initialize entry_price from signal
            
            # Strict confidence thresholds to prevent excessive trading
            min_confidence_thresholds = {
                'BUY': 0.80,      # 80% confidence required for BUY
                'SELL': 0.80,     # 80% confidence required for SELL
                'STRONG_BUY': 0.85,  # 85% confidence for strong signals
                'STRONG_SELL': 0.85,
                'HOLD': 0.60      # HOLD signals can be lower confidence
            }
            
            min_required_confidence = min_confidence_thresholds.get(signal_type, 0.80)
            
            if confidence < min_required_confidence:
                logger.info(f"[CONFIDENCE_FILTER] ❌ Rejecting {symbol} {signal_type}: confidence {confidence:.3f} < required {min_required_confidence:.3f}")
                return None
            
            # NO FALLBACKS: Price must be valid from signal source
            if entry_price == 0 or entry_price is None:
                raise Exception(f"Signal price is zero/null for {symbol} {signal_type} - no price fallbacks allowed")
            
            # Check if this is a strong signal that might trigger rebalancing
            is_strong_signal = (
                signal_type in ['BUY', 'STRONG_BUY'] and 
                confidence >= self.position_manager.config['strong_signal_threshold']
            )
            
            # Standard risk management based on confidence - optimized for $2,265 portfolio
            if confidence >= 0.8:
                stop_loss_pct = 0.02  # 2% stop loss for high confidence
                take_profit_pct = 0.06  # 6% take profit
                # Use database configuration for sustainable sizing within USD balance
                base_position_size = self.get_config_value('high_confidence_position_percent', 4.0)  # 4% of portfolio = ~$90
            elif confidence >= 0.7:
                stop_loss_pct = 0.03  # 3% stop loss
                take_profit_pct = 0.05  # 5% take profit  
                base_position_size = self.get_config_value('medium_confidence_position_percent', 3.0)  # 3% of portfolio = ~$68
            else:
                stop_loss_pct = 0.04  # 4% stop loss for medium confidence
                take_profit_pct = 0.04  # 4% take profit
                base_position_size = self.get_config_value('low_confidence_position_percent', 2.0)  # 2% of portfolio = ~$45
            
            # Dynamic position sizing for strong signals
            if is_strong_signal:
                dynamic_position_size = self.position_manager.calculate_dynamic_position_size(
                    confidence, base_position_size
                )
                logger.info(f"🔥 Strong {signal_type} signal for {symbol} (confidence: {confidence:.3f})")
                logger.info(f"📈 Position size adjusted: {base_position_size:.1f}% → {dynamic_position_size:.1f}%")
                base_position_size = dynamic_position_size
            
            if signal['signal_type'] in ['BUY', 'STRONG_BUY']:
                stop_loss = entry_price * (1 - stop_loss_pct)
                take_profit = entry_price * (1 + take_profit_pct)
            elif signal['signal_type'] in ['SELL', 'STRONG_SELL']:
                stop_loss = entry_price * (1 + stop_loss_pct)
                take_profit = entry_price * (1 - take_profit_pct)
            else:  # HOLD
                # For HOLD signals, set minimal stop/take profit ranges
                stop_loss = entry_price * (1 - stop_loss_pct * 0.5)  # Tighter stops
                take_profit = entry_price * (1 + take_profit_pct * 0.5)  # Smaller targets
            
            # Calculate actual USD amount for the trade
            # DYNAMIC TRADE SIZING - Use calculate_trade_amount which handles USD balance properly
            calculated_amount = self.calculate_trade_amount(confidence, base_position_size, signal_type, symbol)
            
            # ISSUE B FIX: Don't override dynamic sizing with arbitrary minimums
            # The calculate_trade_amount function already handles minimums based on USD balance
            if calculated_amount <= 0:
                logger.info(f"[DYNAMIC_SIZING] ❌ Rejecting {symbol} {signal_type}: calculated amount ${calculated_amount:.2f} (insufficient funds or position)")
                return None
            
            amount_usd = calculated_amount  # Use the carefully calculated amount
            
            logger.info(f"[DYNAMIC_SIZING] ✅ {symbol} {signal_type}: ${amount_usd:.2f} (based on available funds)")
            
            recommendation = {
                'signal_id': signal.get('id'),  # Add signal ID for tracking
                'symbol': symbol,
                'action': signal.get('action', signal_type),  # Use signal action field, fallback to signal_type for compatibility
                'confidence': confidence,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'position_size_percent': base_position_size,
                'amount_usd': amount_usd,
                'reasoning': f"Enhanced XGBoost signal (confidence: {confidence:.3f})",
                'is_mock': False,
                'is_strong_signal': is_strong_signal,
                'requires_rebalancing': is_strong_signal
            }
            
            return recommendation
            
        except Exception as e:
            logger.error(f"Error converting signal to recommendation: {e}")
            return None
    
    def get_risk_adjusted_position_size(self, symbol: str, base_size: float, current_positions: Dict) -> Optional[float]:
        """Get risk-adjusted position size from advanced risk management service"""
        try:
            # Prepare request data
            request_data = {
                'symbol': symbol,
                'base_size': base_size,
                'current_positions': current_positions
            }
            
            response = requests.post(
                f"{self.risk_management_url}/optimize_position_size",
                json=request_data,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                final_size = result.get('final_size', base_size)
                reasoning = result.get('reasoning', {})
                
                logger.info(f"[RISK] {symbol} size adjusted: ${base_size:.2f} → ${final_size:.2f}")
                logger.debug(f"[RISK] Factors - Volatility: {reasoning.get('volatility_factor', 1.0):.3f}, "
                           f"Correlation: {reasoning.get('correlation_factor', 1.0):.3f}, "
                           f"Heat: {reasoning.get('heat_factor', 1.0):.3f}")
                
                return final_size
            else:
                logger.warning(f"Risk management service unavailable: {response.status_code}")
                return base_size
                
        except Exception as e:
            logger.warning(f"Error getting risk-adjusted position size: {e}")
            return base_size
    
    def check_for_recent_recommendations(self, symbol: str, action: str, time_window_minutes: int = 5) -> Dict:
        """ROBUST DEDUPLICATION: Check for recent recommendations with shorter window for rapid duplicates"""
        try:
            conn = mysql.connector.connect(**self.trades_db_config)
            cursor = conn.cursor()
            
            # STRENGTHENED: Use shorter 5-minute window for rapid duplicate detection
            # and more granular amount checking
            query = """
                SELECT id, amount_usd, created_at, execution_status
                FROM trade_recommendations
                WHERE symbol = %s AND action = %s
                AND created_at >= DATE_SUB(NOW(), INTERVAL %s MINUTE)
                AND execution_status IN ('PENDING', 'EXECUTED')
                ORDER BY created_at DESC
            """
            
            cursor.execute(query, (symbol, action, time_window_minutes))
            recent_recs = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            # ENHANCED ANALYSIS: More granular duplicate detection
            has_recent = len(recent_recs) > 0
            total_amount = sum(rec[1] for rec in recent_recs if rec[1])
            
            # AGGRESSIVE DEDUPLICATION: Block if ANY recent recommendation exists
            should_consolidate = len(recent_recs) > 0
            
            return {
                'has_recent': has_recent,
                'count': len(recent_recs),
                'total_amount': total_amount,
                'should_consolidate': should_consolidate,
                'recent_recommendations': recent_recs
            }
            
        except Exception as e:
            logger.error(f"Error checking recent recommendations: {e}")
            return {'has_recent': False, 'should_consolidate': False}

    def save_recommendation(self, recommendation: Dict) -> Optional[int]:
        """ROBUST SAVE: Trade recommendation with enhanced deduplication and database locking"""
        try:
            symbol = recommendation['symbol']
            action = recommendation['action']
            amount_usd = recommendation.get('amount_usd', 0)
            
            # DATABASE TRANSACTION with LOCKING to prevent race conditions
            conn = mysql.connector.connect(**self.trades_db_config)
            conn.start_transaction()
            cursor = conn.cursor()
            
            try:
                # STEP 1: LOCK and check for recent duplicates within 5 minutes
                lock_query = """
                    SELECT id, amount_usd, created_at 
                    FROM trade_recommendations 
                    WHERE symbol = %s AND action = %s
                    AND created_at >= DATE_SUB(NOW(), INTERVAL 5 MINUTE)
                    AND execution_status IN ('PENDING', 'EXECUTED')
                    FOR UPDATE
                """
                
                cursor.execute(lock_query, (symbol, action))
                recent_recs = cursor.fetchall()
                
                # AGGRESSIVE DEDUPLICATION: Block if any recent exists
                if len(recent_recs) > 0:
                    most_recent = recent_recs[0]
                    rec_id, rec_amount, rec_time = most_recent
                    
                    logger.warning(f"[ROBUST_DEDUP] ❌ Blocking duplicate {symbol} {action} recommendation")
                    logger.warning(f"[ROBUST_DEDUP] Recent: ID {rec_id} ${rec_amount:.2f} at {rec_time}")
                    logger.warning(f"[ROBUST_DEDUP] Would create: ${amount_usd:.2f} (BLOCKED)")
                    
                    conn.rollback()
                    cursor.close()
                    conn.close()
                    return None
                
                # STEP 2: No duplicates found - proceed with insertion
                insert_query = """
                INSERT INTO trade_recommendations 
                (symbol, action, confidence, entry_price, stop_loss, take_profit, 
                 position_size_percent, amount_usd, reasoning, is_mock, execution_status, generated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'PENDING', NOW())
                """
                
                values = (
                    recommendation['symbol'],
                    recommendation['action'],
                    recommendation['confidence'],
                    recommendation['entry_price'],
                    recommendation['stop_loss'],
                    recommendation['take_profit'],
                    recommendation['position_size_percent'],
                    recommendation.get('amount_usd', 0.0),
                    recommendation['reasoning'] + ' [ROBUST_DEDUP_CHECKED]',  # Mark as robustly checked
                    recommendation['is_mock']
                )
                
                cursor.execute(insert_query, values)
                recommendation_id = cursor.lastrowid
                
                # COMMIT the transaction
                conn.commit()
                
                cursor.close()
                conn.close()
                
                # Mark the signal as processed
                if recommendation.get('signal_id'):
                    self.mark_signal_as_processed(recommendation['signal_id'])
                
                logger.info(f"[ROBUST_DEDUP] ✅ Saved UNIQUE recommendation {recommendation_id} for {symbol} {action}")
                logger.info(f"[ROBUST_DEDUP] Amount: ${recommendation.get('amount_usd', 0):.2f}")
                return recommendation_id
                
            except Exception as e:
                conn.rollback()
                cursor.close()
                conn.close()
                raise e
                
        except Exception as e:
            logger.error(f"Error saving recommendation with robust deduplication: {e}")
            return None
    
    def mark_signal_as_processed(self, signal_id: int) -> bool:
        """Mark a signal as processed in the database"""
        try:
            conn = mysql.connector.connect(**self.signals_db_config)
            cursor = conn.cursor()
            
            update_query = """
            UPDATE trading_signals 
            SET processed = 1, processed_at = NOW()
            WHERE id = %s
            """
            
            cursor.execute(update_query, (signal_id,))
            conn.commit()
            
            cursor.close()
            conn.close()
            
            logger.debug(f"Marked signal {signal_id} as processed")
            return True
            
        except Exception as e:
            logger.error(f"Error marking signal {signal_id} as processed: {e}")
            return False
    
    def get_risk_limits(self) -> Dict:
        """Get risk management limits from database configuration"""
        try:
            # First try to get from trade execution service
            response = requests.get(f"{self.trade_execution_url}/portfolio", timeout=15)
            if response.status_code == 200:
                portfolio_data = response.json()
                if 'risk_limits' in portfolio_data:
                    return portfolio_data['risk_limits']
        except Exception as e:
            logger.warning(f"Could not fetch risk limits from trading engine: {e}")
        
        # Fallback to database configuration
        try:
            self.load_trade_configuration()  # Refresh config if needed
            return {
                'max_position_size_usd': self.max_position_size_usd,
                'max_daily_trades': self.max_daily_trades,
                'max_daily_loss_usd': self.max_daily_loss_usd,
                'balance_utilization_percent': self.balance_utilization_percent,
                'min_confidence_threshold': self.min_confidence_threshold
            }
        except Exception as e:
            logger.warning(f"Error loading risk limits from database: {e}")
            # Final fallback to hardcoded defaults
            return {
                'max_position_size_usd': 500.0,
                'max_daily_trades': 300,
                'max_daily_loss_usd': 500.0,
                'balance_utilization_percent': 95.0,
                'min_confidence_threshold': 0.75  # ANTI-CHURNING: 75% threshold prevents churning while allowing legitimate trades
            }
    
    def get_portfolio_status(self) -> Dict:
        """Get current portfolio status including USD balance and positions with caching"""
        try:
            current_time = time.time()
            
            # Check if we have a recent cached result
            if (current_time - self.last_portfolio_fetch < self.portfolio_cache_ttl and 
                self.portfolio_cache):
                return self.portfolio_cache
            
            response = requests.get(f"{self.trade_execution_url}/portfolio", timeout=15)
            if response.status_code == 200:
                raw_data = response.json()
                
                # Convert positions list to dictionary format for easier access
                positions_data = raw_data.get('positions', {})
                if isinstance(positions_data, list):
                    # Convert list format to dict format
                    positions_dict = {}
                    for pos in positions_data:
                        if isinstance(pos, dict) and 'currency' in pos:
                            positions_dict[pos['currency']] = {
                                'balance': pos.get('available_balance', 0),
                                'value_usd': pos.get('value_usd', 0),
                                'current_price': pos.get('current_price', 0)
                            }
                    positions_data = positions_dict
                
                # Normalize the response to match expected format
                normalized_data = {
                    'cash_balance': raw_data.get('usd_balance', 0.0),  # Use usd_balance from trading engine
                    'usd_balance': raw_data.get('usd_balance', 0.0),   # Bridge compatibility
                    'total_portfolio_value': raw_data.get('total_portfolio_value', 0.0),
                    'positions': positions_data,  # Use converted dictionary format
                    'is_mock': raw_data.get('is_mock', False)
                }
                
                # Cache the result
                self.portfolio_cache = normalized_data
                self.last_portfolio_fetch = current_time
                
                return normalized_data
            else:
                logger.error(f"Could not fetch portfolio status: {response.status_code}")
                raise Exception(f"Portfolio fetch failed with status {response.status_code}")
        except Exception as e:
            logger.error(f"Error fetching portfolio status: {e}")
            raise Exception(f"Portfolio fetch failed: {e}")
    
    def should_sell_for_rebalancing(self, new_symbol: str, new_confidence: float, trade_size_usd: float) -> Optional[str]:
        """Determine if we should sell an existing position to fund a new purchase.
        Only sell if the new opportunity is significantly stronger than existing positions."""
        try:
            portfolio = self.get_portfolio_status()
            usd_balance = portfolio.get('usd_balance', 0.0)
            positions = portfolio.get('positions', {})
            
            # If we have enough USD balance, no need to sell
            if usd_balance >= trade_size_usd:
                return None
            
            # ENHANCED LOGIC: Only sell for STRONG buy opportunities
            # Define strong buy threshold - only consider selling for high-confidence signals
            STRONG_BUY_THRESHOLD = 0.75  # ANTI-CHURNING: Minimum confidence for new buy to justify selling
            CONFIDENCE_ADVANTAGE_REQUIRED = 0.20  # New signal must be 20% higher than existing
            
            if new_confidence < STRONG_BUY_THRESHOLD:
                logger.info(f"[SMART-REBALANCE] {new_symbol} confidence {new_confidence:.3f} below strong buy threshold {STRONG_BUY_THRESHOLD} - not selling existing positions")
                return None
            
            # Positions should already be in dict format from get_portfolio_status()
            if not isinstance(positions, dict):
                logger.warning(f"[REBALANCING] Unexpected positions format: {type(positions)}")
                return None
            
            # Find the weakest position that could fund this strong buy opportunity
            best_candidate = None
            best_candidate_weakness = 0.0
            
            logger.info(f"[SMART-REBALANCE] Evaluating positions for strong buy {new_symbol} (confidence: {new_confidence:.3f})")
            
            for symbol, position_data in positions.items():
                if isinstance(position_data, dict):
                    balance = position_data.get('balance', 0)
                    value_usd = position_data.get('value_usd', 0)
                    
                    # Skip if position too small to fund the trade
                    if balance <= 0 or value_usd < trade_size_usd:
                        continue
                    
                    # Skip if it's the same symbol we want to buy
                    if symbol == new_symbol:
                        continue
                    
                    # Get recent signal confidence for existing position
                    recent_signal_confidence = self.get_recent_signal_confidence(symbol)
                    
                    # Calculate "weakness score" for this position
                    weakness_score = self.calculate_position_weakness(symbol, recent_signal_confidence, value_usd)
                    
                    # Only consider selling if new signal has significant advantage
                    confidence_advantage = new_confidence - (recent_signal_confidence or 0.5)
                    
                    if confidence_advantage >= CONFIDENCE_ADVANTAGE_REQUIRED:
                        logger.info(f"[SMART-REBALANCE] {symbol}: confidence={recent_signal_confidence}, weakness={weakness_score:.3f}, advantage={confidence_advantage:.3f}")
                        
                        # Track the weakest position that meets our criteria
                        if weakness_score > best_candidate_weakness:
                            best_candidate = symbol
                            best_candidate_weakness = weakness_score
                    else:
                        logger.debug(f"[SMART-REBALANCE] {symbol}: insufficient advantage ({confidence_advantage:.3f} < {CONFIDENCE_ADVANTAGE_REQUIRED})")
            
            if best_candidate:
                recent_conf = self.get_recent_signal_confidence(best_candidate)
                logger.info(f"[SMART-REBALANCE] ✅ APPROVED: Sell {best_candidate} (confidence: {recent_conf}) for {new_symbol} (confidence: {new_confidence:.3f})")
                logger.info(f"[SMART-REBALANCE] Confidence advantage: {new_confidence - (recent_conf or 0.5):.3f}, weakness score: {best_candidate_weakness:.3f}")
                return best_candidate
            else:
                logger.info(f"[SMART-REBALANCE] ❌ REJECTED: No suitable positions found to sell for {new_symbol} - maintaining current portfolio")
                return None
            
        except Exception as e:
            logger.error(f"Error in smart rebalancing logic: {e}")
            return None
    
    def calculate_position_weakness(self, symbol: str, recent_confidence: Optional[float], value_usd: float) -> float:
        """Calculate a weakness score for a position (higher = weaker = more suitable to sell).
        Factors: signal age, confidence level, position size, recent performance"""
        try:
            weakness_score = 0.0
            
            # Factor 1: Signal confidence (or lack thereof)
            if recent_confidence is None:
                weakness_score += 0.5  # No recent signal = moderate weakness
                logger.debug(f"[WEAKNESS] {symbol}: No recent signal (+0.5)")
            else:
                # Lower confidence = higher weakness
                confidence_weakness = (1.0 - recent_confidence) * 0.4
                weakness_score += confidence_weakness
                logger.debug(f"[WEAKNESS] {symbol}: Confidence weakness +{confidence_weakness:.3f}")
            
            # Factor 2: Signal age (older signals = weaker)
            signal_age_weakness = self.get_signal_age_weakness(symbol)
            weakness_score += signal_age_weakness
            
            # Factor 3: Position size consideration (very large positions harder to move)
            if value_usd > 500:
                weakness_score -= 0.1  # Large positions slightly less weak (harder to move)
                logger.debug(f"[WEAKNESS] {symbol}: Large position penalty (-0.1)")
            
            # Factor 4: Recent performance (if we have trading history)
            performance_weakness = self.get_performance_weakness(symbol)
            weakness_score += performance_weakness
            
            return max(0.0, min(1.0, weakness_score))  # Clamp between 0-1
            
        except Exception as e:
            logger.error(f"Error calculating weakness for {symbol}: {e}")
            return 0.3  # Default moderate weakness
    
    def get_signal_age_weakness(self, symbol: str) -> float:
        """Calculate weakness based on signal age (older = weaker)"""
        try:
            conn = mysql.connector.connect(**self.signals_db_config)
            cursor = conn.cursor()
            
            query = """
            SELECT TIMESTAMPDIFF(MINUTE, created_at, NOW()) as age_minutes
            FROM trading_signals 
            WHERE symbol = %s 
            ORDER BY created_at DESC LIMIT 1
            """
            
            cursor.execute(query, (symbol,))
            result = cursor.fetchone()
            
            cursor.close()
            conn.close()
            
            if result and result[0] is not None:
                age_minutes = result[0]
                # Convert to weakness: 0-60 min = 0, 60-180 min = 0.1-0.3, 180+ min = 0.3
                if age_minutes <= 60:
                    age_weakness = 0.0
                elif age_minutes <= 180:
                    age_weakness = (age_minutes - 60) / 120 * 0.3  # Scale 0-0.3
                else:
                    age_weakness = 0.3
                
                logger.debug(f"[WEAKNESS] {symbol}: Signal age {age_minutes}min = +{age_weakness:.3f}")
                return age_weakness
            else:
                # No signal found = old/stale
                logger.debug(f"[WEAKNESS] {symbol}: No recent signal = +0.3")
                return 0.3
                
        except Exception as e:
            logger.error(f"Error getting signal age for {symbol}: {e}")
            return 0.2  # Default moderate age penalty
    
    def get_performance_weakness(self, symbol: str) -> float:
        """Calculate weakness based on recent trading performance (if available)"""
        try:
            # This could be enhanced with actual P&L tracking
            # For now, return minimal impact
            return 0.0
        except Exception as e:
            logger.error(f"Error getting performance weakness for {symbol}: {e}")
            return 0.0
    
    def get_recent_signal_confidence(self, symbol: str) -> Optional[float]:
        """Get the confidence of the most recent signal for a symbol"""
        try:
            conn = mysql.connector.connect(**self.signals_db_config)
            cursor = conn.cursor()
            
            # Get most recent enhanced signal for this symbol
            query = """
            SELECT confidence FROM trading_signals 
            WHERE symbol = %s AND created_at >= DATE_SUB(NOW(), INTERVAL 2 HOUR)
            ORDER BY created_at DESC LIMIT 1
            """
            
            cursor.execute(query, (symbol,))
            result = cursor.fetchone()
            
            cursor.close()
            conn.close()
            
            return float(result[0]) if result else None
            
        except Exception as e:
            logger.error(f"Error getting recent signal confidence for {symbol}: {e}")
            return None
    
    def cleanup_small_positions(self) -> int:
        """Automatically sell positions worth less than minimum threshold"""
        try:
            # Configurable threshold for small positions (default $5)
            SMALL_POSITION_THRESHOLD = float(os.getenv('SMALL_POSITION_THRESHOLD_USD', '5.00'))
            
            # Minimum amount required for Coinbase to process the trade (prevent tiny trades that fail)
            MIN_COINBASE_TRADE_AMOUNT = float(os.getenv('MIN_COINBASE_TRADE_AMOUNT', '1.00'))
            
            portfolio = self.get_portfolio_status()
            positions = portfolio.get('positions', {})
            
            if not positions:
                logger.debug("[CLEANUP] No positions to check for cleanup")
                return 0
            
            small_positions = []
            
            # Identify positions below threshold  
            for symbol, position_data in positions.items():
                if isinstance(position_data, dict):
                    balance = position_data.get('balance', 0)
                    value_usd = position_data.get('value_usd', 0)
                    
                    # DATABASE-DRIVEN ASSET FILTERING: Check if symbol is supported by Coinbase
                    if not is_asset_supported(symbol):
                        logger.info(f"[CLEANUP] Skipping {symbol}: not supported by Coinbase Advanced Trade")
                        continue
                    
                    # Check if position is small but still has value AND is large enough for Coinbase to process
                    if (MIN_COINBASE_TRADE_AMOUNT <= value_usd < SMALL_POSITION_THRESHOLD and 
                        balance > 0 and 
                        symbol not in ['USD', 'USDC', 'USDT']):  # Skip stablecoins
                        
                        small_positions.append({
                            'symbol': symbol,
                            'value_usd': value_usd,
                            'balance': balance
                        })
                    elif 0 < value_usd < MIN_COINBASE_TRADE_AMOUNT:
                        # Log tiny positions that are being skipped
                        logger.debug(f"[CLEANUP] Skipping {symbol}: ${value_usd:.2f} too small for Coinbase (min ${MIN_COINBASE_TRADE_AMOUNT:.2f})")
            
            if not small_positions:
                logger.debug(f"[CLEANUP] No positions found between ${MIN_COINBASE_TRADE_AMOUNT:.2f} and ${SMALL_POSITION_THRESHOLD:.2f} threshold")
                return 0
            
            logger.info(f"[CLEANUP] Found {len(small_positions)} small positions to clean up:")
            for pos in small_positions:
                logger.info(f"[CLEANUP]   {pos['symbol']}: ${pos['value_usd']:.2f} (balance: {pos['balance']:.6f})")
            
            cleaned_count = 0
            
            # Generate sell recommendations for small positions
            for pos in small_positions:
                try:
                    # Create sell recommendation for full position
                    cleanup_recommendation = {
                        'symbol': pos['symbol'],
                        'action': 'SELL',
                        'confidence': 0.95,  # High confidence for cleanup
                        'entry_price': 0,  # Market price
                        'stop_loss': 0,
                        'take_profit': 0,
                        'position_size_percent': 100,  # Sell full position
                        'amount_usd': pos['value_usd'],
                        'value_usd': pos['value_usd'],
                        'reasoning': f'Automatic cleanup: position ${pos["value_usd"]:.2f} < ${SMALL_POSITION_THRESHOLD:.2f} threshold',
                        'is_mock': False,
                        'is_rebalancing': False,
                        'is_consolidation': True,  # Mark as consolidation
                        'consolidation_value': pos['value_usd']
                    }
                    
                    # Save cleanup recommendation
                    rec_id = self.save_recommendation(cleanup_recommendation)
                    if rec_id:
                        # Execute cleanup trade with bypassed minimum trade size
                        if self.execute_trade(cleanup_recommendation, bypass_min_size=True):
                            cleaned_count += 1
                            logger.info(f"[CLEANUP] ✅ Successfully cleaned up {pos['symbol']} (${pos['value_usd']:.2f})")
                        else:
                            logger.warning(f"[CLEANUP] ❌ Failed to execute cleanup trade for {pos['symbol']}")
                        
                        # Small delay between cleanup trades
                        time.sleep(1)
                    else:
                        logger.warning(f"[CLEANUP] Failed to save cleanup recommendation for {pos['symbol']}")
                        
                except Exception as e:
                    logger.error(f"[CLEANUP] Error processing cleanup for {pos['symbol']}: {e}")
            
            if cleaned_count > 0:
                total_value = sum(pos['value_usd'] for pos in small_positions[:cleaned_count])
                logger.info(f"[CLEANUP] 🎉 Successfully cleaned up {cleaned_count} positions worth ${total_value:.2f}")
            
            return cleaned_count
            
        except Exception as e:
            logger.error(f"[CLEANUP] Error in small position cleanup: {e}")
            return 0

    def execute_sell_order(self, symbol: str, target_usd_amount: float) -> bool:
        """Execute a sell order to generate USD for new purchases"""
        try:
            # Get current position for the symbol
            portfolio = self.get_portfolio_status()
            positions = portfolio.get('positions', {})
            
            if symbol not in positions:
                logger.warning(f"[SELL] No position found for {symbol}")
                return False
                
            target_position = positions[symbol]
            if not isinstance(target_position, dict):
                logger.warning(f"[SELL] Invalid position data for {symbol}")
                return False

            current_balance = target_position.get('balance', 0)
            current_value = target_position.get('value_usd', 0)
            
            if current_balance <= 0 or current_value <= 0:
                logger.warning(f"[SELL] No balance or value available for {symbol}")
                return False
            
            # Calculate how much to sell (try to get the target USD amount)
            # Sell enough to get target amount, but not more than 50% of position
            sell_percentage = min(target_usd_amount / current_value, 0.5)
            sell_size_usd = current_value * sell_percentage
            
            logger.info(f"[SELL] Selling ${sell_size_usd:.2f} of {symbol} (balance: {current_balance})")
            
            # Create sell trade request
            trade_data = {
                'symbol': symbol,
                'action': 'sell',
                'size_usd': sell_size_usd,
                'order_type': 'MARKET'
            }
            
            response = requests.post(
                f"{self.trade_execution_url}/execute_trade",
                json=trade_data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success', False):
                    logger.info(f"[SELL] Successfully sold {symbol}: {result}")
                    return True
                else:
                    logger.warning(f"[SELL] Sell order rejected: {result}")
                    return False
            else:
                logger.error(f"[SELL] Sell order failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing sell order for {symbol}: {e}")
            return False
    
    def check_trade_cooldown(self, symbol: str, action: str) -> bool:
        """Check if symbol is in cooldown period to prevent excessive trading"""
        try:
            conn = mysql.connector.connect(**self.trades_db_config)
            cursor = conn.cursor()
            
            # Check for any recent trades on this symbol
            query = """
            SELECT action, created_at, TIMESTAMPDIFF(MINUTE, created_at, NOW()) as minutes_ago
            FROM trades 
            WHERE symbol = %s 
            AND created_at >= DATE_SUB(NOW(), INTERVAL %s MINUTE)
            ORDER BY created_at DESC LIMIT 1
            """
            
            cursor.execute(query, (symbol, self.trade_cooldown_minutes))
            recent_trade = cursor.fetchone()
            
            if recent_trade:
                action_db, created_at, minutes_ago = recent_trade
                logger.info(f"[COOLDOWN] {symbol}: Last trade was {action_db} {minutes_ago} minutes ago")
                
                # If it's the same action within cooldown period, block it
                if action.upper() == action_db.upper():
                    logger.warning(f"[COOLDOWN] {symbol}: {action} blocked - same action within {self.trade_cooldown_minutes} minutes")
                    return True  # Block the trade
                
                # If it's opposite action, use shorter cooldown
                if (action.upper() in ['BUY', 'STRONG_BUY'] and action_db.upper() in ['SELL', 'STRONG_SELL']) or \
                   (action.upper() in ['SELL', 'STRONG_SELL'] and action_db.upper() in ['BUY', 'STRONG_BUY']):
                    if minutes_ago < self.contradictory_trade_cooldown_minutes:
                        logger.warning(f"[COOLDOWN] {symbol}: {action} blocked - contradictory to {action_db} {minutes_ago} minutes ago (min: {self.contradictory_trade_cooldown_minutes})")
                        return True  # Block contradictory trades
            
            cursor.close()
            conn.close()
            return False  # No cooldown, allow trade
            
        except Exception as e:
            logger.error(f"Error checking trade cooldown for {symbol}: {e}")
            return False  # Default to allow if check fails

    def check_daily_trade_count(self) -> bool:
        """Check if we've exceeded daily trade limit"""
        try:
            conn = mysql.connector.connect(**self.trades_db_config)
            cursor = conn.cursor()
            
            # Count trades executed today
            query = "SELECT COUNT(*) FROM trades WHERE DATE(created_at) = CURDATE()"
            cursor.execute(query)
            daily_count = cursor.fetchone()[0]
            
            cursor.close()
            conn.close()
            
            max_daily = getattr(self, 'max_daily_trades', 50)
            if daily_count >= max_daily:
                logger.warning(f"[DAILY-LIMIT] Daily trade limit reached: {daily_count}/{max_daily}")
                return True  # Block further trades
            
            logger.info(f"[DAILY-COUNT] Trades today: {daily_count}/{max_daily}")
            return False  # Allow trade
            
        except Exception as e:
            logger.error(f"Error checking daily trade count: {e}")
            return False  # Default to allow if check fails

    def should_block_immediate_rebalancing(self, symbol: str, action: str) -> bool:
        """Check if rebalancing should be blocked due to recent contrary trades"""
        try:
            conn = mysql.connector.connect(**self.trades_db_config)
            cursor = conn.cursor()
            
            # Define opposite actions
            if action in ['SELL', 'STRONG_SELL']:
                opposite_actions = ['BUY', 'STRONG_BUY']
                block_window_minutes = 60  # Block SELL for 60 minutes after BUY
            elif action in ['BUY', 'STRONG_BUY']:
                opposite_actions = ['SELL', 'STRONG_SELL']
                block_window_minutes = 30  # Block BUY for 30 minutes after SELL
            else:
                return False  # No blocking for HOLD actions
            
            # Check for recent opposite trades within the block window
            query = """
            SELECT action, created_at FROM trade_recommendations 
            WHERE symbol = %s AND action IN %s
            AND created_at >= DATE_SUB(NOW(), INTERVAL %s MINUTE)
            ORDER BY created_at DESC LIMIT 1
            """
            
            cursor.execute(query, (symbol, opposite_actions, block_window_minutes))
            recent_opposite = cursor.fetchone()
            
            cursor.close()
            conn.close()
            
            if recent_opposite:
                opposite_action, created_at = recent_opposite
                time_since_opposite = datetime.now() - created_at
                minutes_since_opposite = time_since_opposite.total_seconds() / 60
                
                logger.info(f"🚫 {symbol}: Blocking {action} - recent {opposite_action} {minutes_since_opposite:.1f} minutes ago (min window: {block_window_minutes})")
                return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Error checking rebalancing cooldown for {symbol}: {e}")
            return False  # Default to allowing if check fails

    def calculate_proper_position_size(self, recommendation: Dict) -> Optional[float]:
        """Calculate proper position size based on portfolio value and risk management"""
        try:
            symbol = recommendation['symbol']
            position_size_percent = recommendation['position_size_percent']
            
            # Get current portfolio information
            try:
                portfolio_response = requests.get(f'{self.trade_execution_url}/portfolio', timeout=15)
                if portfolio_response.status_code != 200:
                    logger.warning(f"[POSITION-SIZE] Cannot get portfolio value for {symbol}, using fallback")
                    return None
                    
                portfolio_data = portfolio_response.json()
                actual_portfolio_value = portfolio_data.get('total_portfolio_value', 0)
                usd_balance = portfolio_data.get('usd_balance', 0)  # Use correct key from trading engine response
                positions = portfolio_data.get('positions', {})
                
                if actual_portfolio_value <= 0:
                    logger.warning(f"[POSITION-SIZE] Invalid portfolio value for {symbol}: ${actual_portfolio_value}")
                    return None
                    
            except Exception as e:
                logger.error(f"[POSITION-SIZE] Error getting portfolio for {symbol}: {e}")
                return None
            
            # Calculate USD amount based on actual portfolio value
            calculated_size_usd = actual_portfolio_value * (position_size_percent / 100.0)
            logger.info(f"[POSITION-SIZE] {symbol}: Portfolio ${actual_portfolio_value:.2f} * {position_size_percent:.1f}% = ${calculated_size_usd:.2f}")
            
            # Apply risk management caps
            max_position_size = self.max_position_size_usd  # Use direct reference instead of non-existent method
            size_usd = min(calculated_size_usd, max_position_size)
            
            # Get risk-adjusted position size
            try:
                risk_adjusted_size = self.get_risk_adjusted_position_size(symbol, size_usd, positions)
                if risk_adjusted_size:
                    size_usd = risk_adjusted_size
                    logger.info(f"[POSITION-SIZE] {symbol}: Risk-adjusted to ${size_usd:.2f}")
            except Exception as e:
                logger.warning(f"[POSITION-SIZE] {symbol}: Risk adjustment failed ({e}), using base calculation")
                # Continue with the base calculation
            
            # Apply balance constraints
            if usd_balance < size_usd:
                max_available = usd_balance * (self.balance_utilization_percent / 100.0)
                size_usd = min(size_usd, max_available)
                logger.info(f"[POSITION-SIZE] {symbol}: Balance-limited to ${size_usd:.2f}")
            
            # Apply minimum trade size check based on action type
            action = recommendation.get('action', 'BUY')
            if action in ['SELL', 'STRONG_SELL']:
                min_trade_size = 5.0  # Lower minimum for SELL orders
            else:
                min_trade_size = float(os.getenv('MIN_TRADE_SIZE_USD', '25.00'))  # Higher minimum for BUY orders
            
            min_coinbase_amount = float(os.getenv('MIN_COINBASE_TRADE_AMOUNT', '1.00'))
            
            if size_usd < min_coinbase_amount:
                logger.warning(f"[POSITION-SIZE] {symbol}: ${size_usd:.2f} below Coinbase minimum ${min_coinbase_amount:.2f}")
                return None
                
            if size_usd < min_trade_size:
                logger.warning(f"[POSITION-SIZE] {symbol}: ${size_usd:.2f} below {action} minimum ${min_trade_size:.2f}")
                return None
            
            # Final precision rounding
            size_usd = round(size_usd, 2)
            logger.info(f"[POSITION-SIZE] {symbol}: Final calculated amount ${size_usd:.2f}")
            
            return size_usd
            
        except Exception as e:
            logger.error(f"[POSITION-SIZE] Error calculating position size for {recommendation.get('symbol', 'unknown')}: {e}")
            raise Exception(f"Position size calculation failed: {e}")

    def execute_trade(self, recommendation: Dict, bypass_min_size: bool = False) -> bool:
        """Execute trade via trade execution service with portfolio-aware logic"""
        try:
            action = recommendation['action']
            symbol = recommendation['symbol']
            
            # COOLDOWN CHECKS: Prevent excessive and contradictory trading
            
            # 1. Check daily trade limit first
            if self.check_daily_trade_count():
                logger.warning(f"[LIMIT] {symbol}: Trade blocked - daily limit reached")
                return False
            
            # 2. Check symbol-specific cooldown periods
            if self.check_trade_cooldown(symbol, action):
                logger.warning(f"[COOLDOWN] {symbol}: {action} trade blocked - cooldown active")
                return False
            
            # Handle HOLD signals - save recommendation but don't execute trade
            if action == 'HOLD':
                logger.info(f"[HOLD] {symbol}: Recommendation saved but no trade executed (HOLD signal)")
                return True  # Consider this successful since we saved the recommendation
            
            # Get portfolio status and risk limits
            portfolio = self.get_portfolio_status()
            risk_limits = self.get_risk_limits()
            max_position_size = risk_limits.get('max_position_size_usd', self.max_position_size_usd)
            usd_balance = portfolio.get('usd_balance', 0.0)
            positions = portfolio.get('positions', {})
            
            # Handle SELL signals - check if we have position to sell
            if action in ['SELL', 'STRONG_SELL']:
                # REBALANCING PREVENTION: Check if this SELL should be blocked due to recent BUY
                if self.should_block_immediate_rebalancing(symbol, action):
                    logger.info(f"[REBALANCE-BLOCK] {symbol}: SELL blocked due to recent BUY order")
                    return False  # Block this SELL to prevent immediate buy-sell cycling
                
                if symbol not in positions:
                    logger.warning(f"[SELL] No position in {symbol} to sell")
                    return False
                    
                position_data = positions[symbol]
                if not isinstance(position_data, dict):
                    logger.warning(f"[SELL] Invalid position data for {symbol}")
                    return False
                    
                current_balance = position_data.get('balance', 0)
                current_value = position_data.get('value_usd', 0)
                
                if current_balance <= 0 or current_value <= 0:
                    logger.warning(f"[SELL] No balance to sell for {symbol}")
                    return False

                # Check if this is a rebalancing or consolidation trade
                is_rebalancing = recommendation.get('is_rebalancing', False)
                is_consolidation = recommendation.get('is_consolidation', False)
                
                if is_rebalancing or is_consolidation:
                    # For rebalancing/consolidation, use the exact USD value from the signal if available
                    signal_value_usd = recommendation.get('value_usd', 0) or recommendation.get('amount_usd', 0)
                    # Round the signal value immediately for Coinbase API precision requirements
                    signal_value_usd = round(signal_value_usd, 2)
                    logger.info(f"[SELL] Debug - value_usd: {recommendation.get('value_usd')}, amount_usd: {recommendation.get('amount_usd')}, rounded: {signal_value_usd}")
                    if signal_value_usd > 0:
                        # Use the value from the rebalancing signal directly (already rounded for precision)
                        size_usd = min(signal_value_usd, max_position_size)
                        logger.info(f"[SELL] {'Rebalancing' if is_rebalancing else 'Consolidation'} - selling ${size_usd:.2f} of {symbol} (signal value: ${signal_value_usd:.2f})")
                    else:
                        # Fallback to current position value
                        size_usd = round(min(current_value, max_position_size), 2)
                        logger.info(f"[SELL] {'Rebalancing' if is_rebalancing else 'Consolidation'} - selling ${size_usd:.2f} of {symbol} position (full: ${current_value:.2f})")
                else:
                    # For regular sell orders, use a percentage of current position
                    size_usd = round(min(current_value * 0.5, max_position_size), 2)  # Sell up to 50% of position
                    logger.info(f"[SELL] Regular sell - selling ${size_usd:.2f} of {symbol} position (50% of ${current_value:.2f})")
                
            else:  # BUY or STRONG_BUY
                # REBALANCING PREVENTION: Check if this BUY should be blocked due to recent SELL
                if self.should_block_immediate_rebalancing(symbol, action):
                    logger.info(f"[REBALANCE-BLOCK] {symbol}: BUY blocked due to recent SELL order")
                    return False  # Block this BUY to prevent immediate sell-buy cycling
                # Calculate initial trade size based on ACTUAL portfolio value
                position_size_percent = recommendation['position_size_percent']
                entry_price = recommendation['entry_price']
                
                # Get actual portfolio value from trading engine
                try:
                    portfolio_response = requests.get(f'{self.trade_execution_url}/portfolio', timeout=15)
                    if portfolio_response.status_code == 200:
                        portfolio_data = portfolio_response.json()
                        actual_portfolio_value = portfolio_data.get('total_portfolio_value', 0)
                        
                        # Calculate USD amount based on actual portfolio value
                        calculated_size_usd = actual_portfolio_value * (position_size_percent / 100.0)
                        
                        logger.info(f"[PORTFOLIO] Actual portfolio value: ${actual_portfolio_value:.2f}")
                        logger.info(f"[POSITION] Calculated trade size: ${calculated_size_usd:.2f} ({position_size_percent:.1f}% of portfolio)")
                    else:
                        # NO FALLBACKS: Portfolio data is required for position sizing
                        raise Exception("Portfolio value required for position sizing - no fallbacks allowed")
                        
                except Exception as e:
                    logger.error(f"[PORTFOLIO] Error getting portfolio value: {e}")
                    raise Exception(f"Portfolio value calculation failed: {e}")
                
                # Cap the trade size to respect risk management limits
                size_usd = min(calculated_size_usd, max_position_size)
                
                # Get risk-adjusted position size from advanced risk management
                risk_adjusted_size = self.get_risk_adjusted_position_size(symbol, size_usd, positions)
                if risk_adjusted_size:
                    size_usd = risk_adjusted_size
                    logger.info(f"[RISK] Risk-adjusted position size for {symbol}: ${size_usd:.2f}")
                
                # Further cap based on actual available USD balance
                if usd_balance < size_usd:
                    logger.info(f"[PORTFOLIO] USD balance ${usd_balance:.2f} < desired ${size_usd:.2f}")
                    
                    # Check if we should sell something to fund this purchase
                    symbol_to_sell = self.should_sell_for_rebalancing(
                        recommendation['symbol'], 
                        recommendation['confidence'], 
                        size_usd
                    )
                    
                    if symbol_to_sell:
                        # Execute sell order first
                        sell_success = self.execute_sell_order(symbol_to_sell, size_usd)
                        if not sell_success:
                            logger.warning(f"[REBALANCE] Failed to sell {symbol_to_sell}, reducing trade size to available balance")
                            max_available = usd_balance * (self.balance_utilization_percent / 100.0)
                            size_usd = min(size_usd, max_available)
                    else:
                        # No rebalancing possible, reduce trade size to available balance with utilization limit
                        max_available = usd_balance * (self.balance_utilization_percent / 100.0)
                        logger.info(f"[PORTFOLIO] No rebalancing opportunity, reducing trade size to ${max_available:.2f} ({self.balance_utilization_percent}% of ${usd_balance:.2f})")
                        size_usd = min(size_usd, max_available)
                
                # Skip trade if size is too small - use environment variable or default
                min_trade_size = float(os.getenv('MIN_TRADE_SIZE_USD', '5.00'))  # Increased from 2.00 to 5.00 to prevent wasteful micro-trades
                if size_usd < min_trade_size:
                    logger.warning(f"[PORTFOLIO] Trade size ${size_usd:.2f} too small, skipping (min: ${min_trade_size})")
                    return False
                
                # Log if we're capping the trade size
                if calculated_size_usd > max_position_size:
                    logger.info(f"[RISK-CAP] Trade size capped: {calculated_size_usd:.2f} → {size_usd:.2f} (max: {max_position_size})")
            
            # SYMBOL STANDARDIZATION: Ensure consistent symbol format
            # standardized_symbol = normalize_symbol(recommendation['symbol'])  # Symbol utils not available
            standardized_symbol = recommendation['symbol']  # Use symbol as-is
            logger.debug(f"[SYMBOL] Standardized '{recommendation['symbol']}' -> '{standardized_symbol}'")
            
            # Final precision rounding to ensure Coinbase API compatibility
            size_usd = round(size_usd, 2)
            
            # MINIMUM TRADE SIZE CHECK: Skip trades below minimum threshold (unless bypassed for cleanup)
            min_trade_size = float(os.getenv('MIN_TRADE_SIZE_USD', '5.00'))  # Increased from 0.50 to 5.00 to prevent wasteful micro-trades
            min_coinbase_amount = float(os.getenv('MIN_COINBASE_TRADE_AMOUNT', '1.00'))  # Minimum for Coinbase to process
            
            # Always enforce Coinbase minimum, even for cleanup operations
            if size_usd < min_coinbase_amount:
                logger.warning(f"[COINBASE-MIN] {standardized_symbol}: Trade size ${size_usd:.2f} below Coinbase minimum ${min_coinbase_amount:.2f} - skipping trade")
                return True  # Return success to avoid retry loops, but don't execute
            
            if not bypass_min_size and size_usd < min_trade_size:
                logger.info(f"[SKIP] {standardized_symbol}: Trade size ${size_usd:.2f} below minimum ${min_trade_size:.2f} - skipping trade")
                return True  # Return success to avoid retry loops, but don't execute
            elif bypass_min_size and size_usd < min_trade_size:
                logger.info(f"[CLEANUP] {standardized_symbol}: Bypassing minimum trade size ${min_trade_size:.2f} for cleanup (${size_usd:.2f})")
                logger.info(f"[CLEANUP] {standardized_symbol}: But still respecting Coinbase minimum ${min_coinbase_amount:.2f}")
            
            trade_data = {
                'symbol': standardized_symbol,  # Use standardized symbol
                'action': recommendation['action'].lower(),
                'confidence': recommendation['confidence'],
                'stop_loss': recommendation['stop_loss'],
                'take_profit': recommendation['take_profit'],
                'size_usd': size_usd
            }
            
            # Standardize the entire trade data structure
            # trade_data = SymbolStandardizer.standardize_trade_data(trade_data)  # Symbol utils not available
            
            response = requests.post(
                f"{self.trade_execution_url}/execute_trade",
                json=trade_data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success', False):
                    logger.info(f"Trade executed successfully: {result}")
                    # Send trade notification
                    self.send_trade_notification(recommendation, result)
                else:
                    logger.warning(f"Trade rejected by trading engine: {result}")
                return result.get('success', False)
            else:
                logger.error(f"Trade execution failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return False
    
    def send_trade_notification(self, recommendation: Dict, result: Dict):
        """Send notification when a trade is executed successfully"""
        try:
            symbol = recommendation['symbol']
            action = recommendation['action']
            confidence = recommendation.get('confidence', 0)
            size_usd = result.get('size_usd', 0)
            entry_price = recommendation.get('entry_price', 0)
            
            # Create notification message
            message = (
                f"🚀 TRADE EXECUTED!\n"
                f"Symbol: {symbol}\n"
                f"Action: {action}\n"
                f"Size: ${size_usd:.2f}\n"
                f"Price: ${entry_price:.4f}\n"
                f"Confidence: {confidence:.1f}%\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            
            title = f"CryptoAI Trade Alert: {action} {symbol}"
            
            # Send notification via notification service
            notification_data = {
                'title': title,
                'message': message,
                'priority': 'high',
                'channel': 'trading'
            }
            
            response = requests.post(
                f"http://192.168.230.163:8038/send",
                json=notification_data,
                timeout=5
            )
            
            if response.status_code == 200:
                logger.info(f"✅ Trade notification sent for {symbol} {action}")
            else:
                logger.warning(f"⚠️ Failed to send trade notification: {response.status_code}")
                
        except Exception as e:
            logger.warning(f"⚠️ Error sending trade notification: {e}")
    
    def filter_duplicate_signals(self, signals: List[Dict]) -> List[Dict]:
        """Filter out signals that are duplicates of recent processed signals"""
        try:
            if not signals:
                return []
            
            conn = mysql.connector.connect(**self.trades_db_config)
            cursor = conn.cursor()
            
            # Check for recent recommendations within the last hour
            signal_dedup_minutes = int(os.environ.get('SIGNAL_DEDUP_MINUTES', 30))
            
            filtered_signals = []
            for signal in signals:
                # Check if we've processed a similar signal recently
                query = """
                SELECT id, action, confidence, created_at
                FROM trade_recommendations 
                WHERE symbol = %s 
                AND action = %s
                AND created_at >= DATE_SUB(NOW(), INTERVAL %s MINUTE)
                ORDER BY created_at DESC LIMIT 1
                """
                
                action = signal.get('signal_type', '').upper()
                cursor.execute(query, (signal['symbol'], action, signal_dedup_minutes))
                recent_rec = cursor.fetchone()
                
                if recent_rec:
                    rec_id, rec_action, rec_confidence, rec_created_at = recent_rec
                    minutes_ago = (datetime.now() - rec_created_at).total_seconds() / 60
                    
                    # Check if it's essentially the same signal (same action, similar confidence)
                    confidence_diff = abs(signal['confidence'] - rec_confidence)
                    if confidence_diff < 0.05:  # Less than 5% confidence difference
                        logger.info(f"[DEDUP] {signal['symbol']}: Skipping {action} signal - similar signal processed {minutes_ago:.1f} min ago (conf: {signal['confidence']:.3f} vs {rec_confidence:.3f})")
                        continue
                    
                # Signal passed deduplication check
                filtered_signals.append(signal)
            
            cursor.close()
            conn.close()
            
            return filtered_signals
            
        except Exception as e:
            logger.error(f"Error filtering duplicate signals: {e}")
            # If filter fails, return original signals to not block processing
            return signals

    def process_signals(self):
        """Main processing loop for converting signals to recommendations with signal coherence and dynamic rebalancing"""
        try:
            # Clean up expired data first to ensure we work with fresh data only
            self._cleanup_expired_data()
            
            signals = self.get_unprocessed_enhanced_signals()
            
            if not signals:
                logger.debug("No new enhanced signals to process")
                # Still run cleanup and consolidation even if no signals
                self._run_cleanup_and_consolidation()
                return

            # TEMPORAL DEDUPLICATION: Filter out signals that are too similar to recent ones
            filtered_signals = self.filter_duplicate_signals(signals)
            if len(filtered_signals) < len(signals):
                logger.info(f"[DEDUP] Filtered {len(signals) - len(filtered_signals)} duplicate signals, processing {len(filtered_signals)}")
            
            if not filtered_signals:
                logger.debug("No signals remaining after deduplication filter")
                # Still run cleanup and consolidation even if no signals to process
                self._run_cleanup_and_consolidation()
                return
            
            # SIGNAL COHERENCE: Group signals by symbol and process only one per symbol per cycle
            signals_by_symbol = {}
            for signal in filtered_signals:
                symbol = signal['symbol']
                if symbol not in signals_by_symbol:
                    signals_by_symbol[symbol] = []
                signals_by_symbol[symbol].append(signal)
            
            # For each symbol, prioritize highest confidence signal
            prioritized_signals = []
            for symbol, symbol_signals in signals_by_symbol.items():
                if len(symbol_signals) > 1:
                    # Sort by confidence descending
                    symbol_signals.sort(key=lambda x: x['confidence'], reverse=True)
                    best_signal = symbol_signals[0]
                    logger.info(f"🎯 {symbol}: Processing highest confidence signal {best_signal['signal_type']} ({best_signal['confidence']:.3f}) from {len(symbol_signals)} candidates")
                    prioritized_signals.append(best_signal)
                    
                    # Mark other signals as processed to avoid duplicates
                    for signal in symbol_signals[1:]:
                        self.processed_signals.add(signal['id'])
                        logger.debug(f"🚫 {symbol}: Skipping lower confidence {signal['signal_type']} ({signal['confidence']:.3f})")
                else:
                    prioritized_signals.append(symbol_signals[0])
            
            processed_count = 0
            executed_count = 0
            rebalancing_count = 0
            
            # Process signals and identify strong BUY signals that need rebalancing
            strong_buy_signals = []
            
            for signal in prioritized_signals:
                # Check if already converted
                if self.signal_already_exists_as_recommendation(
                    signal['id'], signal['symbol'], signal['signal_type']
                ):
                    logger.debug(f"Signal {signal['id']} already converted, skipping")
                    self.processed_signals.add(signal['id'])
                    continue
                
                # Convert to recommendation
                recommendation = self.convert_signal_to_recommendation(signal)
                if not recommendation:
                    # Mark signal as processed even if filtered out (e.g., by asset filter)
                    self.processed_signals.add(signal['id'])
                    self.mark_signal_as_processed(signal['id'])  # Update database
                    logger.debug(f"Signal {signal['id']} ({signal['symbol']}) processed but no recommendation created")
                    continue
                
                # Calculate proper position size before saving
                proper_amount_usd = self.calculate_proper_position_size(recommendation)
                if proper_amount_usd is not None:
                    recommendation['amount_usd'] = proper_amount_usd
                    logger.info(f"[POSITION-FIX] Updated {recommendation['symbol']} amount from ${recommendation.get('amount_usd', 0):.2f} to ${proper_amount_usd:.2f}")
                
                # Collect strong BUY signals for potential rebalancing
                if recommendation.get('requires_rebalancing', False):
                    strong_buy_signals.append((signal, recommendation))
                
                # Save recommendation
                rec_id = self.save_recommendation(recommendation)
                if not rec_id:
                    continue
                
                processed_count += 1
                
                # Mark as processed
                self.processed_signals.add(signal['id'])
            
            # Handle position rebalancing for strong BUY signals
            if strong_buy_signals and self.position_manager.config['enable_dynamic_rebalancing']:
                logger.info(f"🔄 Processing {len(strong_buy_signals)} strong BUY signals for potential rebalancing")
                
                for signal, recommendation in strong_buy_signals:
                    try:
                        # Calculate required funding for this position
                        portfolio = self.position_manager.get_current_portfolio()
                        if not portfolio:
                            logger.warning(f"Cannot get portfolio for rebalancing {signal['symbol']}")
                            continue
                        
                        total_value = portfolio.get('total_portfolio_value', 0)
                        required_funding = total_value * recommendation['position_size_percent'] / 100
                        standard_funding = total_value * 2.0 / 100  # Standard 2% position
                        
                        # Only rebalance if we need more than standard position size
                        if required_funding > standard_funding * 1.2:  # 20% threshold
                            logger.info(f"💰 {signal['symbol']}: Requires ${required_funding:.2f} (standard: ${standard_funding:.2f})")
                            
                            # Generate rebalancing signals
                            rebalancing_signals = self.position_manager.generate_rebalancing_signals(
                                signal, required_funding - standard_funding
                            )
                            
                            if rebalancing_signals:
                                logger.info(f"🔄 Generated {len(rebalancing_signals)} rebalancing signals for {signal['symbol']}")
                                
                                # Process rebalancing sell signals first
                                for rebal_signal in rebalancing_signals:
                                    rebal_recommendation = {
                                        'symbol': rebal_signal['symbol'],
                                        'action': rebal_signal['signal_type'].upper(),  # Convert to uppercase
                                        'confidence': rebal_signal['confidence'],
                                        'entry_price': 0,  # Market price for sells
                                        'stop_loss': 0,
                                        'take_profit': 0,
                                        'position_size_percent': 0,  # Not used for rebalancing sells
                                        'amount_usd': rebal_signal['value_usd'],  # Use exact USD amount from signal
                                        'value_usd': rebal_signal['value_usd'],  # Keep for execute_trade method
                                        'reasoning': rebal_signal['reasoning'],
                                        'is_mock': False,
                                        'is_rebalancing': True,
                                        'rebalancing_for': signal['symbol']
                                    }
                                    
                                    # Save rebalancing recommendation
                                    rebal_rec_id = self.save_recommendation(rebal_recommendation)
                                    if rebal_rec_id:
                                        # Execute rebalancing trade
                                        if self.execute_trade(rebal_recommendation):
                                            rebalancing_count += 1
                                            logger.info(f"✅ Rebalancing SELL executed: {rebal_signal['symbol']}")
                                        
                                        # Small delay between rebalancing trades
                                        time.sleep(2)
                        
                        # Now execute the strong BUY signal
                        if self.execute_trade(recommendation):
                            executed_count += 1
                            logger.info(f"✅ Strong BUY executed: {signal['symbol']} (${required_funding:.2f})")
                        
                        # Delay between strong signals
                        time.sleep(3)
                        
                    except Exception as e:
                        logger.error(f"Error processing rebalancing for {signal['symbol']}: {e}")
                        # Still try to execute the original signal
                        if self.execute_trade(recommendation):
                            executed_count += 1
            
            else:
                # Process remaining signals normally (non-strong signals)
                for signal in prioritized_signals:
                    if signal['id'] in self.processed_signals:
                        continue
                    
                    recommendation = self.convert_signal_to_recommendation(signal)
                    if not recommendation or recommendation.get('requires_rebalancing', False):
                        continue
                    
                    rec_id = self.save_recommendation(recommendation)
                    if rec_id and self.execute_trade(recommendation):
                        executed_count += 1
                    
                    time.sleep(1)
            
            if processed_count > 0:
                log_message = f"BRIDGE CYCLE: Processed {processed_count} signals, executed {executed_count} trades"
                if rebalancing_count > 0:
                    log_message += f", performed {rebalancing_count} rebalancing trades"
                logger.info(log_message)
                
                # Update health metrics
                health_status["signals_processed_today"] += processed_count
                health_status["total_signals_processed"] += processed_count
                health_status["trades_executed_today"] += executed_count + rebalancing_count
                health_status["total_trades_executed"] += executed_count + rebalancing_count
                
                if executed_count > 0:
                    health_status["last_successful_trade"] = datetime.now().isoformat()
            
            # Check for portfolio consolidation and cleanup
            self._run_cleanup_and_consolidation()
            
        except Exception as e:
            logger.error(f"Error in process_signals: {e}")
    
    def _cleanup_expired_data(self):
        """Clean up expired signals and recommendations to keep system running on fresh data only"""
        try:
            conn = mysql.connector.connect(**self.trades_db_config)
            cursor = conn.cursor()
            
            # Mark signals older than 6 hours as processed (expired)
            expired_signals_cutoff = datetime.now() - timedelta(hours=6)
            cursor.execute("""
                UPDATE trading_signals 
                SET processed = 1, 
                    reasoning = CONCAT(COALESCE(reasoning, ''), ' [AUTO-EXPIRED: >6h old]')
                WHERE processed = 0 
                AND created_at < %s
            """, (expired_signals_cutoff,))
            expired_signals = cursor.rowcount
            
            # Mark recommendations older than 4 hours as expired
            expired_recs_cutoff = datetime.now() - timedelta(hours=4)
            cursor.execute("""
                UPDATE trade_recommendations 
                SET execution_status = 'EXPIRED',
                    reasoning = CONCAT(COALESCE(reasoning, ''), ' [AUTO-EXPIRED: >4h old]')
                WHERE execution_status = 'PENDING'
                AND created_at < %s
            """, (expired_recs_cutoff,))
            expired_recommendations = cursor.rowcount
            
            conn.commit()
            cursor.close()
            conn.close()
            
            if expired_signals > 0 or expired_recommendations > 0:
                logger.info(f"🕐 EXPIRED DATA CLEANUP: {expired_signals} signals, {expired_recommendations} recommendations marked as expired")
            
        except Exception as e:
            logger.error(f"Error in expired data cleanup: {e}")

    def _run_cleanup_and_consolidation(self):
        """Run portfolio cleanup and consolidation tasks"""
        global health_status
        
        # Check for portfolio consolidation (periodically)
        self.check_portfolio_consolidation()
        
        # Check for small position cleanup every cycle
        try:
            logger.info("[CLEANUP] Starting small position cleanup check...")
            cleanup_count = self.cleanup_small_positions()
            if cleanup_count > 0:
                logger.info(f"🧹 Cleaned up {cleanup_count} small positions")
                health_status["trades_executed_today"] += cleanup_count
                health_status["total_trades_executed"] += cleanup_count
                health_status["last_successful_trade"] = datetime.now().isoformat()
            else:
                logger.debug("[CLEANUP] No small positions cleaned up this cycle")
        except Exception as e:
            logger.error(f"Error in small position cleanup: {e}")
            import traceback
            logger.error(f"Cleanup traceback: {traceback.format_exc()}")
    
    def check_portfolio_consolidation(self):
        """Check if it's time to consolidate small positions"""
        try:
            current_time = time.time()
            
            # Check if enough time has passed since last consolidation
            if current_time - self.last_consolidation < self.consolidation_interval:
                return
            
            logger.info("🔄 Starting portfolio consolidation check...")
            
            # Get consolidation signals from position manager
            consolidation_signals = self.position_manager.consolidate_small_positions(
                min_position_value_usd=self.min_consolidation_value
            )
            
            if not consolidation_signals:
                logger.info("✅ No small positions found for consolidation")
                self.last_consolidation = current_time
                return
            
            consolidation_count = 0
            total_consolidated_value = 0.0
            
            # Process consolidation signals
            for signal in consolidation_signals:
                try:
                    # Convert consolidation signal to recommendation
                    recommendation = {
                        'symbol': signal['symbol'],
                        'action': signal['signal_type'],
                        'confidence': signal['confidence'],
                        'entry_price': 0,  # Market price for sells
                        'stop_loss': 0,
                        'take_profit': 0,
                        'position_size_percent': 0,  # Full position liquidation
                        'amount_usd': signal['amount_usd'],  # CRITICAL FIX: Copy amount_usd from signal
                        'reasoning': signal['reasoning'],
                        'is_mock': False,
                        'is_consolidation': True,
                        'consolidation_value': signal['consolidation_value']
                    }
                    
                    # Save consolidation recommendation
                    rec_id = self.save_recommendation(recommendation)
                    if rec_id:
                        # Execute consolidation trade
                        if self.execute_trade(recommendation):
                            consolidation_count += 1
                            total_consolidated_value += signal['consolidation_value']
                            logger.info(f"✅ Consolidated {signal['symbol']}: ${signal['consolidation_value']:.2f}")
                        
                        # Small delay between consolidation trades
                        time.sleep(2)
                        
                except Exception as e:
                    logger.error(f"Error processing consolidation for {signal['symbol']}: {e}")
                    continue
            
            # Update tracking
            self.last_consolidation = current_time
            
            if consolidation_count > 0:
                logger.info(f"🎯 Portfolio consolidation complete: {consolidation_count} positions, ${total_consolidated_value:.2f} total value")
                
                # Update health metrics
                health_status["trades_executed_today"] += consolidation_count
                health_status["total_trades_executed"] += consolidation_count
                health_status["consolidations_performed_today"] += 1
                health_status["total_consolidations_performed"] += 1
                health_status["last_consolidation"] = datetime.now().isoformat()
                health_status["last_successful_trade"] = datetime.now().isoformat()
            else:
                logger.info("ℹ️ Portfolio consolidation attempted but no trades executed")
                
        except Exception as e:
            logger.error(f"Error in check_portfolio_consolidation: {e}")
    
    def run_continuous_monitoring(self):
        """Run continuous monitoring and processing"""
        logger.info("Starting continuous signal monitoring...")
        logger.info(f"Monitoring interval: {self.bridge_interval} seconds")
        logger.info(f"Trade execution URL: {self.trade_execution_url}")
        
        health_status["processing_cycle_seconds"] = self.bridge_interval
        
        while True:
            try:
                # Reset daily counters if new day
                today = datetime.now().date()
                if hasattr(self, 'last_date') and self.last_date != today:
                    health_status["signals_processed_today"] = 0
                    health_status["trades_executed_today"] = 0
                    health_status["consolidations_performed_today"] = 0
                self.last_date = today
                
                # Update status to processing
                health_status["status"] = "processing"
                
                # Process signals
                signals_before = health_status["total_signals_processed"]
                trades_before = health_status["total_trades_executed"]
                
                self.process_signals()
                
                # Update health status after processing
                health_status["last_signal_processing"] = datetime.now().isoformat()
                health_status["status"] = "healthy"
                health_status["last_error"] = None
                
                # Note: Signal and trade counts will be updated in process_signals method
                
                time.sleep(self.bridge_interval)
                
            except KeyboardInterrupt:
                logger.info("Received shutdown signal, stopping...")
                health_status["status"] = "stopped"
                break
            except Exception as e:
                health_status["status"] = "error"
                health_status["last_error"] = f"Monitoring loop error: {e}"
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.bridge_interval)

def main():
    """Main function with health monitoring"""
    global bridge_instance
    
    # Start health server first
    start_health_server()
    
    logger.info("Enhanced Signal Bridge Service Starting...")
    logger.info("=" * 50)
    
    # Test database connections
    try:
        test_bridge = AutomatedSignalBridge()
        # Test signals database
        conn = mysql.connector.connect(**test_bridge.signals_db_config)
        conn.close()
        # Test trades database  
        conn = mysql.connector.connect(**test_bridge.trades_db_config)
        conn.close()
        health_status["database_connected"] = True
        logger.info("✓ Database connections verified")
    except Exception as e:
        health_status["database_connected"] = False
        health_status["status"] = "error"
        health_status["last_error"] = f"Database connection failed: {e}"
        logger.error(f"❌ Database connection failed: {e}")
        return
    
    # Test trading engine connection
    try:
        response = requests.get(f"{test_bridge.trade_execution_url}/health", timeout=5)
        if response.status_code == 200:
            health_status["trading_engine_connected"] = True
            health_status["status"] = "healthy"
            logger.info("✓ Trading engine connection verified")
        else:
            health_status["trading_engine_connected"] = False
            health_status["status"] = "error"
            health_status["last_error"] = f"Trading engine returned {response.status_code}"
            logger.warning(f"⚠️ Trading engine returned {response.status_code}")
    except Exception as e:
        health_status["trading_engine_connected"] = False
        health_status["status"] = "warning"
        health_status["last_error"] = f"Trading engine connection failed: {e}"
        logger.warning(f"⚠️ Trading engine connection failed: {e}")
    
    # Initialize bridge and make it available to API endpoints
    bridge_instance = AutomatedSignalBridge()
    logger.info("🚀 Dynamic position management system initialized")
    logger.info(f"💡 Strong signal threshold: {bridge_instance.position_manager.config['strong_signal_threshold']}")
    logger.info(f"📈 Max position multiplier: {bridge_instance.position_manager.config['max_position_multiplier']}")
    
    bridge_instance.run_continuous_monitoring()

if __name__ == "__main__":
    main()
