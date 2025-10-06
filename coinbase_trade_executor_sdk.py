#!/usr/bin/env python3
"""
Coinbase Advanced Trade Executor using Official SDK
"""
import os
import uvicorn
import time
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mysql.connector
from datetime import datetime
from typing import Optional
from coinbase.rest import RESTClient

app = FastAPI(title="Coinbase Advanced Trade Service (SDK)")

class TradeRequest(BaseModel):
    symbol: str
    action: str
    size_usd: float
    order_type: str = "MARKET"

def get_coinbase_client():
    """Initialize Coinbase REST client"""
    # Try environment variables first
    api_key = os.getenv('COINBASE_API_KEY')
    api_secret = os.getenv('COINBASE_PRIVATE_KEY', '')
    
    # If not in environment, try to load from JSON file
    if not api_key or not api_secret:
        try:
            with open('coinbase_api_key.json', 'r') as f:
                creds = json.load(f)
            api_key = creds['name']
            api_secret = creds['privateKey']
            print(f"Loaded API credentials from coinbase_api_key.json")
        except Exception as e:
            print(f"Error loading credentials from file: {e}")
    
    if not api_key or not api_secret:
        raise Exception("Missing Coinbase API credentials")
    
    return RESTClient(api_key=api_key, api_secret=api_secret)

def get_coinbase_accounts():
    """Get Coinbase account balances using SDK"""
    try:
        client = get_coinbase_client()
        accounts = client.get_accounts()
        return accounts
    except Exception as e:
        raise Exception(f"Failed to get accounts: {e}")

def get_current_price(symbol):
    """Get current price for a symbol"""
    try:
        client = get_coinbase_client()
        
        # Ensure symbol is in correct format (e.g., BTC-USD)
        if '-USD' not in symbol:
            symbol = f"{symbol}-USD"
        
        # Get product details
        product = client.get_product(symbol)
        return float(product.price)
    except Exception as e:
        print(f"Warning: Could not get price for {symbol}: {e}")
        return None

def place_coinbase_order(symbol, side, size_usd):
    """Place a real order using Coinbase SDK"""
    try:
        client = get_coinbase_client()
        
        # Ensure symbol is in correct format (e.g., BTC-USD)
        if '-USD' not in symbol:
            symbol = f"{symbol}-USD"
        
        # Create order configuration for market order
        if side.upper() == "BUY":
            # For buy orders, specify quote size (USD amount)
            order_config = {
                "market_market_ioc": {
                    "quote_size": str(size_usd)
                }
            }
        else:
            # For sell orders, specify base size (crypto amount)
            # Get current price to calculate correct crypto amount
            current_price = get_current_price(symbol)
            if current_price:
                crypto_amount = size_usd / current_price
                # Round to appropriate precision (DOGE typically uses 0 decimal places)
                crypto_amount = round(crypto_amount, 0)
                order_config = {
                    "market_market_ioc": {
                        "base_size": str(crypto_amount)
                    }
                }
            else:
                # Fallback to rough estimate if price unavailable
                estimated_crypto_amount = size_usd / 0.1  # Rough estimate
                estimated_crypto_amount = round(estimated_crypto_amount, 0)
                order_config = {
                    "market_market_ioc": {
                        "base_size": str(estimated_crypto_amount)
                    }
                }
        
        # Place the order
        order_result = client.create_order(
            client_order_id=f"trade_{symbol}_{side}_{int(time.time())}",
            product_id=symbol,
            side=side.upper(),
            order_configuration=order_config
        )
        
        print(f"Order API Response: {order_result}")
        return order_result
        
    except Exception as e:
        raise Exception(f"Failed to place order: {e}")

@app.get("/health")
async def health():
    try:
        # Test API connection
        accounts = get_coinbase_accounts()
        return {
            "status": "healthy", 
            "service": "coinbase-advanced-trade-sdk",
            "api_connected": True,
            "api_version": "v3_advanced_trade_sdk",
            "accounts_count": len(accounts.accounts) if hasattr(accounts, 'accounts') else 0
        }
    except Exception as e:
        return {
            "status": "unhealthy", 
            "service": "coinbase-advanced-trade-sdk",
            "api_connected": False,
            "error": str(e)
        }

@app.get("/accounts")
async def get_accounts():
    """Get account balances"""
    try:
        accounts_data = get_coinbase_accounts()
        accounts = accounts_data.accounts if hasattr(accounts_data, 'accounts') else []
        
        balances = {}
        for account in accounts:
            currency = account.currency if hasattr(account, 'currency') else account.get('currency')
            available = account.available_balance if hasattr(account, 'available_balance') else account.get('available_balance', {})
            if hasattr(available, 'value'):
                balance = float(available.value)
            else:
                balance = float(available.get('value', 0))
            if balance > 0:
                balances[currency] = balance
        
        return {"success": True, "balances": balances}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/execute_trade")
async def execute_trade(trade: TradeRequest):
    """Execute a REAL trade on Coinbase Advanced Trade using SDK"""
    try:
        print(f"EXECUTING REAL TRADE: {trade.symbol} {trade.action} ${trade.size_usd}")
        
        # Convert action to Coinbase side format
        side = "buy" if trade.action.upper() == "BUY" else "sell"
        
        # Place the real order
        order_result = place_coinbase_order(trade.symbol, side, trade.size_usd)
        
        # Check if order was successful
        success = True  # SDK handles errors by raising exceptions
        order_id = None
        
        # Try to extract order ID from response
        if hasattr(order_result, 'order_id'):
            order_id = order_result.order_id
        elif hasattr(order_result, 'id'):
            order_id = order_result.id
        elif isinstance(order_result, dict):
            order_id = order_result.get('order_id') or order_result.get('id')
        else:
            order_id = f"coinbase_{int(time.time())}"
        
        result = {
            "success": success,
            "message": f"REAL {trade.action} order for {trade.symbol} ${trade.size_usd}",
            "order_id": order_id,
            "coinbase_response": str(order_result),
            "executed_amount": trade.size_usd,
            "timestamp": datetime.now().isoformat()
        }
        
        if success:
            print(f"REAL TRADE SUCCESS: {trade.symbol} {trade.action} ${trade.size_usd} - Order ID: {order_id}")
        else:
            print(f"REAL TRADE FAILED: {trade.symbol} {trade.action} ${trade.size_usd}")
            print(f"   Response: {order_result}")
        
        return result
        
    except Exception as e:
        error_msg = str(e)
        print(f"REAL TRADE ERROR: {error_msg}")
        return {"success": False, "error": error_msg}

if __name__ == "__main__":
    print("Starting Coinbase Advanced Trade Service (SDK)")
    uvicorn.run(app, host="0.0.0.0", port=8024)
