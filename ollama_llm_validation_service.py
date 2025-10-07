#!/usr/bin/env python3
"""
Ollama LLM Validation Service
Validates trade recommendations using Ollama LLM models
"""

import os
import logging
import time
import json
import requests
import aiohttp
import asyncio
import mysql.connector
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
import uvicorn
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title='Ollama LLM Validation Service')

# Configuration
OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://ollama:11434')
DEFAULT_MODEL = os.getenv('OLLAMA_DEFAULT_MODEL', 'llama2:7b')

# --- Database Functions ---
def get_db_connection():
    try:
        db_config = {
            'host': os.getenv('DB_HOST', '172.22.32.1'),
            'user': os.getenv('DB_USER', 'news_collector'),
            'password': os.getenv('DB_PASSWORD'),
            'database': os.getenv('DB_NAME_PRICES', 'crypto_prices')
        }
        return mysql.connector.connect(**db_config)
    except Exception as e:
        logger.error(f'Database connection failed: {e}')
        return None

# --- Ollama Functions ---
async def check_ollama_models():
    """Check available Ollama models"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f'{OLLAMA_URL}/api/tags', timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    models = [model['name'] for model in data.get('models', [])]
                    logger.info(f'Available Ollama models: {models}')
                    return models
                else:
                    logger.warning(f'Failed to get Ollama models: {response.status}')
                    return []
    except Exception as e:
        logger.error(f'Error checking Ollama models: {e}')
        return []

async def validate_with_ollama(recommendation, market_context):
    """Validate trade recommendation using Ollama LLM"""
    try:
        # Get available models
        available_models = await check_ollama_models()
        
        if not available_models:
            logger.warning('No Ollama models available, using fallback validation')
            return fallback_validation(recommendation, market_context)
        
        # Use the first available model
        model = available_models[0]
        
        # Create prompt for LLM validation
        symbol = recommendation['symbol']
        signal_type = recommendation['signal_type']
        confidence = recommendation['confidence']
        amount = recommendation['amount_usd']
        
        current_price = market_context.get('current_price', 'N/A')
        volume_24h = market_context.get('volume_24h', 'N/A')
        price_trend = market_context.get('price_trend', 'N/A')
        rsi = market_context.get('rsi', 'N/A')
        sentiment = market_context.get('sentiment', 'N/A')
        vix = market_context.get('vix', 'N/A')
        sentiment_score = market_context.get('sentiment_score', 'N/A')
        
        prompt = f"""
As a cryptocurrency trading expert, analyze this trade recommendation:

SYMBOL: {symbol}
SIGNAL TYPE: {signal_type}
CONFIDENCE: {confidence:.3f}
AMOUNT: ${amount}

MARKET CONTEXT:
- Current Price: ${current_price}
- Volume 24h: {volume_24h}
- Price Trend: {price_trend}
- RSI: {rsi}
- Market Sentiment: {sentiment}
- VIX: {vix}
- Sentiment Score: {sentiment_score}

Please provide a JSON response with:
1. validation: "APPROVE", "REJECT", or "MODIFY"
2. confidence: 0.0 to 1.0
3. reasoning: Brief explanation
4. risk_assessment: "LOW", "MEDIUM", or "HIGH"
5. suggested_amount: Recommended trade amount (if MODIFY)

Respond only with valid JSON:
{{
    "validation": "APPROVE|REJECT|MODIFY",
    "confidence": 0.0-1.0,
    "reasoning": "explanation",
    "risk_assessment": "LOW|MEDIUM|HIGH",
    "suggested_amount": 0.0
}}
"""
        
        # Call Ollama API
        async with aiohttp.ClientSession() as session:
            payload = {
                'model': model,
                'prompt': prompt,
                'stream': False,
                'options': {
                    'temperature': 0.3,
                    'top_p': 0.9,
                    'num_ctx': 2048
                }
            }
            
            async with session.post(f'{OLLAMA_URL}/api/generate', json=payload, timeout=60) as response:
                if response.status == 200:
                    result = await response.json()
                    response_text = result.get('response', '')
                    
                    # Parse JSON response
                    try:
                        # Extract JSON from response
                        json_start = response_text.find('{')
                        json_end = response_text.rfind('}') + 1
                        if json_start != -1 and json_end > json_start:
                            json_text = response_text[json_start:json_end]
                            validation_result = json.loads(json_text)
                            
                            # Validate response format
                            required_fields = ['validation', 'confidence', 'reasoning', 'risk_assessment']
                            if all(field in validation_result for field in required_fields):
                                logger.info(f'Ollama validation for {symbol}: {validation_result["validation"]} (confidence: {validation_result["confidence"]:.3f})')
                                return validation_result
                    
                    except json.JSONDecodeError as e:
                        logger.error(f'Failed to parse Ollama JSON response: {e}')
                        logger.error(f'Response text: {response_text}')
                
                else:
                    logger.error(f'Ollama API error: {response.status}')
        
        # Fallback if Ollama fails
        return fallback_validation(recommendation, market_context)
        
    except Exception as e:
        logger.error(f'Error in Ollama validation: {e}')
        return fallback_validation(recommendation, market_context)

def fallback_validation(recommendation, market_context):
    """Fallback validation using simple rules"""
    symbol = recommendation['symbol']
    signal_type = recommendation['signal_type']
    confidence = recommendation['confidence']
    amount = recommendation['amount_usd']
    
    # Simple validation rules
    validation_score = 0.0
    reasoning_parts = []
    
    # Confidence check
    if confidence > 0.7:
        validation_score += 0.3
        reasoning_parts.append('High confidence signal')
    elif confidence > 0.5:
        validation_score += 0.2
        reasoning_parts.append('Moderate confidence signal')
    else:
        reasoning_parts.append('Low confidence signal')
    
    # RSI check
    rsi = market_context.get('rsi', 50)
    if signal_type == 'BUY' and rsi < 70:
        validation_score += 0.2
        reasoning_parts.append('RSI not overbought')
    elif signal_type == 'SELL' and rsi > 30:
        validation_score += 0.2
        reasoning_parts.append('RSI not oversold')
    
    # Sentiment check
    sentiment = market_context.get('sentiment', 'neutral')
    if sentiment == 'bullish' and signal_type == 'BUY':
        validation_score += 0.2
        reasoning_parts.append('Bullish sentiment supports buy')
    elif sentiment == 'bearish' and signal_type == 'SELL':
        validation_score += 0.2
        reasoning_parts.append('Bearish sentiment supports sell')
    
    # Amount check
    if amount <= 100:
        validation_score += 0.1
        reasoning_parts.append('Reasonable trade size')
    elif amount > 1000:
        validation_score -= 0.2
        reasoning_parts.append('Large trade size - high risk')
    
    # Determine validation result
    if validation_score >= 0.6:
        validation = 'APPROVE'
    elif validation_score >= 0.3:
        validation = 'MODIFY'
        suggested_amount = amount * 0.5
    else:
        validation = 'REJECT'
        suggested_amount = 0.0
    
    # Risk assessment
    if validation_score >= 0.7:
        risk = 'LOW'
    elif validation_score >= 0.4:
        risk = 'MEDIUM'
    else:
        risk = 'HIGH'
    
    return {
        'validation': validation,
        'confidence': min(1.0, max(0.0, validation_score)),
        'reasoning': '; '.join(reasoning_parts),
        'risk_assessment': risk,
        'suggested_amount': suggested_amount if validation == 'MODIFY' else amount
    }

def get_market_context(symbol):
    """Get comprehensive market context for a symbol"""
    conn = get_db_connection()
    if not conn:
        return {}
    
    try:
        cursor = conn.cursor(dictionary=True)
        
        # Get recent price data
        cursor.execute('''
            SELECT current_price, volume_24h, timestamp_iso
            FROM ml_features_materialized 
            WHERE symbol = %s 
            ORDER BY timestamp_iso DESC 
            LIMIT 5
        ''', (symbol,))
        price_data = cursor.fetchall()
        
        # Get technical indicators
        cursor.execute('''
            SELECT rsi, crypto_sentiment, vix, sentiment_score
            FROM ml_features_materialized 
            WHERE symbol = %s 
            ORDER BY timestamp_iso DESC 
            LIMIT 1
        ''', (symbol,))
        indicators = cursor.fetchone()
        
        context = {
            'symbol': symbol,
            'current_price': price_data[0]['current_price'] if price_data else None,
            'volume_24h': price_data[0]['volume_24h'] if price_data else None,
            'price_trend': 'up' if len(price_data) > 1 and price_data[0]['current_price'] > price_data[1]['current_price'] else 'down',
            'rsi': indicators['rsi'] if indicators else None,
            'sentiment': indicators['crypto_sentiment'] if indicators else None,
            'vix': indicators['vix'] if indicators else None,
            'sentiment_score': indicators['sentiment_score'] if indicators else None
        }
        
        return context
    except Exception as e:
        logger.error(f'Error getting market context for {symbol}: {e}')
        return {}
    finally:
        if conn:
            conn.close()

def validate_recommendation(recommendation_id):
    """Validate a specific recommendation"""
    conn = get_db_connection()
    if not conn:
        return None
    
    try:
        cursor = conn.cursor(dictionary=True)
        
        # Get recommendation details
        cursor.execute('''
            SELECT * FROM trade_recommendations 
            WHERE id = %s
        ''', (recommendation_id,))
        recommendation = cursor.fetchone()
        
        if not recommendation:
            return None
        
        # Get market context
        market_context = get_market_context(recommendation['symbol'])
        
        # Validate with Ollama (async)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        validation_result = loop.run_until_complete(validate_with_ollama(recommendation, market_context))
        loop.close()
        
        # Update recommendation with validation results
        cursor.execute('''
            UPDATE trade_recommendations 
            SET llm_validation = %s,
                llm_confidence = %s,
                llm_reasoning = %s,
                risk_assessment = %s,
                suggested_amount = %s,
                validation_timestamp = NOW()
            WHERE id = %s
        ''', (
            validation_result['validation'],
            validation_result['confidence'],
            validation_result['reasoning'],
            validation_result['risk_assessment'],
            validation_result['suggested_amount'],
            recommendation_id
        ))
        
        conn.commit()
        
        logger.info(f'Validated recommendation {recommendation_id}: {validation_result["validation"]} (confidence: {validation_result["confidence"]:.3f})')
        
        return validation_result
        
    except Exception as e:
        logger.error(f'Error validating recommendation {recommendation_id}: {e}')
        return None
    finally:
        if conn:
            conn.close()

def validate_pending_recommendations():
    """Validate all pending recommendations"""
    conn = get_db_connection()
    if not conn:
        return
    
    try:
        cursor = conn.cursor(dictionary=True)
        
        # Get pending recommendations that haven't been validated
        cursor.execute('''
            SELECT id FROM trade_recommendations 
            WHERE execution_status = 'PENDING'
            AND (llm_validation IS NULL OR llm_validation = '')
            AND created_at >= NOW() - INTERVAL 30 MINUTE
            ORDER BY created_at DESC
            LIMIT 5
        ''')
        
        recommendations = cursor.fetchall()
        
        validated_count = 0
        for rec in recommendations:
            result = validate_recommendation(rec['id'])
            if result:
                validated_count += 1
        
        logger.info(f'Validated {validated_count} recommendations with Ollama')
        
    except Exception as e:
        logger.error(f'Error in validation cycle: {e}')
    finally:
        if conn:
            conn.close()

# --- FastAPI Endpoints ---
@app.get('/health')
def health_check():
    return {
        'status': 'healthy',
        'service': 'ollama_llm_validation',
        'ollama_url': OLLAMA_URL,
        'default_model': DEFAULT_MODEL,
        'timestamp': datetime.now().isoformat()
    }

@app.post('/validate/{recommendation_id}')
async def validate_endpoint(recommendation_id: int):
    result = validate_recommendation(recommendation_id)
    if result:
        return result
    else:
        raise HTTPException(status_code=404, detail='Recommendation not found or validation failed')

@app.get('/status')
def get_status():
    conn = get_db_connection()
    if not conn:
        return {'status': 'unhealthy', 'error': 'Database not connected'}
    
    try:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT COUNT(*) FROM trade_recommendations 
            WHERE execution_status = 'PENDING'
            AND (llm_validation IS NULL OR llm_validation = '')
        ''')
        pending_validation = cursor.fetchone()[0]
        
        cursor.execute('''
            SELECT COUNT(*) FROM trade_recommendations 
            WHERE llm_validation = 'APPROVE'
            AND execution_status = 'PENDING'
        ''')
        approved_pending = cursor.fetchone()[0]
        
        return {
            'status': 'healthy',
            'pending_validation': pending_validation,
            'approved_pending': approved_pending,
            'ollama_url': OLLAMA_URL,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        return {'status': 'unhealthy', 'error': str(e)}
    finally:
        if conn:
            conn.close()

@app.on_event('startup')
async def startup_event():
    logger.info('Starting Ollama LLM Validation Service...')
    
    # Check Ollama connection
    available_models = await check_ollama_models()
    if available_models:
        logger.info(f'Connected to Ollama with models: {available_models}')
    else:
        logger.warning('No Ollama models available, will use fallback validation')
    
    # Start validation worker
    def validation_worker():
        while True:
            try:
                validate_pending_recommendations()
                time.sleep(60)  # Validate every minute
            except Exception as e:
                logger.error(f'Error in validation worker: {e}')
                time.sleep(60)
    
    threading.Thread(target=validation_worker, daemon=True).start()
    logger.info('Ollama LLM Validation Service started')

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8050)
