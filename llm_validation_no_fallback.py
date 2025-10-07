#!/usr/bin/env python3
"""
Ollama LLM Validation Service - NO FALLBACK MODE
Real AI validation only - no fallback rules
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
app = FastAPI(title='Ollama LLM Validation Service - NO FALLBACK')

# Configuration
OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://ollama:11434')
DEFAULT_MODEL = os.getenv('OLLAMA_DEFAULT_MODEL', 'tinyllama:1.1b')

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
        raise Exception(f'Database connection failed: {e}')

def log_llm_validation_attempt(recommendation_id, symbol, signal_type, confidence, amount_usd, 
                              market_context, prompt, response, error=None, processing_time=None):
    """Log LLM validation attempt with full details"""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        
        # Create validation log entry
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS llm_validation_logs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                recommendation_id INT,
                symbol VARCHAR(10),
                signal_type VARCHAR(10),
                confidence DECIMAL(5,3),
                amount_usd DECIMAL(15,8),
                market_context JSON,
                prompt TEXT,
                response TEXT,
                error TEXT,
                processing_time DECIMAL(8,3),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_recommendation_id (recommendation_id),
                INDEX idx_symbol (symbol),
                INDEX idx_timestamp (timestamp)
            )
        ''')
        
        cursor.execute('''
            INSERT INTO llm_validation_logs 
            (recommendation_id, symbol, signal_type, confidence, amount_usd, 
             market_context, prompt, response, error, processing_time, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ''', (
            recommendation_id, symbol, signal_type, confidence, amount_usd,
            json.dumps(market_context), prompt, response, error, processing_time, datetime.now()
        ))
        
        conn.commit()
        logger.info(f'Logged LLM validation attempt for recommendation {recommendation_id}')
        
    except Exception as e:
        logger.error(f'Error logging LLM validation attempt: {e}')
        conn.rollback()
    finally:
        if conn:
            conn.close()

# --- Ollama Functions ---
async def check_ollama_models():
    """Check available Ollama models - NO FALLBACK"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f'{OLLAMA_URL}/api/tags', timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    models = [model['name'] for model in data.get('models', [])]
                    logger.info(f'Available Ollama models: {models}')
                    return models
                else:
                    logger.error(f'Failed to get Ollama models: {response.status}')
                    raise Exception(f'Ollama API error: {response.status}')
    except Exception as e:
        logger.error(f'Error checking Ollama models: {e}')
        raise Exception(f'Cannot connect to Ollama: {e}')

async def validate_with_ollama(recommendation, market_context):
    """Validate trade recommendation using Ollama LLM - NO FALLBACK"""
    try:
        # Get available models - MUST WORK
        available_models = await check_ollama_models()
        
        if not available_models:
            raise Exception('No Ollama models available - system must fail')
        
        # Use the first available model
        model = available_models[0]
        logger.info(f'Using Ollama model: {model}')
        
        # Create comprehensive prompt for LLM validation
        symbol = recommendation['symbol']
        signal_type = recommendation['signal_type']
        confidence = recommendation['confidence']
        amount = recommendation['amount_usd']
        
        current_price = market_context.get('current_price', 'N/A')
        volume_24h = market_context.get('volume_24h', 'N/A')
        rsi = market_context.get('rsi', 'N/A')
        sentiment = market_context.get('sentiment', 'N/A')
        vix = market_context.get('vix', 'N/A')
        sentiment_score = market_context.get('sentiment_score', 'N/A')
        
        prompt = f'''You are an expert cryptocurrency trading analyst with deep knowledge of market dynamics, technical analysis, and risk management. Analyze this trade recommendation comprehensively.

TRADE RECOMMENDATION:
- Symbol: {symbol}
- Signal Type: {signal_type}
- Confidence: {confidence:.3f}
- Amount: ${amount}

MARKET CONTEXT:
- Current Price: ${current_price}
- Volume 24h: {volume_24h}
- RSI: {rsi}
- Market Sentiment: {sentiment}
- VIX: {vix}
- Sentiment Score: {sentiment_score}

ANALYSIS REQUIREMENTS:
1. Evaluate the technical indicators and their reliability
2. Assess the risk-reward ratio based on current market conditions
3. Consider market volatility and overall sentiment
4. Analyze the confidence level appropriateness
5. Evaluate the trade size relative to market conditions and volatility
6. Consider correlation with broader market trends

RESPONSE FORMAT (JSON only):
{{
    "validation": "APPROVE|REJECT|MODIFY",
    "confidence": 0.0-1.0,
    "reasoning": "Detailed analysis of market conditions, technical indicators, risk assessment, and trade appropriateness",
    "risk_assessment": "LOW|MEDIUM|HIGH",
    "suggested_amount": 0.0,
    "market_analysis": "Analysis of current market conditions and trends",
    "technical_analysis": "Analysis of technical indicators and signals"
}}

Provide only the JSON response with no additional text.'''
        
        # Call Ollama API - NO FALLBACK
        async with aiohttp.ClientSession() as session:
            payload = {
                'model': model,
                'prompt': prompt,
                'stream': False,
                'options': {
                    'temperature': 0.2,
                    'top_p': 0.9,
                    'num_ctx': 4096
                }
            }
            
            async with session.post(f'{OLLAMA_URL}/api/generate', json=payload, timeout=120) as response:
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
                            required_fields = ['validation', 'confidence', 'reasoning']
                            if all(field in validation_result for field in required_fields):
                                # Provide defaults for optional fields
                                validation_result.setdefault('risk_assessment', 'MEDIUM')
                                validation_result.setdefault('suggested_amount', amount)
                                
                                logger.info(f'Ollama validation for {symbol}: {validation_result["validation"]} (confidence: {validation_result["confidence"]:.3f})')
                                return validation_result
                            else:
                                raise Exception(f'Invalid LLM response format: missing required fields: {required_fields}')
                    except json.JSONDecodeError as e:
                        logger.error(f'Failed to parse Ollama JSON response: {e}')
                        logger.error(f'Response text: {response_text}')
                        raise Exception(f'Invalid LLM response format: {e}')
                
                else:
                    logger.error(f'Ollama API error: {response.status}')
                    raise Exception(f'Ollama API error: {response.status}')
        
    except Exception as e:
        logger.error(f'Error in Ollama validation: {e}')
        raise Exception(f'LLM validation failed: {e}')

def get_market_context(symbol):
    """Get comprehensive market context for a symbol"""
    conn = get_db_connection()
    try:
        cursor = conn.cursor(dictionary=True)
        
        # Get recent price data and technical indicators
        cursor.execute('''
            SELECT current_price, volume_24h, rsi_14, avg_cryptobert_score, vix_index, sentiment_fear_greed_index
            FROM ml_features_materialized 
            WHERE symbol = %s 
            ORDER BY timestamp_iso DESC 
            LIMIT 1
        ''', (symbol,))
        data = cursor.fetchone()
        
        if data:
            return {
                'symbol': symbol,
                'current_price': data['current_price'],
                'volume_24h': data['volume_24h'],
                'rsi': data['rsi_14'],
                'sentiment': data['avg_cryptobert_score'],
                'vix': data['vix_index'],
                'sentiment_score': data['sentiment_fear_greed_index']
            }
        else:
            raise Exception(f'No market data found for {symbol}')
    except Exception as e:
        logger.error(f'Error getting market context for {symbol}: {e}')
        raise Exception(f'Cannot get market context for {symbol}: {e}')
    finally:
        if conn:
            conn.close()

def validate_recommendation(recommendation_id):
    """Validate a specific recommendation using real LLM - NO FALLBACK"""
    conn = get_db_connection()
    try:
        cursor = conn.cursor(dictionary=True)
        
        # Get recommendation details
        cursor.execute('''
            SELECT * FROM trade_recommendations 
            WHERE id = %s
        ''', (recommendation_id,))
        recommendation = cursor.fetchone()
        
        if not recommendation:
            raise Exception(f'Recommendation {recommendation_id} not found')
        
        # Get market context
        market_context = get_market_context(recommendation['symbol'])
        
        # Validate with Ollama - NO FALLBACK
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
            validation_result.get('suggested_amount', recommendation['amount_usd']),
            recommendation_id
        ))
        
        conn.commit()
        
        logger.info(f'Validated recommendation {recommendation_id}: {validation_result["validation"]} (confidence: {validation_result["confidence"]:.3f})')
        
        return validation_result
        
    except Exception as e:
        logger.error(f'Error validating recommendation {recommendation_id}: {e}')
        raise Exception(f'LLM validation failed for recommendation {recommendation_id}: {e}')
    finally:
        if conn:
            conn.close()

def validate_pending_recommendations():
    """Validate all pending recommendations using real LLM - NO FALLBACK"""
    conn = get_db_connection()
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
            try:
                result = validate_recommendation(rec['id'])
                if result:
                    validated_count += 1
            except Exception as e:
                logger.error(f'Failed to validate recommendation {rec["id"]}: {e}')
                # Continue with other recommendations
        
        logger.info(f'Validated {validated_count} recommendations with Ollama LLM')
        
    except Exception as e:
        logger.error(f'Error in validation cycle: {e}')
        raise Exception(f'LLM validation cycle failed: {e}')
    finally:
        if conn:
            conn.close()

# --- FastAPI Endpoints ---
@app.get('/health')
def health_check():
    return {
        'status': 'healthy',
        'service': 'ollama_llm_validation_no_fallback',
        'ollama_url': OLLAMA_URL,
        'default_model': DEFAULT_MODEL,
        'timestamp': datetime.now().isoformat()
    }

@app.post('/validate/{recommendation_id}')
async def validate_endpoint(recommendation_id: int):
    try:
        result = validate_recommendation(recommendation_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'LLM validation failed: {str(e)}')

@app.get('/status')
def get_status():
    conn = get_db_connection()
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
            'model': DEFAULT_MODEL,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        return {'status': 'unhealthy', 'error': str(e)}
    finally:
        if conn:
            conn.close()

@app.on_event('startup')
async def startup_event():
    logger.info('Starting Ollama LLM Validation Service - NO FALLBACK MODE...')
    
    # Check Ollama connection - MUST WORK
    try:
        available_models = await check_ollama_models()
        if available_models:
            logger.info(f'Connected to Ollama with models: {available_models}')
        else:
            raise Exception('No Ollama models available - system cannot start')
    except Exception as e:
        logger.error(f'Failed to connect to Ollama: {e}')
        raise Exception(f'Cannot start without Ollama: {e}')
    
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
    logger.info('Ollama LLM Validation Service started - NO FALLBACK MODE')

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8050)
