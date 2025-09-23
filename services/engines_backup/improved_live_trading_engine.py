#!/usr/bin/env python3

import pickle
import mysql.connector
import requests
import logging
import time
import json
from datetime import datetime, timedelta
from collections import defaultdict
import math
import sys
import os
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('live_trading_engine.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class OptimizedXGBoostPredictor:
    """Optimized XGBoost model wrapper for trading predictions"""
    
    def __init__(self, timeframe='4h'):
        self.timeframe = timeframe
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.params = None
        
        # Model paths
        self.model_dir = './ml_models_optimized'
        self.model_path = os.path.join(self.model_dir, f'xgboost_optimized_{timeframe}.joblib')
        self.scaler_path = os.path.join(self.model_dir, f'scaler_optimized_{timeframe}.joblib')
        self.params_path = os.path.join(self.model_dir, f'params_optimized_{timeframe}.joblib')
        
    def load_model(self):
        """Load the optimized XGBoost model and components"""
        try:
            # Load model
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                logger.info(f"✅ Loaded XGBoost model: {self.model_path}")
            else:
                logger.error(f"❌ Model file not found: {self.model_path}")
                return False
            
            # Load scaler
            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
                logger.info(f"✅ Loaded scaler: {self.scaler_path}")
            else:
                logger.warning(f"⚠️ Scaler not found: {self.scaler_path}")
            
            # Load parameters
            if os.path.exists(self.params_path):
                self.params = joblib.load(self.params_path)
                logger.info(f"✅ Loaded parameters: {self.params_path}")
            
            logger.info(f"🎯 Using {self.timeframe} timeframe model")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load XGBoost model: {e}")
            return False
    
    def predict_proba(self, X):
        """Make probability predictions"""
        try:
            if self.model is None:
                raise ValueError("Model not loaded")
            
            # Convert to DataFrame if needed
            if isinstance(X, list):
                X = np.array(X)
            if len(X.shape) == 1:
                X = X.reshape(1, -1)
            
            # Scale features if scaler is available
            if self.scaler is not None:
                X_scaled = self.scaler.transform(X)
            else:
                X_scaled = X
            
            # Make prediction
            probabilities = self.model.predict_proba(X_scaled)
            
            return probabilities
            
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}")
            return [[0.5, 0.5]]  # Return neutral prediction on error

class RealTimeDataProvider:
    """Provides real-time market data with enhanced error handling"""
    
    def __init__(self):
        self.coinbase_base_url = "https://api.exchange.coinbase.com"
        self.binance_base_url = "https://api.binance.com/api/v3"
        self.rate_limit_delay = 0.2
        
        # Load symbol mappings from database
        self.symbol_mappings = self._load_symbol_mappings_from_db()
        
        self.last_successful_data = {}
        self.api_error_count = 0
    
    def _load_symbol_mappings_from_db(self):
        """Load cryptocurrency symbol mappings from database with exchange information"""
        try:
            db_config = {
                'host': 'host.docker.internal',
                'user': 'news_collector',
                'password': '99Rules!',
                'database': 'crypto_market_data'
            }
            
            connection = mysql.connector.connect(**db_config)
            cursor = connection.cursor()
            
            # Get all active crypto assets
            query = "SELECT symbol, name, aliases FROM crypto_assets WHERE is_active = 1"
            cursor.execute(query)
            
            symbol_mappings = {}
            
            for symbol, name, aliases_json in cursor.fetchall():
                # Parse aliases if they exist
                aliases = []
                if aliases_json:
                    try:
                        aliases = json.loads(aliases_json) if isinstance(aliases_json, str) else aliases_json
                    except:
                        aliases = []
                
                # Generate exchange mappings based on common patterns
                coinbase_symbol = self._get_coinbase_mapping(symbol, name, aliases)
                binance_symbol = self._get_binance_mapping(symbol, name, aliases)
                
                symbol_mappings[symbol] = {
                    'coinbase': coinbase_symbol,
                    'binance': binance_symbol
                }
            
            cursor.close()
            connection.close()
            
            logger.info(f"📋 Loaded symbol mappings for {len(symbol_mappings)} cryptocurrencies from database")
            return symbol_mappings
            
        except Exception as e:
            logger.error(f"❌ Error loading symbol mappings from database: {e}")
            logger.warning("⚠️ Using fallback hardcoded symbol mappings")
            return self._get_fallback_symbol_mappings()
    
    def _get_coinbase_mapping(self, symbol, name, aliases):
        """Generate Coinbase symbol mapping based on known patterns"""
        # Symbols known to NOT be on Coinbase
        coinbase_blacklist = {'BNB', 'TRX', 'VET', 'FTM', 'XMR', '888', 'APC', 'BRETT', 
                              'BTFD', 'FARTCOIN', 'GOHOME', 'SPX6900', 'BOME'}
        
        if symbol in coinbase_blacklist:
            return None
            
        # Most symbols follow SYMBOL-USD pattern on Coinbase
        return f"{symbol}-USD"
    
    def _get_binance_mapping(self, symbol, name, aliases):
        """Generate Binance symbol mapping based on known patterns"""
        # Symbols known to NOT be on Binance
        binance_blacklist = {'888', 'APC', 'BRETT', 'BTFD', 'FARTCOIN', 'GOHOME', 'SPX6900'}
        
        if symbol in binance_blacklist:
            return None
            
        # Most symbols follow SYMBOLUSDT pattern on Binance
        return f"{symbol}USDT"
    
    def _get_fallback_symbol_mappings(self):
        """Fallback hardcoded symbol mappings if database fails"""
    def _get_fallback_symbol_mappings(self):
        """Fallback hardcoded symbol mappings if database fails"""
        return {
            # Major cryptocurrencies
            'BTC': {'coinbase': 'BTC-USD', 'binance': 'BTCUSDT'},
            'ETH': {'coinbase': 'ETH-USD', 'binance': 'ETHUSDT'},
            'SOL': {'coinbase': 'SOL-USD', 'binance': 'SOLUSDT'},
            'ADA': {'coinbase': 'ADA-USD', 'binance': 'ADAUSDT'},
            'MATIC': {'coinbase': 'MATIC-USD', 'binance': 'MATICUSDT'},
            'DOGE': {'coinbase': 'DOGE-USD', 'binance': 'DOGEUSDT'},
            'DOT': {'coinbase': 'DOT-USD', 'binance': 'DOTUSDT'},
            'AVAX': {'coinbase': 'AVAX-USD', 'binance': 'AVAXUSDT'},
            'LINK': {'coinbase': 'LINK-USD', 'binance': 'LINKUSDT'},
            'UNI': {'coinbase': 'UNI-USD', 'binance': 'UNIUSDT'},
            'LTC': {'coinbase': 'LTC-USD', 'binance': 'LTCUSDT'},
            'ATOM': {'coinbase': 'ATOM-USD', 'binance': 'ATOMUSDT'},
            'XRP': {'coinbase': 'XRP-USD', 'binance': 'XRPUSDT'},
            'BNB': {'coinbase': None, 'binance': 'BNBUSDT'},
            'TRX': {'coinbase': None, 'binance': 'TRXUSDT'},
            'XLM': {'coinbase': 'XLM-USD', 'binance': 'XLMUSDT'},
            'VET': {'coinbase': None, 'binance': 'VETUSDT'},
            'FIL': {'coinbase': 'FIL-USD', 'binance': 'FILUSDT'},
            'ALGO': {'coinbase': 'ALGO-USD', 'binance': 'ALGOUSDT'},
            'SHIB': {'coinbase': 'SHIB-USD', 'binance': 'SHIBUSDT'},
            'NEAR': {'coinbase': 'NEAR-USD', 'binance': 'NEARUSDT'},
            'CRV': {'coinbase': 'CRV-USD', 'binance': 'CRVUSDT'},
            'COMP': {'coinbase': 'COMP-USD', 'binance': 'COMPUSDT'},
            'MKR': {'coinbase': 'MKR-USD', 'binance': 'MKRUSDT'},
            'AAVE': {'coinbase': 'AAVE-USD', 'binance': 'AAVEUSDT'},
            'SUSHI': {'coinbase': 'SUSHI-USD', 'binance': 'SUSHIUSDT'},
            'YFI': {'coinbase': 'YFI-USD', 'binance': 'YFIUSDT'},
            'SUI': {'coinbase': 'SUI-USD', 'binance': 'SUIUSDT'},
            'OP': {'coinbase': 'OP-USD', 'binance': 'OPUSDT'},
            'ARB': {'coinbase': 'ARB-USD', 'binance': 'ARBUSDT'},
            'APT': {'coinbase': 'APT-USD', 'binance': 'APTUSDT'},
            'XTZ': {'coinbase': 'XTZ-USD', 'binance': 'XTZUSDT'},
            'EOS': {'coinbase': 'EOS-USD', 'binance': 'EOSUSDT'},
            'FTM': {'coinbase': None, 'binance': 'FTMUSDT'},
            'XMR': {'coinbase': None, 'binance': 'XMRUSDT'}
        }
        
        self.last_successful_data = {}
        self.api_error_count = 0
    
    def get_coinbase_ticker(self, symbol):
        """Get real-time ticker from Coinbase"""
        try:
            coinbase_symbol = self.symbol_mappings.get(symbol, {}).get('coinbase')
            if not coinbase_symbol:
                return None
                
            url = f"{self.coinbase_base_url}/products/{coinbase_symbol}/ticker"
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            result = {
                'price': float(data['price']),
                'volume_24h': float(data['volume']) * float(data['price']),  # Convert to USD volume
                'timestamp': datetime.now(),
                'source': 'coinbase',
                'bid': float(data.get('bid', data['price'])),
                'ask': float(data.get('ask', data['price']))
            }
            
            # Cache successful data
            self.last_successful_data[symbol] = result
            self.api_error_count = 0
            
            return result
            
        except Exception as e:
            self.api_error_count += 1
            logger.warning(f"Coinbase API error for {symbol}: {e}")
            return None
    
    def get_binance_ticker(self, symbol):
        """Get real-time ticker from Binance"""
        try:
            binance_symbol = self.symbol_mappings.get(symbol, {}).get('binance')
            if not binance_symbol:
                return None
                
            url = f"{self.binance_base_url}/ticker/24hr"
            params = {'symbol': binance_symbol}
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            result = {
                'price': float(data['lastPrice']),
                'volume_24h': float(data['quoteVolume']),
                'timestamp': datetime.now(),
                'source': 'binance',
                'price_change_24h': float(data['priceChangePercent'])
            }
            
            # Cache successful data
            self.last_successful_data[symbol] = result
            self.api_error_count = 0
            
            return result
            
        except Exception as e:
            self.api_error_count += 1
            logger.warning(f"Binance API error for {symbol}: {e}")
            return None
    
    def get_real_time_data(self, symbol):
        """Get real-time data with intelligent fallback"""
        
        # Try Coinbase first
        data = self.get_coinbase_ticker(symbol)
        if data:
            return data
        
        time.sleep(self.rate_limit_delay)
        
        # Try Binance as fallback
        data = self.get_binance_ticker(symbol)
        if data:
            return data
        
        # If both fail, use cached data if available and recent
        cached_data = self.last_successful_data.get(symbol)
        if cached_data:
            time_diff = (datetime.now() - cached_data['timestamp']).total_seconds()
            if time_diff < 300:  # Use cached data if less than 5 minutes old
                logger.warning(f"⚠️ Using cached data for {symbol} ({time_diff:.0f}s old)")
                cached_data['source'] = f"cached_{cached_data['source']}"
                return cached_data
        
        logger.error(f"❌ No real-time data available for {symbol}")
        return None
    
    def get_current_price(self, symbol):
        """Get current price in the format expected by the trading engine"""
        data = self.get_real_time_data(symbol)
        if data:
            return {
                'price': data['price'],
                'volume_24h': data.get('volume_24h'),
                'source': data['source'],
                'timestamp': data['timestamp']
            }
        return None

class ImprovedLiveTradingEngine:
    """Improved live trading engine with optimized XGBoost models"""
    
    def __init__(self, use_real_time=True, symbols=['ETH', 'BTC'], timeframe='4h'):
        self.model = OptimizedXGBoostPredictor(timeframe=timeframe)
        self.db_connection = None
        self.real_time_provider = RealTimeDataProvider() if use_real_time else None
        self.use_real_time = use_real_time
        self.symbols = symbols
        self.timeframe = timeframe
        
        self.signal_history = []
        self.last_signal_times = {}
        
        # Performance tracking
        self.performance_metrics = {
            'total_signals': 0,
            'buy_signals': 0,
            'hold_signals': 0,
            'avg_confidence': 0.0,
            'session_start': datetime.now(),
            'last_signal_time': None,
            'api_errors': 0,
            'db_errors': 0
        }
        
        # Trading configuration based on our optimization results
        self.config = {
            'bull_threshold': 0.65,      # Threshold for bull market (optimized)
            'bear_threshold': 0.75,      # Threshold for bear market (optimized)
            'min_signal_gap_minutes': 30, # Minimum time between signals for same symbol
            'max_daily_signals': 10,     # Risk management
            'feature_timeout_minutes': 60, # Max age for database features
        }
        
        logger.info(f"🎯 Engine initialized for symbols: {self.symbols}")
        logger.info(f"📊 Using {timeframe} timeframe XGBoost model")
    
    def load_model(self):
        """Load the optimized XGBoost model"""
        return self.model.load_model()
    
    def connect_database(self):
        """Connect to MySQL database with retry logic"""
        
        max_retries = 3
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                self.db_connection = mysql.connector.connect(
                    host='host.docker.internal',  # Updated for WSL/Docker compatibility
                    user='news_collector',
                    password='99Rules!',
                    database='crypto_prices',
                    autocommit=True,
                    connection_timeout=10
                )
                
                # Test connection
                cursor = self.db_connection.cursor()
                cursor.execute("SELECT 1")
                cursor.fetchone()
                cursor.close()
                
                logger.info("✅ Database connected successfully")
                return True
                
            except Exception as e:
                logger.warning(f"Database connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
        
        logger.error("❌ Database connection failed after all retries")
        return False
    
    def get_latest_features_from_db(self, symbol, max_age_minutes=60):
        """Get latest enhanced features from database for XGBoost model"""
        
        if not self.db_connection:
            return None
        
        try:
            cursor = self.db_connection.cursor(dictionary=True)
            
            # Get latest materialized features with all columns needed for XGBoost
            query = """
            SELECT *,
                TIMESTAMPDIFF(MINUTE, timestamp_iso, NOW()) as age_minutes
            FROM ml_features_materialized 
            WHERE symbol = %s 
                AND TIMESTAMPDIFF(MINUTE, timestamp_iso, NOW()) < %s
            ORDER BY timestamp_iso DESC 
            LIMIT 1
            """
            
            cursor.execute(query, (symbol, max_age_minutes))
            result = cursor.fetchone()
            cursor.close()
            
            if result:
                logger.info(f"📊 DB features for {symbol}: {result['timestamp_iso']} "
                           f"(${result['current_price']:.2f}, age: {result['age_minutes']}min)")
                return result
            else:
                logger.warning(f"⚠️ No recent features found for {symbol} (within {max_age_minutes}min)")
                return None
                
        except Exception as e:
            self.performance_metrics['db_errors'] += 1
            logger.error(f"Database features error for {symbol}: {e}")
            return None
    
    def create_enhanced_features_for_xgboost(self, symbol, db_features, current_price, volume_24h):
        """Create enhanced feature vector matching XGBoost training data EXACTLY"""
        
        try:
            # Current timestamp
            now = datetime.now()
            
            # Helper function to safely convert to float
            def safe_float(value, default=0):
                try:
                    if value is None:
                        return float(default)
                    # Handle Decimal types from database
                    if hasattr(value, '__float__'):
                        return float(value)
                    return float(value)
                except (ValueError, TypeError):
                    return float(default)
            
            # Initialize all features in the EXACT order expected by the model
            features = {}
            
            # Convert current price safely
            current_price_float = safe_float(current_price, 50000)
            
            # Base features (first group)
            features['id'] = 0  # Placeholder ID
            features['price_hour'] = now.hour * current_price_float / 24
            features['current_price'] = current_price_float
            features['volume_24h'] = safe_float(volume_24h or db_features.get('volume_24h'), 1000000)
            features['hourly_volume'] = safe_float(db_features.get('hourly_volume'), features['volume_24h'] / 24)
            features['market_cap'] = safe_float(db_features.get('market_cap'), current_price_float * 1000000)
            features['price_change_24h'] = safe_float(db_features.get('price_change_24h'), 0)
            features['price_change_percentage_24h'] = safe_float(db_features.get('price_change_percentage_24h'), 0)
            
            # Technical indicators
            features['rsi_14'] = safe_float(db_features.get('rsi_14'), 50)
            features['sma_20'] = safe_float(db_features.get('sma_20'), current_price_float)
            features['sma_50'] = safe_float(db_features.get('sma_50'), current_price_float)
            features['ema_12'] = safe_float(db_features.get('ema_12'), current_price_float)
            features['ema_26'] = safe_float(db_features.get('ema_26'), current_price_float)
            features['macd_line'] = safe_float(db_features.get('macd_line'), 0)
            features['macd_signal'] = safe_float(db_features.get('macd_signal'), 0)
            features['macd_histogram'] = safe_float(db_features.get('macd_histogram'), 0)
            features['bb_upper'] = safe_float(db_features.get('bb_upper'), current_price_float * 1.02)
            features['bb_middle'] = safe_float(db_features.get('bb_middle'), current_price_float)
            features['bb_lower'] = safe_float(db_features.get('bb_lower'), current_price_float * 0.98)
            features['stoch_k'] = safe_float(db_features.get('stoch_k'), 50)
            features['stoch_d'] = safe_float(db_features.get('stoch_d'), 50)
            features['atr_14'] = safe_float(db_features.get('atr_14'), current_price_float * 0.02)
            features['vwap'] = safe_float(db_features.get('vwap'), current_price_float)
            
            # Macro features
            features['vix'] = safe_float(db_features.get('vix'), 20)
            features['spx'] = safe_float(db_features.get('spx'), 4000)
            features['dxy'] = safe_float(db_features.get('dxy'), 100)
            features['fed_funds_rate'] = safe_float(db_features.get('fed_funds_rate'), 5)
            
            # Sentiment features (crypto-specific)
            features['crypto_sentiment_count'] = safe_float(db_features.get('crypto_sentiment_count'), 10)
            features['avg_cryptobert_score'] = safe_float(db_features.get('avg_cryptobert_score'), 0)
            features['avg_vader_score'] = safe_float(db_features.get('avg_vader_score'), 0)
            features['avg_textblob_score'] = safe_float(db_features.get('avg_textblob_score'), 0)
            features['avg_crypto_keywords_score'] = safe_float(db_features.get('avg_crypto_keywords_score'), 0)
            
            # Stock sentiment features
            features['stock_sentiment_count'] = safe_float(db_features.get('stock_sentiment_count'), 10)
            features['avg_finbert_sentiment_score'] = safe_float(db_features.get('avg_finbert_sentiment_score'), 0)
            features['avg_fear_greed_score'] = safe_float(db_features.get('avg_fear_greed_score'), 50)
            features['avg_volatility_sentiment'] = safe_float(db_features.get('avg_volatility_sentiment'), 0)
            features['avg_risk_appetite'] = safe_float(db_features.get('avg_risk_appetite'), 0)
            features['avg_crypto_correlation'] = safe_float(db_features.get('avg_crypto_correlation'), 0)
            features['data_quality_score'] = safe_float(db_features.get('data_quality_score'), 0.8)
            
            # General sentiment features (most important)
            features['general_crypto_sentiment_count'] = safe_float(db_features.get('general_crypto_sentiment_count'), 10)
            features['avg_general_cryptobert_score'] = safe_float(db_features.get('avg_general_cryptobert_score'), 0)
            features['avg_general_vader_score'] = safe_float(db_features.get('avg_general_vader_score'), 0)
            features['avg_general_textblob_score'] = safe_float(db_features.get('avg_general_textblob_score'), 0)
            features['avg_general_crypto_keywords_score'] = safe_float(db_features.get('avg_general_crypto_keywords_score'), 0)
            
            # Social features
            features['social_post_count'] = safe_float(db_features.get('social_post_count'), 100)
            features['social_avg_sentiment'] = safe_float(db_features.get('social_avg_sentiment'), 0)
            features['social_total_engagement'] = safe_float(db_features.get('social_total_engagement'), 1000)
            features['social_unique_authors'] = safe_float(db_features.get('social_unique_authors'), 50)
            features['social_avg_confidence'] = safe_float(db_features.get('social_avg_confidence'), 0.5)
            
            # Time-based features
            features['hour'] = now.hour
            features['day_of_week'] = now.weekday()
            features['month'] = now.month
            features['quarter'] = (now.month - 1) // 3 + 1
            features['is_weekend'] = 1 if now.weekday() >= 5 else 0
            features['is_month_start'] = 1 if now.day <= 3 else 0
            features['is_month_end'] = 1 if now.day >= 28 else 0
            features['is_quarter_end'] = 1 if now.month in [3, 6, 9, 12] and now.day >= 28 else 0
            
            # Market session features
            features['is_us_market_hours'] = 1 if 14 <= now.hour <= 21 else 0
            features['is_asian_session'] = 1 if now.hour >= 23 or now.hour <= 8 else 0
            features['is_european_session'] = 1 if 8 <= now.hour <= 16 else 0
            
            # Enhanced temporal features
            features['hour_sin'] = np.sin(2 * np.pi * now.hour / 24)
            features['hour_cos'] = np.cos(2 * np.pi * now.hour / 24)
            features['day_sin'] = np.sin(2 * np.pi * now.weekday() / 7)
            features['day_cos'] = np.cos(2 * np.pi * now.weekday() / 7)
            features['month_sin'] = np.sin(2 * np.pi * now.month / 12)
            features['month_cos'] = np.cos(2 * np.pi * now.month / 12)
            
            # Moving averages and price ratios
            for window in [5, 10, 20, 50, 100]:
                ma_val = safe_float(db_features.get(f'ma_{window}'), current_price_float)
                features[f'ma_{window}'] = ma_val
                features[f'price_vs_ma_{window}'] = current_price_float / max(ma_val, 1)
                features[f'ma_{window}_slope'] = safe_float(db_features.get(f'ma_{window}_slope'), 0)
            
            # Volatility features
            for window in [5, 10, 20, 50]:
                vol_val = safe_float(db_features.get(f'volatility_{window}'), 0.02)
                features[f'volatility_{window}'] = vol_val
                features[f'volatility_{window}_rank'] = min(vol_val * 10, 1.0)
            
            # Momentum features
            for periods in [1, 3, 5, 10, 20, 50]:
                momentum_val = safe_float(db_features.get(f'momentum_{periods}'), 0)
                features[f'momentum_{periods}'] = momentum_val
                features[f'momentum_{periods}_rank'] = max(0, min(momentum_val + 0.5, 1.0))
            
            # RSI features
            rsi_val = features['rsi_14']
            features['rsi_oversold'] = 1 if rsi_val < 30 else 0
            features['rsi_overbought'] = 1 if rsi_val > 70 else 0
            features['rsi_momentum'] = safe_float(db_features.get('rsi_momentum'), 0)
            
            # Volume features
            for window in [5, 10, 20]:
                features[f'volume_ratio_{window}'] = safe_float(db_features.get(f'volume_ratio_{window}'), 1.0)
            features['volume_price_corr'] = safe_float(db_features.get('volume_price_corr'), 0)
            
            # Support/Resistance features
            features['price_position'] = safe_float(db_features.get('price_position'), 0.5)
            features['near_high'] = safe_float(db_features.get('near_high'), 0)
            features['near_low'] = safe_float(db_features.get('near_low'), 0)
            
            # Bollinger Band features
            features['bb_position'] = safe_float(db_features.get('bb_position'), 0.5)
            features['bb_squeeze'] = safe_float(db_features.get('bb_squeeze'), 0)
            
            # Cross-asset features
            features['btc_correlation'] = safe_float(db_features.get('btc_correlation'), 0.5 if symbol != 'BTC' else 1.0)
            features['btc_beta'] = safe_float(db_features.get('btc_beta'), 1.0)
            features['btc_relative_performance'] = safe_float(db_features.get('btc_relative_performance'), 0)
            
            # Macro interaction features
            vix_val = features['vix']
            spx_val = features['spx']
            features['risk_on_off'] = safe_float(db_features.get('risk_on_off'), 1 if vix_val < 20 and spx_val > 4000 else 0)
            features['dollar_strength'] = safe_float(db_features.get('dollar_strength'), 0)
            features['macro_stress'] = 1 if vix_val > 25 else 0
            
            # Feature interactions
            features['rsi_momentum_interaction'] = features['rsi_14'] * features['momentum_5']
            
            # Create DataFrame with features in exact order
            feature_df = pd.DataFrame([features])
            
            # Ensure we have exactly the features expected by the model
            expected_features = [
                'id', 'price_hour', 'current_price', 'volume_24h', 'hourly_volume', 'market_cap',
                'price_change_24h', 'price_change_percentage_24h', 'rsi_14', 'sma_20', 'sma_50',
                'ema_12', 'ema_26', 'macd_line', 'macd_signal', 'macd_histogram', 'bb_upper',
                'bb_middle', 'bb_lower', 'stoch_k', 'stoch_d', 'atr_14', 'vwap', 'vix', 'spx',
                'dxy', 'fed_funds_rate', 'crypto_sentiment_count', 'avg_cryptobert_score',
                'avg_vader_score', 'avg_textblob_score', 'avg_crypto_keywords_score',
                'stock_sentiment_count', 'avg_finbert_sentiment_score', 'avg_fear_greed_score',
                'avg_volatility_sentiment', 'avg_risk_appetite', 'avg_crypto_correlation',
                'data_quality_score', 'general_crypto_sentiment_count', 'avg_general_cryptobert_score',
                'avg_general_vader_score', 'avg_general_textblob_score', 'avg_general_crypto_keywords_score',
                'social_post_count', 'social_avg_sentiment', 'social_total_engagement',
                'social_unique_authors', 'social_avg_confidence', 'hour', 'day_of_week', 'month',
                'quarter', 'is_weekend', 'is_month_start', 'is_month_end', 'is_quarter_end',
                'is_us_market_hours', 'is_asian_session', 'is_european_session', 'hour_sin',
                'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos', 'ma_5',
                'price_vs_ma_5', 'ma_5_slope', 'ma_10', 'price_vs_ma_10', 'ma_10_slope',
                'ma_20', 'price_vs_ma_20', 'ma_20_slope', 'ma_50', 'price_vs_ma_50',
                'ma_50_slope', 'ma_100', 'price_vs_ma_100', 'ma_100_slope', 'volatility_5',
                'volatility_5_rank', 'volatility_10', 'volatility_10_rank', 'volatility_20',
                'volatility_20_rank', 'volatility_50', 'volatility_50_rank', 'momentum_1',
                'momentum_1_rank', 'momentum_3', 'momentum_3_rank', 'momentum_5', 'momentum_5_rank',
                'momentum_10', 'momentum_10_rank', 'momentum_20', 'momentum_20_rank', 'momentum_50',
                'momentum_50_rank', 'rsi_oversold', 'rsi_overbought', 'rsi_momentum',
                'volume_ratio_5', 'volume_ratio_10', 'volume_ratio_20', 'volume_price_corr',
                'price_position', 'near_high', 'near_low', 'bb_position', 'bb_squeeze',
                'btc_correlation', 'btc_beta', 'btc_relative_performance', 'risk_on_off',
                'dollar_strength', 'macro_stress', 'rsi_momentum_interaction'
            ]
            
            # Reorder columns to match expected order
            feature_df = feature_df[expected_features]
            
            logger.info(f"✅ Created {len(feature_df.columns)} enhanced features for {symbol} (exact model match)")
            return feature_df
            
        except Exception as e:
            logger.error(f"Enhanced feature creation error for {symbol}: {e}")
            return None
    
    def get_latest_raw_price(self, symbol, max_age_minutes=10):
        """Get latest raw price data from price_data table"""
        
        if not self.db_connection:
            return None
        
        try:
            cursor = self.db_connection.cursor(dictionary=True)
            
            query = """
            SELECT current_price, timestamp_iso,
                   TIMESTAMPDIFF(MINUTE, timestamp_iso, NOW()) as age_minutes
            FROM price_data 
            WHERE symbol = %s 
                AND TIMESTAMPDIFF(MINUTE, timestamp_iso, NOW()) < %s
            ORDER BY timestamp_iso DESC 
            LIMIT 1
            """
            
            cursor.execute(query, (symbol, max_age_minutes))
            result = cursor.fetchone()
            cursor.close()
            
            if result:
                logger.info(f"💰 Latest {symbol} price: ${result['current_price']:.2f} "
                           f"(age: {result['age_minutes']}min)")
                return result
            else:
                logger.warning(f"⚠️ No recent price data for {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Raw price query error for {symbol}: {e}")
            return None
    
    def create_hybrid_features(self, symbol):
        """Create features from multiple data sources for XGBoost model"""
        
        # Strategy: Real-time price + Database features
        real_time_data = None
        db_features = None
        raw_price = None
        
        # 1. Try to get real-time price
        real_time_data = None
        if self.use_real_time and self.real_time_provider:
            real_time_data = self.real_time_provider.get_current_price(symbol)
            if not real_time_data:
                self.performance_metrics['api_errors'] += 1
        
        # 2. Get database features (technical indicators, sentiment)
        db_features = self.get_latest_features_from_db(symbol)
        
        # 3. Fallback to latest raw price from database
        if not real_time_data:
            raw_price = self.get_latest_raw_price(symbol)
        
        # 4. Combine data sources
        if not db_features:
            logger.error(f"❌ No database features available for {symbol}")
            return None
        
        # Use real-time price if available, otherwise database price
        current_price = None
        volume_24h = None
        data_source = None
        
        if real_time_data:
            current_price = real_time_data['price']
            volume_24h = real_time_data.get('volume_24h') or db_features.get('volume_24h', 1000000)
            data_source = f"realtime_{real_time_data['source']}"
        elif raw_price:
            current_price = raw_price['current_price']
            volume_24h = raw_price.get('volume_24h') or db_features.get('volume_24h', 1000000)
            data_source = "database_fresh"
        else:
            current_price = db_features['current_price']
            volume_24h = db_features.get('volume_24h', 1000000)
            data_source = "database_cached"
        
        # 5. Create enhanced feature DataFrame for XGBoost
        feature_df = self.create_enhanced_features_for_xgboost(
            symbol=symbol,
            db_features=db_features,
            current_price=current_price,
            volume_24h=volume_24h
        )
        
        return {
            'features': feature_df,
            'current_price': current_price,
            'volume_24h': volume_24h,
            'data_source': data_source,
            'db_features': db_features,
            'real_time_available': real_time_data is not None
        }
    
    def detect_market_regime(self, current_price, db_features):
        """Enhanced market regime detection using XGBoost-style features"""
        
        try:
            sma_20 = float(db_features.get('sma_20') or current_price)
            rsi = float(db_features.get('rsi_14') or 50)
            crypto_sentiment = float(db_features.get('avg_general_cryptobert_score') or 0)
            vix = float(db_features.get('vix') or 20)
            macd = float(db_features.get('macd_line') or 0)
            
            # Multiple regime indicators (enhanced for XGBoost features)
            bull_score = 0
            
            # Price trend
            if current_price > sma_20 * 1.05:
                bull_score += 2
            elif current_price > sma_20:
                bull_score += 1
            
            # Momentum
            if rsi > 55:
                bull_score += 1
            if macd > 0:
                bull_score += 1
            
            # Sentiment (using general sentiment which is most important)
            if crypto_sentiment > 0.1:
                bull_score += 1
            if crypto_sentiment > 0.2:
                bull_score += 1
            
            # Fear index
            if vix < 25:
                bull_score += 1
            
            # Determine regime and threshold (optimized for XGBoost performance)
            if bull_score >= 5:
                regime = 'strong_bull'
                threshold = self.config['bull_threshold'] - 0.05  # More aggressive
            elif bull_score >= 3:
                regime = 'bull'
                threshold = self.config['bull_threshold']
            else:
                regime = 'bear'
                threshold = self.config['bear_threshold']
            
            logger.info(f"🎚️ Market regime: {regime} (score: {bull_score}/7, threshold: {threshold:.2f})")
            return regime, threshold
            
        except Exception as e:
            logger.error(f"Regime detection error: {e}")
            return 'neutral', self.config['bull_threshold']
    
    def should_generate_signal(self, symbol):
        """Check if we should generate a signal (rate limiting)"""
        
        last_signal_time = self.last_signal_times.get(symbol)
        if last_signal_time:
            time_diff = (datetime.now() - last_signal_time).total_seconds() / 60
            if time_diff < self.config['min_signal_gap_minutes']:
                logger.info(f"⏭️ Skipping {symbol} - last signal {time_diff:.1f}min ago")
                return False
        
        return True
    
    def generate_signal(self, symbol):
        """Generate live trading signal for symbol using optimized XGBoost model"""
        
        if not self.model.model:
            logger.error("❌ XGBoost model not loaded")
            return None
        
        if not self.should_generate_signal(symbol):
            return None
        
        # Get hybrid data with enhanced features
        data = self.create_hybrid_features(symbol)
        if not data or data['features'] is None:
            logger.error(f"❌ Could not create features for {symbol}")
            return None
        
        try:
            # Make prediction using XGBoost model
            prediction_proba = self.model.predict_proba(data['features'])[0]
            confidence = prediction_proba[1]  # Probability of UP movement
            
            # Detect market regime
            regime, threshold = self.detect_market_regime(data['current_price'], data['db_features'])
            
            # Generate signal based on optimized thresholds
            signal_type = 'BUY' if confidence >= threshold else 'HOLD'
            
            # Create enhanced signal object
            signal = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'price': data['current_price'],
                'signal_type': signal_type,
                'confidence': confidence,
                'threshold': threshold,
                'regime': regime,
                'data_source': data['data_source'],
                'real_time_available': data['real_time_available'],
                'volume_24h': data['volume_24h'],
                'rsi': data['db_features'].get('rsi_14') or 0,
                'crypto_sentiment': data['db_features'].get('avg_general_cryptobert_score') or 0,
                'vix': data['db_features'].get('vix') or 0,
                'model_type': f'xgboost_{self.timeframe}',
                'macd': data['db_features'].get('macd_line') or 0,
                'general_sentiment': data['db_features'].get('avg_general_vader_score') or 0
            }
            
            # Update last signal time
            self.last_signal_times[symbol] = datetime.now()
            
            return signal
            
        except Exception as e:
            logger.error(f"XGBoost signal generation error for {symbol}: {e}")
            return None
    
    def save_signal_to_database(self, signal):
        """Save trading signal to database with enhanced XGBoost schema"""
        
        if not self.db_connection or not signal:
            return False
        
        try:
            cursor = self.db_connection.cursor()
            
            # Insert signal into unified trading_signals table
            insert_query = """
            INSERT INTO trading_signals 
            (timestamp, symbol, price, signal_type, confidence, threshold, regime, model,
             data_source, real_time_available, volume_24h, rsi, crypto_sentiment, vix)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            cursor.execute(insert_query, (
                signal['timestamp'],
                signal['symbol'],
                float(signal['price']),  # Convert to Python float
                signal['signal_type'],
                float(signal['confidence']),  # Convert to Python float
                float(signal['threshold']),  # Convert to Python float
                signal['regime'],
                f'xgboost_{self.timeframe}',  # Model identifier
                signal['data_source'],
                signal['real_time_available'],
                float(signal['volume_24h']) if signal['volume_24h'] is not None else None,
                float(signal['rsi']) if signal['rsi'] is not None else None,
                float(signal['crypto_sentiment']) if signal['crypto_sentiment'] is not None else None,
                float(signal['vix']) if signal['vix'] is not None else None
            ))
            
            cursor.close()
            logger.info(f"💾 XGBoost signal saved: {signal['signal_type']} {signal['symbol']} @ ${signal['price']:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Database save error: {e}")
            return False
    
    def log_signal(self, signal):
        """Enhanced signal logging for XGBoost model"""
        
        emoji = "🚀" if signal['signal_type'] == 'BUY' else "⏸️"
        rt_indicator = "📡" if signal['real_time_available'] else "📊"
        model_indicator = "🧠"
        
        logger.info(f"{emoji} {rt_indicator} {model_indicator} {signal['signal_type']}: {signal['symbol']} @ ${signal['price']:.2f}")
        logger.info(f"   💪 Confidence: {signal['confidence']:.3f} (threshold: {signal['threshold']:.2f})")
        logger.info(f"   🎚️ Regime: {signal['regime']}")
        logger.info(f"   📊 RSI: {signal['rsi']:.1f}, Sentiment: {signal['crypto_sentiment']:.3f}, VIX: {signal['vix']:.1f}")
        logger.info(f"   🧠 Model: {signal.get('model_type', 'xgboost')} ({self.timeframe} timeframe)")
        logger.info(f"   🔗 Source: {signal['data_source']}")
        
        if signal['signal_type'] == 'BUY':
            logger.info("🚨 *** XGBOOST BUY SIGNAL DETECTED ***")
            logger.info(f"   💰 Consider position: ${signal['price']:.2f}")
            logger.info(f"   📈 Confidence: {signal['confidence']*100:.1f}%")
            logger.info(f"   🎯 Optimized {self.timeframe} model prediction")
            
    def run_live_engine(self, interval_seconds=60):
        """Run the continuous live trading engine"""
        
        logger.info("🚀 STARTING IMPROVED LIVE TRADING ENGINE")
        logger.info("=" * 60)
        logger.info(f"🎯 Symbols: {self.symbols}")
        logger.info(f"⏱️ Interval: {interval_seconds}s")
        logger.info(f"📡 Real-time APIs: {'ENABLED' if self.use_real_time else 'DISABLED'}")
        logger.info(f"🎚️ Bull threshold: {self.config['bull_threshold']:.2f}")
        logger.info(f"🎚️ Bear threshold: {self.config['bear_threshold']:.2f}")
        logger.info("=" * 60)
        
        cycle_count = 0
        
        try:
            while True:
                cycle_count += 1
                cycle_start = datetime.now()
                
                logger.info(f"🔄 Cycle {cycle_count} starting...")
                
                signals_generated = 0
                
                for symbol in self.symbols:
                    try:
                        # Generate signal
                        signal = self.generate_signal(symbol)
                        
                        if signal:
                            signals_generated += 1
                            
                            # Log the signal
                            self.log_signal(signal)
                            
                            # Save to database
                            self.save_signal_to_database(signal)
                            
                            # Update performance metrics
                            self.performance_metrics['total_signals'] += 1
                            if signal['signal_type'] == 'BUY':
                                self.performance_metrics['buy_signals'] += 1
                            else:
                                self.performance_metrics['hold_signals'] += 1
                            
                            self.performance_metrics['last_signal_time'] = signal['timestamp']
                            
                            # Add to history
                            self.signal_history.append(signal)
                            if len(self.signal_history) > 1000:  # Keep last 1000 signals
                                self.signal_history = self.signal_history[-1000:]
                        
                        # Small delay between symbols
                        time.sleep(1)
                        
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {e}")
                
                # Cycle summary
                cycle_duration = (datetime.now() - cycle_start).total_seconds()
                logger.info(f"✅ Cycle {cycle_count} complete: {signals_generated} signals in {cycle_duration:.1f}s")
                
                # Performance summary every 10 cycles
                if cycle_count % 10 == 0:
                    self.log_performance_summary()
                
                # Wait for next cycle
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            logger.info("\n⏹️ Live trading engine stopped by user")
        except Exception as e:
            logger.error(f"❌ Critical engine error: {e}")
        finally:
            self.cleanup()
    
    def log_performance_summary(self):
        """Log performance metrics"""
        
        runtime = datetime.now() - self.performance_metrics['session_start']
        runtime_hours = runtime.total_seconds() / 3600
        
        logger.info("📊 PERFORMANCE SUMMARY")
        logger.info("=" * 40)
        logger.info(f"⏱️ Runtime: {runtime_hours:.1f} hours")
        logger.info(f"🎯 Total signals: {self.performance_metrics['total_signals']}")
        logger.info(f"🚀 Buy signals: {self.performance_metrics['buy_signals']}")
        logger.info(f"⏸️ Hold signals: {self.performance_metrics['hold_signals']}")
        
        if self.performance_metrics['total_signals'] > 0:
            buy_rate = self.performance_metrics['buy_signals'] / self.performance_metrics['total_signals'] * 100
            logger.info(f"📈 Buy rate: {buy_rate:.1f}%")
        
        logger.info(f"❌ API errors: {self.performance_metrics['api_errors']}")
        logger.info(f"❌ DB errors: {self.performance_metrics['db_errors']}")
        logger.info("=" * 40)
    
    def cleanup(self):
        """Cleanup resources"""
        
        if self.db_connection:
            try:
                self.db_connection.close()
                logger.info("📊 Database connection closed")
            except:
                pass
        
        self.log_performance_summary()

def get_active_crypto_symbols():
    """Get list of active cryptocurrency symbols from database"""
    try:
        db_config = {
            'host': 'host.docker.internal',
            'user': 'news_collector',
            'password': '99Rules!',
            'database': 'crypto_market_data'
        }
        
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()
        
        query = "SELECT symbol FROM crypto_assets WHERE is_active = 1 ORDER BY symbol"
        cursor.execute(query)
        
        symbols = [row[0] for row in cursor.fetchall()]
        logger.info(f"📋 Loaded {len(symbols)} active crypto symbols from database")
        
        cursor.close()
        connection.close()
        
        return symbols
        
    except Exception as e:
        logger.error(f"❌ Error fetching symbols from database: {e}")
        # Fallback to hardcoded list
        fallback_symbols = [
            'AAVE', 'ADA', 'ALGO', 'APT', 'ARB', 'ATOM', 'AVAX', 'BNB', 'BTC', 'COMP', 
            'CRV', 'DOGE', 'DOT', 'EOS', 'ETH', 'FIL', 'FTM', 'LINK', 'LTC', 'MATIC', 
            'MKR', 'NEAR', 'OP', 'SHIB', 'SOL', 'SUI', 'SUSHI', 'TRX', 'UNI', 'VET', 
            'XLM', 'XMR', 'XRP', 'XTZ', 'YFI'
        ]
        logger.warning(f"⚠️ Using fallback symbol list with {len(fallback_symbols)} symbols")
        return fallback_symbols

def main():
    """Main function - Updated for XGBoost optimized models"""
    
    # Configuration - Using optimized XGBoost settings
    USE_REAL_TIME = True
    TIMEFRAME = '4h'  # Best performing model from optimization (53.51% accuracy, +2.46% improvement)
    SYMBOLS = get_active_crypto_symbols()
    INTERVAL = 300  # 5 minutes between signal generation cycles (optimized for 4h model)
    
    logger.info("🚀 STARTING OPTIMIZED XGBOOST LIVE TRADING ENGINE")
    logger.info("=" * 70)
    logger.info(f"🧠 Model: XGBoost {TIMEFRAME} timeframe (Best performer: +2.46% vs baseline)")
    logger.info(f"🎯 Symbols: {len(SYMBOLS)} cryptocurrencies")
    logger.info(f"⏱️ Interval: {INTERVAL}s")
    logger.info(f"📡 Real-time APIs: {'ENABLED' if USE_REAL_TIME else 'DISABLED'}")
    logger.info("=" * 70)
    
    # Initialize engine with XGBoost model
    engine = ImprovedLiveTradingEngine(
        use_real_time=USE_REAL_TIME,
        symbols=SYMBOLS,
        timeframe=TIMEFRAME
    )
    
    # Load optimized XGBoost model
    if not engine.load_model():
        logger.error("❌ Failed to load XGBoost model - exiting")
        return
    
    # Connect to database
    if not engine.connect_database():
        logger.error("❌ Failed to connect to database - exiting")
        return
    
    logger.info("✅ Optimized XGBoost live trading engine ready!")
    logger.info(f"🎯 Using optimized {TIMEFRAME} XGBoost model with enhanced features")
    logger.info("🚀 Key features: General sentiment, temporal patterns, macro indicators")
    
    # Start the engine
    engine.run_live_engine(interval_seconds=INTERVAL)

if __name__ == "__main__":
    main()
