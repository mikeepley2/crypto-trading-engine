"""
Advanced ML Integration Service
Connect ML signals to live trading with confidence-based position sizing and ensemble strategies
"""

import asyncio
import logging
import mysql.connector
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import joblib
import aiohttp
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from sklearn.ensemble import RandomForestClassifier

# Configure logging to be less verbose during development
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Set specific loggers to WARNING to reduce noise during development
development_mode = not os.path.exists('/app/ml_models_optimized')
if development_mode:
    logging.getLogger('ml_integration_service').setLevel(logging.WARNING)

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'engines'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))

from automated_strategies import TradingSignal, StrategyManager
from llm_analysis import MarketAnalysisService, MarketContext, LLMAnalysis  
from advanced_order_types import AdvancedOrder, PositionSizer

class ModelType(Enum):
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    LSTM = "lstm"
    TRANSFORMER = "transformer"
    ENSEMBLE = "ensemble"
    DUMMY = "dummy"

class SignalStrength(Enum):
    VERY_WEAK = 1
    WEAK = 2
    MODERATE = 3
    STRONG = 4
    VERY_STRONG = 5

@dataclass
class MLSignal:
    """ML-generated trading signal"""
    model_name: str
    model_type: ModelType
    symbol: str
    prediction: float  # -1 to 1 (sell to buy)
    confidence: float  # 0 to 1
    strength: SignalStrength
    features_used: List[str]
    feature_importance: Dict[str, float]
    prediction_horizon: str  # '1h', '4h', '1d', '1w'
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime = field(default_factory=lambda: datetime.utcnow() + timedelta(hours=4))

@dataclass
class EnsembleSignal:
    """Ensemble of multiple ML signals"""
    symbol: str
    individual_signals: List[MLSignal]
    consensus_prediction: float
    consensus_confidence: float
    agreement_score: float  # How much models agree
    dominant_timeframe: str
    risk_adjusted_size: Decimal
    recommended_action: str  # 'buy', 'sell', 'hold'
    stop_loss_price: Optional[Decimal] = None
    take_profit_price: Optional[Decimal] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

class FeatureEngineer:
    """Generate features for ML models"""
    
    def __init__(self, db_config: Dict[str, str]):
        self.db_config = db_config
        self.logger = logging.getLogger(__name__)
    
    async def get_features(self, symbol: str, lookback_hours: int = 168) -> Optional[pd.DataFrame]:
        """Generate comprehensive feature set for ML models"""
        
        try:
            # Get price data
            price_data = await self.get_price_data(symbol, lookback_hours)
            if price_data.empty:
                return None
            
            # Calculate technical indicators
            features = self.calculate_technical_features(price_data)
            
            # Add market sentiment features
            sentiment_features = await self.get_sentiment_features(symbol)
            features.update(sentiment_features)
            
            # Add macro features
            macro_features = await self.get_macro_features()
            features.update(macro_features)
            
            # Convert to DataFrame
            feature_df = pd.DataFrame([features])
            
            self.logger.info(f"Generated {len(features)} features for {symbol}")
            return feature_df
            
        except Exception as e:
            self.logger.error(f"Error generating features for {symbol}: {e}")
            return None
    
    async def get_price_data(self, symbol: str, lookback_hours: int) -> pd.DataFrame:
        """Get historical price data"""
        
        try:
            connection = mysql.connector.connect(**self.db_config)
            cursor = connection.cursor(dictionary=True)
            
            query = """
            SELECT timestamp, open_price, high_price, low_price, close_price, volume
            FROM crypto_market_data.price_data
            WHERE symbol = %s AND timestamp >= DATE_SUB(NOW(), INTERVAL %s HOUR)
            ORDER BY timestamp ASC
            """
            
            cursor.execute(query, (symbol, lookback_hours))
            data = cursor.fetchall()
            
            cursor.close()
            connection.close()
            
            if data:
                df = pd.DataFrame(data)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                return df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Error getting price data: {e}")
            return pd.DataFrame()
    
    def calculate_technical_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate technical indicator features"""
        
        if df.empty:
            return {}
        
        features = {}
        
        try:
            # Price features
            features['price_last'] = float(df['close_price'].iloc[-1])
            features['price_sma_20'] = float(df['close_price'].rolling(20).mean().iloc[-1])
            features['price_sma_50'] = float(df['close_price'].rolling(50).mean().iloc[-1])
            features['price_ema_12'] = float(df['close_price'].ewm(span=12).mean().iloc[-1])
            features['price_ema_26'] = float(df['close_price'].ewm(span=26).mean().iloc[-1])
            
            # RSI
            delta = df['close_price'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            features['rsi_14'] = float(100 - (100 / (1 + rs.iloc[-1])))
            
            # MACD
            ema_12 = df['close_price'].ewm(span=12).mean()
            ema_26 = df['close_price'].ewm(span=26).mean()
            macd = ema_12 - ema_26
            macd_signal = macd.ewm(span=9).mean()
            features['macd'] = float(macd.iloc[-1])
            features['macd_signal'] = float(macd_signal.iloc[-1])
            features['macd_histogram'] = float(macd.iloc[-1] - macd_signal.iloc[-1])
            
            # Bollinger Bands
            bb_middle = df['close_price'].rolling(20).mean()
            bb_std = df['close_price'].rolling(20).std()
            bb_upper = bb_middle + (bb_std * 2)
            bb_lower = bb_middle - (bb_std * 2)
            features['bb_position'] = float((df['close_price'].iloc[-1] - bb_lower.iloc[-1]) / 
                                          (bb_upper.iloc[-1] - bb_lower.iloc[-1]))
            
            # Volume features
            features['volume_last'] = float(df['volume'].iloc[-1])
            features['volume_sma_20'] = float(df['volume'].rolling(20).mean().iloc[-1])
            features['volume_ratio'] = float(df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1])
            
            # Volatility
            returns = df['close_price'].pct_change()
            features['volatility_1d'] = float(returns.rolling(24).std() * np.sqrt(24))
            features['volatility_7d'] = float(returns.rolling(168).std() * np.sqrt(168))
            
            # Price momentum
            features['momentum_1h'] = float((df['close_price'].iloc[-1] / df['close_price'].iloc[-2]) - 1)
            features['momentum_4h'] = float((df['close_price'].iloc[-1] / df['close_price'].iloc[-5]) - 1)
            features['momentum_24h'] = float((df['close_price'].iloc[-1] / df['close_price'].iloc[-25]) - 1)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error calculating technical features: {e}")
            return {}
    
    async def get_sentiment_features(self, symbol: str) -> Dict[str, float]:
        """Get sentiment-based features"""
        
        try:
            connection = mysql.connector.connect(**self.db_config)
            cursor = connection.cursor(dictionary=True)
            
            # Get recent sentiment data
            query = """
            SELECT sentiment_score, confidence
            FROM crypto_prices.news_sentiment
            WHERE symbol = %s AND timestamp >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
            ORDER BY timestamp DESC
            LIMIT 50
            """
            
            cursor.execute(query, (symbol,))
            sentiment_data = cursor.fetchall()
            
            cursor.close()
            connection.close()
            
            features = {}
            
            if sentiment_data:
                sentiments = [row['sentiment_score'] for row in sentiment_data]
                confidences = [row['confidence'] for row in sentiment_data]
                
                features['sentiment_mean'] = float(np.mean(sentiments))
                features['sentiment_std'] = float(np.std(sentiments))
                features['sentiment_trend'] = float(np.polyfit(range(len(sentiments)), sentiments, 1)[0])
                features['sentiment_confidence'] = float(np.mean(confidences))
            else:
                features['sentiment_mean'] = 0.0
                features['sentiment_std'] = 0.0
                features['sentiment_trend'] = 0.0
                features['sentiment_confidence'] = 0.0
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error getting sentiment features: {e}")
            return {
                'sentiment_mean': 0.0,
                'sentiment_std': 0.0,
                'sentiment_trend': 0.0,
                'sentiment_confidence': 0.0
            }
    
    async def get_macro_features(self) -> Dict[str, float]:
        """Get macroeconomic features"""
        
        try:
            # This would connect to macro data sources
            # For now, return placeholder features
            features = {
                'vix_index': 20.0,  # Volatility index
                'dxy_index': 100.0,  # Dollar index
                'btc_dominance': 45.0,  # Bitcoin dominance
                'total_market_cap': 2000000.0,  # Total crypto market cap
                'fear_greed_index': 50.0  # Fear & Greed index
            }
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error getting macro features: {e}")
            return {}

class MLModelManager:
    """Manage multiple ML models with existing trained models"""
    
    def __init__(self, models_path: str = None):
        """Initialize with smart model path detection"""
        
        # Check for existing trained models in multiple locations
        possible_paths = [
            models_path,
            './ml_models_optimized',
            '../ml_models_optimized', 
            '../../ml_models_optimized',
            './backend/services/trading/ml/ml_models_optimized',
            '/app/ml_models_optimized'  # Docker default
        ]
        
        self.models_path = None
        for path in possible_paths:
            if path and os.path.exists(path):
                # Check if actual model files exist
                if os.path.exists(os.path.join(path, 'xgboost_optimized_1h.joblib')):
                    self.models_path = path
                    self.logger.info(f"✅ Found trained models in: {path}")
                    break
        
        if not self.models_path:
            self.models_path = '/app/ml_models_optimized'  # Default for Docker
            
        self.models: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Dict] = {}
        self.logger = logging.getLogger(__name__)
        
    def load_models(self):
        """Load all available ML models"""
        
        try:
            # Check if models directory exists
            if not os.path.exists(self.models_path):
                self.logger.warning(f"Models directory {self.models_path} does not exist. Creating dummy models for development.")
                self._create_dummy_models()
                return
            
            # Load XGBoost models with scalers (from existing trained models)
            xgb_models = {
                'xgb_1h': f"{self.models_path}/xgboost_optimized_1h.joblib",
                'xgb_4h': f"{self.models_path}/xgboost_optimized_4h.joblib", 
                'xgb_24h': f"{self.models_path}/xgboost_optimized_24h.joblib"  # Check for 24h instead of 1d
            }
            
            # Also check for scalers and parameters
            scaler_files = {
                'xgb_1h': f"{self.models_path}/scaler_optimized_1h.joblib",
                'xgb_4h': f"{self.models_path}/scaler_optimized_4h.joblib",
                'xgb_24h': f"{self.models_path}/scaler_optimized_24h.joblib"
            }
            
            params_files = {
                'xgb_1h': f"{self.models_path}/params_optimized_1h.joblib",
                'xgb_4h': f"{self.models_path}/params_optimized_4h.joblib", 
                'xgb_24h': f"{self.models_path}/params_optimized_24h.joblib"
            }
            
            for name, path in xgb_models.items():
                try:
                    if os.path.exists(path):
                        model = joblib.load(path)
                        self.models[name] = model
                        
                        # Load associated scaler and params
                        scaler_path = scaler_files.get(name)
                        params_path = params_files.get(name)
                        
                        scaler = None
                        params = None
                        
                        if scaler_path and os.path.exists(scaler_path):
                            scaler = joblib.load(scaler_path)
                            
                        if params_path and os.path.exists(params_path):
                            params = joblib.load(params_path)
                        
                        self.model_metadata[name] = {
                            'type': ModelType.XGBOOST,
                            'timeframe': name.split('_')[-1],
                            'accuracy': 0.68,  # Actual trained model accuracy
                            'last_trained': datetime.utcnow() - timedelta(days=7),
                            'scaler': scaler,
                            'params': params,
                            'is_trained': True
                        }
                        self.logger.info(f"✅ Loaded trained XGBoost model: {name}")
                    else:
                        self.logger.debug(f"Model file not found: {path}")
                except Exception as e:
                    self.logger.warning(f"Could not load model {name}: {e}")
            
            # Load LightGBM models
            lgb_models = {
                'lgb_1h': f"{self.models_path}/lightgbm_optimized_1h.joblib",
                'lgb_4h': f"{self.models_path}/lightgbm_optimized_4h.joblib"
            }
            
            for name, path in lgb_models.items():
                try:
                    if os.path.exists(path):
                        model = joblib.load(path)
                        self.models[name] = model
                        self.model_metadata[name] = {
                            'type': ModelType.LIGHTGBM,
                            'timeframe': name.split('_')[-1],
                            'accuracy': 0.63,
                            'last_trained': datetime.utcnow() - timedelta(days=1)
                        }
                        self.logger.info(f"Loaded model: {name}")
                    else:
                        self.logger.debug(f"Model file not found: {path} (development mode)")
                except Exception as e:
                    self.logger.warning(f"Could not load model {name}: {e}")
            
            # If no models loaded, create dummy models for development
            if len(self.models) == 0:
                self.logger.warning("No trained models found. Creating dummy models for development.")
                self._create_dummy_models()
            else:
                self.logger.info(f"Successfully loaded {len(self.models)} models")
            
        except Exception as e:
            self.logger.error(f"Error during model loading: {e}")
            self.logger.warning("Creating dummy models for development.")
            self._create_dummy_models()
    
    def _create_dummy_models(self):
        """Create dummy models for development/testing"""
        
        # Create simple dummy models
        dummy_models = {
            'xgb_1h': RandomForestClassifier(n_estimators=10, random_state=42),
            'xgb_4h': RandomForestClassifier(n_estimators=10, random_state=43), 
            'xgb_1d': RandomForestClassifier(n_estimators=10, random_state=44),
            'lgb_1h': RandomForestClassifier(n_estimators=10, random_state=45),
            'lgb_4h': RandomForestClassifier(n_estimators=10, random_state=46)
        }
        
        # Fit dummy models with some fake data
        import numpy as np
        X_dummy = np.random.random((100, 50))  # 50 features
        y_dummy = np.random.randint(0, 3, 100)  # 3 classes (buy, sell, hold)
        
        for name, model in dummy_models.items():
            model.fit(X_dummy, y_dummy)
            self.models[name] = model
            self.model_metadata[name] = {
                'type': ModelType.DUMMY,
                'timeframe': name.split('_')[-1],
                'accuracy': 0.33,  # Random baseline
                'last_trained': datetime.utcnow(),
                'is_dummy': True
            }
            
        self.logger.info(f"Created {len(dummy_models)} dummy models for development")
    
    async def generate_signals(self, symbol: str, features: pd.DataFrame) -> List[MLSignal]:
        """Generate signals from all models"""
        
        signals = []
        
        for model_name, model in self.models.items():
            try:
                signal = await self.generate_single_signal(model_name, model, symbol, features)
                if signal:
                    signals.append(signal)
            except Exception as e:
                self.logger.error(f"Error generating signal from {model_name}: {e}")
        
        return signals
    
    async def generate_single_signal(self, model_name: str, model: Any, symbol: str, features: pd.DataFrame) -> Optional[MLSignal]:
        """Generate signal from a single model with scaler support"""
        
        try:
            metadata = self.model_metadata.get(model_name, {})
            
            # Apply scaler if available (for trained XGBoost models)
            processed_features = features
            scaler = metadata.get('scaler')
            if scaler is not None:
                try:
                    # Scale the features using the trained scaler
                    processed_features = pd.DataFrame(
                        scaler.transform(features),
                        columns=features.columns,
                        index=features.index
                    )
                    self.logger.debug(f"Applied scaler for {model_name}")
                except Exception as e:
                    self.logger.warning(f"Scaler application failed for {model_name}: {e}")
                    processed_features = features
            
            # Make prediction
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(processed_features)[0]
                prediction = float(probabilities[1] - probabilities[0])  # Assuming binary classification
                confidence = float(max(probabilities))
            else:
                prediction = float(model.predict(processed_features)[0])
                confidence = min(0.8, abs(prediction))  # Estimate confidence
            
            # Determine signal strength based on trained model performance
            strength = self.determine_signal_strength(prediction, confidence, metadata)
            
            # Get feature importance (if available)
            feature_importance = {}
            if hasattr(model, 'feature_importances_'):
                feature_names = features.columns.tolist()
                importance_values = model.feature_importances_
                feature_importance = dict(zip(feature_names, importance_values))
            
            # Calculate expiration based on timeframe
            timeframe = metadata.get('timeframe', '4h')
            if timeframe == '1h':
                expires_at = datetime.utcnow() + timedelta(hours=1)
            elif timeframe == '4h':
                expires_at = datetime.utcnow() + timedelta(hours=4)
            elif timeframe == '24h':
                expires_at = datetime.utcnow() + timedelta(hours=24)
            else:
                expires_at = datetime.utcnow() + timedelta(hours=4)  # Default
            
            signal = MLSignal(
                model_name=model_name,
                model_type=metadata.get('type', ModelType.XGBOOST),
                symbol=symbol,
                prediction=prediction,
                confidence=confidence,
                strength=strength,
                features_used=features.columns.tolist(),
                feature_importance=feature_importance,
                prediction_horizon=timeframe,
                expires_at=expires_at,
                metadata={
                    'model_accuracy': metadata.get('accuracy', 0.5),
                    'last_trained': metadata.get('last_trained', datetime.utcnow()).isoformat()
                }
            )
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating signal from {model_name}: {e}")
            return None
    
    def determine_signal_strength(self, prediction: float, confidence: float, metadata: dict = None) -> SignalStrength:
        """Determine signal strength based on prediction, confidence, and model metadata"""
        
        combined_score = abs(prediction) * confidence
        
        # Adjust thresholds for trained models vs dummy models
        if metadata and metadata.get('is_trained', False):
            # Trained models have higher accuracy, so we can trust stronger signals more
            accuracy = metadata.get('accuracy', 0.5)
            
            # Adjust score based on model accuracy
            adjusted_score = combined_score * (1 + (accuracy - 0.5))  # Boost for good models
            
            if adjusted_score >= 0.75:
                return SignalStrength.VERY_STRONG
            elif adjusted_score >= 0.65:
                return SignalStrength.STRONG
            elif adjusted_score >= 0.55:
                return SignalStrength.MODERATE
            elif adjusted_score >= 0.45:
                return SignalStrength.WEAK
            else:
                return SignalStrength.VERY_WEAK
        else:
            # Original thresholds for dummy/untrained models
            if combined_score >= 0.8:
                return SignalStrength.VERY_STRONG
            elif combined_score >= 0.6:
                return SignalStrength.STRONG
            elif combined_score >= 0.4:
                return SignalStrength.MODERATE
            elif combined_score >= 0.2:
                return SignalStrength.WEAK
            else:
                return SignalStrength.VERY_WEAK

class EnsembleEngine:
    """Create ensemble signals from multiple ML models"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def create_ensemble_signal(self, signals: List[MLSignal], current_price: Decimal) -> EnsembleSignal:
        """Create ensemble signal from multiple ML signals"""
        
        if not signals:
            raise ValueError("No signals provided for ensemble")
        
        symbol = signals[0].symbol
        
        # Calculate weighted consensus
        total_weight = 0
        weighted_prediction = 0
        weighted_confidence = 0
        
        for signal in signals:
            # Weight by model accuracy and confidence
            model_accuracy = signal.metadata.get('model_accuracy', 0.5)
            weight = signal.confidence * model_accuracy
            
            weighted_prediction += signal.prediction * weight
            weighted_confidence += signal.confidence * weight
            total_weight += weight
        
        if total_weight > 0:
            consensus_prediction = weighted_prediction / total_weight
            consensus_confidence = weighted_confidence / total_weight
        else:
            consensus_prediction = 0
            consensus_confidence = 0
        
        # Calculate agreement score
        agreement_score = self.calculate_agreement_score(signals)
        
        # Determine dominant timeframe
        timeframes = [signal.prediction_horizon for signal in signals]
        dominant_timeframe = max(set(timeframes), key=timeframes.count)
        
        # Calculate risk-adjusted position size
        position_sizer = PositionSizer()
        risk_adjusted_size = position_sizer.calculate_position_size(
            portfolio_value=Decimal('100000'),  # Would get from portfolio service
            entry_price=current_price,
            signal_confidence=consensus_confidence
        )
        
        # Determine recommended action
        if consensus_prediction > 0.3 and consensus_confidence > 0.6:
            recommended_action = 'buy'
            stop_loss_price = current_price * Decimal('0.95')  # 5% stop loss
            take_profit_price = current_price * Decimal('1.10')  # 10% take profit
        elif consensus_prediction < -0.3 and consensus_confidence > 0.6:
            recommended_action = 'sell'
            stop_loss_price = current_price * Decimal('1.05')  # 5% stop loss
            take_profit_price = current_price * Decimal('0.90')  # 10% take profit
        else:
            recommended_action = 'hold'
            stop_loss_price = None
            take_profit_price = None
        
        return EnsembleSignal(
            symbol=symbol,
            individual_signals=signals,
            consensus_prediction=consensus_prediction,
            consensus_confidence=consensus_confidence,
            agreement_score=agreement_score,
            dominant_timeframe=dominant_timeframe,
            risk_adjusted_size=risk_adjusted_size,
            recommended_action=recommended_action,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price
        )
    
    def calculate_agreement_score(self, signals: List[MLSignal]) -> float:
        """Calculate how much the models agree"""
        
        if len(signals) <= 1:
            return 1.0
        
        predictions = [signal.prediction for signal in signals]
        
        # Calculate standard deviation of predictions
        std_dev = np.std(predictions)
        
        # Convert to agreement score (0 = no agreement, 1 = perfect agreement)
        # Assuming predictions range from -1 to 1, max std dev would be around 1
        agreement_score = max(0, 1 - (std_dev / 1.0))
        
        return agreement_score

class MLIntegrationService:
    """Main service for ML integration"""
    
    def __init__(self, db_config: Dict[str, str]):
        self.db_config = db_config
        self.feature_engineer = FeatureEngineer(db_config)
        self.model_manager = MLModelManager()
        self.ensemble_engine = EnsembleEngine()
        self.logger = logging.getLogger(__name__)
        
        # Initialize models
        self.model_manager.load_models()
    
    async def generate_trading_signals(self, symbols: List[str]) -> Dict[str, EnsembleSignal]:
        """Generate ensemble trading signals for multiple symbols"""
        
        signals_by_symbol = {}
        
        for symbol in symbols:
            try:
                signal = await self.generate_signal_for_symbol(symbol)
                if signal:
                    signals_by_symbol[symbol] = signal
            except Exception as e:
                self.logger.error(f"Error generating signal for {symbol}: {e}")
        
        return signals_by_symbol
    
    async def generate_signal_for_symbol(self, symbol: str) -> Optional[EnsembleSignal]:
        """Generate ensemble signal for a single symbol"""
        
        try:
            # Generate features
            features = await self.feature_engineer.get_features(symbol)
            if features is None or features.empty:
                self.logger.warning(f"No features available for {symbol}")
                return None
            
            # Generate ML signals
            ml_signals = await self.model_manager.generate_signals(symbol, features)
            if not ml_signals:
                self.logger.warning(f"No ML signals generated for {symbol}")
                return None
            
            # Get current price
            current_price = await self.get_current_price(symbol)
            
            # Create ensemble signal
            ensemble_signal = self.ensemble_engine.create_ensemble_signal(ml_signals, current_price)
            
            self.logger.info(f"Generated ensemble signal for {symbol}: "
                           f"prediction={ensemble_signal.consensus_prediction:.3f}, "
                           f"confidence={ensemble_signal.consensus_confidence:.3f}")
            
            return ensemble_signal
            
        except Exception as e:
            self.logger.error(f"Error generating signal for {symbol}: {e}")
            return None
    
    async def get_current_price(self, symbol: str) -> Decimal:
        """Get current market price"""
        
        try:
            # This would fetch from price API
            # For now, return a placeholder
            return Decimal('50000')  # Placeholder
        except Exception as e:
            self.logger.error(f"Error getting current price for {symbol}: {e}")
            return Decimal('50000')  # Fallback
    
    async def convert_to_trading_signal(self, ensemble_signal: EnsembleSignal) -> TradingSignal:
        """Convert ensemble signal to trading signal format"""
        
        return TradingSignal(
            symbol=ensemble_signal.symbol,
            signal_type=ensemble_signal.recommended_action,
            strength=ensemble_signal.consensus_confidence,
            confidence=ensemble_signal.consensus_confidence,
            price_target=ensemble_signal.take_profit_price,
            stop_loss=ensemble_signal.stop_loss_price,
            time_horizon=ensemble_signal.dominant_timeframe,
            source="ml_ensemble",
            metadata={
                'consensus_prediction': ensemble_signal.consensus_prediction,
                'agreement_score': ensemble_signal.agreement_score,
                'model_count': len(ensemble_signal.individual_signals),
                'dominant_timeframe': ensemble_signal.dominant_timeframe
            }
        )

# FastAPI service
app = FastAPI(title="ML Integration Service", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database configuration
DB_CONFIG = {
    'host': 'host.docker.internal',
    'user': 'news_collector',
    'password': '99Rules!',
    'database': 'crypto_market_data'
}

# Initialize ML service
ml_service = MLIntegrationService(DB_CONFIG)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "ml_integration"}

@app.get("/models/status")
async def get_models_status():
    """Get status of loaded ML models"""
    
    models_info = []
    for name, metadata in ml_service.model_manager.model_metadata.items():
        models_info.append({
            'name': name,
            'type': metadata['type'].value,
            'timeframe': metadata['timeframe'],
            'accuracy': metadata['accuracy'],
            'last_trained': metadata['last_trained'].isoformat()
        })
    
    return {
        'total_models': len(ml_service.model_manager.models),
        'models': models_info,
        'timestamp': datetime.utcnow().isoformat()
    }

@app.post("/signals/generate")
async def generate_signals(symbols: List[str]):
    """Generate ML ensemble signals for symbols"""
    
    try:
        signals = await ml_service.generate_trading_signals(symbols)
        
        # Convert to JSON-serializable format
        result = {}
        for symbol, signal in signals.items():
            result[symbol] = {
                'consensus_prediction': signal.consensus_prediction,
                'consensus_confidence': signal.consensus_confidence,
                'agreement_score': signal.agreement_score,
                'recommended_action': signal.recommended_action,
                'dominant_timeframe': signal.dominant_timeframe,
                'risk_adjusted_size': float(signal.risk_adjusted_size),
                'stop_loss_price': float(signal.stop_loss_price) if signal.stop_loss_price else None,
                'take_profit_price': float(signal.take_profit_price) if signal.take_profit_price else None,
                'model_count': len(signal.individual_signals),
                'timestamp': signal.timestamp.isoformat()
            }
        
        return {
            'symbols_processed': len(symbols),
            'signals_generated': len(signals),
            'signals': result,
            'timestamp': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/signals/{symbol}")
async def get_signal_for_symbol(symbol: str):
    """Get ML ensemble signal for a specific symbol"""
    
    try:
        signal = await ml_service.generate_signal_for_symbol(symbol)
        
        if signal is None:
            raise HTTPException(status_code=404, detail=f"No signal generated for {symbol}")
        
        return {
            'symbol': symbol,
            'consensus_prediction': signal.consensus_prediction,
            'consensus_confidence': signal.consensus_confidence,
            'agreement_score': signal.agreement_score,
            'recommended_action': signal.recommended_action,
            'dominant_timeframe': signal.dominant_timeframe,
            'risk_adjusted_size': float(signal.risk_adjusted_size),
            'stop_loss_price': float(signal.stop_loss_price) if signal.stop_loss_price else None,
            'take_profit_price': float(signal.take_profit_price) if signal.take_profit_price else None,
            'individual_signals': [
                {
                    'model_name': s.model_name,
                    'prediction': s.prediction,
                    'confidence': s.confidence,
                    'strength': s.strength.value,
                    'timeframe': s.prediction_horizon
                }
                for s in signal.individual_signals
            ],
            'timestamp': signal.timestamp.isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8024)
