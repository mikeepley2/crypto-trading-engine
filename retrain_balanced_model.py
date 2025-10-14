#!/usr/bin/env python3
"""
Retrain ML Model with Balanced Data
This script will retrain the model to ensure it generates both BUY and SELL signals
"""

import os
import sys
import logging
import mysql.connector
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_db_connection():
    """Get database connection"""
    try:
        return mysql.connector.connect(
            host=os.getenv('DB_HOST', '172.22.32.1'),
            user=os.getenv('DB_USER', 'news_collector'),
            password=os.getenv('DB_PASSWORD', '99Rules!'),
            database=os.getenv('DB_NAME_PRICES', 'crypto_prices')
        )
    except Exception as e:
        logger.error(f'Database connection error: {e}')
        return None

def get_training_data():
    """Get training data from database"""
    logger.info("Fetching training data from database...")
    
    conn = get_db_connection()
    if not conn:
        return None, None
    
    try:
        # Get data from the last 2 years for training
        query = """
        SELECT 
            symbol,
            timestamp_iso,
            current_price,
            rsi_14,
            sma_20,
            sma_50,
            ema_12,
            ema_26,
            macd_line,
            macd_signal,
            macd_histogram,
            bb_upper,
            bb_middle,
            bb_lower,
            stoch_k,
            stoch_d,
            atr_14,
            vwap,
            vix,
            spx,
            dxy,
            tnx,
            fed_funds_rate,
            avg_cryptobert_score,
            avg_vader_score,
            avg_textblob_score,
            avg_crypto_keywords_score,
            avg_finbert_sentiment_score,
            avg_fear_greed_score,
            avg_volatility_sentiment,
            avg_risk_appetite,
            avg_crypto_correlation,
            data_quality_score,
            avg_general_cryptobert_score,
            avg_general_vader_score,
            avg_general_textblob_score,
            avg_general_crypto_keywords_score,
            social_post_count,
            social_avg_sentiment,
            social_weighted_sentiment,
            social_engagement_weighted_sentiment,
            social_verified_user_sentiment,
            social_total_engagement,
            social_unique_authors,
            social_avg_confidence,
            treasury_10y,
            vix_index,
            dxy_index,
            spx_price,
            gold_price,
            oil_price,
            btc_fear_greed,
            market_cap_usd,
            total_volume_24h,
            active_addresses_24h,
            transaction_count_24h,
            exchange_net_flow_24h,
            price_volatility_7d,
            onchain_market_cap_usd,
            onchain_volume_24h,
            onchain_price_volatility_7d,
            market_cap_rank,
            unemployment_rate,
            inflation_rate,
            social_sentiment,
            news_sentiment,
            reddit_sentiment,
            open_price,
            high_price,
            low_price,
            close_price,
            ohlc_volume,
            percent_change_1h,
            percent_change_24h,
            percent_change_7d,
            sentiment_positive,
            sentiment_negative,
            sentiment_neutral,
            sentiment_fear_greed_index,
            sentiment_volume_weighted,
            sentiment_social_dominance,
            sentiment_news_impact,
            sentiment_whale_movement,
            onchain_active_addresses,
            onchain_transaction_volume,
            onchain_avg_transaction_value,
            onchain_nvt_ratio,
            onchain_mvrv_ratio,
            onchain_whale_transactions,
            gdp_growth,
            cpi_inflation,
            interest_rate,
            employment_rate,
            consumer_confidence,
            retail_sales,
            industrial_production
        FROM ml_features_materialized 
        WHERE timestamp_iso >= NOW() - INTERVAL 2 YEAR
        AND current_price IS NOT NULL 
        AND current_price > 0
        AND symbol IN ('BTC', 'ETH', 'XRP', 'ADA', 'SOL', 'DOT', 'MATIC', 'LINK', 'UNI', 'LTC')
        ORDER BY symbol, timestamp_iso
        """
        
        df = pd.read_sql(query, conn)
        logger.info(f"Fetched {len(df)} rows of training data")
        
        if len(df) == 0:
            logger.error("No training data found!")
            return None, None
        
        return df, conn
        
    except Exception as e:
        logger.error(f"Error fetching training data: {e}")
        return None, None

def create_labels(df):
    """Create balanced labels based on future price movements"""
    logger.info("Creating balanced labels...")
    
    # Sort by symbol and timestamp
    df = df.sort_values(['symbol', 'timestamp_iso']).reset_index(drop=True)
    
    # Calculate future price changes (1 hour, 4 hours, 24 hours)
    df['price_1h_future'] = df.groupby('symbol')['current_price'].shift(-1)
    df['price_4h_future'] = df.groupby('symbol')['current_price'].shift(-4)
    df['price_24h_future'] = df.groupby('symbol')['current_price'].shift(-24)
    
    # Calculate percentage changes
    df['change_1h'] = (df['price_1h_future'] - df['current_price']) / df['current_price'] * 100
    df['change_4h'] = (df['price_4h_future'] - df['current_price']) / df['current_price'] * 100
    df['change_24h'] = (df['price_24h_future'] - df['current_price']) / df['current_price'] * 100
    
    # Create balanced labels
    # BUY (1): Strong positive movement in any timeframe
    # SELL (0): Strong negative movement in any timeframe
    # HOLD: Everything else (will be filtered out)
    
    buy_condition = (
        (df['change_1h'] > 1.0) |  # 1%+ in 1 hour
        (df['change_4h'] > 2.0) |  # 2%+ in 4 hours
        (df['change_24h'] > 3.0)   # 3%+ in 24 hours
    )
    
    sell_condition = (
        (df['change_1h'] < -1.0) |  # -1%+ in 1 hour
        (df['change_4h'] < -2.0) |  # -2%+ in 4 hours
        (df['change_24h'] < -3.0)   # -3%+ in 24 hours
    )
    
    # Create labels
    df['label'] = np.where(buy_condition, 1, 
                          np.where(sell_condition, 0, -1))  # -1 for HOLD
    
    # Remove HOLD samples to create balanced dataset
    df_labeled = df[df['label'] != -1].copy()
    
    # Check class distribution
    class_counts = df_labeled['label'].value_counts()
    logger.info(f"Class distribution: {class_counts}")
    logger.info(f"BUY ratio: {class_counts.get(1, 0) / len(df_labeled):.3f}")
    logger.info(f"SELL ratio: {class_counts.get(0, 0) / len(df_labeled):.3f}")
    
    return df_labeled

def prepare_features(df):
    """Prepare features for training"""
    logger.info("Preparing features...")
    
    # Select feature columns (exclude metadata and target columns)
    feature_columns = [
        'rsi_14', 'sma_20', 'sma_50', 'ema_12', 'ema_26',
        'macd_line', 'macd_signal', 'macd_histogram',
        'bb_upper', 'bb_middle', 'bb_lower',
        'stoch_k', 'stoch_d', 'atr_14', 'vwap',
        'vix', 'spx', 'dxy', 'tnx', 'fed_funds_rate',
        'avg_cryptobert_score', 'avg_vader_score', 'avg_textblob_score',
        'avg_crypto_keywords_score', 'avg_finbert_sentiment_score',
        'avg_fear_greed_score', 'avg_volatility_sentiment',
        'avg_risk_appetite', 'avg_crypto_correlation',
        'data_quality_score', 'avg_general_cryptobert_score',
        'avg_general_vader_score', 'avg_general_textblob_score',
        'avg_general_crypto_keywords_score', 'social_post_count',
        'social_avg_sentiment', 'social_weighted_sentiment',
        'social_engagement_weighted_sentiment', 'social_verified_user_sentiment',
        'social_total_engagement', 'social_unique_authors',
        'social_avg_confidence', 'treasury_10y', 'vix_index',
        'dxy_index', 'spx_price', 'gold_price', 'oil_price',
        'btc_fear_greed', 'market_cap_usd', 'total_volume_24h',
        'active_addresses_24h', 'transaction_count_24h',
        'exchange_net_flow_24h', 'price_volatility_7d',
        'onchain_market_cap_usd', 'onchain_volume_24h',
        'onchain_price_volatility_7d', 'market_cap_rank',
        'unemployment_rate', 'inflation_rate', 'social_sentiment',
        'news_sentiment', 'reddit_sentiment', 'open_price',
        'high_price', 'low_price', 'close_price', 'ohlc_volume',
        'percent_change_1h', 'percent_change_24h', 'percent_change_7d',
        'sentiment_positive', 'sentiment_negative', 'sentiment_neutral',
        'sentiment_fear_greed_index', 'sentiment_volume_weighted',
        'sentiment_social_dominance', 'sentiment_news_impact',
        'sentiment_whale_movement', 'onchain_active_addresses',
        'onchain_transaction_volume', 'onchain_avg_transaction_value',
        'onchain_nvt_ratio', 'onchain_mvrv_ratio', 'onchain_whale_transactions',
        'gdp_growth', 'cpi_inflation', 'interest_rate',
        'employment_rate', 'consumer_confidence', 'retail_sales',
        'industrial_production'
    ]
    
    # Select available features
    available_features = [col for col in feature_columns if col in df.columns]
    logger.info(f"Using {len(available_features)} features")
    
    # Prepare feature matrix
    X = df[available_features].copy()
    
    # Handle missing values
    X = X.fillna(0)
    
    # Handle infinite values
    X = X.replace([np.inf, -np.inf], 0)
    
    # Prepare labels
    y = df['label'].values
    
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Labels shape: {y.shape}")
    
    return X, y, available_features

def train_model(X, y):
    """Train balanced XGBoost model"""
    logger.info("Training balanced XGBoost model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Calculate class weights for balancing
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    
    logger.info(f"Class weights: {class_weight_dict}")
    
    # XGBoost parameters for balanced training
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'scale_pos_weight': class_weight_dict[1] / class_weight_dict[0]  # Balance classes
    }
    
    # Train model
    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    logger.info("Model Evaluation:")
    logger.info(f"Accuracy: {model.score(X_test_scaled, y_test):.4f}")
    logger.info(f"AUC: {roc_auc_score(y_test, y_pred_proba[:, 1]):.4f}")
    
    # Classification report
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, y_pred))
    
    # Confusion matrix
    logger.info("\nConfusion Matrix:")
    logger.info(confusion_matrix(y_test, y_pred))
    
    # Test prediction distribution
    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)
    
    logger.info(f"\nTraining predictions - Class 0: {np.sum(train_pred == 0)}, Class 1: {np.sum(train_pred == 1)}")
    logger.info(f"Test predictions - Class 0: {np.sum(test_pred == 0)}, Class 1: {np.sum(test_pred == 1)}")
    
    return model, scaler, X.columns.tolist()

def save_model(model, scaler, feature_names, stats):
    """Save the trained model and metadata"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model
    model_filename = f"balanced_retrained_model_{timestamp}.joblib"
    joblib.dump(model, model_filename)
    logger.info(f"Model saved as: {model_filename}")
    
    # Save scaler
    scaler_filename = f"balanced_retrained_scaler_{timestamp}.joblib"
    joblib.dump(scaler, scaler_filename)
    logger.info(f"Scaler saved as: {scaler_filename}")
    
    # Save feature names
    import json
    features_filename = f"balanced_retrained_features_{timestamp}.json"
    with open(features_filename, 'w') as f:
        json.dump(feature_names, f, indent=2)
    logger.info(f"Features saved as: {features_filename}")
    
    # Save stats
    stats_filename = f"balanced_retrained_stats_{timestamp}.json"
    with open(stats_filename, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Stats saved as: {stats_filename}")
    
    return model_filename, scaler_filename, features_filename, stats_filename

def test_model_predictions(model, scaler, feature_names):
    """Test model with random data to ensure it generates both classes"""
    logger.info("Testing model predictions...")
    
    # Generate random test data
    np.random.seed(42)
    test_data = np.random.randn(100, len(feature_names))
    test_data_scaled = scaler.transform(test_data)
    
    # Make predictions
    predictions = model.predict(test_data_scaled)
    probabilities = model.predict_proba(test_data_scaled)
    
    # Check distribution
    class_0_count = np.sum(predictions == 0)
    class_1_count = np.sum(predictions == 1)
    
    logger.info(f"Test predictions - Class 0 (SELL): {class_0_count}, Class 1 (BUY): {class_1_count}")
    logger.info(f"Class 0 probability range: {np.min(probabilities[:, 0]):.3f} - {np.max(probabilities[:, 0]):.3f}")
    logger.info(f"Class 1 probability range: {np.min(probabilities[:, 1]):.3f} - {np.max(probabilities[:, 1]):.3f}")
    
    # Test signal generation logic
    buy_signals = 0
    sell_signals = 0
    hold_signals = 0
    
    for i in range(len(predictions)):
        prediction = predictions[i]
        confidence = max(probabilities[i])
        
        if prediction == 1 and confidence > 0.5:
            buy_signals += 1
        elif prediction == 0 and confidence > 0.6:
            sell_signals += 1
        else:
            hold_signals += 1
    
    logger.info(f"Signal distribution - BUY: {buy_signals}, SELL: {sell_signals}, HOLD: {hold_signals}")
    
    if sell_signals == 0:
        logger.warning("⚠️ Model is not generating SELL signals!")
        return False
    elif buy_signals == 0:
        logger.warning("⚠️ Model is not generating BUY signals!")
        return False
    else:
        logger.info("✅ Model generates both BUY and SELL signals!")
        return True

def main():
    """Main training pipeline"""
    logger.info("🚀 Starting balanced model retraining...")
    
    # Get training data
    df, conn = get_training_data()
    if df is None:
        logger.error("Failed to get training data")
        return False
    
    try:
        # Create balanced labels
        df_labeled = create_labels(df)
        if len(df_labeled) == 0:
            logger.error("No labeled data created")
            return False
        
        # Prepare features
        X, y, feature_names = prepare_features(df_labeled)
        
        # Train model
        model, scaler, _ = train_model(X, y)
        
        # Create stats
        stats = {
            'training_samples': len(X),
            'feature_count': len(feature_names),
            'class_distribution': {
                'class_0': int(np.sum(y == 0)),
                'class_1': int(np.sum(y == 1))
            },
            'class_balance_ratio': np.sum(y == 1) / np.sum(y == 0),
            'label_definition': 'Balanced BUY/SELL based on future price movements',
            'timestamp': datetime.now().isoformat()
        }
        
        # Save model
        model_files = save_model(model, scaler, feature_names, stats)
        
        # Test model
        success = test_model_predictions(model, scaler, feature_names)
        
        if success:
            logger.info("🎉 Model retraining completed successfully!")
            logger.info(f"Model files: {model_files}")
            return True
        else:
            logger.error("❌ Model retraining failed - model not generating balanced signals")
            return False
            
    except Exception as e:
        logger.error(f"Error during training: {e}")
        return False
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


