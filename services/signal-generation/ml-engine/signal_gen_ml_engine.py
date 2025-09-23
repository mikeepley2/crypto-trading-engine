#!/usr/bin/env python3
"""
Signal Generation ML Engine Service
Handles XGBoost model operations and ML predictions for trading signals

This microservice extracts the core ML functionality from the monolithic enhanced_signal_generator.py
Responsibilities:
- Load and manage XGBoost model
- Process features for model predictions
- Generate ML-based buy/sell/hold predictions
- Provide confidence scores and signal strength
"""

import os
import sys
import json
import logging
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import asyncio
import aiohttp
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
import uvicorn
from pydantic import BaseModel
import mysql.connector
from mysql.connector import pooling
# Temporarily disable shared logging to test if it's causing the model loading issue
# from shared.loki_logging import setup_logging, log_request_info, log_response_info, log_model_operation, log_health_check

# Setup basic logging instead of Loki logging for debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MLPredictionRequest(BaseModel):
    """Request model for ML predictions"""
    symbol: str
    features: Dict[str, float]
    timestamp: Optional[str] = None

class MLPredictionResponse(BaseModel):
    """Response model for ML predictions"""
    symbol: str
    prediction_probability: float  # Raw ML prediction (0-1)
    confidence: float  # Confidence score
    signal_type: str  # BUY, SELL, HOLD
    signal_strength: str  # WEAK, MODERATE, STRONG, VERY_STRONG
    features_used: int
    model_version: str
    timestamp: str

class ModelInfoResponse(BaseModel):
    """Model information response"""
    model_version: str
    feature_count: int
    model_type: str
    accuracy: Optional[float] = None
    loaded_at: str
    model_file: str

class SignalGenMLEngine:
    """Core ML Engine for signal generation"""
    
    def __init__(self):
        self.app = FastAPI(
            title="Signal Generation ML Engine",
            description="Core ML predictions for crypto trading signals",
            version="1.0.0"
        )
        
        # Setup Prometheus metrics
        self.instrumentator = Instrumentator()
        self.instrumentator.instrument(self.app).expose(self.app)
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self.model = None
        self.model_metadata = None
        self.feature_columns = []
        self.model_file = None
        self.loaded_at = None
        
        self.setup_routes()
        self.load_model()
    
    def find_latest_model(self) -> Tuple[Optional[str], Optional[str]]:
        """Find the most recent model and metadata files"""
        try:
            # Look for model files in current directory and parent directories
            search_paths = ['.', '..', '../..', '../../..', '../../../..']
            
            model_file = None
            metadata_file = None
            
            for search_path in search_paths:
                if os.path.exists(search_path):
                    files = os.listdir(search_path)
                    
                    # Find model files
                    model_files = [f for f in files if f.startswith('full_dataset_gpu_xgboost_model_') and f.endswith('.joblib')]
                    if not model_files:
                        # Try alternative naming
                        model_files = [f for f in files if 'xgboost' in f.lower() and f.endswith('.joblib')]
                    
                    # Find metadata files
                    metadata_files = [f for f in files if f.startswith('full_dataset_gpu_xgboost_metadata_') and f.endswith('.json')]
                    if not metadata_files:
                        # Try alternative naming
                        metadata_files = [f for f in files if 'metadata' in f.lower() and 'xgboost' in f.lower() and f.endswith('.json')]
                    
                    if model_files and metadata_files:
                        # Get the most recent files
                        model_file = os.path.join(search_path, sorted(model_files)[-1])
                        metadata_file = os.path.join(search_path, sorted(metadata_files)[-1])
                        break
            
            return model_file, metadata_file
            
        except Exception as e:
            logger.error(f"Error finding model files: {e}")
            return None, None
    
    def load_model(self):
        """Load the XGBoost model and metadata"""
        try:
            model_file, metadata_file = self.find_latest_model()
            
            if not model_file or not metadata_file:
                logger.error("No model or metadata files found")
                return
            
            # Load model with XGBoost compatibility handling
            try:
                self.model = joblib.load(model_file)
                
                # Advanced XGBoost compatibility fix
                if hasattr(self.model, '_Booster'):
                    # For XGBoost models, set the deprecated attribute
                    if not hasattr(self.model, 'use_label_encoder'):
                        self.model.use_label_encoder = False
                    
                    # Try to recreate the classifier if needed
                    import xgboost as xgb
                    if isinstance(self.model, xgb.XGBClassifier):
                        # Get the booster and recreate with current XGBoost version
                        booster = self.model.get_booster()
                        new_model = xgb.XGBClassifier()
                        new_model._Booster = booster
                        new_model.use_label_encoder = False
                        self.model = new_model
                        logger.info("✅ Recreated XGBoost model for compatibility")
                        
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                # Try to load just the booster and create a new classifier
                try:
                    import xgboost as xgb
                    booster = xgb.Booster()
                    booster.load_model(model_file.replace('.joblib', '.json'))
                    self.model = xgb.XGBClassifier()
                    self.model._Booster = booster
                    self.model.use_label_encoder = False
                    logger.info("✅ Loaded model as XGBoost booster")
                except Exception as e2:
                    logger.error(f"Failed to load model: {e2}")
                    return
                
            self.model_file = model_file
            self.loaded_at = datetime.now()
            logger.info(f"✅ Loaded model from {model_file}")
            
            # Load metadata
            with open(metadata_file, 'r') as f:
                self.model_metadata = json.load(f)
            
            # Extract feature columns
            if 'features' in self.model_metadata:
                self.feature_columns = self.model_metadata['features']
            elif 'feature_columns' in self.model_metadata:
                self.feature_columns = self.model_metadata['feature_columns']
            elif hasattr(self.model, 'feature_names_in_') and self.model.feature_names_in_ is not None:
                self.feature_columns = list(self.model.feature_names_in_)
            else:
                # Fallback - create generic feature names
                n_features = getattr(self.model, 'n_features_in_', 120)
                self.feature_columns = [f'feature_{i}' for i in range(n_features)]
            
            logger.info(f"✅ Loaded metadata from {metadata_file} with {len(self.feature_columns)} features")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            self.model = None
            self.model_metadata = None
    
    def prepare_features(self, features: Dict[str, float]) -> np.ndarray:
        """Prepare features for model prediction"""
        try:
            # Create feature array with correct order
            feature_array = np.zeros(len(self.feature_columns))
            
            for i, feature_name in enumerate(self.feature_columns):
                if feature_name in features:
                    feature_array[i] = features[feature_name]
                else:
                    # Use default value of 0
                    feature_array[i] = 0.0
            
            return feature_array.reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            raise
    
    def get_signal_type(self, prediction_probability: float) -> str:
        """Determine signal type based on prediction probability"""
        if prediction_probability > 0.6:
            return "BUY"
        elif prediction_probability < 0.4:
            return "SELL"
        else:
            return "HOLD"
    
    def get_signal_strength(self, confidence: float) -> str:
        """Determine signal strength based on confidence"""
        if confidence < 0.6:
            return "WEAK"
        elif confidence < 0.75:
            return "MODERATE"
        elif confidence < 0.85:
            return "STRONG"
        else:
            return "VERY_STRONG"
    
    async def predict(self, request: MLPredictionRequest) -> MLPredictionResponse:
        """Generate ML prediction for given features"""
        try:
            if not self.model or not self.feature_columns:
                raise HTTPException(status_code=503, detail="ML model not loaded")
            
            # Prepare features
            feature_array = self.prepare_features(request.features)
            
            # Make prediction with XGBoost compatibility handling
            try:
                prediction_proba = self.model.predict_proba(feature_array)[0]
            except AttributeError as e:
                if 'use_label_encoder' in str(e):
                    # Try to fix the model on the fly
                    import xgboost as xgb
                    if hasattr(self.model, '_Booster'):
                        logger.warning("Attempting to fix XGBoost model compatibility")
                        self.model.use_label_encoder = False
                        # Force re-initialization of internal state
                        if hasattr(self.model, '_validate_params'):
                            self.model._validate_params()
                        prediction_proba = self.model.predict_proba(feature_array)[0]
                    else:
                        raise HTTPException(status_code=503, detail=f"XGBoost compatibility error: {str(e)}")
                else:
                    raise
            
            # Get the probability of positive class (usually index 1)
            if len(prediction_proba) > 1:
                prediction_probability = float(prediction_proba[1])
            else:
                prediction_probability = float(prediction_proba[0])
            
            # Calculate confidence (max probability)
            confidence = float(np.max(prediction_proba))
            
            # Adjust confidence for HOLD signals (when prediction is close to 0.5)
            if 0.4 <= prediction_probability <= 0.6:
                confidence = max(0.6, confidence)  # HOLD signals get reasonable confidence
            
            # Get signal type and strength
            signal_type = self.get_signal_type(prediction_probability)
            signal_strength = self.get_signal_strength(confidence)
            
            # Log prediction
            logger.info(f"🤖 ML Prediction for {request.symbol}: {signal_type} "
                       f"(prob: {prediction_probability:.3f}, conf: {confidence:.3f}, {signal_strength})")
            
            return MLPredictionResponse(
                symbol=request.symbol,
                prediction_probability=prediction_probability,
                confidence=confidence,
                signal_type=signal_type,
                signal_strength=signal_strength,
                features_used=len(self.feature_columns),
                model_version=self.model_metadata.get('model_version', 'xgboost_v1') if self.model_metadata else 'xgboost_v1',
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"❌ Error in ML prediction for {request.symbol}: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    async def batch_predict(self, requests: List[MLPredictionRequest]) -> List[MLPredictionResponse]:
        """Generate batch ML predictions"""
        try:
            if not self.model or not self.feature_columns:
                raise HTTPException(status_code=503, detail="ML model not loaded")
            
            results = []
            for request in requests:
                try:
                    result = await self.predict(request)
                    results.append(result)
                except Exception as e:
                    logger.error(f"❌ Error predicting {request.symbol}: {e}")
                    # Continue with other predictions
                    continue
            
            logger.info(f"📊 Batch prediction completed: {len(results)}/{len(requests)} successful")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error in batch prediction: {e}")
            raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")
    
    def setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            model_status = "loaded" if self.model else "not_loaded"
            
            return {
                "status": "healthy" if self.model else "degraded",
                "service": "signal-gen-ml-engine",
                "model_status": model_status,
                "features_count": len(self.feature_columns),
                "timestamp": datetime.now().isoformat(),
                "uptime_seconds": (datetime.now() - self.loaded_at).total_seconds() if self.loaded_at else 0
            }
        
        @self.app.post("/predict", response_model=MLPredictionResponse)
        async def predict_endpoint(request: MLPredictionRequest):
            """Single prediction endpoint"""
            return await self.predict(request)
        
        @self.app.post("/batch_predict")
        async def batch_predict_endpoint(requests: List[MLPredictionRequest]):
            """Batch prediction endpoint"""
            return await self.batch_predict(requests)
        
        @self.app.get("/model_info", response_model=ModelInfoResponse)
        async def model_info():
            """Get model information"""
            if not self.model_metadata:
                raise HTTPException(status_code=503, detail="Model metadata not available")
            
            return ModelInfoResponse(
                model_version=self.model_metadata.get('model_version', 'xgboost_v1'),
                feature_count=len(self.feature_columns),
                model_type="XGBoost",
                accuracy=self.model_metadata.get('accuracy'),
                loaded_at=self.loaded_at.isoformat() if self.loaded_at else datetime.now().isoformat(),
                model_file=os.path.basename(self.model_file) if self.model_file else "unknown"
            )
        
        @self.app.get("/features")
        async def get_features():
            """Get all feature column names"""
            return {
                "feature_columns": self.feature_columns,
                "total_count": len(self.feature_columns),
                "sample_features": self.feature_columns[:10] if self.feature_columns else []
            }
        
        @self.app.get("/status")
        async def get_status():
            """Detailed status information"""
            return {
                "service": "signal-gen-ml-engine",
                "version": "1.0.0",
                "model_loaded": bool(self.model),
                "model_file": os.path.basename(self.model_file) if self.model_file else None,
                "loaded_at": self.loaded_at.isoformat() if self.loaded_at else None,
                "feature_count": len(self.feature_columns),
                "model_metadata": self.model_metadata is not None,
                "timestamp": datetime.now().isoformat()
            }

        @self.app.post("/reload_model")
        async def reload_model():
            """Manually reload the ML model"""
            try:
                logger.info("🔄 Manual model reload requested")
                self.load_model()
                if self.model:
                    return {
                        "status": "success", 
                        "message": "Model reloaded successfully",
                        "model_file": os.path.basename(self.model_file) if self.model_file else None,
                        "feature_count": len(self.feature_columns)
                    }
                else:
                    return {
                        "status": "error", 
                        "message": "Model reload failed - no model loaded"
                    }
            except Exception as e:
                logger.error(f"Error during manual model reload: {e}")
                return {
                    "status": "error", 
                    "message": f"Model reload failed: {str(e)}"
                }

def main():
    """Main function to run the ML Engine service"""
    try:
        logger.info("🚀 Starting Signal Generation ML Engine...")
        
        ml_engine = SignalGenMLEngine()
        
        # Get port from environment or use default
        port = int(os.getenv('ML_ENGINE_PORT', 8051))
        
        logger.info(f"🔥 ML Engine service starting on port {port}")
        
        # Run the FastAPI application
        uvicorn.run(
            ml_engine.app,
            host="0.0.0.0",
            port=port,
            log_level="info"
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to start ML Engine: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()