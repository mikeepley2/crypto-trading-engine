#!/usr/bin/env python3
"""
Simple LLM Trade Validation Service - Minimal Version
"""

import logging
import os
from datetime import datetime
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LLM Trade Validation Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TradeValidationRequest(BaseModel):
    symbol: str
    action: str
    size_usd: float
    current_price: float
    ml_signal_confidence: float = None
    ml_signal_direction: str = None

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "llm_trade_validation",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/validate")
async def validate_trade(request: TradeValidationRequest):
    """Validate a trade using basic rule-based logic"""
    
    try:
        logger.info(f"Validating trade: {request.symbol} {request.action} ${request.size_usd}")
        
        # Basic validation logic
        warnings = []
        confidence = 0.8
        
        # Position size check
        if request.size_usd > 1000:
            warnings.append("Large position size")
            confidence *= 0.8
        
        # Price validation
        if request.current_price <= 0:
            return {
                "decision": "reject",
                "confidence": 0.0,
                "reasoning": "Invalid price data",
                "risk_assessment": "high",
                "warnings": ["Invalid price"],
                "suggestions": [],
                "timestamp": datetime.now().isoformat()
            }
        
        # Default approval with basic checks
        return {
            "decision": "approve",
            "confidence": confidence,
            "reasoning": f"Basic validation passed for {request.symbol} {request.action}",
            "risk_assessment": "low" if not warnings else "medium",
            "warnings": warnings,
            "suggestions": ["Monitor position size", "Set stop-loss orders"],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {e}")

@app.get("/models")
async def get_available_models():
    """Get available models (placeholder)"""
    return {
        "available_models": ["basic_rules"],
        "model_specializations": {"basic_rules": "rule_based_validation"}
    }

if __name__ == "__main__":
    port = int(os.getenv('LLM_VALIDATION_PORT', 8035))
    uvicorn.run(app, host="0.0.0.0", port=port)