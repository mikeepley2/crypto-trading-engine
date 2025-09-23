#!/usr/bin/env python3
"""
LLM-Enhanced Signal Generation Service
FastAPI microservice for generating trading signals with LLM assessment
"""

import asyncio
import logging
import os
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import json

# Import our enhanced signal generator
from llm_enhanced_signal_generator import LLMEnhancedSignalGenerator
from signal_analytics_tracker import SignalAnalyticsTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [LLM-SIGNALS] %(message)s'
)
logger = logging.getLogger(__name__)

# Global service state
service_state = {
    "status": "starting",
    "last_signal_generation": None,
    "signals_generated_today": 0,
    "total_signals_generated": 0,
    "database_connected": False,
    "llm_service_connected": False,
    "last_error": None,
    "start_time": datetime.now(),
    "service_name": "llm-enhanced-signal-generator",
    "generation_cycle_minutes": 5,
    "last_llm_assessments": 0,
    "total_llm_assessments": 0,
    "llm_assessment_enabled": True
}

# FastAPI app
app = FastAPI(
    title="LLM-Enhanced Signal Generation Service",
    description="Generate trading signals with LLM-based confidence assessment",
    version="1.0.0"
)

# Global generator instance
generator = None

# Pydantic models
class SignalGenerationRequest(BaseModel):
    symbols: Optional[List[str]] = None
    enable_llm_assessment: bool = True
    confidence_threshold: Optional[float] = 0.4

class SignalResponse(BaseModel):
    symbol: str
    signal_type: str
    confidence: float
    original_ml_confidence: Optional[float] = None
    llm_confidence_adjustment: Optional[float] = None
    llm_reasoning: Optional[str] = None
    llm_enhanced: bool = False
    timestamp: str

class ServiceStatus(BaseModel):
    status: str
    service_name: str
    uptime_seconds: float
    signals_generated_today: int
    total_signals_generated: int
    last_signal_generation: Optional[str]
    llm_assessments_today: int
    llm_assessment_enabled: bool
    database_connected: bool
    llm_service_connected: bool

@app.on_event("startup")
async def startup_event():
    """Initialize the service on startup"""
    global generator
    
    try:
        logger.info("🚀 Starting LLM-Enhanced Signal Generation Service...")
        
        # Initialize the generator
        generator = LLMEnhancedSignalGenerator()
        
        # Test database connection
        portfolio = generator.get_current_portfolio()
        service_state["database_connected"] = True
        
        # Test LLM service availability
        try:
            # Quick test of LLM service
            service_state["llm_service_connected"] = bool(
                os.getenv("OPENAI_API_KEY") or os.getenv("XAI_API_KEY")
            )
        except:
            service_state["llm_service_connected"] = False
        
        service_state["status"] = "ready"
        service_state["llm_assessment_enabled"] = generator.enable_llm_assessment
        
        logger.info("✅ LLM-Enhanced Signal Generation Service ready")
        logger.info(f"   Database: {'✅ Connected' if service_state['database_connected'] else '❌ Disconnected'}")
        logger.info(f"   LLM Service: {'✅ Available' if service_state['llm_service_connected'] else '❌ Unavailable'}")
        logger.info(f"   LLM Assessment: {'✅ Enabled' if service_state['llm_assessment_enabled'] else '❌ Disabled'}")
        
        # Start background signal generation
        start_background_generation()
        
    except Exception as e:
        service_state["status"] = "error"
        service_state["last_error"] = str(e)
        logger.error(f"❌ Failed to start service: {e}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    is_healthy = (
        service_state["database_connected"] and
        service_state["status"] not in ["error", "critical"]
    )
    
    return {
        "status": "healthy" if is_healthy else "unhealthy",
        "service": service_state["service_name"],
        "timestamp": datetime.now().isoformat(),
        "llm_assessment_enabled": service_state["llm_assessment_enabled"]
    }

@app.get("/status", response_model=ServiceStatus)
async def get_status():
    """Detailed status information"""
    uptime = datetime.now() - service_state["start_time"]
    
    return ServiceStatus(
        status=service_state["status"],
        service_name=service_state["service_name"],
        uptime_seconds=uptime.total_seconds(),
        signals_generated_today=service_state["signals_generated_today"],
        total_signals_generated=service_state["total_signals_generated"],
        last_signal_generation=service_state["last_signal_generation"],
        llm_assessments_today=service_state["last_llm_assessments"],
        llm_assessment_enabled=service_state["llm_assessment_enabled"],
        database_connected=service_state["database_connected"],
        llm_service_connected=service_state["llm_service_connected"]
    )

@app.get("/metrics")
async def get_metrics():
    """Service metrics for monitoring"""
    uptime = datetime.now() - service_state["start_time"]
    
    # Calculate LLM assessment rate
    llm_assessment_rate = 0.0
    if service_state["total_signals_generated"] > 0:
        llm_assessment_rate = service_state["total_llm_assessments"] / service_state["total_signals_generated"]
    
    return {
        "signals_generated_today": service_state["signals_generated_today"],
        "total_signals_generated": service_state["total_signals_generated"],
        "llm_assessments_total": service_state["total_llm_assessments"],
        "llm_assessment_rate": llm_assessment_rate,
        "uptime_seconds": uptime.total_seconds(),
        "generation_cycle_minutes": service_state["generation_cycle_minutes"],
        "last_signal_generation": service_state["last_signal_generation"],
        "error_count": 1 if service_state["last_error"] else 0,
        "llm_enabled": service_state["llm_assessment_enabled"],
        "timestamp": datetime.now().isoformat()
    }

@app.post("/generate-signals")
async def generate_signals_endpoint(request: SignalGenerationRequest) -> Dict:
    """Generate trading signals with optional LLM assessment"""
    
    if not generator:
        raise HTTPException(status_code=503, detail="Signal generator not initialized")
    
    try:
        # Set LLM assessment preference
        original_llm_setting = generator.enable_llm_assessment
        generator.enable_llm_assessment = request.enable_llm_assessment
        
        # Set confidence threshold
        if request.confidence_threshold:
            generator.llm_assessment_threshold = request.confidence_threshold
        
        # Generate signals
        signals = await generator.generate_signals_with_llm_assessment(request.symbols)
        
        # Restore original setting
        generator.enable_llm_assessment = original_llm_setting
        
        # Update metrics
        service_state["signals_generated_today"] += len(signals)
        service_state["total_signals_generated"] += len(signals)
        service_state["last_signal_generation"] = datetime.now().isoformat()
        
        # Count LLM assessments
        llm_enhanced_count = sum(1 for s in signals if s.get('llm_enhanced'))
        service_state["last_llm_assessments"] = llm_enhanced_count
        service_state["total_llm_assessments"] += llm_enhanced_count
        
        # Convert signals to response format
        signal_responses = []
        for signal in signals:
            signal_responses.append(SignalResponse(
                symbol=signal['symbol'],
                signal_type=signal['signal_type'],
                confidence=signal['confidence'],
                original_ml_confidence=signal.get('original_ml_confidence'),
                llm_confidence_adjustment=signal.get('llm_confidence_adjustment'),
                llm_reasoning=signal.get('llm_reasoning'),
                llm_enhanced=signal.get('llm_enhanced', False),
                timestamp=signal.get('timestamp', datetime.now().isoformat())
            ))
        
        return {
            "signals_generated": len(signals),
            "llm_enhanced_signals": llm_enhanced_count,
            "signals": signal_responses,
            "generation_time": datetime.now().isoformat()
        }
        
    except Exception as e:
        service_state["last_error"] = str(e)
        logger.error(f"Error generating signals: {e}")
        raise HTTPException(status_code=500, detail=f"Signal generation failed: {e}")

@app.get("/signals/latest")
async def get_latest_signals(limit: int = 10, symbol: Optional[str] = None):
    """Get latest generated signals from database"""
    
    if not generator:
        raise HTTPException(status_code=503, detail="Signal generator not initialized")
    
    try:
        # Get recent signals from database
        recent_signals = generator.get_recent_signals(limit=limit, symbol=symbol)
        
        return {
            "signals": recent_signals,
            "count": len(recent_signals),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error retrieving latest signals: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve signals: {e}")

@app.post("/assess-signal")
async def assess_single_signal(symbol: str, signal_type: str, confidence: float):
    """Assess a single trading signal with LLM"""
    
    if not generator:
        raise HTTPException(status_code=503, detail="Signal generator not initialized")
    
    try:
        # Create a mock signal for assessment
        mock_signal = {
            'symbol': symbol,
            'signal_type': signal_type,
            'confidence': confidence,
            'price': 0,  # Would need to fetch current price
            'sentiment_score': 0,
            'sentiment_confidence': 0,
            'sentiment_trend': 'STABLE'
        }
        
        # Get portfolio context
        portfolio = generator.get_current_portfolio()
        
        # Create market context
        market_context = generator._create_market_context_from_signal(mock_signal, portfolio)
        
        # Get LLM assessment
        assessment = await generator.llm_assessor.assess_signal(
            market_context, signal_type, confidence
        )
        
        return {
            "symbol": symbol,
            "original_confidence": confidence,
            "adjusted_confidence": assessment.adjusted_confidence,
            "confidence_adjustment": assessment.confidence_adjustment,
            "reasoning": assessment.adjustment_reasoning,
            "key_factors": assessment.key_factors,
            "risk_factors": assessment.risk_factors,
            "overall_assessment": assessment.overall_assessment,
            "assessment_time": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error assessing signal: {e}")
        raise HTTPException(status_code=500, detail=f"Signal assessment failed: {e}")

def start_background_generation():
    """Start background signal generation"""
    def background_worker():
        while True:
            try:
                if service_state["status"] == "ready" and generator:
                    logger.info("🔄 Running automated signal generation...")
                    
                    # Run signal generation
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    signals = loop.run_until_complete(
                        generator.generate_signals_with_llm_assessment()
                    )
                    
                    loop.close()
                    
                    # Update metrics
                    service_state["signals_generated_today"] += len(signals)
                    service_state["total_signals_generated"] += len(signals)
                    service_state["last_signal_generation"] = datetime.now().isoformat()
                    
                    llm_enhanced_count = sum(1 for s in signals if s.get('llm_enhanced'))
                    service_state["last_llm_assessments"] = llm_enhanced_count
                    service_state["total_llm_assessments"] += llm_enhanced_count
                    
                    logger.info(f"✅ Generated {len(signals)} signals ({llm_enhanced_count} LLM-enhanced)")
                    
            except Exception as e:
                logger.error(f"Error in background signal generation: {e}")
                service_state["last_error"] = str(e)
            
            # Wait for next generation cycle
            time.sleep(service_state["generation_cycle_minutes"] * 60)
    
    # Start background thread
    thread = threading.Thread(target=background_worker, daemon=True)
    thread.start()
    logger.info(f"🔄 Background signal generation started (every {service_state['generation_cycle_minutes']} minutes)")

if __name__ == "__main__":
    # Configure service
    port = int(os.getenv("LLM_SIGNAL_SERVICE_PORT", "8045"))
    
    logger.info(f"🚀 Starting LLM-Enhanced Signal Service on port {port}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
