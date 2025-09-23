#!/usr/bin/env python3
"""
LLM Trade Validation Service
Comprehensive trade validation using multiple LLM models with full context analysis
"""

import asyncio
import aiohttp
import json
import logging
import mysql.connector
import os
import requests
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ValidationDecision(Enum):
    APPROVE = "approve"
    REJECT = "reject"
    MODIFY = "modify"
    DEFER = "defer"

@dataclass
class TradeContext:
    """Comprehensive trading context for LLM analysis"""
    # Trade Details
    symbol: str
    action: str  # BUY/SELL
    size_usd: float
    current_price: float
    
    # Portfolio Context
    total_portfolio_value: float
    usd_balance: float
    current_positions: Dict[str, Any]
    position_concentration: Dict[str, float]
    
    # Recent Trading History
    recent_trades: List[Dict]
    daily_trading_volume: float
    daily_trade_count: int
    recent_pnl: float
    
    # Market Data
    price_change_24h: float
    volume_24h: float
    volatility_7d: float
    rsi_14: Optional[float] = None
    
    # Sentiment & ML Data
    crypto_sentiment: Optional[float] = None
    stock_sentiment: Optional[float] = None
    ml_signal_confidence: Optional[float] = None
    ml_signal_direction: Optional[str] = None
    
    # Risk Metrics
    portfolio_drawdown: float
    daily_loss_limit: float
    position_size_limit: float
    correlation_risk: Optional[float] = None
    
    # External Context
    news_sentiment: Optional[float] = None
    social_sentiment: Optional[float] = None
    market_conditions: str = "normal"  # bull/bear/volatile/normal
    
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ValidationResult:
    """LLM validation result"""
    decision: ValidationDecision
    confidence: float  # 0.0 to 1.0
    reasoning: str
    risk_assessment: str  # low/medium/high/critical
    modifications: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    alternative_suggestions: List[str] = field(default_factory=list)
    model_consensus: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

class OllamaLLMService:
    """Ollama LLM service integration"""
    
    def __init__(self, base_url: str = "http://ollama.crypto-trading.svc.cluster.local:11434"):
        self.base_url = base_url
        self.available_models = []
        self.model_specializations = {
            "llama2:7b": "general_reasoning",
            "deepseek-coder:1.3b": "technical_analysis",
            "neural-chat:7b": "communication",
            "qwen:1.8b": "fast_analysis",
            "phi3:3.8b": "risk_assessment",
            "gemma:7b": "market_analysis",
            "zephyr:7b-beta": "sentiment_analysis",
            "mistral:7b": "decision_making"
        }
        self.check_available_models()
    
    def check_available_models(self):
        """Check which Ollama models are available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get('models', [])
                self.available_models = [model['name'] for model in models]
                logger.info(f"Available Ollama models: {self.available_models}")
            else:
                logger.warning(f"Failed to get Ollama models: {response.status_code}")
        except Exception as e:
            logger.error(f"Error checking Ollama models: {e}")
    
    async def analyze_trade(self, context: TradeContext, model: str, focus: str) -> Dict[str, Any]:
        """Analyze trade using specific Ollama model"""
        
        prompt = self._build_analysis_prompt(context, focus)
        
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "top_p": 0.9,
                        "num_ctx": 2048
                    }
                }
                
                async with session.post(f"{self.base_url}/api/generate", json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        analysis_text = result.get('response', '')
                        
                        # Parse the structured response
                        analysis = self._parse_llm_response(analysis_text, model, focus)
                        return analysis
                    else:
                        logger.error(f"Ollama API error for {model}: {response.status}")
                        return self._get_fallback_analysis(model, focus)
                        
        except Exception as e:
            logger.error(f"Error analyzing with {model}: {e}")
            return self._get_fallback_analysis(model, focus)
    
    def _build_analysis_prompt(self, context: TradeContext, focus: str) -> str:
        """Build comprehensive analysis prompt"""
        
        prompt = f"""
TRADING DECISION ANALYSIS REQUEST

TRADE DETAILS:
- Symbol: {context.symbol}
- Action: {context.action}
- Size: ${context.size_usd:,.2f}
- Current Price: ${context.current_price:,.2f}
- Price Change 24h: {context.price_change_24h:+.2%}

PORTFOLIO CONTEXT:
- Total Value: ${context.total_portfolio_value:,.2f}
- USD Balance: ${context.usd_balance:,.2f}
- Trade Size vs Portfolio: {(context.size_usd / context.total_portfolio_value * 100):.1f}%
- Active Positions: {len(context.current_positions)}

RECENT TRADING:
- Daily Volume: ${context.daily_trading_volume:,.2f}
- Daily Trades: {context.daily_trade_count}
- Recent P&L: ${context.recent_pnl:+,.2f}

MARKET CONDITIONS:
- Volatility (7d): {context.volatility_7d:.2%}
- Volume 24h: ${context.volume_24h:,.0f}
- Market Conditions: {context.market_conditions}

SENTIMENT DATA:
- Crypto Sentiment: {context.crypto_sentiment or 'N/A'}
- Stock Sentiment: {context.stock_sentiment or 'N/A'}
- ML Signal: {context.ml_signal_direction} ({context.ml_signal_confidence or 'N/A'})

RISK METRICS:
- Portfolio Drawdown: {context.portfolio_drawdown:.2%}
- Daily Loss Limit: ${context.daily_loss_limit:,.2f}
- Position Size Limit: ${context.position_size_limit:,.2f}

ANALYSIS FOCUS: {focus.upper()}

Please provide a structured analysis with:
1. DECISION: APPROVE/REJECT/MODIFY/DEFER
2. CONFIDENCE: 0.0-1.0 score
3. REASONING: Detailed explanation
4. RISK_LEVEL: low/medium/high/critical
5. WARNINGS: Any concerns or risks
6. SUGGESTIONS: Alternative approaches

Format your response as:
DECISION: [your decision]
CONFIDENCE: [0.0-1.0]
REASONING: [detailed explanation]
RISK_LEVEL: [risk level]
WARNINGS: [bullet points if any]
SUGGESTIONS: [bullet points if any]
"""
        
        return prompt
    
    def _parse_llm_response(self, response: str, model: str, focus: str) -> Dict[str, Any]:
        """Parse structured LLM response"""
        
        try:
            lines = response.strip().split('\n')
            analysis = {
                'model': model,
                'focus': focus,
                'decision': 'defer',
                'confidence': 0.5,
                'reasoning': 'Unable to parse response',
                'risk_level': 'medium',
                'warnings': [],
                'suggestions': []
            }
            
            current_section = None
            for line in lines:
                line = line.strip()
                if line.startswith('DECISION:'):
                    analysis['decision'] = line.split(':', 1)[1].strip().lower()
                elif line.startswith('CONFIDENCE:'):
                    try:
                        analysis['confidence'] = float(line.split(':', 1)[1].strip())
                    except:
                        analysis['confidence'] = 0.5
                elif line.startswith('REASONING:'):
                    analysis['reasoning'] = line.split(':', 1)[1].strip()
                elif line.startswith('RISK_LEVEL:'):
                    analysis['risk_level'] = line.split(':', 1)[1].strip().lower()
                elif line.startswith('WARNINGS:'):
                    current_section = 'warnings'
                elif line.startswith('SUGGESTIONS:'):
                    current_section = 'suggestions'
                elif line.startswith('- ') or line.startswith('• '):
                    if current_section == 'warnings':
                        analysis['warnings'].append(line[2:])
                    elif current_section == 'suggestions':
                        analysis['suggestions'].append(line[2:])
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error parsing LLM response from {model}: {e}")
            return self._get_fallback_analysis(model, focus)
    
    def _get_fallback_analysis(self, model: str, focus: str) -> Dict[str, Any]:
        """Fallback analysis when LLM fails"""
        return {
            'model': model,
            'focus': focus,
            'decision': 'defer',
            'confidence': 0.3,
            'reasoning': f'LLM analysis failed for {model}, deferring to risk management rules',
            'risk_level': 'medium',
            'warnings': ['LLM analysis unavailable'],
            'suggestions': ['Use conservative position sizing', 'Monitor market conditions closely']
        }

class TradeValidationService:
    """Main trade validation service using multiple LLM models"""
    
    def __init__(self):
        self.ollama_service = OllamaLLMService()
        self.db_config = {
            'host': os.getenv('DATABASE_HOST', 'host.docker.internal'),
            'user': os.getenv('DATABASE_USER', 'news_collector'),
            'password': os.getenv('DATABASE_PASSWORD', '99Rules!'),
            'database': os.getenv('DATABASE_NAME', 'crypto_transactions')
        }
        
        # Risk limits
        self.max_daily_trades = int(os.getenv('MAX_DAILY_TRADES', '50'))
        self.max_daily_loss = float(os.getenv('MAX_DAILY_LOSS_USD', '500'))
        self.max_position_size = float(os.getenv('MAX_POSITION_SIZE_USD', '1000'))
        
        logger.info("LLM Trade Validation Service initialized")
    
    async def validate_trade(self, context: TradeContext) -> ValidationResult:
        """Comprehensive trade validation using multiple LLM models"""
        
        try:
            # Run parallel analysis with different models
            analysis_tasks = []
            
            # Select models based on availability and specialization
            model_assignments = {
                "phi3:3.8b": "risk_assessment",
                "mistral:7b": "decision_making", 
                "llama2:7b": "general_reasoning",
                "deepseek-coder:1.3b": "technical_analysis"
            }
            
            for model, focus in model_assignments.items():
                if model in self.ollama_service.available_models:
                    task = self.ollama_service.analyze_trade(context, model, focus)
                    analysis_tasks.append(task)
            
            # Run analyses in parallel
            if analysis_tasks:
                analyses = await asyncio.gather(*analysis_tasks, return_exceptions=True)
                
                # Filter out exceptions
                valid_analyses = [a for a in analyses if not isinstance(a, Exception)]
                
                if valid_analyses:
                    # Calculate consensus
                    consensus = self._calculate_consensus(valid_analyses)
                    
                    # Apply additional rule-based validation
                    final_result = await self._apply_final_validation(context, consensus)
                    
                    return final_result
            
            # Fallback to rule-based validation only
            logger.warning("No LLM analyses available, using rule-based validation")
            return await self._rule_based_validation(context)
            
        except Exception as e:
            logger.error(f"Error in trade validation: {e}")
            return ValidationResult(
                decision=ValidationDecision.REJECT,
                confidence=0.0,
                reasoning=f"Validation error: {e}",
                risk_assessment="critical"
            )
    
    def _calculate_consensus(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate consensus from multiple LLM analyses"""
        
        if not analyses:
            return {
                'decision': 'reject',
                'confidence': 0.0,
                'reasoning': 'No analyses available',
                'risk_level': 'critical'
            }
        
        # Count decisions
        decisions = [a.get('decision', 'defer') for a in analyses]
        decision_counts = {}
        for decision in decisions:
            decision_counts[decision] = decision_counts.get(decision, 0) + 1
        
        # Get most common decision
        consensus_decision = max(decision_counts, key=decision_counts.get)
        
        # Calculate average confidence
        confidences = [a.get('confidence', 0.5) for a in analyses]
        avg_confidence = sum(confidences) / len(confidences)
        
        # Adjust confidence based on consensus strength
        consensus_strength = decision_counts[consensus_decision] / len(analyses)
        adjusted_confidence = avg_confidence * consensus_strength
        
        # Combine reasoning
        reasonings = [a.get('reasoning', '') for a in analyses]
        combined_reasoning = " | ".join(reasonings[:3])  # Limit length
        
        # Get most severe risk level
        risk_levels = [a.get('risk_level', 'medium') for a in analyses]
        risk_hierarchy = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
        highest_risk = max(risk_levels, key=lambda x: risk_hierarchy.get(x, 2))
        
        # Combine warnings and suggestions
        all_warnings = []
        all_suggestions = []
        for analysis in analyses:
            all_warnings.extend(analysis.get('warnings', []))
            all_suggestions.extend(analysis.get('suggestions', []))
        
        return {
            'decision': consensus_decision,
            'confidence': adjusted_confidence,
            'reasoning': combined_reasoning,
            'risk_level': highest_risk,
            'warnings': list(set(all_warnings))[:5],  # Unique, limited
            'suggestions': list(set(all_suggestions))[:5],
            'consensus_strength': consensus_strength,
            'model_count': len(analyses)
        }
    
    async def _apply_final_validation(self, context: TradeContext, consensus: Dict[str, Any]) -> ValidationResult:
        """Apply final validation rules and create result"""
        
        decision = ValidationDecision(consensus['decision'])
        confidence = consensus['confidence']
        warnings = consensus.get('warnings', [])
        
        # Apply hard risk limits
        if context.size_usd > self.max_position_size:
            decision = ValidationDecision.REJECT
            confidence = 0.0
            warnings.append(f"Trade size ${context.size_usd:,.2f} exceeds limit ${self.max_position_size:,.2f}")
        
        if context.daily_trade_count >= self.max_daily_trades:
            decision = ValidationDecision.REJECT
            confidence = 0.0
            warnings.append(f"Daily trade limit reached ({context.daily_trade_count}/{self.max_daily_trades})")
        
        if context.recent_pnl <= -self.max_daily_loss:
            decision = ValidationDecision.REJECT
            confidence = 0.0
            warnings.append(f"Daily loss limit exceeded (${context.recent_pnl:,.2f})")
        
        # Reduce confidence for high-risk scenarios
        if consensus['risk_level'] == 'high':
            confidence *= 0.7
        elif consensus['risk_level'] == 'critical':
            confidence *= 0.3
            if decision == ValidationDecision.APPROVE:
                decision = ValidationDecision.DEFER
        
        # Portfolio concentration check
        if context.action == 'BUY':
            position_percentage = (context.size_usd / context.total_portfolio_value) * 100
            if position_percentage > 20:  # 20% concentration limit
                warnings.append(f"High position concentration: {position_percentage:.1f}%")
                confidence *= 0.8
        
        return ValidationResult(
            decision=decision,
            confidence=confidence,
            reasoning=consensus['reasoning'],
            risk_assessment=consensus['risk_level'],
            warnings=warnings,
            alternative_suggestions=consensus.get('suggestions', []),
            model_consensus={f"model_{i}": a.get('decision', 'defer') for i, a in enumerate(consensus.get('analyses', []))},
        )
    
    async def _rule_based_validation(self, context: TradeContext) -> ValidationResult:
        """Fallback rule-based validation when LLM is unavailable"""
        
        warnings = []
        
        # Basic risk checks
        if context.size_usd > self.max_position_size:
            return ValidationResult(
                decision=ValidationDecision.REJECT,
                confidence=1.0,
                reasoning=f"Trade size ${context.size_usd:,.2f} exceeds maximum ${self.max_position_size:,.2f}",
                risk_assessment="high",
                warnings=["Position size limit exceeded"]
            )
        
        if context.daily_trade_count >= self.max_daily_trades:
            return ValidationResult(
                decision=ValidationDecision.REJECT,
                confidence=1.0,
                reasoning="Daily trade limit reached",
                risk_assessment="medium",
                warnings=["Daily trade limit exceeded"]
            )
        
        # Portfolio checks
        position_percentage = (context.size_usd / context.total_portfolio_value) * 100
        confidence = 1.0
        
        if position_percentage > 15:
            warnings.append(f"High position concentration: {position_percentage:.1f}%")
            confidence *= 0.8
        
        if context.volatility_7d > 0.3:  # 30% weekly volatility
            warnings.append("High market volatility detected")
            confidence *= 0.9
        
        return ValidationResult(
            decision=ValidationDecision.APPROVE,
            confidence=confidence,
            reasoning="Rule-based validation passed (LLM unavailable)",
            risk_assessment="low" if not warnings else "medium",
            warnings=warnings
        )
    
    async def get_portfolio_context(self, symbol: str) -> Dict[str, Any]:
        """Get current portfolio context from database"""
        
        try:
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor(dictionary=True)
            
            # Get current positions
            cursor.execute("""
                SELECT symbol, quantity, current_value, unrealized_pnl
                FROM portfolio_positions 
                WHERE quantity > 0
            """)
            positions = cursor.fetchall()
            
            # Get recent trades (last 24h)
            cursor.execute("""
                SELECT symbol, action, size_usd, price, timestamp
                FROM trades 
                WHERE timestamp >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
                ORDER BY timestamp DESC
                LIMIT 20
            """)
            recent_trades = cursor.fetchall()
            
            # Calculate portfolio metrics
            total_value = sum(float(pos['current_value']) for pos in positions)
            daily_volume = sum(float(trade['size_usd']) for trade in recent_trades)
            daily_trades = len(recent_trades)
            daily_pnl = sum(float(pos['unrealized_pnl']) for pos in positions)
            
            cursor.close()
            conn.close()
            
            return {
                'total_portfolio_value': total_value,
                'positions': positions,
                'recent_trades': recent_trades,
                'daily_trading_volume': daily_volume,
                'daily_trade_count': daily_trades,
                'recent_pnl': daily_pnl
            }
            
        except Exception as e:
            logger.error(f"Error getting portfolio context: {e}")
            return {
                'total_portfolio_value': 10000.0,  # Fallback
                'positions': [],
                'recent_trades': [],
                'daily_trading_volume': 0.0,
                'daily_trade_count': 0,
                'recent_pnl': 0.0
            }

# FastAPI application
app = FastAPI(title="LLM Trade Validation Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global service instance
validation_service = None

@app.on_event("startup")
async def startup_event():
    """Initialize validation service on startup"""
    global validation_service
    validation_service = TradeValidationService()
    logger.info("LLM Trade Validation Service started")

class TradeValidationRequest(BaseModel):
    symbol: str
    action: str
    size_usd: float
    current_price: float
    ml_signal_confidence: Optional[float] = None
    ml_signal_direction: Optional[str] = None

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "llm_trade_validation",
        "timestamp": datetime.now().isoformat(),
        "available_models": validation_service.ollama_service.available_models if validation_service else []
    }

@app.post("/validate")
async def validate_trade(request: TradeValidationRequest):
    """Validate a trade using LLM analysis"""
    
    if not validation_service:
        raise HTTPException(status_code=500, detail="Validation service not initialized")
    
    try:
        # Get portfolio context
        portfolio_context = await validation_service.get_portfolio_context(request.symbol)
        
        # Build comprehensive context
        context = TradeContext(
            symbol=request.symbol,
            action=request.action,
            size_usd=request.size_usd,
            current_price=request.current_price,
            total_portfolio_value=portfolio_context['total_portfolio_value'],
            usd_balance=portfolio_context.get('usd_balance', 1000.0),
            current_positions=portfolio_context['positions'],
            position_concentration={},  # Would calculate from positions
            recent_trades=portfolio_context['recent_trades'],
            daily_trading_volume=portfolio_context['daily_trading_volume'],
            daily_trade_count=portfolio_context['daily_trade_count'],
            recent_pnl=portfolio_context['recent_pnl'],
            price_change_24h=0.0,  # Would get from market data
            volume_24h=0.0,  # Would get from market data
            volatility_7d=0.1,  # Would calculate from price history
            ml_signal_confidence=request.ml_signal_confidence,
            ml_signal_direction=request.ml_signal_direction,
            portfolio_drawdown=0.0,  # Would calculate
            daily_loss_limit=validation_service.max_daily_loss,
            position_size_limit=validation_service.max_position_size
        )
        
        # Perform validation
        result = await validation_service.validate_trade(context)
        
        return {
            "decision": result.decision.value,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
            "risk_assessment": result.risk_assessment,
            "warnings": result.warnings,
            "suggestions": result.alternative_suggestions,
            "timestamp": result.timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {e}")

@app.get("/models")
async def get_available_models():
    """Get available Ollama models"""
    if not validation_service:
        raise HTTPException(status_code=500, detail="Service not initialized")
    
    return {
        "available_models": validation_service.ollama_service.available_models,
        "model_specializations": validation_service.ollama_service.model_specializations
    }

if __name__ == "__main__":
    port = int(os.getenv('LLM_VALIDATION_PORT', 8035))
    uvicorn.run(app, host="0.0.0.0", port=port)