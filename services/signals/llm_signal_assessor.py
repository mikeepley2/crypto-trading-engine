#!/usr/bin/env python3
"""
LLM Signal Assessor Service
Evaluates and adjusts ML trading signals based on LLM contextual analysis
"""

import asyncio
import aiohttp
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MarketContext:
    """Market context for LLM assessment"""
    symbol: str
    current_price: float
    price_change_24h: float
    volume_24h: float
    market_cap: Optional[float]
    technical_indicators: Dict
    sentiment_data: Dict
    news_impact: Dict
    ml_prediction: Dict
    portfolio_context: Dict

@dataclass
class LLMAssessment:
    """LLM assessment result"""
    original_confidence: float
    adjusted_confidence: float
    confidence_adjustment: float
    adjustment_reasoning: str
    key_factors: List[str]
    risk_factors: List[str]
    market_context_score: float
    technical_alignment_score: float
    sentiment_alignment_score: float
    overall_assessment: str

class LLMSignalAssessor:
    """
    LLM-powered signal assessment and adjustment service
    
    This service takes ML-generated trading signals and uses LLM reasoning
    to evaluate market context and adjust signal confidence accordingly.
    """
    
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.xai_api_key = os.getenv("XAI_API_KEY", "")
        self.openai_url = "https://api.openai.com/v1/chat/completions"
        self.xai_url = "https://api.x.ai/v1/chat/completions"
        
        # Assessment configuration
        self.max_adjustment = 0.3  # Maximum confidence adjustment (±30%)
        self.min_confidence = 0.1  # Minimum confidence after adjustment
        self.max_confidence = 0.95  # Maximum confidence after adjustment
        
        logger.info("🧠 LLM Signal Assessor initialized")
    
    async def assess_signal(self, market_context: MarketContext, 
                          signal_type: str, original_confidence: float) -> LLMAssessment:
        """
        Assess and potentially adjust a trading signal using LLM analysis
        
        Args:
            market_context: Complete market context for the signal
            signal_type: 'BUY', 'SELL', or 'HOLD'
            original_confidence: Original ML confidence (0.0 to 1.0)
            
        Returns:
            LLMAssessment with adjusted confidence and reasoning
        """
        try:
            # Create assessment prompt
            prompt = self._create_assessment_prompt(market_context, signal_type, original_confidence)
            
            # Get LLM assessment
            llm_response = await self._call_llm_api(prompt)
            
            # Parse LLM response
            assessment = self._parse_llm_assessment(llm_response, original_confidence)
            
            # Apply confidence adjustment with safety limits
            adjusted_confidence = self._apply_confidence_adjustment(
                original_confidence, assessment.confidence_adjustment
            )
            
            # Update assessment with final values
            assessment.adjusted_confidence = adjusted_confidence
            
            logger.info(f"🧠 {market_context.symbol} LLM Assessment: "
                       f"{original_confidence:.3f} → {adjusted_confidence:.3f} "
                       f"({assessment.confidence_adjustment:+.3f}) - {assessment.overall_assessment}")
            
            return assessment
            
        except Exception as e:
            logger.error(f"Error in LLM signal assessment for {market_context.symbol}: {e}")
            
            # Return neutral assessment on error
            return LLMAssessment(
                original_confidence=original_confidence,
                adjusted_confidence=original_confidence,
                confidence_adjustment=0.0,
                adjustment_reasoning="LLM assessment unavailable - using original ML confidence",
                key_factors=["ML_SIGNAL_ONLY"],
                risk_factors=["LLM_UNAVAILABLE"],
                market_context_score=0.5,
                technical_alignment_score=0.5,
                sentiment_alignment_score=0.5,
                overall_assessment="NEUTRAL"
            )
    
    def _create_assessment_prompt(self, context: MarketContext, 
                                signal_type: str, confidence: float) -> str:
        """Create detailed assessment prompt for LLM"""
        
        ml_data = context.ml_prediction
        tech_data = context.technical_indicators
        sentiment_data = context.sentiment_data
        portfolio_data = context.portfolio_context
        
        prompt = f"""
You are an expert cryptocurrency trading analyst. Assess the following ML trading signal and adjust its confidence based on your analysis of market conditions.

CURRENT ML SIGNAL:
- Symbol: {context.symbol}
- Signal: {signal_type} 
- ML Confidence: {confidence:.3f} ({confidence*100:.1f}%)
- ML Model: XGBoost with 120 features

MARKET DATA:
- Current Price: ${context.current_price:.2f}
- 24h Change: {context.price_change_24h:.2f}%
- Volume: ${context.volume_24h:,.0f}
- Market Cap: ${context.market_cap or 'N/A'}

TECHNICAL ANALYSIS:
- RSI: {tech_data.get('rsi', 'N/A')}
- MACD: {tech_data.get('macd', 'N/A')}
- Volume Trend: {tech_data.get('volume_trend', 'N/A')}
- Price Momentum: {tech_data.get('momentum', 'N/A')}
- Support/Resistance: {tech_data.get('support_resistance', 'N/A')}

SENTIMENT ANALYSIS:
- Sentiment Score: {sentiment_data.get('sentiment_score', 0):.3f}
- Sentiment Trend: {sentiment_data.get('sentiment_trend', 'STABLE')}
- News Impact: {context.news_impact.get('impact_score', 0):.2f}
- Social Volume: {sentiment_data.get('volume_score', 0):.2f}

PORTFOLIO CONTEXT:
- Current Position: {portfolio_data.get('position_weight', 0):.1f}% of portfolio
- Cash Available: {portfolio_data.get('cash_percentage', 0):.1f}%
- Portfolio Heat: {portfolio_data.get('portfolio_heat', 0):.1f}%
- Correlation Risk: {portfolio_data.get('correlation_risk', 'LOW')}

MARKET REGIME:
- Overall Regime: {context.ml_prediction.get('market_regime', 'SIDEWAYS')}
- Volatility: {context.technical_indicators.get('volatility', 'NORMAL')}
- Risk Environment: {portfolio_data.get('risk_environment', 'NORMAL')}

ASSESSMENT TASK:
Analyze all the above information and determine if the ML signal confidence should be:
1. INCREASED (if market conditions strongly support the ML signal)
2. DECREASED (if market conditions contradict or weaken the ML signal)  
3. MAINTAINED (if market conditions are neutral or mixed)

Provide your assessment in this JSON format:
{{
    "confidence_adjustment": -0.3 to +0.3,
    "overall_assessment": "BULLISH/BEARISH/NEUTRAL",
    "market_context_score": 0.0 to 1.0,
    "technical_alignment_score": 0.0 to 1.0, 
    "sentiment_alignment_score": 0.0 to 1.0,
    "adjustment_reasoning": "Detailed explanation of your reasoning",
    "key_supporting_factors": ["factor1", "factor2", "factor3"],
    "key_risk_factors": ["risk1", "risk2", "risk3"]
}}

Consider these guidelines:
- Positive adjustment (+): ML signal is supported by strong market context
- Negative adjustment (-): ML signal contradicts market conditions
- Zero adjustment (0): Mixed or insufficient evidence for adjustment
- Maximum adjustment: ±0.3 (±30%)
"""
        
        return prompt
    
    async def _call_llm_api(self, prompt: str) -> str:
        """Call LLM API (OpenAI or XAI/Grok)"""
        
        # Try OpenAI GPT-4 first if available
        if self.openai_api_key:
            try:
                return await self._call_openai(prompt)
            except Exception as e:
                logger.warning(f"OpenAI API failed: {e}, trying XAI/Grok")
        
        # Fallback to XAI/Grok if available
        if self.xai_api_key:
            try:
                return await self._call_xai(prompt)
            except Exception as e:
                logger.warning(f"XAI API failed: {e}")
        
        # If no APIs available, return mock response
        return self._generate_mock_assessment(prompt)
    
    async def _call_openai(self, prompt: str) -> str:
        """Call OpenAI GPT-4 API"""
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "gpt-4",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert cryptocurrency trading analyst with 20+ years of experience. Provide precise, data-driven assessments of trading signals based on comprehensive market analysis."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "temperature": 0.2,  # Low temperature for consistent analysis
            "max_tokens": 800
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(self.openai_url, headers=headers, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]
                else:
                    error_text = await response.text()
                    raise Exception(f"OpenAI API error: {response.status} - {error_text}")
    
    async def _call_xai(self, prompt: str) -> str:
        """Call XAI/Grok API"""
        headers = {
            "Authorization": f"Bearer {self.xai_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "grok-beta",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a sophisticated cryptocurrency trading analyst. Analyze market conditions and provide objective assessments of trading signal strength."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.2,
            "max_tokens": 800
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(self.xai_url, headers=headers, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]
                else:
                    error_text = await response.text()
                    raise Exception(f"XAI API error: {response.status} - {error_text}")
    
    def _generate_mock_assessment(self, prompt: str) -> str:
        """Generate mock LLM assessment for testing"""
        
        # Extract symbol from prompt for context-aware mock
        symbol = "BTC"
        for s in ["BTC", "ETH", "SOL", "ADA", "MATIC"]:
            if s in prompt:
                symbol = s
                break
        
        # Generate realistic mock assessment
        import random
        
        adjustment = random.uniform(-0.2, 0.2)
        assessment = random.choice(["BULLISH", "BEARISH", "NEUTRAL"])
        
        return f"""
{{
    "confidence_adjustment": {adjustment:.3f},
    "overall_assessment": "{assessment}",
    "market_context_score": {random.uniform(0.4, 0.8):.2f},
    "technical_alignment_score": {random.uniform(0.4, 0.8):.2f},
    "sentiment_alignment_score": {random.uniform(0.4, 0.8):.2f},
    "adjustment_reasoning": "Mock LLM assessment: Market conditions show mixed signals for {symbol}. Technical indicators suggest moderate {assessment.lower()} bias with sentiment alignment supporting the adjustment.",
    "key_supporting_factors": ["Technical momentum", "Sentiment trend", "Volume confirmation"],
    "key_risk_factors": ["Market volatility", "Correlation risk", "Liquidity concerns"]
}}
"""
    
    def _parse_llm_assessment(self, llm_response: str, original_confidence: float) -> LLMAssessment:
        """Parse LLM response into structured assessment"""
        
        try:
            # Extract JSON from response
            json_start = llm_response.find('{')
            json_end = llm_response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = llm_response[json_start:json_end]
                data = json.loads(json_str)
                
                return LLMAssessment(
                    original_confidence=original_confidence,
                    adjusted_confidence=0.0,  # Will be set later
                    confidence_adjustment=float(data.get('confidence_adjustment', 0.0)),
                    adjustment_reasoning=data.get('adjustment_reasoning', ''),
                    key_factors=data.get('key_supporting_factors', []),
                    risk_factors=data.get('key_risk_factors', []),
                    market_context_score=float(data.get('market_context_score', 0.5)),
                    technical_alignment_score=float(data.get('technical_alignment_score', 0.5)),
                    sentiment_alignment_score=float(data.get('sentiment_alignment_score', 0.5)),
                    overall_assessment=data.get('overall_assessment', 'NEUTRAL')
                )
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
        
        # Fallback assessment
        return LLMAssessment(
            original_confidence=original_confidence,
            adjusted_confidence=original_confidence,
            confidence_adjustment=0.0,
            adjustment_reasoning="Failed to parse LLM response - maintaining original confidence",
            key_factors=["PARSE_ERROR"],
            risk_factors=["LLM_RESPONSE_ERROR"],
            market_context_score=0.5,
            technical_alignment_score=0.5,
            sentiment_alignment_score=0.5,
            overall_assessment="NEUTRAL"
        )
    
    def _apply_confidence_adjustment(self, original_confidence: float, 
                                   adjustment: float) -> float:
        """Apply confidence adjustment with safety limits"""
        
        # Clamp adjustment to maximum allowed
        clamped_adjustment = max(-self.max_adjustment, min(self.max_adjustment, adjustment))
        
        # Apply adjustment
        adjusted = original_confidence + clamped_adjustment
        
        # Apply min/max confidence limits
        final_confidence = max(self.min_confidence, min(self.max_confidence, adjusted))
        
        return final_confidence
    
    def create_market_context(self, symbol: str, signal_data: Dict, 
                            portfolio_data: Dict, market_data: Dict) -> MarketContext:
        """Create market context from various data sources"""
        
        return MarketContext(
            symbol=symbol,
            current_price=signal_data.get('price', 0),
            price_change_24h=signal_data.get('price_change_24h', 0),
            volume_24h=market_data.get('volume_24h', 0),
            market_cap=market_data.get('market_cap'),
            technical_indicators={
                'rsi': signal_data.get('rsi'),
                'macd': signal_data.get('macd'),
                'volume_trend': signal_data.get('volume_trend'),
                'momentum': signal_data.get('momentum'),
                'support_resistance': signal_data.get('support_resistance'),
                'volatility': signal_data.get('volatility')
            },
            sentiment_data={
                'sentiment_score': signal_data.get('sentiment_score', 0),
                'sentiment_trend': signal_data.get('sentiment_trend', 'STABLE'),
                'confidence': signal_data.get('sentiment_confidence', 0),
                'volume_score': signal_data.get('volume_score', 0)
            },
            news_impact={
                'impact_score': signal_data.get('news_impact', 0),
                'headline_count': signal_data.get('news_count', 0)
            },
            ml_prediction={
                'confidence': signal_data.get('confidence', 0),
                'prediction_probability': signal_data.get('prediction_probability', 0),
                'model_version': signal_data.get('model_version', ''),
                'market_regime': signal_data.get('market_regime', 'SIDEWAYS')
            },
            portfolio_context={
                'position_weight': portfolio_data.get('position_weight', 0),
                'cash_percentage': portfolio_data.get('cash_percentage', 0),
                'portfolio_heat': portfolio_data.get('portfolio_heat', 0),
                'correlation_risk': portfolio_data.get('correlation_risk', 'LOW'),
                'risk_environment': portfolio_data.get('risk_environment', 'NORMAL')
            }
        )
