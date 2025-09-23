"""
LLM-Powered Market Analysis Service
Intelligent market analysis using Large Language Models for enhanced trading decisions
"""

import asyncio
import aiohttp
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from decimal import Decimal

class AnalysisType(Enum):
    SENTIMENT = "sentiment"
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    RISK_ASSESSMENT = "risk_assessment"
    TRADE_REASONING = "trade_reasoning"
    MARKET_CONDITIONS = "market_conditions"

class Sentiment(Enum):
    VERY_BEARISH = -2
    BEARISH = -1
    NEUTRAL = 0
    BULLISH = 1
    VERY_BULLISH = 2

@dataclass
class MarketContext:
    """Market context data for LLM analysis"""
    symbol: str
    current_price: Decimal
    price_change_24h: float
    volume_24h: Decimal
    market_cap: Optional[Decimal] = None
    technical_indicators: Dict[str, Any] = field(default_factory=dict)
    recent_news: List[Dict] = field(default_factory=list)
    social_sentiment: Optional[float] = None
    fear_greed_index: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class LLMAnalysis:
    """LLM analysis result"""
    analysis_type: AnalysisType
    symbol: str
    confidence: float  # 0.0 to 1.0
    sentiment: Sentiment
    reasoning: str
    key_factors: List[str]
    risk_level: str  # 'low', 'medium', 'high'
    recommended_action: str  # 'buy', 'sell', 'hold', 'avoid'
    price_target: Optional[Decimal] = None
    stop_loss_suggestion: Optional[Decimal] = None
    time_horizon: str = "short"  # 'short', 'medium', 'long'
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

class LLMProvider:
    """Abstract base class for LLM providers"""
    
    async def analyze_market(self, context: MarketContext, analysis_type: AnalysisType) -> LLMAnalysis:
        raise NotImplementedError

class OpenAIProvider(LLMProvider):
    """OpenAI GPT-4 provider for market analysis"""
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.openai.com/v1/chat/completions"
        self.logger = logging.getLogger(__name__)
    
    async def analyze_market(self, context: MarketContext, analysis_type: AnalysisType) -> LLMAnalysis:
        """Perform market analysis using OpenAI GPT-4"""
        
        try:
            prompt = self._build_analysis_prompt(context, analysis_type)
            
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": self._get_system_prompt(analysis_type)},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,  # Lower temperature for more consistent analysis
                    "max_tokens": 1000
                }
                
                async with session.post(self.base_url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        analysis_text = result["choices"][0]["message"]["content"]
                        return self._parse_analysis_response(analysis_text, context, analysis_type)
                    else:
                        error_text = await response.text()
                        self.logger.error(f"OpenAI API error: {response.status} - {error_text}")
                        return self._create_fallback_analysis(context, analysis_type)
        
        except Exception as e:
            self.logger.error(f"Error in OpenAI analysis: {e}")
            return self._create_fallback_analysis(context, analysis_type)
    
    def _get_system_prompt(self, analysis_type: AnalysisType) -> str:
        """Get system prompt based on analysis type"""
        
        base_prompt = """You are an expert cryptocurrency trader and analyst with 20+ years of experience in financial markets. You provide detailed, actionable analysis based on multiple data sources including technical indicators, market sentiment, news, and fundamental analysis.

Always provide:
1. Clear confidence level (0.0 to 1.0)
2. Specific reasoning with key factors
3. Risk assessment (low/medium/high)
4. Recommended action (buy/sell/hold/avoid)
5. Price targets and stop-loss levels when appropriate
6. Time horizon for the analysis

Be objective, data-driven, and acknowledge uncertainty when it exists."""
        
        type_specific = {
            AnalysisType.SENTIMENT: "Focus on market sentiment analysis from news, social media, and market psychology indicators.",
            AnalysisType.TECHNICAL: "Focus on technical analysis using price action, chart patterns, and technical indicators.",
            AnalysisType.FUNDAMENTAL: "Focus on fundamental analysis including technology, adoption, partnerships, and intrinsic value.",
            AnalysisType.RISK_ASSESSMENT: "Focus on risk analysis including volatility, correlation, market conditions, and potential downside scenarios.",
            AnalysisType.TRADE_REASONING: "Focus on providing clear reasoning for specific trade recommendations with entry/exit strategies.",
            AnalysisType.MARKET_CONDITIONS: "Focus on overall market conditions and their impact on cryptocurrency prices."
        }
        
        return f"{base_prompt}\n\n{type_specific.get(analysis_type, '')}"
    
    def _build_analysis_prompt(self, context: MarketContext, analysis_type: AnalysisType) -> str:
        """Build analysis prompt with market context"""
        
        prompt = f"""Please analyze {context.symbol} based on the following market data:

CURRENT MARKET DATA:
- Symbol: {context.symbol}
- Current Price: ${context.current_price}
- 24h Change: {context.price_change_24h:.2f}%
- 24h Volume: ${context.volume_24h:,.2f}
- Market Cap: ${context.market_cap:,.2f if context.market_cap else 'N/A'}

TECHNICAL INDICATORS:
{json.dumps(context.technical_indicators, indent=2)}

RECENT NEWS:
{self._format_news(context.recent_news)}

MARKET SENTIMENT:
- Social Sentiment Score: {context.social_sentiment or 'N/A'}
- Fear & Greed Index: {context.fear_greed_index or 'N/A'}

Please provide a detailed {analysis_type.value} analysis in the following JSON format:
{{
    "confidence": 0.0-1.0,
    "sentiment": "very_bearish/bearish/neutral/bullish/very_bullish",
    "reasoning": "detailed explanation of your analysis",
    "key_factors": ["factor1", "factor2", "factor3"],
    "risk_level": "low/medium/high",
    "recommended_action": "buy/sell/hold/avoid",
    "price_target": null or price,
    "stop_loss_suggestion": null or price,
    "time_horizon": "short/medium/long"
}}"""
        
        return prompt
    
    def _format_news(self, news: List[Dict]) -> str:
        """Format news for prompt"""
        if not news:
            return "No recent news available"
        
        formatted = []
        for item in news[:5]:  # Limit to 5 most recent
            formatted.append(f"- {item.get('title', 'No title')} ({item.get('source', 'Unknown source')})")
        
        return "\n".join(formatted)
    
    def _parse_analysis_response(self, response: str, context: MarketContext, analysis_type: AnalysisType) -> LLMAnalysis:
        """Parse LLM response into structured analysis"""
        
        try:
            # Try to extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)
                
                # Map sentiment string to enum
                sentiment_map = {
                    "very_bearish": Sentiment.VERY_BEARISH,
                    "bearish": Sentiment.BEARISH,
                    "neutral": Sentiment.NEUTRAL,
                    "bullish": Sentiment.BULLISH,
                    "very_bullish": Sentiment.VERY_BULLISH
                }
                
                return LLMAnalysis(
                    analysis_type=analysis_type,
                    symbol=context.symbol,
                    confidence=float(data.get('confidence', 0.5)),
                    sentiment=sentiment_map.get(data.get('sentiment', 'neutral'), Sentiment.NEUTRAL),
                    reasoning=data.get('reasoning', ''),
                    key_factors=data.get('key_factors', []),
                    risk_level=data.get('risk_level', 'medium'),
                    recommended_action=data.get('recommended_action', 'hold'),
                    price_target=Decimal(str(data['price_target'])) if data.get('price_target') else None,
                    stop_loss_suggestion=Decimal(str(data['stop_loss_suggestion'])) if data.get('stop_loss_suggestion') else None,
                    time_horizon=data.get('time_horizon', 'short'),
                    metadata={'raw_response': response}
                )
            else:
                # Fallback if no JSON found
                return self._create_fallback_analysis(context, analysis_type, response)
                
        except Exception as e:
            self.logger.error(f"Error parsing LLM response: {e}")
            return self._create_fallback_analysis(context, analysis_type, response)
    
    def _create_fallback_analysis(self, context: MarketContext, analysis_type: AnalysisType, raw_response: str = "") -> LLMAnalysis:
        """Create fallback analysis when LLM fails"""
        
        return LLMAnalysis(
            analysis_type=analysis_type,
            symbol=context.symbol,
            confidence=0.3,  # Low confidence for fallback
            sentiment=Sentiment.NEUTRAL,
            reasoning="LLM analysis unavailable - using fallback logic",
            key_factors=["API_ERROR", "FALLBACK_MODE"],
            risk_level="medium",
            recommended_action="hold",
            metadata={'error': True, 'raw_response': raw_response}
        )

class MarketAnalysisService:
    """Main service for LLM-powered market analysis"""
    
    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider
        self.analysis_cache: Dict[str, LLMAnalysis] = {}
        self.cache_timeout = timedelta(minutes=15)  # Cache analysis for 15 minutes
        self.logger = logging.getLogger(__name__)
    
    async def comprehensive_analysis(self, context: MarketContext) -> Dict[AnalysisType, LLMAnalysis]:
        """Perform comprehensive analysis across multiple dimensions"""
        
        analyses = {}
        analysis_types = [
            AnalysisType.SENTIMENT,
            AnalysisType.TECHNICAL,
            AnalysisType.RISK_ASSESSMENT,
            AnalysisType.TRADE_REASONING
        ]
        
        # Run analyses in parallel for efficiency
        tasks = [
            self.get_analysis(context, analysis_type)
            for analysis_type in analysis_types
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for analysis_type, result in zip(analysis_types, results):
            if isinstance(result, Exception):
                self.logger.error(f"Error in {analysis_type.value} analysis: {result}")
                analyses[analysis_type] = self._create_error_analysis(context, analysis_type)
            else:
                analyses[analysis_type] = result
        
        return analyses
    
    async def get_analysis(self, context: MarketContext, analysis_type: AnalysisType) -> LLMAnalysis:
        """Get analysis with caching"""
        
        cache_key = f"{context.symbol}_{analysis_type.value}_{context.timestamp.strftime('%Y%m%d_%H%M')}"
        
        # Check cache
        cached_analysis = self.analysis_cache.get(cache_key)
        if cached_analysis and (datetime.utcnow() - cached_analysis.timestamp) < self.cache_timeout:
            self.logger.info(f"Using cached analysis for {cache_key}")
            return cached_analysis
        
        # Get fresh analysis
        analysis = await self.llm_provider.analyze_market(context, analysis_type)
        
        # Cache result
        self.analysis_cache[cache_key] = analysis
        
        # Cleanup old cache entries
        self._cleanup_cache()
        
        return analysis
    
    def _cleanup_cache(self):
        """Remove expired cache entries"""
        current_time = datetime.utcnow()
        expired_keys = [
            key for key, analysis in self.analysis_cache.items()
            if (current_time - analysis.timestamp) > self.cache_timeout
        ]
        
        for key in expired_keys:
            del self.analysis_cache[key]
    
    def _create_error_analysis(self, context: MarketContext, analysis_type: AnalysisType) -> LLMAnalysis:
        """Create error analysis when LLM fails"""
        
        return LLMAnalysis(
            analysis_type=analysis_type,
            symbol=context.symbol,
            confidence=0.1,
            sentiment=Sentiment.NEUTRAL,
            reasoning="Analysis service error",
            key_factors=["SERVICE_ERROR"],
            risk_level="high",
            recommended_action="avoid",
            metadata={'error': True}
        )
    
    async def validate_trade_decision(self, 
                                    symbol: str,
                                    action: str,
                                    quantity: Decimal,
                                    price: Decimal,
                                    context: MarketContext) -> Dict[str, Any]:
        """Validate a trade decision using LLM analysis"""
        
        try:
            # Get comprehensive analysis
            analyses = await self.comprehensive_analysis(context)
            
            # Calculate consensus
            consensus = self._calculate_consensus(analyses)
            
            # Validate trade against analysis
            validation = {
                'approved': True,
                'confidence': consensus['confidence'],
                'reasoning': consensus['reasoning'],
                'warnings': [],
                'analysis_summary': consensus
            }
            
            # Check if action aligns with recommendations
            recommended_actions = [analysis.recommended_action for analysis in analyses.values()]
            if action not in recommended_actions and 'hold' not in recommended_actions:
                validation['warnings'].append(f"Action '{action}' not recommended by LLM analysis")
                validation['confidence'] *= 0.7  # Reduce confidence
            
            # Check risk level
            risk_levels = [analysis.risk_level for analysis in analyses.values()]
            if 'high' in risk_levels:
                validation['warnings'].append("High risk level detected in analysis")
                validation['confidence'] *= 0.8
            
            # Check if confidence is too low
            if consensus['confidence'] < 0.3:
                validation['warnings'].append("Low confidence in analysis")
                if consensus['confidence'] < 0.2:
                    validation['approved'] = False
            
            return validation
            
        except Exception as e:
            self.logger.error(f"Error validating trade decision: {e}")
            return {
                'approved': False,
                'confidence': 0.0,
                'reasoning': f"Validation error: {e}",
                'warnings': ['Trade validation failed'],
                'analysis_summary': {}
            }
    
    def _calculate_consensus(self, analyses: Dict[AnalysisType, LLMAnalysis]) -> Dict[str, Any]:
        """Calculate consensus from multiple analyses"""
        
        if not analyses:
            return {
                'confidence': 0.0,
                'sentiment': 'neutral',
                'reasoning': 'No analysis available',
                'recommended_action': 'hold'
            }
        
        # Calculate average confidence
        confidences = [analysis.confidence for analysis in analyses.values()]
        avg_confidence = sum(confidences) / len(confidences)
        
        # Calculate sentiment consensus
        sentiments = [analysis.sentiment.value for analysis in analyses.values()]
        avg_sentiment = sum(sentiments) / len(sentiments)
        
        # Get most common recommended action
        actions = [analysis.recommended_action for analysis in analyses.values()]
        most_common_action = max(set(actions), key=actions.count)
        
        # Combine reasoning
        reasonings = [analysis.reasoning for analysis in analyses.values()]
        combined_reasoning = " | ".join(reasonings[:3])  # Limit length
        
        return {
            'confidence': avg_confidence,
            'sentiment': self._sentiment_value_to_string(avg_sentiment),
            'reasoning': combined_reasoning,
            'recommended_action': most_common_action,
            'analysis_count': len(analyses)
        }
    
    def _sentiment_value_to_string(self, value: float) -> str:
        """Convert sentiment value to string"""
        if value <= -1.5:
            return 'very_bearish'
        elif value <= -0.5:
            return 'bearish'
        elif value <= 0.5:
            return 'neutral'
        elif value <= 1.5:
            return 'bullish'
        else:
            return 'very_bullish'
