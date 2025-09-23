"""
Shared Token Optimization Library for All LLM Services
Unified approach to reduce token usage across the trading system
"""
import os
import json
import time
import logging
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class UnifiedTokenOptimizer:
    """Unified token optimization for all LLM services"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes default
        self.usage_stats = {
            'total_calls': 0,
            'cached_calls': 0,
            'tokens_saved': 0,
            'cost_saved': 0.0
        }
        
        # Service-specific templates
        self.templates = self._load_service_templates()
        
        # Common optimization settings
        self.cost_per_1k_input = 0.00015  # gpt-4o-mini
        self.cost_per_1k_output = 0.0006
        
    def _load_service_templates(self) -> Dict[str, Dict]:
        """Load templates specific to each service type"""
        
        templates = {
            'sentiment_analysis': {
                'system': 'Crypto sentiment analyst. Return JSON: {"score": -1 to 1, "confidence": 0-1, "summary": "brief"}',
                'user_template': 'Analyze sentiment: {text}',
                'max_tokens': 150,
                'temperature': 0.2
            },
            
            'risk_assessment': {
                'system': 'Expert crypto risk analyst. Return JSON: {"risk_score": 0-100, "position_multiplier": 0.5-2.0, "key_risks": [], "recommendations": [], "confidence": 0-100, "reasoning": "brief"}',
                'user_template': 'Risk analysis {symbol}: Price=${price}, 24h={change:.1%}, RSI={rsi}, Sentiment={sentiment}',
                'max_tokens': 500,
                'temperature': 0.3
            },
            
            'trade_reasoning': {
                'system': 'Expert crypto trading advisor. Analyze data and provide specific trading recommendations with clear reasoning.',
                'user_template': 'Trading analysis {coin}: Price=${price}, Change={change:.1%}, Vol=${volume:,.0f}, RSI={rsi}, ML={ml_prediction}, Sentiment={sentiment}. Recommend action with entry/exit/stop levels.',
                'max_tokens': 800,
                'temperature': 0.3
            },
            
            'news_impact': {
                'system': 'Crypto news impact analyst. Return JSON: {"impact_score": 0-1, "direction": "bullish/bearish/neutral", "confidence": 0-1}',
                'user_template': 'News impact for {symbol}: {headline}',
                'max_tokens': 200,
                'temperature': 0.2
            },
            
            'market_regime': {
                'system': 'Market regime analyst. Return JSON: {"regime": "bull/bear/sideways", "confidence": 0-1, "duration_estimate": "days"}',
                'user_template': 'Market regime: BTC=${btc_price}, VIX={vix}, Volume={volume}, Sentiment={sentiment}',
                'max_tokens': 300,
                'temperature': 0.2
            }
        }
        
        return templates
    
    def get_optimized_prompt(self, template_type: str, **kwargs) -> Tuple[str, str, Dict]:
        """Get optimized system/user prompts and settings"""
        
        if template_type not in self.templates:
            # Fallback to basic optimization
            return self._basic_optimization(**kwargs)
        
        template = self.templates[template_type]
        
        # Build optimized prompts
        system_prompt = template['system']
        user_prompt = template['user_template'].format(**kwargs)
        
        settings = {
            'max_tokens': template['max_tokens'],
            'temperature': template['temperature'],
            'model': os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
        }
        
        return system_prompt, user_prompt, settings
    
    def _basic_optimization(self, **kwargs) -> Tuple[str, str, Dict]:
        """Basic fallback optimization"""
        system = "Expert crypto analyst. Provide concise, structured analysis."
        user = f"Analyze: {json.dumps(kwargs, separators=(',', ':'))}"
        settings = {
            'max_tokens': int(os.getenv('OPENAI_MAX_TOKENS', '500')),
            'temperature': float(os.getenv('OPENAI_TEMPERATURE', '0.3')),
            'model': os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
        }
        return system, user, settings
    
    def should_use_cache(self, cache_key: str) -> Tuple[bool, Any]:
        """Check if cached result should be used"""
        if cache_key in self.cache:
            cached_time, cached_result = self.cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                self.usage_stats['cached_calls'] += 1
                logger.debug(f"[{self.service_name}] Using cached result for {cache_key}")
                return True, cached_result
        
        return False, None
    
    def cache_result(self, cache_key: str, result: Any, tokens_used: int = 0):
        """Cache a result and update stats"""
        self.cache[cache_key] = (time.time(), result)
        self.usage_stats['total_calls'] += 1
        
        # Estimate tokens saved by caching
        if cache_key in self.cache:
            self.usage_stats['tokens_saved'] += tokens_used * 0.8  # Estimate 80% savings on cache hit
        
        # Cleanup old cache entries
        self._cleanup_cache()
    
    def _cleanup_cache(self):
        """Remove expired cache entries"""
        current_time = time.time()
        expired_keys = [
            key for key, (timestamp, _) in self.cache.items()
            if current_time - timestamp > self.cache_ttl
        ]
        for key in expired_keys:
            del self.cache[key]
    
    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation"""
        return len(text) // 4
    
    def calculate_savings(self, original_tokens: int, optimized_tokens: int) -> Dict[str, float]:
        """Calculate token and cost savings"""
        tokens_saved = original_tokens - optimized_tokens
        cost_saved = (tokens_saved / 1000) * self.cost_per_1k_input
        savings_percent = (tokens_saved / original_tokens) * 100 if original_tokens > 0 else 0
        
        return {
            'tokens_saved': tokens_saved,
            'cost_saved': cost_saved,
            'savings_percent': savings_percent
        }
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for this service"""
        cache_hit_rate = 0
        if self.usage_stats['total_calls'] > 0:
            cache_hit_rate = (self.usage_stats['cached_calls'] / self.usage_stats['total_calls']) * 100
        
        return {
            'service': self.service_name,
            'total_calls': self.usage_stats['total_calls'],
            'cached_calls': self.usage_stats['cached_calls'],
            'cache_hit_rate': cache_hit_rate,
            'estimated_tokens_saved': self.usage_stats['tokens_saved'],
            'estimated_cost_saved': self.usage_stats['cost_saved'],
            'cache_entries': len(self.cache)
        }

class OpenAIClientOptimizer:
    """Optimized OpenAI client wrapper"""
    
    def __init__(self, service_name: str):
        from openai import OpenAI
        
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.client = OpenAI(api_key=api_key)
        self.optimizer = UnifiedTokenOptimizer(service_name)
        self.service_name = service_name
        
    async def chat_completion_optimized(self, template_type: str, cache_key: str = None, **template_kwargs) -> Tuple[str, Dict]:
        """Optimized chat completion with caching and templates"""
        
        # Check cache first
        if cache_key:
            use_cache, cached_result = self.optimizer.should_use_cache(cache_key)
            if use_cache:
                return cached_result, {'cached': True}
        
        # Get optimized prompts
        system_prompt, user_prompt, settings = self.optimizer.get_optimized_prompt(template_type, **template_kwargs)
        
        # Estimate token usage
        estimated_input_tokens = self.optimizer.estimate_tokens(system_prompt + user_prompt)
        
        try:
            # Make API call
            response = self.client.chat.completions.create(
                model=settings['model'],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=settings['max_tokens'],
                temperature=settings['temperature']
            )
            
            result = response.choices[0].message.content
            
            # Calculate actual usage
            actual_tokens = response.usage.total_tokens if hasattr(response, 'usage') else estimated_input_tokens
            
            # Cache result
            if cache_key:
                self.optimizer.cache_result(cache_key, result, actual_tokens)
            
            # Return result and metadata
            metadata = {
                'cached': False,
                'tokens_used': actual_tokens,
                'estimated_input_tokens': estimated_input_tokens,
                'model': settings['model'],
                'service': self.service_name
            }
            
            logger.info(f"[{self.service_name}] LLM call: {actual_tokens} tokens, template: {template_type}")
            
            return result, metadata
            
        except Exception as e:
            logger.error(f"[{self.service_name}] OpenAI API error: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        return self.optimizer.get_usage_stats()

# Example usage patterns for different services
class ServiceTemplateExamples:
    """Examples of how each service should use the optimizer"""
    
    @staticmethod
    def sentiment_analysis_example():
        """Example for sentiment analysis services"""
        optimizer = OpenAIClientOptimizer("crypto-sentiment")
        
        # Traditional approach (400+ tokens)
        traditional_prompt = """
        You are a cryptocurrency sentiment analysis expert. Analyze the following text for market sentiment related to crypto assets.
        
        Consider factors like:
        - Market emotion (fear, greed, optimism, pessimism)
        - Price movement implications
        - Investment sentiment
        - Community confidence
        - Regulatory concerns
        
        Return a sentiment score from -1 (very negative) to 1 (very positive) along with confidence and summary.
        
        Text to analyze: "Bitcoin showing strong momentum with institutional adoption increasing"
        
        Format your response as JSON with score, confidence, and summary fields.
        """
        
        # Optimized approach (80% fewer tokens)
        async def optimized_sentiment():
            text = "Bitcoin showing strong momentum with institutional adoption increasing"
            cache_key = f"sentiment_{hash(text)}"
            
            result, metadata = await optimizer.chat_completion_optimized(
                template_type="sentiment_analysis",
                cache_key=cache_key,
                text=text
            )
            
            return result, metadata
        
        return optimized_sentiment
    
    @staticmethod
    def risk_assessment_example():
        """Example for risk assessment services"""
        optimizer = OpenAIClientOptimizer("llm-risk-manager")
        
        async def optimized_risk_analysis():
            cache_key = f"risk_BTC_{int(time.time() // 300)}"  # 5-min cache
            
            result, metadata = await optimizer.chat_completion_optimized(
                template_type="risk_assessment",
                cache_key=cache_key,
                symbol="BTC",
                price=63847.12,
                change=0.024,
                rsi=52.3,
                sentiment=0.72
            )
            
            return result, metadata
        
        return optimized_risk_analysis
    
    @staticmethod
    def trade_reasoning_example():
        """Example for trade reasoning services"""
        optimizer = OpenAIClientOptimizer("llm-reasoning")
        
        async def optimized_trade_reasoning():
            cache_key = f"trade_BTC_{int(time.time() // 300)}"
            
            result, metadata = await optimizer.chat_completion_optimized(
                template_type="trade_reasoning",
                cache_key=cache_key,
                coin="BTC",
                price=63847.12,
                change=0.024,
                volume=28500000000,
                rsi=52.3,
                ml_prediction="UP",
                sentiment=0.72
            )
            
            return result, metadata
        
        return optimized_trade_reasoning

# Global optimization statistics
class GlobalOptimizationStats:
    """Track optimization across all services"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.services = {}
        return cls._instance
    
    def register_service(self, service_name: str, optimizer: OpenAIClientOptimizer):
        """Register a service for global tracking"""
        self.services[service_name] = optimizer
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get aggregated stats across all services"""
        total_stats = {
            'total_calls': 0,
            'total_cached_calls': 0,
            'total_tokens_saved': 0,
            'total_cost_saved': 0.0,
            'services': {}
        }
        
        for service_name, optimizer in self.services.items():
            stats = optimizer.get_stats()
            total_stats['services'][service_name] = stats
            total_stats['total_calls'] += stats['total_calls']
            total_stats['total_cached_calls'] += stats['cached_calls']
            total_stats['total_tokens_saved'] += stats['estimated_tokens_saved']
            total_stats['total_cost_saved'] += stats['estimated_cost_saved']
        
        # Calculate global cache hit rate
        if total_stats['total_calls'] > 0:
            total_stats['global_cache_hit_rate'] = (
                total_stats['total_cached_calls'] / total_stats['total_calls']
            ) * 100
        else:
            total_stats['global_cache_hit_rate'] = 0
        
        return total_stats

# Export the main classes
__all__ = [
    'UnifiedTokenOptimizer',
    'OpenAIClientOptimizer', 
    'ServiceTemplateExamples',
    'GlobalOptimizationStats'
]