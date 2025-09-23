#!/usr/bin/env python3
"""
Enhanced LLM Manager with Robust Fallback and Notification System
Handles API quota exhaustion with local alternatives and alerts.
"""

import asyncio
import json
import logging
import os
import time
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import hashlib
import pickle
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LLMProvider(Enum):
    """Available LLM providers."""
    GROK = "grok"
    OPENAI = "openai"
    LOCAL_RULE_BASED = "local_rule_based"
    CACHED = "cached"
    OLLAMA = "ollama"
    FALLBACK = "fallback"

@dataclass
class LLMResponse:
    """Standardized LLM response format."""
    content: str
    provider: LLMProvider
    tokens_used: int
    confidence: float
    cached: bool = False
    error: Optional[str] = None

@dataclass
class QuotaStatus:
    """API quota monitoring."""
    provider: LLMProvider
    requests_today: int
    tokens_today: int
    quota_exceeded: bool
    last_error: Optional[str] = None
    last_error_time: Optional[datetime] = None

class LLMFallbackManager:
    """
    Comprehensive LLM management with intelligent fallback and caching.
    """
    
    def __init__(self):
        self.xai_api_key = os.getenv('XAI_API_KEY')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.ollama_url = os.getenv('OLLAMA_URL', 'http://localhost:11434')
        
        # Cache configuration
        self.cache_dir = Path("temp/llm_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl = int(os.getenv('LLM_CACHE_TTL', '3600'))  # 1 hour default
        
        # Quota tracking
        self.quota_status = {
            LLMProvider.GROK: QuotaStatus(LLMProvider.GROK, 0, 0, False),
            LLMProvider.OPENAI: QuotaStatus(LLMProvider.OPENAI, 0, 0, False),
            LLMProvider.OLLAMA: QuotaStatus(LLMProvider.OLLAMA, 0, 0, False)
        }
        
        # Notification settings
        self.notification_webhook = os.getenv('NOTIFICATION_WEBHOOK_URL')
        self.admin_email = os.getenv('ADMIN_EMAIL')
        
        # Determine provider priority
        self.provider_priority = self._determine_provider_priority()
        
        logger.info(f"🚀 LLM Fallback Manager initialized")
        logger.info(f"📊 Provider priority: {[p.value for p in self.provider_priority]}")
    
    def _determine_provider_priority(self) -> List[LLMProvider]:
        """Determine the optimal provider priority based on available APIs."""
        priority = []
        
        # Check available providers
        if self.xai_api_key:
            priority.append(LLMProvider.GROK)
        
        if self.openai_api_key:
            priority.append(LLMProvider.OPENAI)
            
        # Check if Ollama is available
        if self._check_ollama_availability():
            priority.append(LLMProvider.OLLAMA)
        
        # Always have local fallback
        priority.extend([LLMProvider.CACHED, LLMProvider.LOCAL_RULE_BASED])
        
        return priority
    
    def _check_ollama_availability(self) -> bool:
        """Check if Ollama is available."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def _get_cache_key(self, prompt: str, provider: LLMProvider) -> str:
        """Generate cache key for prompt."""
        content = f"{provider.value}:{prompt[:500]}"  # Use first 500 chars for cache key
        return hashlib.md5(content.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[LLMResponse]:
        """Load response from cache if valid."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            # Check if cache is still valid
            if datetime.now() - cached_data['timestamp'] < timedelta(seconds=self.cache_ttl):
                response = cached_data['response']
                response.cached = True
                logger.info(f"✅ Using cached response from {response.provider.value}")
                return response
            else:
                # Remove expired cache
                cache_file.unlink()
                return None
                
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return None
    
    def _save_to_cache(self, cache_key: str, response: LLMResponse):
        """Save response to cache."""
        try:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            cache_data = {
                'timestamp': datetime.now(),
                'response': response
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
                
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    async def _call_grok_api(self, prompt: str, max_tokens: int = 1000) -> LLMResponse:
        """Call Grok API with quota tracking."""
        if not self.xai_api_key:
            raise ValueError("XAI_API_KEY not configured")
        
        if self.quota_status[LLMProvider.GROK].quota_exceeded:
            raise Exception("Grok API quota exceeded")
        
        headers = {
            "Authorization": f"Bearer {self.xai_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "grok-beta",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert cryptocurrency analysis assistant. Provide precise, actionable insights."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3,
            "max_tokens": max_tokens
        }
        
        try:
            response = requests.post(
                "https://api.x.ai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                tokens_used = result.get("usage", {}).get("total_tokens", max_tokens)
                
                # Update quota tracking
                self.quota_status[LLMProvider.GROK].requests_today += 1
                self.quota_status[LLMProvider.GROK].tokens_today += tokens_used
                
                return LLMResponse(
                    content=content,
                    provider=LLMProvider.GROK,
                    tokens_used=tokens_used,
                    confidence=0.9
                )
            
            elif response.status_code == 429:
                # Quota exceeded
                await self._handle_quota_exceeded(LLMProvider.GROK, response.text)
                raise Exception("Grok API quota exceeded")
            
            else:
                raise Exception(f"Grok API error {response.status_code}: {response.text}")
                
        except Exception as e:
            if "quota" in str(e).lower() or "429" in str(e):
                await self._handle_quota_exceeded(LLMProvider.GROK, str(e))
            raise
    
    async def _call_openai_api(self, prompt: str, max_tokens: int = 1000) -> LLMResponse:
        """Call OpenAI API with quota tracking."""
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not configured")
        
        if self.quota_status[LLMProvider.OPENAI].quota_exceeded:
            raise Exception("OpenAI API quota exceeded")
        
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "gpt-4o-mini",  # Use cheaper model to conserve quota
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert cryptocurrency analysis assistant. Provide precise, actionable insights."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3,
            "max_tokens": max_tokens
        }
        
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                tokens_used = result.get("usage", {}).get("total_tokens", max_tokens)
                
                # Update quota tracking
                self.quota_status[LLMProvider.OPENAI].requests_today += 1
                self.quota_status[LLMProvider.OPENAI].tokens_today += tokens_used
                
                return LLMResponse(
                    content=content,
                    provider=LLMProvider.OPENAI,
                    tokens_used=tokens_used,
                    confidence=0.9
                )
            
            elif response.status_code == 429:
                # Quota exceeded
                await self._handle_quota_exceeded(LLMProvider.OPENAI, response.text)
                raise Exception("OpenAI API quota exceeded")
            
            else:
                raise Exception(f"OpenAI API error {response.status_code}: {response.text}")
                
        except Exception as e:
            if "quota" in str(e).lower() or "429" in str(e):
                await self._handle_quota_exceeded(LLMProvider.OPENAI, str(e))
            raise
    
    async def _call_ollama_api(self, prompt: str, max_tokens: int = 1000) -> LLMResponse:
        """Call local Ollama API."""
        try:
            payload = {
                "model": "llama2:7b",  # Default model, can be configured
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": max_tokens
                }
            }
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=60  # Longer timeout for local processing
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result.get("response", "")
                
                return LLMResponse(
                    content=content,
                    provider=LLMProvider.OLLAMA,
                    tokens_used=len(content.split()),  # Approximate token count
                    confidence=0.8  # Local model confidence
                )
            else:
                raise Exception(f"Ollama API error {response.status_code}")
                
        except Exception as e:
            logger.error(f"Ollama API call failed: {e}")
            raise
    
    def _generate_rule_based_response(self, prompt: str, context: Dict = None) -> LLMResponse:
        """Generate rule-based response for risk analysis when LLMs are unavailable."""
        
        # Extract context for risk analysis
        risk_keywords_high = ['crash', 'dump', 'bearish', 'sell', 'fear', 'panic', 'volatile']
        risk_keywords_low = ['bullish', 'buy', 'rally', 'pump', 'moon', 'hodl', 'optimistic']
        
        prompt_lower = prompt.lower()
        
        # Simple risk scoring based on keywords
        high_risk_count = sum(1 for keyword in risk_keywords_high if keyword in prompt_lower)
        low_risk_count = sum(1 for keyword in risk_keywords_low if keyword in prompt_lower)
        
        # Calculate basic risk score
        if high_risk_count > low_risk_count:
            risk_score = 0.7 + (high_risk_count * 0.1)
            market_sentiment = "bearish"
            position_multiplier = 0.6
        elif low_risk_count > high_risk_count:
            risk_score = 0.3 - (low_risk_count * 0.05)
            market_sentiment = "bullish"
            position_multiplier = 1.2
        else:
            risk_score = 0.5
            market_sentiment = "neutral"
            position_multiplier = 1.0
        
        # Clamp values
        risk_score = max(0.1, min(0.9, risk_score))
        position_multiplier = max(0.5, min(2.0, position_multiplier))
        
        # Generate structured response
        response_content = json.dumps({
            "risk_score": risk_score,
            "position_multiplier": position_multiplier,
            "max_position": 100.0,
            "stop_loss": 0.05,
            "take_profit": 0.10,
            "confidence": 0.6,
            "risk_factors": ["Rule-based analysis", f"Market sentiment: {market_sentiment}"],
            "market_regime": market_sentiment,
            "volatility_adj": 1.1,
            "reasoning": f"Rule-based analysis detected {market_sentiment} sentiment. Risk score: {risk_score:.2f}, Position multiplier: {position_multiplier:.2f}. This is a fallback analysis when LLM services are unavailable."
        })
        
        return LLMResponse(
            content=response_content,
            provider=LLMProvider.LOCAL_RULE_BASED,
            tokens_used=0,
            confidence=0.6
        )
    
    async def _handle_quota_exceeded(self, provider: LLMProvider, error_message: str):
        """Handle API quota exceeded event."""
        self.quota_status[provider].quota_exceeded = True
        self.quota_status[provider].last_error = error_message
        self.quota_status[provider].last_error_time = datetime.now()
        
        logger.error(f"🚨 {provider.value} API quota exceeded: {error_message}")
        
        # Send notifications
        await self._send_quota_notification(provider, error_message)
    
    async def _send_quota_notification(self, provider: LLMProvider, error_message: str):
        """Send notification about quota exhaustion."""
        
        notification_data = {
            "alert": "LLM API Quota Exceeded",
            "provider": provider.value,
            "error": error_message,
            "timestamp": datetime.now().isoformat(),
            "system": "CryptoAI Trading System",
            "action_required": f"Increase {provider.value} API quota or check billing",
            "fallback_status": "Local rule-based analysis activated"
        }
        
        # Webhook notification
        if self.notification_webhook:
            try:
                response = requests.post(
                    self.notification_webhook,
                    json=notification_data,
                    timeout=10
                )
                logger.info(f"📢 Quota notification sent to webhook: {response.status_code}")
            except Exception as e:
                logger.error(f"Failed to send webhook notification: {e}")
        
        # Log to file for monitoring
        try:
            log_file = Path("temp/quota_alerts.log")
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(log_file, "a") as f:
                f.write(f"{datetime.now().isoformat()} - {provider.value} quota exceeded: {error_message}\n")
                
        except Exception as e:
            logger.error(f"Failed to log quota alert: {e}")
        
        # Print alert to console
        print(f"\n🚨 CRITICAL ALERT: {provider.value} API QUOTA EXCEEDED")
        print(f"Error: {error_message}")
        print(f"Time: {datetime.now()}")
        print(f"Action: Please check API billing and increase quota limits")
        print(f"System Status: Fallback to local analysis activated\n")
    
    async def generate_response(self, prompt: str, max_tokens: int = 1000, context: Dict = None) -> LLMResponse:
        """
        Generate LLM response with intelligent fallback chain.
        
        Priority order:
        1. Check cache first
        2. Try available API providers (Grok, OpenAI, Ollama)
        3. Fall back to local rule-based analysis
        """
        
        # Check cache first
        cache_key = self._get_cache_key(prompt, LLMProvider.CACHED)
        cached_response = self._load_from_cache(cache_key)
        if cached_response:
            return cached_response
        
        # Try providers in priority order
        for provider in self.provider_priority:
            
            if provider == LLMProvider.CACHED:
                continue  # Already checked above
            
            if provider == LLMProvider.LOCAL_RULE_BASED:
                # Always works as final fallback
                response = self._generate_rule_based_response(prompt, context)
                logger.info(f"🔧 Using rule-based fallback analysis")
                return response
            
            try:
                if provider == LLMProvider.GROK and self.xai_api_key:
                    if not self.quota_status[provider].quota_exceeded:
                        response = await self._call_grok_api(prompt, max_tokens)
                        logger.info(f"✅ Generated response using {provider.value}")
                        self._save_to_cache(cache_key, response)
                        return response
                
                elif provider == LLMProvider.OPENAI and self.openai_api_key:
                    if not self.quota_status[provider].quota_exceeded:
                        response = await self._call_openai_api(prompt, max_tokens)
                        logger.info(f"✅ Generated response using {provider.value}")
                        self._save_to_cache(cache_key, response)
                        return response
                
                elif provider == LLMProvider.OLLAMA:
                    if self._check_ollama_availability():
                        response = await self._call_ollama_api(prompt, max_tokens)
                        logger.info(f"✅ Generated response using {provider.value}")
                        self._save_to_cache(cache_key, response)
                        return response
                
            except Exception as e:
                logger.warning(f"❌ {provider.value} failed: {e}")
                continue
        
        # If all else fails, use rule-based fallback
        logger.warning("🔧 All LLM providers failed, using rule-based fallback")
        return self._generate_rule_based_response(prompt, context)
    
    def get_system_status(self) -> Dict:
        """Get comprehensive LLM system status."""
        
        status = {
            "timestamp": datetime.now().isoformat(),
            "providers": {},
            "cache_stats": self._get_cache_stats(),
            "fallback_ready": True
        }
        
        for provider, quota in self.quota_status.items():
            status["providers"][provider.value] = {
                "available": self._is_provider_available(provider),
                "quota_exceeded": quota.quota_exceeded,
                "requests_today": quota.requests_today,
                "tokens_today": quota.tokens_today,
                "last_error": quota.last_error,
                "last_error_time": quota.last_error_time.isoformat() if quota.last_error_time else None
            }
        
        return status
    
    def _is_provider_available(self, provider: LLMProvider) -> bool:
        """Check if a provider is available."""
        if provider == LLMProvider.GROK:
            return bool(self.xai_api_key and not self.quota_status[provider].quota_exceeded)
        elif provider == LLMProvider.OPENAI:
            return bool(self.openai_api_key and not self.quota_status[provider].quota_exceeded)
        elif provider == LLMProvider.OLLAMA:
            return self._check_ollama_availability()
        elif provider == LLMProvider.LOCAL_RULE_BASED:
            return True
        return False
    
    def _get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        cache_files = list(self.cache_dir.glob("*.pkl"))
        
        return {
            "cached_responses": len(cache_files),
            "cache_size_mb": sum(f.stat().st_size for f in cache_files) / (1024 * 1024),
            "cache_directory": str(self.cache_dir)
        }
    
    def reset_quota_status(self, provider: LLMProvider = None):
        """Reset quota status (for daily reset or manual override)."""
        if provider:
            self.quota_status[provider].quota_exceeded = False
            self.quota_status[provider].requests_today = 0
            self.quota_status[provider].tokens_today = 0
            self.quota_status[provider].last_error = None
            self.quota_status[provider].last_error_time = None
        else:
            for p in self.quota_status:
                self.reset_quota_status(p)
    
    def clear_cache(self):
        """Clear all cached responses."""
        try:
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            logger.info("🗑️ Cache cleared successfully")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")

# Example usage for integration
async def main():
    """Example usage of the LLM Fallback Manager."""
    
    llm_manager = LLMFallbackManager()
    
    # Example risk analysis prompt
    prompt = """
    Analyze the risk for Bitcoin trading given:
    - Current price: $45,000
    - 24h change: +2.5%
    - Volume increase: 15%
    - News sentiment: Positive regulatory developments
    
    Provide risk score, position sizing recommendations, and reasoning.
    """
    
    try:
        response = await llm_manager.generate_response(prompt)
        
        print(f"🤖 Response from {response.provider.value}:")
        print(f"Confidence: {response.confidence}")
        print(f"Tokens used: {response.tokens_used}")
        print(f"Content: {response.content[:200]}...")
        
        # Get system status
        status = llm_manager.get_system_status()
        print(f"\n📊 System Status: {json.dumps(status, indent=2)}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
