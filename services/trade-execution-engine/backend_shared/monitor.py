#!/usr/bin/env python3
"""
LLM Fallback Monitor Service
Continuously monitors the LLM fallback system health and functionality.
"""

import asyncio
import time
import logging
import os
from datetime import datetime
from backend.shared.llm_fallback_manager import LLMFallbackManager, LLMProvider

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LLMFallbackMonitor:
    """Monitor for LLM fallback system health and performance."""
    
    def __init__(self):
        self.manager = LLMFallbackManager()
        self.check_interval = int(os.getenv('MONITOR_INTERVAL', '300'))  # 5 minutes default
        self.test_prompts = [
            "System health check - respond with 'OK'",
            "What is the current market sentiment?",
            "Analyze this trade recommendation",
        ]
        
    async def run_health_check(self) -> bool:
        """Run a comprehensive health check of the LLM fallback system."""
        try:
            logger.info("🔍 Starting LLM fallback health check...")
            
            # Test basic functionality
            response = await self.manager.generate_response(
                self.test_prompts[0],
                max_tokens=20,
                temperature=0.1
            )
            
            logger.info(f"✅ Health check successful:")
            logger.info(f"   Provider: {response.provider.value}")
            logger.info(f"   Content: {response.content[:100]}")
            logger.info(f"   Tokens: {response.tokens_used}")
            logger.info(f"   Confidence: {response.confidence}")
            logger.info(f"   Cached: {response.cached}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            return False
    
    async def test_provider_fallback(self) -> bool:
        """Test that provider fallback is working correctly."""
        try:
            logger.info("🔄 Testing provider fallback chain...")
            
            # Force a test that might trigger fallback
            response = await self.manager.generate_response(
                "This is a fallback test prompt",
                max_tokens=50,
                temperature=0.5
            )
            
            logger.info(f"✅ Fallback test completed with provider: {response.provider.value}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Fallback test failed: {e}")
            return False
    
    async def check_quota_status(self):
        """Check and log quota status for all providers."""
        try:
            logger.info("📊 Checking quota status...")
            
            # Check if manager has quota tracking
            if hasattr(self.manager, 'quota_tracker'):
                for provider in LLMProvider:
                    if provider in [LLMProvider.GROK, LLMProvider.OPENAI]:
                        status = self.manager.quota_tracker.get_status(provider)
                        logger.info(f"   {provider.value}: {status.remaining_calls} calls remaining")
            else:
                logger.info("   Quota tracking not available")
                
        except Exception as e:
            logger.error(f"❌ Quota check failed: {e}")
    
    async def monitor_loop(self):
        """Main monitoring loop."""
        logger.info(f"🚀 Starting LLM fallback monitor (interval: {self.check_interval}s)")
        
        iteration = 0
        while True:
            try:
                iteration += 1
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                logger.info(f"\n{'='*60}")
                logger.info(f"🔍 Monitor Check #{iteration} - {timestamp}")
                logger.info(f"{'='*60}")
                
                # Run health check
                health_ok = await self.run_health_check()
                
                # Test fallback functionality
                fallback_ok = await self.test_provider_fallback()
                
                # Check quota status
                await self.check_quota_status()
                
                # Overall status
                if health_ok and fallback_ok:
                    logger.info(f"✅ Overall system status: HEALTHY")
                else:
                    logger.warning(f"⚠️ Overall system status: DEGRADED")
                
                logger.info(f"⏰ Next check in {self.check_interval} seconds...")
                
            except Exception as e:
                logger.error(f"❌ Monitor loop error: {e}")
                logger.info(f"🔄 Continuing monitoring despite error...")
            
            # Wait for next check
            await asyncio.sleep(self.check_interval)

async def main():
    """Main entry point."""
    try:
        monitor = LLMFallbackMonitor()
        await monitor.monitor_loop()
    except KeyboardInterrupt:
        logger.info("👋 Monitor shutdown requested")
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
