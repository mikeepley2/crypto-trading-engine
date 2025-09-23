#!/usr/bin/env python3
"""
OpenAI Integration for Enhanced Trading Analysis
AI-powered market analysis and trade reasoning
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import openai
import mysql.connector
from dotenv import load_dotenv

# Load environment variables
load_dotenv('../../../.env.live')

logger = logging.getLogger(__name__)

class OpenAITradingAnalyst:
    """OpenAI-powered trading analysis and market insights"""
    
    def __init__(self):
        # Initialize OpenAI client
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        openai.api_key = self.api_key
        self.model = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
        self.max_tokens = int(os.getenv('OPENAI_MAX_TOKENS', '1000'))
        
        # Database configuration
        self.db_config = {
            'host': os.getenv('DB_HOST', 'host.docker.internal'),
            'user': os.getenv('DB_USER', 'news_collector'),
            'password': os.getenv('DB_PASSWORD', '99Rules!'),
            'database': os.getenv('DB_NAME_PRICES', 'crypto_prices')
        }
        
        logger.info("✅ OpenAI Trading Analyst initialized")
    
    def get_market_data(self, symbol: str, hours_back: int = 24) -> Dict:
        """Get comprehensive market data for analysis"""
        try:
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor(dictionary=True)
            
            # Get recent price data
            cursor.execute("""
                SELECT current_price as price, volume_usd_24h as volume, timestamp_iso 
                FROM price_data 
                WHERE symbol = %s AND timestamp_iso >= DATE_SUB(NOW(), INTERVAL %s HOUR)
                ORDER BY timestamp_iso DESC 
                LIMIT 100
            """, (symbol, hours_back))
            price_data = cursor.fetchall()
            
            # Get technical indicators
            cursor.execute("""
                SELECT rsi, sma_20, sma_50, bb_upper, bb_lower, macd, signal_line, timestamp_iso
                FROM technical_indicators 
                WHERE symbol = %s AND timestamp_iso >= DATE_SUB(NOW(), INTERVAL %s HOUR)
                ORDER BY timestamp_iso DESC 
                LIMIT 50
            """, (symbol, hours_back))
            technical_data = cursor.fetchall()
            
            # Get trading signals
            cursor.execute("""
                SELECT signal_type, signal_strength, confidence, timestamp
                FROM trading_signals 
                WHERE symbol = %s AND timestamp >= DATE_SUB(NOW(), INTERVAL %s HOUR)
                ORDER BY timestamp DESC 
                LIMIT 20
            """, (symbol, hours_back))
            signals_data = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            return {
                'symbol': symbol,
                'price_data': price_data[:10],  # Last 10 price points
                'technical_indicators': technical_data[:5],  # Last 5 technical readings
                'trading_signals': signals_data[:10],  # Last 10 signals
                'data_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get market data for {symbol}: {e}")
            return {}
    
    def get_sentiment_data(self, symbol: str, hours_back: int = 24) -> Dict:
        """Get sentiment data for analysis"""
        try:
            # Switch to sentiment database
            sentiment_db_config = self.db_config.copy()
            sentiment_db_config['database'] = 'crypto_prices'
            
            conn = mysql.connector.connect(**sentiment_db_config)
            cursor = conn.cursor(dictionary=True)
            
            # Get unified sentiment data from the correct table
            cursor.execute("""
                SELECT symbol, composite_sentiment, confidence_score, 
                       sentiment_label, signal_strength, content_timestamp,
                       source_type
                FROM unified_sentiment_data 
                WHERE symbol = %s AND content_timestamp >= DATE_SUB(NOW(), INTERVAL %s HOUR)
                ORDER BY content_timestamp DESC 
                LIMIT 10
            """, (symbol, hours_back))
            sentiment_data = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            return {
                'symbol': symbol,
                'sentiment_data': sentiment_data,
                'sentiment_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get sentiment data for {symbol}: {e}")
            return {}
    
    def analyze_trade_opportunity(self, symbol: str, action: str, market_data: Dict, sentiment_data: Dict) -> Dict:
        """Analyze a trade opportunity using OpenAI"""
        try:
            # Prepare the prompt
            prompt = self._create_analysis_prompt(symbol, action, market_data, sentiment_data)
            
            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert cryptocurrency trading analyst. Provide concise, data-driven analysis for trading decisions. Be specific about risks and opportunities."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=0.3
            )
            
            analysis = response.choices[0].message.content
            
            # Extract structured insights
            insights = self._parse_analysis(analysis)
            
            return {
                'symbol': symbol,
                'action': action,
                'analysis': analysis,
                'insights': insights,
                'confidence': insights.get('confidence', 0.5),
                'recommendation': insights.get('recommendation', 'HOLD'),
                'risk_level': insights.get('risk_level', 'MEDIUM'),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ OpenAI analysis failed for {symbol}: {e}")
            return {
                'symbol': symbol,
                'action': action,
                'error': str(e),
                'confidence': 0.0,
                'recommendation': 'HOLD',
                'timestamp': datetime.now().isoformat()
            }
    
    def _create_analysis_prompt(self, symbol: str, action: str, market_data: Dict, sentiment_data: Dict) -> str:
        """Create analysis prompt for OpenAI"""
        
        # Extract key data points
        latest_price = market_data.get('price_data', [{}])[0].get('price', 'N/A')
        latest_rsi = market_data.get('technical_indicators', [{}])[0].get('rsi', 'N/A')
        latest_sentiment = sentiment_data.get('sentiment_data', [{}])[0].get('composite_sentiment', 'N/A')
        
        prompt = f"""
Analyze this cryptocurrency trading opportunity:

SYMBOL: {symbol}
PROPOSED ACTION: {action}
CURRENT PRICE: ${latest_price}

TECHNICAL ANALYSIS:
- RSI: {latest_rsi}
- Recent price movement: {[p.get('price') for p in market_data.get('price_data', [])[:5]]}
- Technical indicators: {market_data.get('technical_indicators', [{}])[0]}

SENTIMENT ANALYSIS:
- Current sentiment score: {latest_sentiment}
- Sentiment trend: {sentiment_data.get('sentiment_data', [{}])[0].get('trend_direction', 'N/A')}
- Data quality: {sentiment_data.get('sentiment_data', [{}])[0].get('data_quality_score', 'N/A')}

TRADING SIGNALS:
- Recent signals: {[s.get('signal_type') for s in market_data.get('trading_signals', [])[:3]]}

Please provide:
1. RECOMMENDATION: BUY/SELL/HOLD with confidence (0.0-1.0)
2. RISK LEVEL: LOW/MEDIUM/HIGH
3. KEY FACTORS: Most important factors influencing this decision
4. RISKS: Primary risks to consider
5. PRICE TARGETS: Suggested entry/exit levels if applicable

Keep analysis concise and data-focused.
"""
        
        return prompt
    
    def _parse_analysis(self, analysis: str) -> Dict:
        """Parse OpenAI analysis response into structured data"""
        insights = {
            'confidence': 0.5,
            'recommendation': 'HOLD',
            'risk_level': 'MEDIUM',
            'key_factors': [],
            'risks': [],
            'price_targets': {}
        }
        
        try:
            lines = analysis.split('\n')
            
            for line in lines:
                line = line.strip()
                
                # Extract recommendation and confidence
                if 'RECOMMENDATION:' in line.upper():
                    parts = line.split(':')[-1].strip()
                    if 'BUY' in parts.upper():
                        insights['recommendation'] = 'BUY'
                    elif 'SELL' in parts.upper():
                        insights['recommendation'] = 'SELL'
                    
                    # Look for confidence score
                    if '(' in parts and ')' in parts:
                        try:
                            conf_str = parts.split('(')[1].split(')')[0]
                            insights['confidence'] = float(conf_str)
                        except:
                            pass
                
                # Extract risk level
                elif 'RISK LEVEL:' in line.upper():
                    risk = line.split(':')[-1].strip().upper()
                    if risk in ['LOW', 'MEDIUM', 'HIGH']:
                        insights['risk_level'] = risk
                
                # Extract key factors
                elif 'KEY FACTORS:' in line.upper():
                    factors = line.split(':')[-1].strip()
                    insights['key_factors'] = [factors] if factors else []
                
                # Extract risks
                elif 'RISKS:' in line.upper():
                    risks = line.split(':')[-1].strip()
                    insights['risks'] = [risks] if risks else []
        
        except Exception as e:
            logger.warning(f"⚠️ Failed to parse analysis insights: {e}")
        
        return insights
    
    def enhance_trade_recommendation(self, recommendation_id: int) -> Dict:
        """Enhance an existing trade recommendation with AI analysis"""
        try:
            # Get the recommendation from database
            conn = mysql.connector.connect(**{**self.db_config, 'database': 'crypto_transactions'})
            cursor = conn.cursor(dictionary=True)
            
            cursor.execute("""
                SELECT * FROM trade_recommendations WHERE id = %s
            """, (recommendation_id,))
            
            recommendation = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if not recommendation:
                return {'error': 'Recommendation not found'}
            
            symbol = recommendation['symbol']
            action = recommendation['action']
            
            # Get market and sentiment data
            market_data = self.get_market_data(symbol)
            sentiment_data = self.get_sentiment_data(symbol)
            
            # Perform AI analysis
            ai_analysis = self.analyze_trade_opportunity(symbol, action, market_data, sentiment_data)
            
            # Update the recommendation with AI insights
            self._update_recommendation_with_ai(recommendation_id, ai_analysis)
            
            return {
                'recommendation_id': recommendation_id,
                'symbol': symbol,
                'action': action,
                'ai_analysis': ai_analysis,
                'enhanced': True
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to enhance recommendation {recommendation_id}: {e}")
            return {'error': str(e)}
    
    def _update_recommendation_with_ai(self, recommendation_id: int, ai_analysis: Dict):
        """Update recommendation with AI analysis"""
        try:
            conn = mysql.connector.connect(**{**self.db_config, 'database': 'crypto_transactions'})
            cursor = conn.cursor()
            
            # Update reasoning field with AI analysis
            reasoning = f"AI Analysis: {ai_analysis.get('recommendation', 'N/A')} " \
                       f"(Confidence: {ai_analysis.get('confidence', 0.0):.2f}, " \
                       f"Risk: {ai_analysis.get('risk_level', 'N/A')})\n" \
                       f"Analysis: {ai_analysis.get('analysis', '')[:500]}..."
            
            cursor.execute("""
                UPDATE trade_recommendations 
                SET reasoning = %s, confidence = %s
                WHERE id = %s
            """, (reasoning, ai_analysis.get('confidence', 0.5), recommendation_id))
            
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info(f"✅ Updated recommendation {recommendation_id} with AI analysis")
            
        except Exception as e:
            logger.error(f"❌ Failed to update recommendation with AI: {e}")
    
    def generate_market_summary(self, symbols: List[str]) -> Dict:
        """Generate market summary for multiple symbols"""
        try:
            market_summaries = {}
            
            for symbol in symbols[:5]:  # Limit to 5 symbols to avoid API limits
                market_data = self.get_market_data(symbol, hours_back=4)
                sentiment_data = self.get_sentiment_data(symbol, hours_back=4)
                
                if market_data and sentiment_data:
                    analysis = self.analyze_trade_opportunity(symbol, 'ANALYZE', market_data, sentiment_data)
                    market_summaries[symbol] = {
                        'current_price': market_data.get('price_data', [{}])[0].get('price'),
                        'sentiment_score': sentiment_data.get('sentiment_data', [{}])[0].get('composite_sentiment'),
                        'ai_recommendation': analysis.get('recommendation'),
                        'confidence': analysis.get('confidence'),
                        'risk_level': analysis.get('risk_level')
                    }
            
            return {
                'market_summary': market_summaries,
                'generated_at': datetime.now().isoformat(),
                'symbols_analyzed': len(market_summaries)
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to generate market summary: {e}")
            return {'error': str(e)}

if __name__ == "__main__":
    # Test the OpenAI integration
    try:
        analyst = OpenAITradingAnalyst()
        
        # Test market data retrieval
        market_data = analyst.get_market_data('BTC')
        print(f"✅ Market data retrieved for BTC: {len(market_data.get('price_data', []))} price points")
        
        # Test sentiment data
        sentiment_data = analyst.get_sentiment_data('BTC')
        print(f"✅ Sentiment data retrieved for BTC: {len(sentiment_data.get('sentiment_data', []))} sentiment points")
        
        # Test AI analysis
        analysis = analyst.analyze_trade_opportunity('BTC', 'BUY', market_data, sentiment_data)
        print(f"✅ AI analysis completed - Recommendation: {analysis.get('recommendation')}, Confidence: {analysis.get('confidence')}")
        
        print("🤖 OpenAI Trading Analyst ready for live trading!")
        
    except Exception as e:
        print(f"❌ OpenAI integration test failed: {e}")
