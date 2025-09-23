#!/usr/bin/env python3
"""
Real-time Sentiment Analysis Service
Analyzes Twitter and Reddit sentiment for trending cryptocurrencies
Provides sentiment scores and trending analysis to trading engine
"""

import os
import sys
import asyncio
import logging
import json
import re
import mysql.connector
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uvicorn
import aiohttp
import time
from collections import defaultdict, deque

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Real-time Sentiment Analysis Service", version="1.0.0")

class SentimentRequest(BaseModel):
    symbol: str
    lookback_minutes: int = 60

class SentimentResponse(BaseModel):
    symbol: str
    sentiment_score: float
    sentiment_trend: str
    volume_score: float
    confidence: float
    sources: Dict[str, Dict]
    timestamp: str

class TrendingCryptoResponse(BaseModel):
    trending_symbols: List[Dict]
    sentiment_analysis: Dict[str, Dict]
    timestamp: str

class RealTimeSentimentAnalyzer:
    def __init__(self):
        self.db_config = {
            'host': os.environ.get('DATABASE_HOST', 'host.docker.internal'),
            'user': os.environ.get('DATABASE_USER', 'news_collector'),
            'password': os.environ.get('DATABASE_PASSWORD', '99Rules!'),
            'database': os.environ.get('DATABASE_NAME', 'crypto_prices'),
            'port': int(os.environ.get('DATABASE_PORT', 3306))
        }
        
        # Sentiment analysis parameters
        self.sentiment_cache = {}
        self.trending_cache = {}
        self.cache_duration = 300  # 5 minutes
        
        # Sentiment keywords for basic analysis
        self.positive_keywords = {
            'moon', 'bullish', 'pump', 'rocket', 'diamond hands', 'hodl', 'buy', 
            'rally', 'surge', 'breakout', 'gains', 'profit', 'bullrun', 'to the moon',
            'green', 'up', 'rise', 'strong', 'support', 'bounce', 'recovery',
            'accumulate', 'oversold', 'undervalued', 'gem', 'potential'
        }
        
        self.negative_keywords = {
            'bearish', 'dump', 'crash', 'sell', 'drop', 'fall', 'red', 'down',
            'panic', 'fear', 'liquidation', 'loss', 'bear market', 'correction',
            'resistance', 'rejection', 'weak', 'oversold', 'overvalued',
            'bubble', 'scam', 'rug pull', 'exit', 'capitulation'
        }
        
        # Cryptocurrency symbols to monitor
        self.crypto_symbols = {
            'BTC', 'ETH', 'ADA', 'SOL', 'DOT', 'MATIC', 'AVAX', 'ATOM', 'LINK',
            'UNI', 'ALGO', 'XRP', 'LTC', 'BCH', 'ETC', 'XLM', 'FIL', 'AAVE',
            'COMP', 'MKR', 'YFI', 'SUSHI', 'CRV', 'BAL', 'FARM'
        }
        
    def analyze_text_sentiment(self, text: str, symbol: str) -> Tuple[float, Dict]:
        """Analyze sentiment of text using keyword-based approach"""
        try:
            text_lower = text.lower()
            
            # Remove URLs and special characters
            text_clean = re.sub(r'http[s]?://\S+', '', text_lower)
            text_clean = re.sub(r'[^\w\s]', ' ', text_clean)
            
            # Count positive and negative keywords
            positive_count = sum(1 for word in self.positive_keywords if word in text_clean)
            negative_count = sum(1 for word in self.negative_keywords if word in text_clean)
            
            # Symbol-specific sentiment boost
            symbol_mentions = text_clean.count(symbol.lower())
            if symbol_mentions > 0:
                positive_count *= (1 + symbol_mentions * 0.2)
                negative_count *= (1 + symbol_mentions * 0.2)
            
            # Calculate sentiment score (-1 to 1)
            total_sentiment_words = positive_count + negative_count
            if total_sentiment_words == 0:
                sentiment_score = 0.0
            else:
                sentiment_score = (positive_count - negative_count) / max(total_sentiment_words, 1)
            
            # Normalize to -1 to 1 range
            sentiment_score = max(-1.0, min(1.0, sentiment_score))
            
            analysis_details = {
                'positive_words': positive_count,
                'negative_words': negative_count,
                'symbol_mentions': symbol_mentions,
                'text_length': len(text_clean.split())
            }
            
            return sentiment_score, analysis_details
            
        except Exception as e:
            logger.error(f"Error analyzing text sentiment: {e}")
            return 0.0, {}
    
    def get_social_sentiment_data(self, symbol: str, lookback_minutes: int = 60) -> Dict:
        """Get sentiment data from social sources"""
        try:
            conn = mysql.connector.connect(**self.db_config)
            
            # Get social sentiment data from last hour
            cutoff_time = datetime.now() - timedelta(minutes=lookback_minutes)
            
            # Query social data tables
            social_query = """
            SELECT content, sentiment_score, source, created_at 
            FROM social_data 
            WHERE symbol = %s AND created_at >= %s 
            ORDER BY created_at DESC 
            LIMIT 500
            """
            
            social_df = pd.read_sql(social_query, conn, params=[symbol, cutoff_time])
            
            conn.close()
            
            if social_df.empty:
                return {
                    'sentiment_score': 0.0,
                    'volume': 0,
                    'confidence': 0.0,
                    'sources': {}
                }
            
            # Analyze sentiment by source
            source_analysis = {}
            total_sentiment = 0.0
            total_weight = 0.0
            
            for source in social_df['source'].unique():
                source_data = social_df[social_df['source'] == source]
                
                # Calculate weighted sentiment (more recent = higher weight)
                now = datetime.now()
                weights = []
                sentiments = []
                
                for _, row in source_data.iterrows():
                    time_diff = (now - row['created_at']).total_seconds() / 3600  # Hours ago
                    weight = max(0.1, 1.0 - (time_diff / 24))  # Decay over 24 hours
                    
                    # Use existing sentiment score if available, otherwise analyze text
                    if pd.notna(row['sentiment_score']) and row['sentiment_score'] != 0:
                        sentiment = float(row['sentiment_score'])
                    else:
                        sentiment, _ = self.analyze_text_sentiment(str(row['content']), symbol)
                    
                    weights.append(weight)
                    sentiments.append(sentiment)
                
                if weights:
                    weighted_sentiment = np.average(sentiments, weights=weights)
                    total_sentiment += weighted_sentiment * len(source_data)
                    total_weight += len(source_data)
                    
                    source_analysis[source] = {
                        'sentiment': weighted_sentiment,
                        'volume': len(source_data),
                        'avg_recency_hours': np.mean([(now - row['created_at']).total_seconds() / 3600 
                                                     for _, row in source_data.iterrows()])
                    }
            
            # Overall sentiment
            overall_sentiment = total_sentiment / total_weight if total_weight > 0 else 0.0
            
            # Calculate confidence based on volume and consistency
            total_volume = len(social_df)
            confidence = min(1.0, (total_volume / 50) * 0.8)  # Higher confidence with more data
            
            return {
                'sentiment_score': overall_sentiment,
                'volume': total_volume,
                'confidence': confidence,
                'sources': source_analysis
            }
            
        except Exception as e:
            logger.error(f"Error getting social sentiment for {symbol}: {e}")
            return {
                'sentiment_score': 0.0,
                'volume': 0,
                'confidence': 0.0,
                'sources': {}
            }
    
    def get_news_sentiment_data(self, symbol: str, lookback_minutes: int = 60) -> Dict:
        """Get sentiment data from news sources"""
        try:
            conn = mysql.connector.connect(**self.db_config)
            
            cutoff_time = datetime.now() - timedelta(minutes=lookback_minutes)
            
            # Query news sentiment
            news_query = """
            SELECT title, content, sentiment_score, source, published_at 
            FROM crypto_news 
            WHERE symbols LIKE %s AND published_at >= %s 
            ORDER BY published_at DESC 
            LIMIT 100
            """
            
            symbol_pattern = f'%{symbol}%'
            news_df = pd.read_sql(news_query, conn, params=[symbol_pattern, cutoff_time])
            
            conn.close()
            
            if news_df.empty:
                return {
                    'sentiment_score': 0.0,
                    'volume': 0,
                    'confidence': 0.0
                }
            
            # Analyze news sentiment with recency weighting
            now = datetime.now()
            weighted_sentiments = []
            weights = []
            
            for _, row in news_df.iterrows():
                time_diff = (now - row['published_at']).total_seconds() / 3600  # Hours ago
                weight = max(0.2, 1.0 - (time_diff / 12))  # Decay over 12 hours
                
                # Use existing sentiment score if available, otherwise analyze title
                if pd.notna(row['sentiment_score']) and row['sentiment_score'] != 0:
                    sentiment = float(row['sentiment_score'])
                else:
                    sentiment, _ = self.analyze_text_sentiment(str(row['title']), symbol)
                
                weighted_sentiments.append(sentiment)
                weights.append(weight)
            
            overall_sentiment = np.average(weighted_sentiments, weights=weights) if weights else 0.0
            confidence = min(1.0, (len(news_df) / 20) * 0.9)  # Higher confidence for news
            
            return {
                'sentiment_score': overall_sentiment,
                'volume': len(news_df),
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"Error getting news sentiment for {symbol}: {e}")
            return {
                'sentiment_score': 0.0,
                'volume': 0,
                'confidence': 0.0
            }
    
    def calculate_combined_sentiment(self, symbol: str, lookback_minutes: int = 60) -> Dict:
        """Calculate combined sentiment from all sources"""
        try:
            # Check cache first
            cache_key = f"{symbol}_{lookback_minutes}"
            if cache_key in self.sentiment_cache:
                cached_data, timestamp = self.sentiment_cache[cache_key]
                if (datetime.now() - timestamp).total_seconds() < self.cache_duration:
                    return cached_data
            
            # Get sentiment from different sources
            social_sentiment = self.get_social_sentiment_data(symbol, lookback_minutes)
            news_sentiment = self.get_news_sentiment_data(symbol, lookback_minutes)
            
            # Combine sentiments with weights
            social_weight = 0.6  # Social sentiment is more real-time
            news_weight = 0.4   # News sentiment is more authoritative
            
            # Calculate weighted average
            total_weight = 0.0
            combined_sentiment = 0.0
            
            if social_sentiment['confidence'] > 0:
                combined_sentiment += social_sentiment['sentiment_score'] * social_weight * social_sentiment['confidence']
                total_weight += social_weight * social_sentiment['confidence']
            
            if news_sentiment['confidence'] > 0:
                combined_sentiment += news_sentiment['sentiment_score'] * news_weight * news_sentiment['confidence']
                total_weight += news_weight * news_sentiment['confidence']
            
            if total_weight > 0:
                combined_sentiment /= total_weight
            
            # Calculate overall confidence
            overall_confidence = (social_sentiment['confidence'] * 0.6 + news_sentiment['confidence'] * 0.4)
            
            # Calculate sentiment trend (comparing to 2x lookback period)
            if lookback_minutes >= 30:
                prev_sentiment = self.calculate_combined_sentiment(symbol, lookback_minutes * 2)
                prev_score = prev_sentiment.get('sentiment_score', 0.0)
                sentiment_change = combined_sentiment - prev_score
                
                if sentiment_change > 0.1:
                    trend = 'IMPROVING'
                elif sentiment_change < -0.1:
                    trend = 'DECLINING'
                else:
                    trend = 'STABLE'
            else:
                trend = 'STABLE'
            
            # Calculate volume score
            total_volume = social_sentiment['volume'] + news_sentiment['volume']
            volume_score = min(1.0, total_volume / 100)  # Normalize to 0-1
            
            result = {
                'sentiment_score': combined_sentiment,
                'sentiment_trend': trend,
                'volume_score': volume_score,
                'confidence': overall_confidence,
                'sources': {
                    'social': social_sentiment,
                    'news': news_sentiment
                },
                'timestamp': datetime.now().isoformat()
            }
            
            # Cache the result
            self.sentiment_cache[cache_key] = (result, datetime.now())
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating combined sentiment for {symbol}: {e}")
            return {
                'sentiment_score': 0.0,
                'sentiment_trend': 'STABLE',
                'volume_score': 0.0,
                'confidence': 0.0,
                'sources': {},
                'timestamp': datetime.now().isoformat()
            }
    
    def get_trending_cryptocurrencies(self) -> Dict:
        """Get trending cryptocurrencies based on social volume and sentiment"""
        try:
            # Check cache
            if 'trending' in self.trending_cache:
                cached_data, timestamp = self.trending_cache['trending']
                if (datetime.now() - timestamp).total_seconds() < self.cache_duration:
                    return cached_data
            
            trending_data = []
            sentiment_analysis = {}
            
            # Analyze each crypto symbol
            for symbol in self.crypto_symbols:
                sentiment_data = self.calculate_combined_sentiment(symbol, 60)
                
                # Calculate trending score based on volume and sentiment
                volume_score = sentiment_data.get('volume_score', 0.0)
                sentiment_score = abs(sentiment_data.get('sentiment_score', 0.0))  # Absolute value for volatility
                confidence = sentiment_data.get('confidence', 0.0)
                
                trending_score = (volume_score * 0.5 + sentiment_score * 0.3 + confidence * 0.2)
                
                if trending_score > 0.1:  # Only include symbols with meaningful activity
                    trending_data.append({
                        'symbol': symbol,
                        'trending_score': trending_score,
                        'sentiment_score': sentiment_data.get('sentiment_score', 0.0),
                        'volume_score': volume_score,
                        'confidence': confidence,
                        'trend': sentiment_data.get('sentiment_trend', 'STABLE')
                    })
                    
                    sentiment_analysis[symbol] = sentiment_data
            
            # Sort by trending score
            trending_data.sort(key=lambda x: x['trending_score'], reverse=True)
            trending_data = trending_data[:10]  # Top 10 trending
            
            result = {
                'trending_symbols': trending_data,
                'sentiment_analysis': sentiment_analysis,
                'timestamp': datetime.now().isoformat()
            }
            
            # Cache the result
            self.trending_cache['trending'] = (result, datetime.now())
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting trending cryptocurrencies: {e}")
            return {
                'trending_symbols': [],
                'sentiment_analysis': {},
                'timestamp': datetime.now().isoformat()
            }

# Global sentiment analyzer instance
sentiment_analyzer = RealTimeSentimentAnalyzer()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "sentiment-analysis", "timestamp": datetime.now().isoformat()}

@app.post("/analyze_sentiment", response_model=SentimentResponse)
async def analyze_sentiment_endpoint(request: SentimentRequest):
    """Analyze sentiment for a specific cryptocurrency"""
    try:
        result = sentiment_analyzer.calculate_combined_sentiment(
            request.symbol.upper(),
            request.lookback_minutes
        )
        
        return SentimentResponse(
            symbol=request.symbol.upper(),
            sentiment_score=result['sentiment_score'],
            sentiment_trend=result['sentiment_trend'],
            volume_score=result['volume_score'],
            confidence=result['confidence'],
            sources=result['sources'],
            timestamp=result['timestamp']
        )
        
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/trending", response_model=TrendingCryptoResponse)
async def get_trending_endpoint():
    """Get trending cryptocurrencies based on sentiment and volume"""
    try:
        result = sentiment_analyzer.get_trending_cryptocurrencies()
        
        return TrendingCryptoResponse(
            trending_symbols=result['trending_symbols'],
            sentiment_analysis=result['sentiment_analysis'],
            timestamp=result['timestamp']
        )
        
    except Exception as e:
        logger.error(f"Error getting trending cryptos: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sentiment_batch/{symbols}")
async def get_sentiment_batch(symbols: str):
    """Get sentiment for multiple symbols (comma-separated)"""
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(',')]
        results = {}
        
        for symbol in symbol_list:
            if symbol in sentiment_analyzer.crypto_symbols:
                sentiment_data = sentiment_analyzer.calculate_combined_sentiment(symbol, 60)
                results[symbol] = sentiment_data
        
        return {
            'results': results,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in batch sentiment analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sentiment_metrics")
async def get_sentiment_metrics():
    """Get overall sentiment metrics and statistics"""
    try:
        # Get trending data
        trending_data = sentiment_analyzer.get_trending_cryptocurrencies()
        
        # Calculate overall market sentiment
        sentiment_scores = []
        for symbol_data in trending_data['sentiment_analysis'].values():
            score = symbol_data.get('sentiment_score', 0.0)
            confidence = symbol_data.get('confidence', 0.0)
            if confidence > 0.3:  # Only include confident scores
                sentiment_scores.append(score)
        
        market_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.0
        
        # Count sentiment distribution
        positive_count = sum(1 for score in sentiment_scores if score > 0.1)
        negative_count = sum(1 for score in sentiment_scores if score < -0.1)
        neutral_count = len(sentiment_scores) - positive_count - negative_count
        
        return {
            'market_sentiment': market_sentiment,
            'sentiment_distribution': {
                'positive': positive_count,
                'negative': negative_count,
                'neutral': neutral_count
            },
            'total_symbols_analyzed': len(sentiment_scores),
            'trending_count': len(trending_data['trending_symbols']),
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting sentiment metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "sentiment_analysis_service:app",
        host="0.0.0.0",
        port=8028,
        log_level="info"
    )
