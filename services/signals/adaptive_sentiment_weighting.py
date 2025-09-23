#!/usr/bin/env python3
"""
Adaptive Sentiment Weighting Module
Provides sentiment source enumeration and adaptive weighting functionality
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class SentimentSource(Enum):
    """Enumeration of sentiment data sources"""
    SOCIAL_TWITTER = "social_twitter"
    SOCIAL_REDDIT = "social_reddit"
    NEWS_CRYPTO = "news_crypto"
    NEWS_TRADITIONAL = "news_traditional"
    TECHNICAL_INDICATORS = "technical_indicators"
    MARKET_SENTIMENT = "market_sentiment"

class AdaptiveSentimentWeighting:
    """
    Adaptive sentiment weighting system that adjusts sentiment source weights
    based on trading performance feedback
    """
    
    def __init__(self):
        """Initialize with default sentiment source weights"""
        self.source_weights = {
            SentimentSource.SOCIAL_TWITTER: 0.15,
            SentimentSource.SOCIAL_REDDIT: 0.10,
            SentimentSource.NEWS_CRYPTO: 0.25,
            SentimentSource.NEWS_TRADITIONAL: 0.20,
            SentimentSource.TECHNICAL_INDICATORS: 0.20,
            SentimentSource.MARKET_SENTIMENT: 0.10
        }
        
        self.performance_history = {}
        self.adjustment_factor = 0.1  # 10% adjustment per feedback cycle
        self.min_weight = 0.05
        self.max_weight = 0.4
        
        logger.info("✅ AdaptiveSentimentWeighting initialized with default weights")
    
    def get_weights(self) -> Dict[SentimentSource, float]:
        """Get current sentiment source weights"""
        return self.source_weights.copy()
    
    def update_performance(self, source: SentimentSource, symbol: str, 
                          performance_score: float, trade_result: str):
        """
        Update performance tracking for a sentiment source
        
        Args:
            source: The sentiment source
            symbol: The trading symbol
            performance_score: Performance score (-1.0 to 1.0)
            trade_result: 'profitable' or 'loss'
        """
        if source not in self.performance_history:
            self.performance_history[source] = {
                'scores': [],
                'trade_count': 0,
                'profitable_trades': 0
            }
        
        history = self.performance_history[source]
        history['scores'].append(performance_score)
        history['trade_count'] += 1
        
        if trade_result == 'profitable':
            history['profitable_trades'] += 1
        
        # Keep only last 100 scores for each source
        if len(history['scores']) > 100:
            history['scores'] = history['scores'][-100:]
        
        logger.debug(f"Updated performance for {source.value}: score={performance_score:.3f}, result={trade_result}")
    
    def adjust_weights(self) -> Dict[SentimentSource, float]:
        """
        Adjust sentiment source weights based on performance history
        
        Returns:
            Updated weights dictionary
        """
        if not self.performance_history:
            return self.source_weights.copy()
        
        # Calculate performance metrics for each source
        adjustments = {}
        for source, history in self.performance_history.items():
            if history['trade_count'] >= 5:  # Need minimum trades for adjustment
                avg_score = sum(history['scores']) / len(history['scores'])
                win_rate = history['profitable_trades'] / history['trade_count']
                
                # Combined performance metric (score and win rate)
                performance_metric = (avg_score * 0.6) + ((win_rate - 0.5) * 2 * 0.4)
                
                # Calculate adjustment
                adjustment = performance_metric * self.adjustment_factor
                adjustments[source] = adjustment
                
                logger.debug(f"{source.value}: avg_score={avg_score:.3f}, win_rate={win_rate:.3f}, "
                           f"performance={performance_metric:.3f}, adjustment={adjustment:.3f}")
        
        # Apply adjustments
        for source, adjustment in adjustments.items():
            old_weight = self.source_weights[source]
            new_weight = old_weight + adjustment
            
            # Apply bounds
            new_weight = max(self.min_weight, min(self.max_weight, new_weight))
            self.source_weights[source] = new_weight
            
            logger.info(f"Adjusted {source.value}: {old_weight:.3f} → {new_weight:.3f} "
                       f"(change: {adjustment:+.3f})")
        
        # Normalize weights to sum to 1.0
        total_weight = sum(self.source_weights.values())
        if total_weight > 0:
            for source in self.source_weights:
                self.source_weights[source] /= total_weight
        
        return self.source_weights.copy()
    
    def get_source_performance(self, source: SentimentSource) -> Dict:
        """Get performance metrics for a specific source"""
        if source not in self.performance_history:
            return {'trade_count': 0, 'avg_score': 0.0, 'win_rate': 0.0}
        
        history = self.performance_history[source]
        avg_score = sum(history['scores']) / len(history['scores']) if history['scores'] else 0.0
        win_rate = history['profitable_trades'] / history['trade_count'] if history['trade_count'] > 0 else 0.0
        
        return {
            'trade_count': history['trade_count'],
            'avg_score': avg_score,
            'win_rate': win_rate,
            'current_weight': self.source_weights[source]
        }
    
    def reset_performance_history(self):
        """Reset all performance history"""
        self.performance_history.clear()
        logger.info("Performance history reset")
    
    def update_performance_feedback(self, source: str, accuracy: float, signal_strength: float):
        """
        Update performance feedback for sentiment sources based on trading results
        
        Args:
            source: Source name (string)
            accuracy: Prediction accuracy (0.0 to 1.0)
            signal_strength: Signal strength (0.0 to 1.0)
        """
        # Map source name to SentimentSource enum
        source_mapping = {
            'social_twitter': SentimentSource.SOCIAL_TWITTER,
            'social_reddit': SentimentSource.SOCIAL_REDDIT,
            'news_crypto': SentimentSource.NEWS_CRYPTO,
            'news_traditional': SentimentSource.NEWS_TRADITIONAL,
            'technical_indicators': SentimentSource.TECHNICAL_INDICATORS,
            'market_sentiment': SentimentSource.MARKET_SENTIMENT,
            'combined': SentimentSource.MARKET_SENTIMENT  # Default for combined sources
        }
        
        if source not in source_mapping:
            logger.warning(f"Unknown sentiment source: {source}")
            return
        
        sentiment_source = source_mapping[source]
        
        # Convert accuracy and signal strength to performance score
        # High accuracy and strong signal = positive performance
        # Low accuracy or weak signal = negative performance
        performance_score = (accuracy - 0.5) * 2 * signal_strength  # Range: -1.0 to 1.0
        trade_result = 'profitable' if accuracy > 0.5 else 'loss'
        
        # Update performance tracking
        self.update_performance(sentiment_source, source, performance_score, trade_result)
        
        # Adjust weights based on updated performance
        updated_weights = self.adjust_weights()
        
        logger.info(f"Updated performance for {source}: accuracy={accuracy:.3f}, "
                   f"signal_strength={signal_strength:.3f}, score={performance_score:.3f}")
        
        return updated_weights
    
    def set_weights(self, weights: Dict[SentimentSource, float]):
        """Manually set sentiment source weights"""
        # Normalize and validate weights
        total = sum(weights.values())
        if total > 0:
            normalized_weights = {k: v/total for k, v in weights.items()}
            self.source_weights.update(normalized_weights)
            logger.info(f"Manually set weights: {normalized_weights}")