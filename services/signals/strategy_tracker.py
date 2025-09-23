"""
Strategy Tracker for Enhanced ML Ensemble Signals

This module tracks which trading strategies contributed to each signal generated
by the enhanced_ml_ensemble model, providing detailed analytics on strategy performance.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class StrategyType(Enum):
    """Enumeration of all trading strategies used in signal generation"""
    CORE_ML = "core_ml"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    MOMENTUM_DETECTION = "momentum_detection"
    VIRAL_COIN_DETECTION = "viral_coin_detection"
    VOLUME_SURGE = "volume_surge"
    MARKET_REGIME = "market_regime"
    SELLOFF_PROTECTION = "selloff_protection"
    RECOVERY_ENHANCEMENT = "recovery_enhancement"
    PORTFOLIO_REBALANCING = "portfolio_rebalancing"
    CASH_DEPLOYMENT = "cash_deployment"
    KELLY_POSITION_SIZING = "kelly_position_sizing"
    COOLDOWN_MANAGEMENT = "cooldown_management"
    NEW_COIN_ALERT = "new_coin_alert"
    MULTI_TIMEFRAME = "multi_timeframe"
    TECHNICAL_INDICATORS = "technical_indicators"

class StrategyImpact(Enum):
    """Impact of strategy on final signal"""
    STRONG_BUY = "strong_buy"
    MODERATE_BUY = "moderate_buy"
    NEUTRAL = "neutral"
    MODERATE_SELL = "moderate_sell"
    STRONG_SELL = "strong_sell"
    NOT_APPLICABLE = "not_applicable"

@dataclass
class StrategyContribution:
    """Individual strategy contribution to signal generation"""
    strategy_type: StrategyType
    was_active: bool
    confidence: float  # 0.0 to 1.0
    impact: StrategyImpact
    weight: float  # Contribution weight to final decision
    details: Dict[str, Any]  # Strategy-specific details
    execution_time_ms: float
    error_message: Optional[str] = None

class StrategyTracker:
    """Tracks strategy contributions during signal generation"""
    
    def __init__(self, symbol: str, signal_generation_id: str = None):
        self.symbol = symbol
        self.signal_generation_id = signal_generation_id or f"{symbol}_{datetime.now().isoformat()}"
        self.contributions: List[StrategyContribution] = []
        self.start_time = datetime.now()
        self.metadata = {
            'symbol': symbol,
            'generation_id': self.signal_generation_id,
            'start_time': self.start_time.isoformat(),
            'total_strategies_evaluated': 0,
            'active_strategies_count': 0,
            'failed_strategies_count': 0
        }
    
    def record_strategy_contribution(self, 
                                   strategy_type: StrategyType,
                                   was_active: bool,
                                   confidence: float = 0.0,
                                   impact: StrategyImpact = StrategyImpact.NOT_APPLICABLE,
                                   weight: float = 0.0,
                                   details: Dict[str, Any] = None,
                                   execution_time_ms: float = 0.0,
                                   error_message: Optional[str] = None) -> None:
        """Record a strategy's contribution to signal generation"""
        
        contribution = StrategyContribution(
            strategy_type=strategy_type,
            was_active=was_active,
            confidence=confidence,
            impact=impact,
            weight=weight,
            details=details or {},
            execution_time_ms=execution_time_ms,
            error_message=error_message
        )
        
        self.contributions.append(contribution)
        self.metadata['total_strategies_evaluated'] += 1
        
        if was_active:
            self.metadata['active_strategies_count'] += 1
        
        if error_message:
            self.metadata['failed_strategies_count'] += 1
            logger.warning(f"Strategy {strategy_type.value} failed for {self.symbol}: {error_message}")
    
    def get_active_strategies(self) -> List[StrategyContribution]:
        """Get list of strategies that were active in signal generation"""
        return [c for c in self.contributions if c.was_active]
    
    def get_strategy_by_type(self, strategy_type: StrategyType) -> Optional[StrategyContribution]:
        """Get specific strategy contribution by type"""
        for contribution in self.contributions:
            if contribution.strategy_type == strategy_type:
                return contribution
        return None
    
    def get_dominant_strategies(self, min_weight: float = 0.1) -> List[StrategyContribution]:
        """Get strategies that had significant impact on final decision"""
        return [c for c in self.contributions if c.was_active and c.weight >= min_weight]
    
    def get_buy_supporting_strategies(self) -> List[StrategyContribution]:
        """Get strategies that supported BUY decision"""
        return [c for c in self.contributions if c.impact in [StrategyImpact.STRONG_BUY, StrategyImpact.MODERATE_BUY]]
    
    def get_sell_supporting_strategies(self) -> List[StrategyContribution]:
        """Get strategies that supported SELL decision"""
        return [c for c in self.contributions if c.impact in [StrategyImpact.STRONG_SELL, StrategyImpact.MODERATE_SELL]]
    
    def calculate_strategy_consensus(self) -> Dict[str, float]:
        """Calculate consensus scores for different signal types"""
        active_contributions = self.get_active_strategies()
        if not active_contributions:
            return {'buy_consensus': 0.0, 'sell_consensus': 0.0, 'neutral_consensus': 0.0}
        
        total_weight = sum(c.weight for c in active_contributions)
        if total_weight == 0:
            return {'buy_consensus': 0.0, 'sell_consensus': 0.0, 'neutral_consensus': 1.0}
        
        buy_weight = sum(c.weight for c in active_contributions 
                        if c.impact in [StrategyImpact.STRONG_BUY, StrategyImpact.MODERATE_BUY])
        sell_weight = sum(c.weight for c in active_contributions 
                         if c.impact in [StrategyImpact.STRONG_SELL, StrategyImpact.MODERATE_SELL])
        neutral_weight = sum(c.weight for c in active_contributions 
                           if c.impact == StrategyImpact.NEUTRAL)
        
        return {
            'buy_consensus': buy_weight / total_weight,
            'sell_consensus': sell_weight / total_weight,
            'neutral_consensus': neutral_weight / total_weight
        }
    
    def generate_strategy_summary(self) -> Dict[str, Any]:
        """Generate comprehensive summary of strategy usage"""
        active_strategies = self.get_active_strategies()
        dominant_strategies = self.get_dominant_strategies()
        consensus = self.calculate_strategy_consensus()
        
        # Create strategy list for easy reading
        active_strategy_names = [s.strategy_type.value for s in active_strategies]
        dominant_strategy_names = [s.strategy_type.value for s in dominant_strategies]
        
        # Performance metrics
        total_execution_time = sum(c.execution_time_ms for c in self.contributions)
        avg_confidence = sum(c.confidence for c in active_strategies) / len(active_strategies) if active_strategies else 0.0
        
        summary = {
            'symbol': self.symbol,
            'generation_id': self.signal_generation_id,
            'timestamp': datetime.now().isoformat(),
            
            # Strategy counts
            'total_strategies_evaluated': len(self.contributions),
            'active_strategies_count': len(active_strategies),
            'dominant_strategies_count': len(dominant_strategies),
            'failed_strategies_count': self.metadata['failed_strategies_count'],
            
            # Strategy lists
            'active_strategies': active_strategy_names,
            'dominant_strategies': dominant_strategy_names,
            
            # Consensus and performance
            'strategy_consensus': consensus,
            'average_confidence': round(avg_confidence, 3),
            'total_execution_time_ms': round(total_execution_time, 2),
            
            # Detailed contributions
            'strategy_contributions': [
                {
                    'strategy': c.strategy_type.value,
                    'active': c.was_active,
                    'confidence': round(c.confidence, 3),
                    'impact': c.impact.value,
                    'weight': round(c.weight, 3),
                    'execution_time_ms': round(c.execution_time_ms, 2),
                    'details': c.details,
                    'error': c.error_message
                }
                for c in self.contributions
            ]
        }
        
        return summary
    
    def get_database_additional_data(self) -> Dict[str, Any]:
        """Get strategy data formatted for database storage in additional_data"""
        summary = self.generate_strategy_summary()
        
        # Create flattened data for easier querying
        active_strategies = self.get_active_strategies()
        dominant_strategies = self.get_dominant_strategies()
        consensus = self.calculate_strategy_consensus()
        
        return {
            # Strategy tracking metadata
            'strategies_total_evaluated': summary['total_strategies_evaluated'],
            'strategies_active_count': summary['active_strategies_count'],
            'strategies_dominant_count': summary['dominant_strategies_count'],
            'strategies_failed_count': summary['failed_strategies_count'],
            
            # Active strategy list (for easy filtering)
            'strategies_active_list': ','.join(summary['active_strategies']),
            'strategies_dominant_list': ','.join(summary['dominant_strategies']),
            
            # Consensus scores
            'strategy_buy_consensus': round(consensus['buy_consensus'], 3),
            'strategy_sell_consensus': round(consensus['sell_consensus'], 3),
            'strategy_neutral_consensus': round(consensus['neutral_consensus'], 3),
            
            # Performance metrics
            'strategies_avg_confidence': summary['average_confidence'],
            'strategies_execution_time_ms': summary['total_execution_time_ms'],
            
            # Individual strategy flags (for easy querying)
            'used_core_ml': any(c.strategy_type == StrategyType.CORE_ML and c.was_active for c in self.contributions),
            'used_sentiment_analysis': any(c.strategy_type == StrategyType.SENTIMENT_ANALYSIS and c.was_active for c in self.contributions),
            'used_momentum_detection': any(c.strategy_type == StrategyType.MOMENTUM_DETECTION and c.was_active for c in self.contributions),
            'used_viral_coin_detection': any(c.strategy_type == StrategyType.VIRAL_COIN_DETECTION and c.was_active for c in self.contributions),
            'used_volume_surge': any(c.strategy_type == StrategyType.VOLUME_SURGE and c.was_active for c in self.contributions),
            'used_market_regime': any(c.strategy_type == StrategyType.MARKET_REGIME and c.was_active for c in self.contributions),
            'used_selloff_protection': any(c.strategy_type == StrategyType.SELLOFF_PROTECTION and c.was_active for c in self.contributions),
            'used_recovery_enhancement': any(c.strategy_type == StrategyType.RECOVERY_ENHANCEMENT and c.was_active for c in self.contributions),
            'used_portfolio_rebalancing': any(c.strategy_type == StrategyType.PORTFOLIO_REBALANCING and c.was_active for c in self.contributions),
            'used_cash_deployment': any(c.strategy_type == StrategyType.CASH_DEPLOYMENT and c.was_active for c in self.contributions),
            'used_kelly_sizing': any(c.strategy_type == StrategyType.KELLY_POSITION_SIZING and c.was_active for c in self.contributions),
            'used_cooldown_management': any(c.strategy_type == StrategyType.COOLDOWN_MANAGEMENT and c.was_active for c in self.contributions),
            'used_new_coin_alert': any(c.strategy_type == StrategyType.NEW_COIN_ALERT and c.was_active for c in self.contributions),
            'used_multi_timeframe': any(c.strategy_type == StrategyType.MULTI_TIMEFRAME and c.was_active for c in self.contributions),
            
            # Full strategy details (JSON for complex queries)
            'strategy_full_details': json.dumps(summary, default=str)
        }

def create_strategy_tracker(symbol: str) -> StrategyTracker:
    """Factory function to create a new strategy tracker"""
    return StrategyTracker(symbol)
