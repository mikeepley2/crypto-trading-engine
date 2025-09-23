#!/usr/bin/env python3
"""
Intelligent Strategy Orchestration Service
Advanced multi-strategy coordination with intelligent weighting and conflict resolution.

Features:
- Dynamic strategy weighting based on market conditions
- Conflict resolution between competing strategies
- Risk budget allocation across strategies
- Performance-based strategy adjustment
- Market regime detection for strategy selection
- Emergency coordination during market stress
- Portfolio optimization across all strategies

Strategy Integration:
1. Enhanced Signal Generator (ML/XGBoost) - Core AI trading
2. Market Selloff Detector - Defensive cash allocation
3. Momentum/Hype Detector - Opportunistic momentum trading
4. Risk Management Service - Overall risk oversight
5. Portfolio Rebalancer - Portfolio optimization
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import mysql.connector
from mysql.connector import Error
import numpy as np
import pandas as pd
import requests
import uvicorn
from enum import Enum
from collections import defaultdict, deque

# Custom JSON encoder for handling enums and datetime objects
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [STRATEGY_ORCHESTRATOR] %(message)s'
)
logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    """Market regime classifications."""
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    SELLOFF = "selloff"
    RECOVERY = "recovery"
    MOMENTUM_PHASE = "momentum_phase"

class StrategyType(Enum):
    """Available trading strategies."""
    ML_SIGNALS = "ml_signals"
    MARKET_SELLOFF = "market_selloff"
    MOMENTUM_HYPE = "momentum_hype"
    RISK_MANAGEMENT = "risk_management"
    PORTFOLIO_REBALANCE = "portfolio_rebalance"

@dataclass
class StrategyWeight:
    """Strategy weighting configuration."""
    strategy: StrategyType
    base_weight: float  # Base allocation (0-1)
    current_weight: float  # Current adjusted weight
    performance_multiplier: float  # Performance-based adjustment
    regime_multiplier: float  # Market regime adjustment
    risk_adjusted_weight: float  # Final risk-adjusted weight
    confidence: float  # Confidence in strategy performance
    last_updated: datetime

@dataclass
class StrategySignal:
    """Unified strategy signal format."""
    timestamp: datetime
    strategy: StrategyType
    symbol: str
    action: str  # BUY, SELL, HOLD, LIQUIDATE
    confidence: float
    strength: float
    position_size: float
    urgency: str  # low, medium, high, emergency
    reason: str
    risk_score: float
    expected_return: float
    holding_period: str
    metadata: Dict[str, Any]

@dataclass
class StrategyConflict:
    """Strategy conflict detection and resolution."""
    timestamp: datetime
    symbol: str
    conflicting_strategies: List[StrategyType]
    conflict_type: str  # action_conflict, size_conflict, timing_conflict
    resolution_method: str
    final_action: str
    final_position_size: float
    confidence_penalty: float
    metadata: Dict[str, Any]

@dataclass
class RiskBudget:
    """Risk budget allocation across strategies."""
    strategy: StrategyType
    allocated_risk: float  # Percentage of total risk budget
    current_risk_usage: float  # Current risk utilization
    max_position_size: float  # Maximum position size for strategy
    max_drawdown_limit: float  # Maximum allowed drawdown
    stop_loss_override: bool  # Whether to apply hard stop losses

class IntelligentStrategyOrchestrator:
    """Advanced strategy coordination and optimization service."""
    
    def __init__(self):
        self.db_config = {
            'host': 'host.docker.internal',
            'user': 'news_collector',
            'password': '99Rules!',
            'database': 'crypto_transactions'
        }
        
        # Service endpoints
        self.ml_signals_url = os.getenv('ML_SIGNALS_URL', 'http://host.docker.internal:8025')
        self.selloff_detector_url = os.getenv('SELLOFF_DETECTOR_URL', 'http://host.docker.internal:8028')
        self.momentum_detector_url = os.getenv('MOMENTUM_DETECTOR_URL', 'http://host.docker.internal:8029')
        self.trading_engine_url = os.getenv('TRADING_ENGINE_URL', 'http://host.docker.internal:8024')
        self.portfolio_service_url = os.getenv('PORTFOLIO_SERVICE_URL', 'http://host.docker.internal:8026')
        
        # Strategy configuration
        self.strategy_weights = self._initialize_strategy_weights()
        self.risk_budgets = self._initialize_risk_budgets()
        
        # Performance tracking
        self.strategy_performance = defaultdict(deque)  # Rolling performance metrics
        self.conflict_history = deque(maxlen=1000)
        self.signal_history = deque(maxlen=5000)
        
        # Market regime detection
        self.current_regime = MarketRegime.SIDEWAYS
        self.regime_confidence = 0.5
        self.regime_history = deque(maxlen=100)
        
        # Configuration parameters
        self.rebalance_frequency = int(os.getenv('REBALANCE_FREQUENCY_MINUTES', '15'))
        self.conflict_resolution_method = os.getenv('CONFLICT_RESOLUTION', 'weighted_average')
        self.max_total_risk = float(os.getenv('MAX_TOTAL_RISK', '0.15'))  # 15% max portfolio risk
        self.emergency_liquidation_threshold = float(os.getenv('EMERGENCY_THRESHOLD', '0.10'))  # 10% drawdown
        
        # Active signals and positions
        self.active_signals = {}
        self.strategy_positions = defaultdict(dict)
        
        self.app = FastAPI(title="Strategy Orchestrator", version="1.0.0")
        self.setup_routes()
    
    def _initialize_strategy_weights(self) -> Dict[StrategyType, StrategyWeight]:
        """Initialize base strategy weights."""
        weights = {
            StrategyType.ML_SIGNALS: StrategyWeight(
                strategy=StrategyType.ML_SIGNALS,
                base_weight=0.60,  # 60% allocation to ML signals
                current_weight=0.60,
                performance_multiplier=1.0,
                regime_multiplier=1.0,
                risk_adjusted_weight=0.60,
                confidence=0.8,
                last_updated=datetime.now()
            ),
            StrategyType.MARKET_SELLOFF: StrategyWeight(
                strategy=StrategyType.MARKET_SELLOFF,
                base_weight=0.25,  # 25% allocation to selloff protection
                current_weight=0.25,
                performance_multiplier=1.0,
                regime_multiplier=1.0,
                risk_adjusted_weight=0.25,
                confidence=0.9,
                last_updated=datetime.now()
            ),
            StrategyType.MOMENTUM_HYPE: StrategyWeight(
                strategy=StrategyType.MOMENTUM_HYPE,
                base_weight=0.15,  # 15% allocation to momentum trading
                current_weight=0.15,
                performance_multiplier=1.0,
                regime_multiplier=1.0,
                risk_adjusted_weight=0.15,
                confidence=0.6,
                last_updated=datetime.now()
            )
        }
        return weights
    
    def _initialize_risk_budgets(self) -> Dict[StrategyType, RiskBudget]:
        """Initialize risk budget allocation."""
        budgets = {
            StrategyType.ML_SIGNALS: RiskBudget(
                strategy=StrategyType.ML_SIGNALS,
                allocated_risk=0.60,  # 60% of total risk budget
                current_risk_usage=0.0,
                max_position_size=0.08,  # 8% max position size
                max_drawdown_limit=0.12,  # 12% max drawdown
                stop_loss_override=False
            ),
            StrategyType.MARKET_SELLOFF: RiskBudget(
                strategy=StrategyType.MARKET_SELLOFF,
                allocated_risk=0.25,  # 25% of total risk budget
                current_risk_usage=0.0,
                max_position_size=1.0,  # Can liquidate everything
                max_drawdown_limit=0.05,  # 5% max drawdown (defensive)
                stop_loss_override=True  # Hard stop losses for protection
            ),
            StrategyType.MOMENTUM_HYPE: RiskBudget(
                strategy=StrategyType.MOMENTUM_HYPE,
                allocated_risk=0.15,  # 15% of total risk budget
                current_risk_usage=0.0,
                max_position_size=0.05,  # 5% max position size (high risk)
                max_drawdown_limit=0.08,  # 8% max drawdown
                stop_loss_override=True  # Hard stop losses for momentum
            )
        }
        return budgets
    
    def setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "service": "strategy-orchestrator",
                "current_regime": self.current_regime.value,
                "regime_confidence": self.regime_confidence,
                "active_strategies": len(self.strategy_weights),
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.get("/strategies/weights")
        async def get_strategy_weights():
            """Get current strategy weights and allocations."""
            try:
                weights_data = {}
                for strategy_type, weight in self.strategy_weights.items():
                    weight_dict = asdict(weight)
                    weight_dict['strategy'] = strategy_type.value  # Convert enum to string
                    weight_dict['last_updated'] = weight.last_updated.isoformat()
                    weights_data[strategy_type.value] = weight_dict
                
                response_data = {
                    "strategy_weights": weights_data,
                    "current_regime": self.current_regime.value,
                    "total_allocated": sum(w.risk_adjusted_weight for w in self.strategy_weights.values()),
                    "timestamp": datetime.now().isoformat()
                }
                
                return JSONResponse(content=json.loads(json.dumps(response_data, cls=CustomJSONEncoder)))
            except Exception as e:
                logger.error(f"Error getting strategy weights: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/strategies/signals")
        async def get_orchestrated_signals():
            """Get current orchestrated trading signals."""
            try:
                signals = await self.get_orchestrated_signals()
                signals_data = []
                for signal in signals:
                    signal_dict = asdict(signal)
                    signal_dict['timestamp'] = signal.timestamp.isoformat()
                    signal_dict['strategy'] = signal.strategy.value
                    signals_data.append(signal_dict)
                
                return JSONResponse(content=json.loads(json.dumps(signals_data, cls=CustomJSONEncoder)))
            except Exception as e:
                logger.error(f"Error getting orchestrated signals: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/strategies/conflicts")
        async def get_recent_conflicts():
            """Get recent strategy conflicts and resolutions."""
            try:
                conflicts = list(self.conflict_history)[-50:]  # Last 50 conflicts
                conflicts_data = []
                for conflict in conflicts:
                    conflict_dict = asdict(conflict)
                    conflict_dict['timestamp'] = conflict.timestamp.isoformat()
                    conflict_dict['conflicting_strategies'] = [s.value for s in conflict.conflicting_strategies]
                    conflicts_data.append(conflict_dict)
                
                return JSONResponse(content=json.loads(json.dumps(conflicts_data, cls=CustomJSONEncoder)))
            except Exception as e:
                logger.error(f"Error getting conflicts: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/strategies/rebalance")
        async def trigger_rebalance():
            """Trigger strategy rebalancing."""
            try:
                await self.rebalance_strategies()
                return {"status": "rebalanced", "timestamp": datetime.now().isoformat()}
            except Exception as e:
                logger.error(f"Error during rebalance: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/market/regime")
        async def get_market_regime():
            """Get current market regime analysis."""
            try:
                regime_data = await self.detect_market_regime()
                return JSONResponse(content=regime_data)
            except Exception as e:
                logger.error(f"Error getting market regime: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    async def collect_strategy_signals(self) -> Dict[StrategyType, List[StrategySignal]]:
        """Collect signals from all strategy services."""
        strategy_signals = defaultdict(list)
        
        try:
            # 1. Collect ML/XGBoost signals
            try:
                response = requests.get(f"{self.ml_signals_url}/signals/current", timeout=10)
                if response.status_code == 200:
                    ml_data = response.json()
                    for signal_data in ml_data.get('signals', []):
                        signal = StrategySignal(
                            timestamp=datetime.now(),
                            strategy=StrategyType.ML_SIGNALS,
                            symbol=signal_data.get('symbol', ''),
                            action=signal_data.get('action', 'HOLD'),
                            confidence=signal_data.get('confidence', 0.5),
                            strength=signal_data.get('strength', 0.5),
                            position_size=signal_data.get('position_size', 0.05),
                            urgency="medium",
                            reason="ML model prediction",
                            risk_score=signal_data.get('risk_score', 0.5),
                            expected_return=signal_data.get('expected_return', 0.0),
                            holding_period="days",
                            metadata=signal_data
                        )
                        strategy_signals[StrategyType.ML_SIGNALS].append(signal)
            except Exception as e:
                logger.warning(f"Failed to collect ML signals: {e}")
            
            # 2. Collect Market Selloff signals
            try:
                response = requests.get(f"{self.selloff_detector_url}/selloff/current", timeout=10)
                if response.status_code == 200:
                    selloff_data = response.json()
                    if selloff_data.get('selloff_detected', False):
                        signal = StrategySignal(
                            timestamp=datetime.now(),
                            strategy=StrategyType.MARKET_SELLOFF,
                            symbol="USD",  # Cash allocation
                            action="LIQUIDATE" if selloff_data.get('severity', 0) > 0.7 else "REDUCE",
                            confidence=selloff_data.get('confidence', 0.8),
                            strength=selloff_data.get('severity', 0.5),
                            position_size=selloff_data.get('cash_allocation_percentage', 0.3),
                            urgency="high" if selloff_data.get('severity', 0) > 0.7 else "medium",
                            reason=f"Market selloff detected: {selloff_data.get('selloff_type', 'general')}",
                            risk_score=0.9,
                            expected_return=0.0,  # Capital preservation
                            holding_period="hours",
                            metadata=selloff_data
                        )
                        strategy_signals[StrategyType.MARKET_SELLOFF].append(signal)
            except Exception as e:
                logger.warning(f"Failed to collect selloff signals: {e}")
            
            # 3. Collect Momentum/Hype signals
            try:
                response = requests.get(f"{self.momentum_detector_url}/momentum/current", timeout=10)
                if response.status_code == 200:
                    momentum_data = response.json()
                    for momentum_signal in momentum_data:
                        signal = StrategySignal(
                            timestamp=datetime.now(),
                            strategy=StrategyType.MOMENTUM_HYPE,
                            symbol=momentum_signal.get('symbol', ''),
                            action="BUY",
                            confidence=momentum_signal.get('entry_confidence', 0.6),
                            strength=momentum_signal.get('momentum_strength', 0.5),
                            position_size=min(0.05, momentum_signal.get('momentum_strength', 0.05)),
                            urgency="high",
                            reason=f"Momentum detected: {momentum_signal.get('momentum_type', 'general')}",
                            risk_score=momentum_signal.get('risk_multiplier', 1.0),
                            expected_return=momentum_signal.get('profit_target_percentage', 0.15),
                            holding_period="minutes" if momentum_signal.get('expected_duration') == "minutes" else "hours",
                            metadata=momentum_signal
                        )
                        strategy_signals[StrategyType.MOMENTUM_HYPE].append(signal)
            except Exception as e:
                logger.warning(f"Failed to collect momentum signals: {e}")
            
            return strategy_signals
            
        except Exception as e:
            logger.error(f"Error collecting strategy signals: {e}")
            return strategy_signals
    
    async def detect_conflicts(self, strategy_signals: Dict[StrategyType, List[StrategySignal]]) -> List[StrategyConflict]:
        """Detect conflicts between strategy signals."""
        conflicts = []
        
        # Group signals by symbol
        symbol_signals = defaultdict(list)
        for strategy_type, signals in strategy_signals.items():
            for signal in signals:
                symbol_signals[signal.symbol].append(signal)
        
        # Check for conflicts on each symbol
        for symbol, signals in symbol_signals.items():
            if len(signals) <= 1:
                continue
            
            # Check for action conflicts
            actions = [s.action for s in signals]
            if len(set(actions)) > 1:
                # We have conflicting actions
                conflict = StrategyConflict(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    conflicting_strategies=[s.strategy for s in signals],
                    conflict_type="action_conflict",
                    resolution_method=self.conflict_resolution_method,
                    final_action="HOLD",  # Will be resolved later
                    final_position_size=0.0,
                    confidence_penalty=0.1,
                    metadata={
                        "conflicting_actions": actions,
                        "signal_count": len(signals),
                        "strategies": [s.strategy.value for s in signals]
                    }
                )
                conflicts.append(conflict)
        
        return conflicts
    
    async def resolve_conflicts(self, conflicts: List[StrategyConflict], strategy_signals: Dict[StrategyType, List[StrategySignal]]) -> List[StrategySignal]:
        """Resolve strategy conflicts and generate final signals."""
        resolved_signals = []
        
        # Collect non-conflicting signals first
        for strategy_type, signals in strategy_signals.items():
            for signal in signals:
                # Check if this signal is involved in any conflict
                in_conflict = any(
                    signal.symbol == conflict.symbol and signal.strategy in conflict.conflicting_strategies
                    for conflict in conflicts
                )
                
                if not in_conflict:
                    resolved_signals.append(signal)
        
        # Resolve each conflict
        for conflict in conflicts:
            try:
                # Get all signals for this symbol
                symbol_signals = []
                for strategy_type, signals in strategy_signals.items():
                    for signal in signals:
                        if signal.symbol == conflict.symbol:
                            symbol_signals.append(signal)
                
                if not symbol_signals:
                    continue
                
                # Apply resolution method
                if self.conflict_resolution_method == "weighted_average":
                    resolved_signal = await self._resolve_weighted_average(symbol_signals, conflict)
                elif self.conflict_resolution_method == "highest_confidence":
                    resolved_signal = await self._resolve_highest_confidence(symbol_signals, conflict)
                elif self.conflict_resolution_method == "risk_priority":
                    resolved_signal = await self._resolve_risk_priority(symbol_signals, conflict)
                else:
                    # Default: highest confidence
                    resolved_signal = await self._resolve_highest_confidence(symbol_signals, conflict)
                
                if resolved_signal:
                    resolved_signals.append(resolved_signal)
                    
                    # Update conflict with resolution
                    conflict.final_action = resolved_signal.action
                    conflict.final_position_size = resolved_signal.position_size
                    
                    # Store conflict for analysis
                    self.conflict_history.append(conflict)
                    
                    logger.info(f"🔀 CONFLICT RESOLVED: {conflict.symbol} - {conflict.final_action} ({conflict.resolution_method})")
                
            except Exception as e:
                logger.error(f"Error resolving conflict for {conflict.symbol}: {e}")
        
        return resolved_signals
    
    async def _resolve_weighted_average(self, signals: List[StrategySignal], conflict: StrategyConflict) -> Optional[StrategySignal]:
        """Resolve conflict using weighted average of signals."""
        if not signals:
            return None
        
        # Calculate weighted averages
        total_weight = 0
        weighted_confidence = 0
        weighted_strength = 0
        weighted_position_size = 0
        weighted_risk = 0
        
        action_weights = defaultdict(float)
        
        for signal in signals:
            strategy_weight = self.strategy_weights[signal.strategy].risk_adjusted_weight
            
            total_weight += strategy_weight
            weighted_confidence += signal.confidence * strategy_weight
            weighted_strength += signal.strength * strategy_weight
            weighted_position_size += signal.position_size * strategy_weight
            weighted_risk += signal.risk_score * strategy_weight
            
            action_weights[signal.action] += strategy_weight
        
        if total_weight == 0:
            return None
        
        # Normalize
        final_confidence = weighted_confidence / total_weight
        final_strength = weighted_strength / total_weight
        final_position_size = weighted_position_size / total_weight
        final_risk = weighted_risk / total_weight
        
        # Choose action with highest weight
        final_action = max(action_weights, key=action_weights.get)
        
        # Create resolved signal
        resolved_signal = StrategySignal(
            timestamp=datetime.now(),
            strategy=StrategyType.ML_SIGNALS,  # Default strategy for orchestrated signals
            symbol=signals[0].symbol,
            action=final_action,
            confidence=final_confidence * 0.9,  # Penalty for conflict
            strength=final_strength,
            position_size=final_position_size,
            urgency="medium",
            reason=f"Orchestrated resolution: {len(signals)} strategies",
            risk_score=final_risk,
            expected_return=np.mean([s.expected_return for s in signals]),
            holding_period="hours",
            metadata={
                "resolution_method": "weighted_average",
                "input_strategies": [s.strategy.value for s in signals],
                "conflict_penalty": 0.1
            }
        )
        
        return resolved_signal
    
    async def _resolve_highest_confidence(self, signals: List[StrategySignal], conflict: StrategyConflict) -> Optional[StrategySignal]:
        """Resolve conflict by choosing highest confidence signal."""
        if not signals:
            return None
        
        # Find signal with highest confidence
        best_signal = max(signals, key=lambda s: s.confidence * self.strategy_weights[s.strategy].risk_adjusted_weight)
        
        # Apply conflict penalty
        best_signal.confidence *= 0.95
        best_signal.metadata["resolution_method"] = "highest_confidence"
        best_signal.metadata["conflict_penalty"] = 0.05
        
        return best_signal
    
    async def _resolve_risk_priority(self, signals: List[StrategySignal], conflict: StrategyConflict) -> Optional[StrategySignal]:
        """Resolve conflict by prioritizing defensive strategies."""
        if not signals:
            return None
        
        # Priority order: selloff protection > risk management > ML signals > momentum
        priority_order = [
            StrategyType.MARKET_SELLOFF,
            StrategyType.RISK_MANAGEMENT,
            StrategyType.ML_SIGNALS,
            StrategyType.MOMENTUM_HYPE
        ]
        
        for strategy_type in priority_order:
            for signal in signals:
                if signal.strategy == strategy_type:
                    signal.confidence *= 0.9  # Small penalty for conflict resolution
                    signal.metadata["resolution_method"] = "risk_priority"
                    signal.metadata["conflict_penalty"] = 0.1
                    return signal
        
        # Fallback to first signal
        return signals[0] if signals else None
    
    async def get_orchestrated_signals(self) -> List[StrategySignal]:
        """Get final orchestrated trading signals after conflict resolution."""
        try:
            # 1. Collect signals from all strategies
            strategy_signals = await self.collect_strategy_signals()
            
            # 2. Detect conflicts
            conflicts = await self.detect_conflicts(strategy_signals)
            
            # 3. Resolve conflicts
            resolved_signals = await self.resolve_conflicts(conflicts, strategy_signals)
            
            # 4. Apply risk budgets
            risk_adjusted_signals = await self.apply_risk_budgets(resolved_signals)
            
            # 5. Apply market regime adjustments
            final_signals = await self.apply_regime_adjustments(risk_adjusted_signals)
            
            # Store signals for analysis
            for signal in final_signals:
                self.signal_history.append(signal)
            
            logger.info(f"📊 ORCHESTRATED SIGNALS: {len(final_signals)} signals generated from {sum(len(s) for s in strategy_signals.values())} input signals")
            
            return final_signals
            
        except Exception as e:
            logger.error(f"Error orchestrating signals: {e}")
            return []
    
    async def apply_risk_budgets(self, signals: List[StrategySignal]) -> List[StrategySignal]:
        """Apply risk budget constraints to signals."""
        adjusted_signals = []
        
        for signal in signals:
            try:
                risk_budget = self.risk_budgets.get(signal.strategy)
                if not risk_budget:
                    adjusted_signals.append(signal)
                    continue
                
                # Check if strategy is over risk budget
                if risk_budget.current_risk_usage >= risk_budget.allocated_risk:
                    logger.warning(f"Strategy {signal.strategy.value} over risk budget, reducing position size")
                    signal.position_size *= 0.5  # Reduce position size
                
                # Apply max position size constraint
                signal.position_size = min(signal.position_size, risk_budget.max_position_size)
                
                # Apply stop loss override if needed
                if risk_budget.stop_loss_override and signal.action in ["BUY"]:
                    signal.metadata["stop_loss_override"] = True
                    signal.metadata["max_drawdown"] = risk_budget.max_drawdown_limit
                
                adjusted_signals.append(signal)
                
            except Exception as e:
                logger.error(f"Error applying risk budget to signal {signal.symbol}: {e}")
                adjusted_signals.append(signal)  # Include original signal if error
        
        return adjusted_signals
    
    async def apply_regime_adjustments(self, signals: List[StrategySignal]) -> List[StrategySignal]:
        """Apply market regime-based adjustments to signals."""
        adjusted_signals = []
        
        for signal in signals:
            try:
                # Apply regime-based position size adjustments
                if self.current_regime == MarketRegime.BEAR_MARKET:
                    signal.position_size *= 0.7  # Reduce position sizes in bear market
                elif self.current_regime == MarketRegime.HIGH_VOLATILITY:
                    signal.position_size *= 0.8  # Reduce position sizes in high volatility
                elif self.current_regime == MarketRegime.SELLOFF:
                    if signal.strategy != StrategyType.MARKET_SELLOFF:
                        signal.position_size *= 0.3  # Heavily reduce non-selloff signals
                elif self.current_regime == MarketRegime.MOMENTUM_PHASE:
                    if signal.strategy == StrategyType.MOMENTUM_HYPE:
                        signal.position_size *= 1.3  # Increase momentum signals in momentum phase
                
                # Apply regime confidence adjustment
                signal.confidence *= self.regime_confidence
                
                signal.metadata["market_regime"] = self.current_regime.value
                signal.metadata["regime_confidence"] = self.regime_confidence
                
                adjusted_signals.append(signal)
                
            except Exception as e:
                logger.error(f"Error applying regime adjustment to signal {signal.symbol}: {e}")
                adjusted_signals.append(signal)
        
        return adjusted_signals
    
    async def detect_market_regime(self) -> Dict:
        """Detect current market regime for strategy adjustment."""
        try:
            # Get market data for regime detection
            # This would normally analyze various market indicators
            # For now, we'll use a simplified approach
            
            regime_data = {
                "current_regime": self.current_regime.value,
                "confidence": self.regime_confidence,
                "indicators": {
                    "volatility": "medium",
                    "trend": "sideways",
                    "volume": "normal",
                    "sentiment": "neutral"
                },
                "regime_changes": list(self.regime_history)[-10:],
                "timestamp": datetime.now().isoformat()
            }
            
            return regime_data
            
        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            return {"error": str(e)}
    
    async def rebalance_strategies(self):
        """Rebalance strategy weights based on performance."""
        try:
            # This would analyze recent performance and adjust strategy weights
            # For now, we'll implement a basic performance-based adjustment
            
            for strategy_type, weight in self.strategy_weights.items():
                # Get recent performance (simplified)
                # In reality, this would calculate actual returns
                recent_performance = 1.0  # Placeholder
                
                # Adjust performance multiplier
                weight.performance_multiplier = min(1.5, max(0.5, recent_performance))
                
                # Recalculate risk-adjusted weight
                weight.risk_adjusted_weight = (
                    weight.base_weight * 
                    weight.performance_multiplier * 
                    weight.regime_multiplier
                )
                
                weight.last_updated = datetime.now()
            
            # Normalize weights to sum to 1.0
            total_weight = sum(w.risk_adjusted_weight for w in self.strategy_weights.values())
            if total_weight > 0:
                for weight in self.strategy_weights.values():
                    weight.risk_adjusted_weight /= total_weight
            
            logger.info("🎯 Strategy weights rebalanced based on performance")
            
        except Exception as e:
            logger.error(f"Error rebalancing strategies: {e}")

async def main():
    """Main function to run the orchestration service."""
    orchestrator = IntelligentStrategyOrchestrator()
    
    # Start background tasks
    async def background_tasks():
        while True:
            try:
                # Rebalance strategies periodically
                await orchestrator.rebalance_strategies()
                await asyncio.sleep(orchestrator.rebalance_frequency * 60)  # Convert to seconds
            except Exception as e:
                logger.error(f"Error in background tasks: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry
    
    # Start background task
    asyncio.create_task(background_tasks())
    
    # Start the FastAPI server
    config = uvicorn.Config(
        orchestrator.app,
        host="0.0.0.0",
        port=8030,
        log_level="info"
    )
    server = uvicorn.Server(config)
    
    logger.info("🎭 Intelligent Strategy Orchestrator starting on port 8030...")
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())
