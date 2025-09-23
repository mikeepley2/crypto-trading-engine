#!/usr/bin/env python3
"""
Exit Strategy Engine for Coinbase Listing Trades
Implements intelligent hold-until-plateau exit strategy

Features:
- Multi-indicator plateau detection
- Dynamic profit targets based on Coinbase Effect research
- Volatility-based exit timing
- Declining momentum detection
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import mysql.connector
from mysql.connector import Error
import logging
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class ExitSignal:
    """Represents an exit signal for a listing trade"""
    symbol: str
    signal_type: str  # PLATEAU, PROFIT_TARGET, DECLINING, STOP_LOSS
    confidence: float
    recommended_action: str  # FULL_EXIT, PARTIAL_EXIT, HOLD
    trigger_price: float
    reasoning: str

class PlateauDetector:
    """Advanced plateau detection for Coinbase listing trades"""
    
    def __init__(self):
        # Plateau detection parameters
        self.volatility_threshold = 0.02  # 2% volatility threshold
        self.trend_threshold = 0.005      # 0.5% trend threshold
        self.min_plateau_duration = 10    # Minutes
        self.price_window = 20            # Price samples for analysis
        
        # Historical price tracking
        self.price_histories: Dict[str, deque] = {}
        self.timestamp_histories: Dict[str, deque] = {}
    
    def update_price_data(self, symbol: str, price: float, timestamp: datetime):
        """Update price and timestamp data for symbol"""
        if symbol not in self.price_histories:
            self.price_histories[symbol] = deque(maxlen=self.price_window)
            self.timestamp_histories[symbol] = deque(maxlen=self.price_window)
        
        self.price_histories[symbol].append(price)
        self.timestamp_histories[symbol].append(timestamp)
    
    def calculate_volatility(self, symbol: str, window: int = 10) -> float:
        """Calculate recent price volatility"""
        if symbol not in self.price_histories or len(self.price_histories[symbol]) < window:
            return 1.0  # High volatility if insufficient data
        
        prices = list(self.price_histories[symbol])[-window:]
        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        
        return np.std(returns) if returns else 1.0
    
    def calculate_trend_strength(self, symbol: str, window: int = 10) -> float:
        """Calculate trend strength (positive = uptrend, negative = downtrend)"""
        if symbol not in self.price_histories or len(self.price_histories[symbol]) < window:
            return 0.0
        
        prices = list(self.price_histories[symbol])[-window:]
        
        # Simple linear regression to find trend
        x = np.array(range(len(prices)))
        y = np.array(prices)
        
        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
            return slope / prices[-1]  # Normalize by current price
        
        return 0.0
    
    def detect_price_plateau(self, symbol: str) -> Tuple[bool, float, str]:
        """
        Detect if price has plateaued
        Returns: (is_plateau, confidence, reasoning)
        """
        if symbol not in self.price_histories or len(self.price_histories[symbol]) < 10:
            return False, 0.0, "Insufficient price data"
        
        # Check volatility
        volatility = self.calculate_volatility(symbol)
        is_low_volatility = volatility < self.volatility_threshold
        
        # Check trend strength
        trend = abs(self.calculate_trend_strength(symbol))
        is_weak_trend = trend < self.trend_threshold
        
        # Check duration of current conditions
        duration_minutes = 0
        if len(self.timestamp_histories[symbol]) >= 2:
            duration = self.timestamp_histories[symbol][-1] - self.timestamp_histories[symbol][0]
            duration_minutes = duration.total_seconds() / 60
        
        is_sufficient_duration = duration_minutes >= self.min_plateau_duration
        
        # Calculate confidence
        confidence = 0.0
        reasoning_parts = []
        
        if is_low_volatility:
            confidence += 0.4
            reasoning_parts.append(f"Low volatility ({volatility:.3f})")
        
        if is_weak_trend:
            confidence += 0.3
            reasoning_parts.append(f"Weak trend ({trend:.4f})")
        
        if is_sufficient_duration:
            confidence += 0.3
            reasoning_parts.append(f"Duration ({duration_minutes:.1f}min)")
        
        is_plateau = confidence >= 0.7
        reasoning = " + ".join(reasoning_parts) if reasoning_parts else "Insufficient conditions"
        
        return is_plateau, confidence, reasoning

class MomentumAnalyzer:
    """Analyzes price momentum for exit timing"""
    
    def __init__(self):
        self.momentum_window = 15
        self.acceleration_window = 5
    
    def calculate_momentum(self, prices: List[float]) -> float:
        """Calculate price momentum (rate of change)"""
        if len(prices) < 2:
            return 0.0
        
        # Simple momentum: (current - previous) / previous
        return (prices[-1] - prices[0]) / prices[0]
    
    def calculate_acceleration(self, prices: List[float]) -> float:
        """Calculate price acceleration (change in momentum)"""
        if len(prices) < self.acceleration_window:
            return 0.0
        
        # Calculate momentum for recent and earlier periods
        recent_momentum = self.calculate_momentum(prices[-self.acceleration_window:])
        earlier_momentum = self.calculate_momentum(prices[-self.acceleration_window*2:-self.acceleration_window])
        
        return recent_momentum - earlier_momentum
    
    def detect_declining_momentum(self, symbol: str, price_history: deque) -> Tuple[bool, float, str]:
        """
        Detect if momentum is declining (good time to exit)
        Returns: (is_declining, confidence, reasoning)
        """
        if len(price_history) < self.momentum_window:
            return False, 0.0, "Insufficient data for momentum analysis"
        
        prices = list(price_history)
        
        # Calculate recent momentum and acceleration
        momentum = self.calculate_momentum(prices[-self.momentum_window:])
        acceleration = self.calculate_acceleration(prices)
        
        # Check for declining momentum patterns
        is_negative_acceleration = acceleration < -0.01  # 1% deceleration
        is_weakening_momentum = momentum > 0 and momentum < 0.05  # Positive but weak
        
        confidence = 0.0
        reasoning_parts = []
        
        if is_negative_acceleration:
            confidence += 0.6
            reasoning_parts.append(f"Negative acceleration ({acceleration:.3f})")
        
        if is_weakening_momentum:
            confidence += 0.4
            reasoning_parts.append(f"Weakening momentum ({momentum:.3f})")
        
        is_declining = confidence >= 0.6
        reasoning = " + ".join(reasoning_parts) if reasoning_parts else "Momentum stable"
        
        return is_declining, confidence, reasoning

class ExitStrategyEngine:
    """Main exit strategy engine for listing trades"""
    
    def __init__(self):
        self.plateau_detector = PlateauDetector()
        self.momentum_analyzer = MomentumAnalyzer()
        
        # Exit thresholds based on Coinbase Effect research
        self.profit_targets = {
            "conservative": 0.30,   # 30% - early exit for risk-averse
            "moderate": 0.50,       # 50% - balanced approach
            "aggressive": 0.91,     # 91% - full Coinbase Effect
            "maximum": 1.50         # 150% - outlier capture
        }
        
        # Risk management
        self.stop_loss = -0.15      # 15% stop loss
        self.trailing_stop = 0.10   # 10% trailing stop after 50% gain
        self.max_hold_hours = 72    # 3 days maximum hold
        
        # Partial exit strategy
        self.partial_exit_levels = [
            (0.30, 0.25),  # Exit 25% at 30% gain
            (0.60, 0.50),  # Exit 50% at 60% gain
            (1.00, 0.75),  # Exit 75% at 100% gain
        ]
    
    def analyze_exit_opportunity(self, symbol: str, entry_price: float, 
                               current_price: float, entry_time: datetime,
                               price_history: deque) -> Optional[ExitSignal]:
        """
        Comprehensive exit analysis for a listing trade
        Returns exit signal if position should be closed
        """
        
        # Update plateau detector with current price
        self.plateau_detector.update_price_data(symbol, current_price, datetime.now())
        
        # Calculate current performance
        pnl_pct = (current_price - entry_price) / entry_price
        hold_duration = datetime.now() - entry_time
        hold_hours = hold_duration.total_seconds() / 3600
        
        # Check stop loss
        if pnl_pct <= self.stop_loss:
            return ExitSignal(
                symbol=symbol,
                signal_type="STOP_LOSS",
                confidence=1.0,
                recommended_action="FULL_EXIT",
                trigger_price=current_price,
                reasoning=f"Stop loss triggered at {pnl_pct:.1%}"
            )
        
        # Check maximum hold time
        if hold_hours >= self.max_hold_hours:
            return ExitSignal(
                symbol=symbol,
                signal_type="MAX_HOLD",
                confidence=0.9,
                recommended_action="FULL_EXIT",
                trigger_price=current_price,
                reasoning=f"Maximum hold time reached ({hold_hours:.1f}h)"
            )
        
        # Plateau detection
        is_plateau, plateau_confidence, plateau_reasoning = self.plateau_detector.detect_price_plateau(symbol)
        
        # Momentum analysis
        is_declining, momentum_confidence, momentum_reasoning = self.momentum_analyzer.detect_declining_momentum(symbol, price_history)
        
        # Profit target analysis
        exit_signal = self._analyze_profit_targets(pnl_pct, is_plateau, plateau_confidence, 
                                                 is_declining, momentum_confidence, symbol, current_price)
        
        if exit_signal:
            return exit_signal
        
        # Check for trailing stop
        if pnl_pct >= 0.50:  # Only activate trailing stop after 50% gain
            trailing_signal = self._check_trailing_stop(symbol, current_price, price_history)
            if trailing_signal:
                return trailing_signal
        
        # No exit signal generated
        return None
    
    def _analyze_profit_targets(self, pnl_pct: float, is_plateau: bool, plateau_confidence: float,
                              is_declining: bool, momentum_confidence: float, 
                              symbol: str, current_price: float) -> Optional[ExitSignal]:
        """Analyze profit targets with plateau and momentum conditions"""
        
        # Aggressive target (91% - full Coinbase Effect)
        if pnl_pct >= self.profit_targets["aggressive"]:
            if is_plateau or is_declining:
                return ExitSignal(
                    symbol=symbol,
                    signal_type="PROFIT_TARGET",
                    confidence=max(plateau_confidence, momentum_confidence),
                    recommended_action="FULL_EXIT",
                    trigger_price=current_price,
                    reasoning=f"Aggressive target reached ({pnl_pct:.1%}) with exit conditions"
                )
        
        # Moderate target (50%)
        elif pnl_pct >= self.profit_targets["moderate"]:
            # Require stronger signal for earlier exit
            if is_plateau and plateau_confidence >= 0.8:
                return ExitSignal(
                    symbol=symbol,
                    signal_type="PROFIT_TARGET",
                    confidence=plateau_confidence,
                    recommended_action="PARTIAL_EXIT",
                    trigger_price=current_price,
                    reasoning=f"Moderate target ({pnl_pct:.1%}) with strong plateau signal"
                )
            
            if is_declining and momentum_confidence >= 0.8:
                return ExitSignal(
                    symbol=symbol,
                    signal_type="DECLINING",
                    confidence=momentum_confidence,
                    recommended_action="PARTIAL_EXIT",
                    trigger_price=current_price,
                    reasoning=f"Declining momentum at {pnl_pct:.1%} gain"
                )
        
        # Conservative target (30%)
        elif pnl_pct >= self.profit_targets["conservative"]:
            # Only exit if very strong signals
            if is_plateau and is_declining and min(plateau_confidence, momentum_confidence) >= 0.9:
                return ExitSignal(
                    symbol=symbol,
                    signal_type="PLATEAU",
                    confidence=min(plateau_confidence, momentum_confidence),
                    recommended_action="PARTIAL_EXIT",
                    trigger_price=current_price,
                    reasoning=f"Strong plateau + declining momentum at {pnl_pct:.1%}"
                )
        
        return None
    
    def _check_trailing_stop(self, symbol: str, current_price: float, 
                           price_history: deque) -> Optional[ExitSignal]:
        """Check for trailing stop loss after significant gains"""
        if len(price_history) < 10:
            return None
        
        prices = list(price_history)
        recent_high = max(prices[-10:])  # High in last 10 periods
        
        # Calculate drawdown from recent high
        drawdown = (current_price - recent_high) / recent_high
        
        if drawdown <= -self.trailing_stop:
            return ExitSignal(
                symbol=symbol,
                signal_type="TRAILING_STOP",
                confidence=0.9,
                recommended_action="FULL_EXIT",
                trigger_price=current_price,
                reasoning=f"Trailing stop triggered: {drawdown:.1%} from recent high"
            )
        
        return None
    
    def get_partial_exit_recommendation(self, pnl_pct: float) -> Optional[Tuple[float, str]]:
        """Get partial exit recommendation based on gain level"""
        for gain_threshold, exit_percentage in self.partial_exit_levels:
            if pnl_pct >= gain_threshold:
                reason = f"Partial exit ({exit_percentage*100}%) at {pnl_pct:.1%} gain"
                return exit_percentage, reason
        
        return None

def main():
    """Test the exit strategy engine"""
    engine = ExitStrategyEngine()
    
    # Simulate price data
    prices = deque([100, 102, 105, 110, 115, 120, 125, 130, 135, 140], maxlen=20)
    
    # Test exit analysis
    signal = engine.analyze_exit_opportunity(
        symbol="TEST",
        entry_price=100,
        current_price=140,
        entry_time=datetime.now() - timedelta(hours=2),
        price_history=prices
    )
    
    if signal:
        print(f"Exit Signal: {signal.signal_type}")
        print(f"Confidence: {signal.confidence:.2f}")
        print(f"Action: {signal.recommended_action}")
        print(f"Reasoning: {signal.reasoning}")
    else:
        print("No exit signal generated")

if __name__ == "__main__":
    main()
