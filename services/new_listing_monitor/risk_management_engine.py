#!/usr/bin/env python3
"""
Risk Management Engine for Coinbase Listing Trades
Implements specialized risk controls for new listing opportunities

Features:
- Position sizing based on confidence and volatility
- Portfolio allocation limits for listing trades
- Dynamic stop losses based on listing characteristics
- Risk-adjusted exposure management
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import mysql.connector
from mysql.connector import Error
import logging
import json

logger = logging.getLogger(__name__)

@dataclass
class RiskMetrics:
    """Risk metrics for a listing trade opportunity"""
    symbol: str
    confidence_score: float
    volatility_estimate: float
    liquidity_score: float
    market_cap_tier: str  # MICRO, SMALL, MID, LARGE
    risk_score: float     # 0-1, higher = riskier
    max_position_size: float
    recommended_stop_loss: float

@dataclass
class PortfolioRisk:
    """Portfolio-level risk metrics"""
    total_value: float
    listing_allocation_pct: float
    active_listing_count: int
    max_single_position_pct: float
    diversification_score: float
    overall_risk_score: float

class RiskProfiler:
    """Profiles risk characteristics of new listing opportunities"""
    
    def __init__(self):
        # Risk scoring parameters
        self.volatility_weights = {
            "price_history": 0.3,
            "market_cap": 0.2,
            "listing_type": 0.3,
            "social_sentiment": 0.2
        }
        
        # Market cap tiers (estimated USD values)
        self.market_cap_tiers = {
            "MICRO": (0, 50_000_000),      # Under $50M
            "SMALL": (50_000_000, 300_000_000),    # $50M - $300M
            "MID": (300_000_000, 2_000_000_000),   # $300M - $2B
            "LARGE": (2_000_000_000, float('inf'))  # Over $2B
        }
        
        # Risk multipliers by listing type
        self.listing_type_risk = {
            "announced": 1.0,       # Official announcement - baseline risk
            "roadmap_added": 1.3,   # Roadmap addition - higher uncertainty
            "base_added": 1.5,      # Base network - highest uncertainty
            "rumor": 2.0           # Rumor - maximum risk
        }
    
    def estimate_market_cap_tier(self, symbol: str, current_price: float = None) -> str:
        """Estimate market cap tier based on available data"""
        try:
            # For new listings, we often don't have reliable market cap data
            # Use heuristics based on listing characteristics
            
            # Check if it's a well-known project
            known_large_caps = ["BTC", "ETH", "ADA", "DOT", "LINK", "UNI"]
            if symbol.upper() in known_large_caps:
                return "LARGE"
            
            # Check if it's a mid-tier project (common DeFi/infrastructure)
            mid_tier_indicators = ["SWAP", "DEX", "BRIDGE", "DAO", "PROTOCOL"]
            if any(indicator in symbol.upper() for indicator in mid_tier_indicators):
                return "MID"
            
            # For new/unknown tokens, assume smaller cap
            return "SMALL"
            
        except Exception as e:
            logger.warning(f"Could not determine market cap tier for {symbol}: {e}")
            return "SMALL"  # Conservative default
    
    def calculate_volatility_estimate(self, symbol: str, listing_data: Dict) -> float:
        """Estimate expected volatility for new listing"""
        base_volatility = 0.15  # 15% base volatility for crypto
        
        # Adjust based on listing type
        listing_type = listing_data.get("listing_type", "announced")
        type_multiplier = self.listing_type_risk.get(listing_type, 1.5)
        
        # Adjust based on confidence
        confidence = listing_data.get("confidence", 0.5)
        confidence_factor = 2.0 - confidence  # Lower confidence = higher volatility
        
        # Adjust based on social sentiment
        sentiment = listing_data.get("sentiment_score", 0.5)
        sentiment_factor = 1.0 + (0.5 - sentiment)  # Negative sentiment increases volatility
        
        estimated_volatility = base_volatility * type_multiplier * confidence_factor * sentiment_factor
        
        # Cap at reasonable bounds
        return min(max(estimated_volatility, 0.05), 0.50)  # 5% to 50%
    
    def calculate_liquidity_score(self, symbol: str, listing_data: Dict) -> float:
        """Estimate liquidity score (0-1, higher = more liquid)"""
        base_score = 0.5
        
        # Higher confidence typically means better initial liquidity
        confidence = listing_data.get("confidence", 0.5)
        confidence_boost = confidence * 0.3
        
        # Official announcements usually have better initial liquidity
        listing_type = listing_data.get("listing_type", "announced")
        type_scores = {
            "announced": 0.8,
            "roadmap_added": 0.6,
            "base_added": 0.4,
            "rumor": 0.2
        }
        type_score = type_scores.get(listing_type, 0.5)
        
        # Social buzz can indicate initial trading interest
        social_buzz = listing_data.get("social_buzz_score", 0.5)
        buzz_boost = social_buzz * 0.2
        
        liquidity_score = base_score + confidence_boost + (type_score - 0.5) + buzz_boost
        
        return min(max(liquidity_score, 0.1), 1.0)
    
    def calculate_risk_score(self, volatility: float, liquidity: float, 
                           confidence: float, market_cap_tier: str) -> float:
        """Calculate overall risk score (0-1, higher = riskier)"""
        
        # Volatility component (40% weight)
        vol_risk = min(volatility / 0.30, 1.0) * 0.4
        
        # Liquidity component (25% weight) - inverse relationship
        liquidity_risk = (1.0 - liquidity) * 0.25
        
        # Confidence component (20% weight) - inverse relationship
        confidence_risk = (1.0 - confidence) * 0.20
        
        # Market cap component (15% weight)
        cap_risk_scores = {
            "LARGE": 0.1,
            "MID": 0.3,
            "SMALL": 0.6,
            "MICRO": 0.9
        }
        cap_risk = cap_risk_scores.get(market_cap_tier, 0.6) * 0.15
        
        total_risk = vol_risk + liquidity_risk + confidence_risk + cap_risk
        
        return min(total_risk, 1.0)

class PositionSizer:
    """Calculates optimal position sizes for listing trades"""
    
    def __init__(self, base_portfolio_value: float = 2571.86):
        self.portfolio_value = base_portfolio_value
        
        # Base position sizing parameters
        self.base_position_pct = 0.05      # 5% base position
        self.max_position_pct = 0.10       # 10% max single position
        self.max_listing_allocation = 0.20  # 20% max total listing allocation
        
        # Risk-adjusted multipliers
        self.confidence_multipliers = {
            (0.9, 1.0): 1.0,    # Very high confidence
            (0.8, 0.9): 0.8,    # High confidence  
            (0.7, 0.8): 0.6,    # Medium confidence
            (0.5, 0.7): 0.4,    # Low confidence
            (0.0, 0.5): 0.2     # Very low confidence
        }
        
        self.risk_multipliers = {
            (0.0, 0.3): 1.0,    # Low risk
            (0.3, 0.5): 0.8,    # Medium-low risk
            (0.5, 0.7): 0.6,    # Medium-high risk
            (0.7, 1.0): 0.4     # High risk
        }
    
    def get_multiplier(self, value: float, multiplier_dict: Dict) -> float:
        """Get multiplier based on value ranges"""
        for (low, high), multiplier in multiplier_dict.items():
            if low <= value < high:
                return multiplier
        return 0.2  # Default conservative multiplier
    
    def calculate_position_size(self, risk_metrics: RiskMetrics, 
                              current_listing_allocation: float) -> float:
        """Calculate optimal position size in USD"""
        
        # Base position size
        base_size = self.portfolio_value * self.base_position_pct
        
        # Apply confidence multiplier
        confidence_mult = self.get_multiplier(risk_metrics.confidence_score, self.confidence_multipliers)
        
        # Apply risk multiplier
        risk_mult = self.get_multiplier(risk_metrics.risk_score, self.risk_multipliers)
        
        # Apply liquidity multiplier
        liquidity_mult = min(risk_metrics.liquidity_score * 1.5, 1.0)
        
        # Calculate adjusted position size
        adjusted_size = base_size * confidence_mult * risk_mult * liquidity_mult
        
        # Apply portfolio constraints
        max_size_by_portfolio = self.portfolio_value * self.max_position_pct
        remaining_listing_capacity = self.portfolio_value * self.max_listing_allocation - current_listing_allocation
        
        # Final position size
        final_size = min(adjusted_size, max_size_by_portfolio, remaining_listing_capacity)
        
        # Ensure minimum viable position
        if final_size < 50.0:
            return 0.0  # Too small to be viable
        
        return final_size

class StopLossCalculator:
    """Calculates dynamic stop losses for listing trades"""
    
    def __init__(self):
        # Base stop loss levels
        self.base_stop_loss = 0.15  # 15% base stop loss
        
        # Adjustments based on risk factors
        self.volatility_adjustments = {
            (0.0, 0.10): -0.02,   # Low vol: tighter stop
            (0.10, 0.20): 0.0,    # Normal vol: base stop
            (0.20, 0.30): 0.05,   # High vol: wider stop
            (0.30, 1.0): 0.10     # Very high vol: much wider stop
        }
        
        self.confidence_adjustments = {
            (0.9, 1.0): -0.03,    # Very high confidence: tighter stop
            (0.8, 0.9): -0.01,    # High confidence: slightly tighter
            (0.7, 0.8): 0.0,      # Medium confidence: base stop
            (0.5, 0.7): 0.02,     # Low confidence: wider stop
            (0.0, 0.5): 0.05      # Very low confidence: much wider stop
        }
    
    def get_adjustment(self, value: float, adjustment_dict: Dict) -> float:
        """Get adjustment factor based on value ranges"""
        for (low, high), adjustment in adjustment_dict.items():
            if low <= value < high:
                return adjustment
        return 0.0
    
    def calculate_stop_loss(self, risk_metrics: RiskMetrics) -> float:
        """Calculate dynamic stop loss percentage"""
        
        # Start with base stop loss
        stop_loss = self.base_stop_loss
        
        # Adjust for volatility
        vol_adjustment = self.get_adjustment(risk_metrics.volatility_estimate, self.volatility_adjustments)
        stop_loss += vol_adjustment
        
        # Adjust for confidence
        conf_adjustment = self.get_adjustment(risk_metrics.confidence_score, self.confidence_adjustments)
        stop_loss += conf_adjustment
        
        # Adjust for liquidity (lower liquidity = wider stop)
        liquidity_adjustment = (1.0 - risk_metrics.liquidity_score) * 0.05
        stop_loss += liquidity_adjustment
        
        # Ensure reasonable bounds
        return min(max(stop_loss, 0.08), 0.25)  # 8% to 25%

class RiskManager:
    """Main risk management engine for listing trades"""
    
    def __init__(self, portfolio_value: float = 2571.86):
        self.risk_profiler = RiskProfiler()
        self.position_sizer = PositionSizer(portfolio_value)
        self.stop_loss_calculator = StopLossCalculator()
        
        self.db_config = {
            'host': '192.168.230.163',
            'user': 'news_collector',
            'password': '99Rules!',
            'database': 'crypto_transactions'
        }
    
    def analyze_listing_risk(self, symbol: str, listing_data: Dict) -> RiskMetrics:
        """Comprehensive risk analysis for listing opportunity"""
        
        # Extract data
        confidence = listing_data.get("confidence", 0.5)
        listing_type = listing_data.get("listing_type", "announced")
        
        # Calculate risk components
        market_cap_tier = self.risk_profiler.estimate_market_cap_tier(symbol)
        volatility = self.risk_profiler.calculate_volatility_estimate(symbol, listing_data)
        liquidity = self.risk_profiler.calculate_liquidity_score(symbol, listing_data)
        risk_score = self.risk_profiler.calculate_risk_score(volatility, liquidity, confidence, market_cap_tier)
        
        # Get current listing allocation
        current_allocation = self.get_current_listing_allocation()
        
        # Calculate position size
        risk_metrics = RiskMetrics(
            symbol=symbol,
            confidence_score=confidence,
            volatility_estimate=volatility,
            liquidity_score=liquidity,
            market_cap_tier=market_cap_tier,
            risk_score=risk_score,
            max_position_size=0.0,  # Calculated below
            recommended_stop_loss=0.0  # Calculated below
        )
        
        # Calculate optimal position size
        risk_metrics.max_position_size = self.position_sizer.calculate_position_size(risk_metrics, current_allocation)
        
        # Calculate stop loss
        risk_metrics.recommended_stop_loss = self.stop_loss_calculator.calculate_stop_loss(risk_metrics)
        
        return risk_metrics
    
    def get_current_listing_allocation(self) -> float:
        """Get current USD allocation to active listing trades"""
        try:
            connection = mysql.connector.connect(**self.db_config)
            cursor = connection.cursor()
            
            cursor.execute("""
                SELECT SUM(position_size_usd) as total_allocation
                FROM listing_trades 
                WHERE status = 'ACTIVE'
            """)
            
            result = cursor.fetchone()
            return float(result[0]) if result and result[0] else 0.0
            
        except Error as e:
            logger.error(f"❌ Error getting listing allocation: {e}")
            return 0.0
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()
    
    def get_portfolio_risk_metrics(self) -> PortfolioRisk:
        """Calculate portfolio-level risk metrics"""
        try:
            connection = mysql.connector.connect(**self.db_config)
            cursor = connection.cursor(dictionary=True)
            
            # Get active listing trades
            cursor.execute("""
                SELECT symbol, position_size_usd, confidence_score 
                FROM listing_trades 
                WHERE status = 'ACTIVE'
            """)
            
            active_trades = cursor.fetchall()
            
            # Calculate metrics
            total_listing_value = sum(trade['position_size_usd'] for trade in active_trades)
            listing_allocation_pct = total_listing_value / self.position_sizer.portfolio_value
            active_count = len(active_trades)
            
            # Max single position
            max_position = max([trade['position_size_usd'] for trade in active_trades], default=0)
            max_position_pct = max_position / self.position_sizer.portfolio_value
            
            # Diversification score (higher = more diversified)
            if active_count <= 1:
                diversification = 0.0
            else:
                position_sizes = [trade['position_size_usd'] for trade in active_trades]
                # Calculate Herfindahl index and convert to diversification score
                hhi = sum((size/total_listing_value)**2 for size in position_sizes if total_listing_value > 0)
                diversification = 1.0 - hhi if hhi > 0 else 0.0
            
            # Overall risk score
            avg_confidence = np.mean([trade['confidence_score'] for trade in active_trades]) if active_trades else 1.0
            allocation_risk = min(listing_allocation_pct / self.position_sizer.max_listing_allocation, 1.0)
            concentration_risk = max_position_pct / self.position_sizer.max_position_pct
            confidence_risk = 1.0 - avg_confidence
            
            overall_risk = (allocation_risk * 0.4 + concentration_risk * 0.3 + 
                          confidence_risk * 0.2 + (1.0 - diversification) * 0.1)
            
            return PortfolioRisk(
                total_value=self.position_sizer.portfolio_value,
                listing_allocation_pct=listing_allocation_pct,
                active_listing_count=active_count,
                max_single_position_pct=max_position_pct,
                diversification_score=diversification,
                overall_risk_score=min(overall_risk, 1.0)
            )
            
        except Error as e:
            logger.error(f"❌ Error calculating portfolio risk: {e}")
            return PortfolioRisk(
                total_value=self.position_sizer.portfolio_value,
                listing_allocation_pct=0.0,
                active_listing_count=0,
                max_single_position_pct=0.0,
                diversification_score=1.0,
                overall_risk_score=0.0
            )
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()
    
    def validate_trade_opportunity(self, symbol: str, listing_data: Dict) -> Tuple[bool, str, Optional[RiskMetrics]]:
        """
        Validate if listing trade opportunity meets risk criteria
        Returns: (approved, reason, risk_metrics)
        """
        
        # DATABASE-DRIVEN ASSET FILTERING: Check if asset is supported for trading
        try:
            from coinbase_asset_filter import is_asset_supported
            if not is_asset_supported(symbol):
                logger.info(f"[ASSET_FILTER] Rejecting listing trade for {symbol}: not supported by Coinbase Advanced Trade")
                return False, f"Asset {symbol} not supported by Coinbase Advanced Trade", None
        except ImportError:
            logger.warning(f"[ASSET_FILTER] Could not import asset filter, proceeding without asset filtering for {symbol}")
        except Exception as e:
            logger.warning(f"[ASSET_FILTER] Error checking asset support for {symbol}: {e}")
        
        # Analyze risk metrics
        risk_metrics = self.analyze_listing_risk(symbol, listing_data)
        
        # Check if position size is viable
        if risk_metrics.max_position_size <= 0:
            return False, "Position size too small or allocation limits exceeded", risk_metrics
        
        # Check overall risk score
        if risk_metrics.risk_score > 0.8:
            return False, f"Risk score too high: {risk_metrics.risk_score:.2f}", risk_metrics
        
        # Check portfolio-level risk
        portfolio_risk = self.get_portfolio_risk_metrics()
        
        if portfolio_risk.overall_risk_score > 0.8:
            return False, f"Portfolio risk too high: {portfolio_risk.overall_risk_score:.2f}", risk_metrics
        
        if portfolio_risk.active_listing_count >= 5:
            return False, f"Too many active listing trades: {portfolio_risk.active_listing_count}", risk_metrics
        
        # All checks passed
        return True, "Trade approved by risk management", risk_metrics
    
    def log_risk_decision(self, symbol: str, approved: bool, reason: str, risk_metrics: RiskMetrics):
        """Log risk management decision"""
        logger.info(f"🛡️ RISK ANALYSIS: {symbol}")
        logger.info(f"   Approved: {approved}")
        logger.info(f"   Reason: {reason}")
        if risk_metrics:
            logger.info(f"   Risk Score: {risk_metrics.risk_score:.2f}")
            logger.info(f"   Position Size: ${risk_metrics.max_position_size:.2f}")
            logger.info(f"   Stop Loss: {risk_metrics.recommended_stop_loss:.1%}")
            logger.info(f"   Confidence: {risk_metrics.confidence_score:.2f}")

def main():
    """Test risk management engine"""
    risk_manager = RiskManager()
    
    # Test listing data
    test_listing = {
        "symbol": "TEST",
        "confidence": 0.8,
        "listing_type": "announced",
        "sentiment_score": 0.7,
        "social_buzz_score": 0.6
    }
    
    # Validate trade
    approved, reason, risk_metrics = risk_manager.validate_trade_opportunity("TEST", test_listing)
    
    risk_manager.log_risk_decision("TEST", approved, reason, risk_metrics)
    
    # Get portfolio risk
    portfolio_risk = risk_manager.get_portfolio_risk_metrics()
    print(f"\nPortfolio Risk Metrics:")
    print(f"Listing Allocation: {portfolio_risk.listing_allocation_pct:.1%}")
    print(f"Active Trades: {portfolio_risk.active_listing_count}")
    print(f"Overall Risk: {portfolio_risk.overall_risk_score:.2f}")

if __name__ == "__main__":
    main()
