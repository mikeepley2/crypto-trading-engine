#!/usr/bin/env python3
"""
Enhanced Signal Generator with LLM Assessment Integration
Adds LLM-based signal strength adjustment to the existing ML signal generation
"""

import sys
import os
import asyncio
import logging
from typing import Dict, List, Optional

# Add the parent directory to the path to import the original generator
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Import the original enhanced signal generator and LLM assessor
from enhanced_signal_generator import EnhancedSignalGenerator
from llm_signal_assessor import LLMSignalAssessor, MarketContext
from signal_analytics_tracker import SignalAnalyticsTracker

# Configure logging
logger = logging.getLogger(__name__)

class LLMEnhancedSignalGenerator(EnhancedSignalGenerator):
    """
    Enhanced Signal Generator with LLM Assessment Integration
    
    Extends the original signal generator to include LLM-based confidence adjustment
    based on contextual market analysis.
    """
    
    def __init__(self, host='host.docker.internal', user='news_collector', 
                 password='99Rules!', database='crypto_prices'):
        """Initialize with LLM assessment capability"""
        
        # Initialize parent class (no parameters needed)
        super().__init__()
        
        # Override database configuration if different parameters provided
        if host != 'host.docker.internal' or user != 'news_collector' or password != '99Rules!' or database != 'crypto_prices':
            self.db_config = {
                'host': host,
                'user': user, 
                'password': password,
                'database': database,
                'port': int(os.environ.get('DATABASE_PORT', 3306))
            }
        
        # Initialize LLM assessor
        self.llm_assessor = LLMSignalAssessor()
        
        # Initialize analytics tracker
        self.analytics_tracker = SignalAnalyticsTracker(self.db_config)
        
        # LLM assessment configuration
        self.enable_llm_assessment = os.getenv('ENABLE_LLM_ASSESSMENT', 'true').lower() == 'true'
        self.llm_assessment_threshold = float(os.getenv('LLM_ASSESSMENT_THRESHOLD', '0.4'))
        
        logger.info(f"🧠 LLM-Enhanced Signal Generator initialized")
        logger.info(f"   LLM Assessment: {'ENABLED' if self.enable_llm_assessment else 'DISABLED'}")
        logger.info(f"   Assessment Threshold: {self.llm_assessment_threshold:.2f}")
    
    async def generate_signals_with_llm_assessment(self, symbols: List[str] = None) -> List[Dict]:
        """
        Generate signals with LLM assessment and confidence adjustment
        
        This is the main entry point for LLM-enhanced signal generation.
        """
        try:
            logger.info(f"🧠 Generating LLM-enhanced signals for: {symbols or 'ALL'}")
            
            # Start new analytics session
            session_id = self.analytics_tracker.start_signal_session()
            
            # Generate base ML signals using parent class
            base_signals = self.generate_signals(symbols)
            
            if not base_signals:
                logger.warning("No base signals generated")
                return []
            
            # Track all base signals
            portfolio = self.get_current_portfolio()
            market_conditions = self._get_market_conditions()
            
            for signal in base_signals:
                self.analytics_tracker.track_base_signal(
                    symbol=signal['symbol'],
                    strategy_name=signal.get('model_version', 'enhanced_ml'),
                    signal_type=signal['signal_type'],
                    confidence=signal['confidence'],
                    reasoning=signal.get('portfolio_reason', 'ML-based signal'),
                    portfolio_context=portfolio,
                    market_conditions=market_conditions
                )
            
            # Apply LLM assessment if enabled
            if self.enable_llm_assessment:
                enhanced_signals = await self._apply_llm_assessment_to_signals(base_signals)
            else:
                enhanced_signals = base_signals
                logger.info("LLM assessment disabled - using base ML signals")
            
            # Track final decisions
            for signal in enhanced_signals:
                strategy_name = signal.get('model_version', 'enhanced_ml')
                if 'llm_confidence_adjustment' in signal:
                    strategy_name += '_llm_enhanced'
                
                selection_reason = "Selected based on "
                if 'llm_confidence_adjustment' in signal:
                    adj = signal['llm_confidence_adjustment']
                    selection_reason += f"LLM assessment (confidence adjusted by {adj:+.3f})"
                else:
                    selection_reason += "base ML confidence score"
                
                self.analytics_tracker.track_final_decision(
                    symbol=signal['symbol'],
                    selected_strategy=strategy_name,
                    final_signal_type=signal['signal_type'],
                    final_confidence=signal['confidence'],
                    selection_reason=selection_reason
                )
            
            # Log session analytics
            self.analytics_tracker.log_session_analytics()
            
            # Save enhanced signals to database
            if enhanced_signals:
                self.save_enhanced_signals_to_db(enhanced_signals)
            
            return enhanced_signals
            
        except Exception as e:
            logger.error(f"Error in LLM-enhanced signal generation: {e}")
            return []
    
    async def _apply_llm_assessment_to_signals(self, base_signals: List[Dict]) -> List[Dict]:
        """Apply LLM assessment to adjust signal confidence"""
        
        enhanced_signals = []
        assessment_tasks = []
        
        # Get current portfolio for context
        portfolio = self.get_current_portfolio()
        
        # Create assessment tasks for each signal
        for signal in base_signals:
            # Skip signals below assessment threshold
            if signal['confidence'] < self.llm_assessment_threshold:
                logger.debug(f"Skipping LLM assessment for {signal['symbol']} "
                           f"(confidence {signal['confidence']:.3f} below threshold)")
                enhanced_signals.append(signal)
                continue
            
            # Create market context for LLM assessment
            market_context = self._create_market_context_from_signal(signal, portfolio)
            
            # Create async assessment task
            task = self._assess_single_signal(signal, market_context)
            assessment_tasks.append((signal, task))
        
        # Execute all assessments concurrently
        if assessment_tasks:
            logger.info(f"🧠 Running LLM assessment for {len(assessment_tasks)} signals...")
            
            for signal, task in assessment_tasks:
                try:
                    enhanced_signal = await task
                    enhanced_signals.append(enhanced_signal)
                except Exception as e:
                    logger.error(f"LLM assessment failed for {signal['symbol']}: {e}")
                    # Use original signal on assessment failure
                    enhanced_signals.append(signal)
        
        # Log assessment summary
        total_adjustments = sum(1 for s in enhanced_signals if 'llm_confidence_adjustment' in s)
        avg_adjustment = sum(s.get('llm_confidence_adjustment', 0) for s in enhanced_signals) / len(enhanced_signals) if enhanced_signals else 0
        
        logger.info(f"🧠 LLM Assessment Complete: {total_adjustments}/{len(enhanced_signals)} signals adjusted, "
                   f"avg adjustment: {avg_adjustment:+.3f}")
        
        return enhanced_signals
    
    async def _assess_single_signal(self, signal: Dict, market_context: MarketContext) -> Dict:
        """Assess a single signal with LLM and return enhanced signal"""
        
        try:
            # Store original confidence for tracking
            original_confidence = signal['confidence']
            
            # Get LLM assessment
            assessment = await self.llm_assessor.assess_signal(
                market_context, 
                signal['signal_type'], 
                signal['confidence']
            )
            
            # Track LLM assessment
            llm_sentiment = "NEUTRAL"
            if assessment.confidence_adjustment > 0.05:
                llm_sentiment = "BULLISH"
            elif assessment.confidence_adjustment < -0.05:
                llm_sentiment = "BEARISH"
            
            self.analytics_tracker.track_llm_assessment(
                symbol=signal['symbol'],
                strategy_name=signal.get('model_version', 'enhanced_ml'),
                pre_score=original_confidence,
                post_score=assessment.adjusted_confidence,
                llm_sentiment=llm_sentiment,
                llm_reasoning=assessment.adjustment_reasoning,
                original_signal_type=signal['signal_type']
            )
            
            # Create enhanced signal with LLM data
            enhanced_signal = signal.copy()
            
            # Update confidence and add LLM metadata
            enhanced_signal.update({
                'confidence': assessment.adjusted_confidence,
                'original_ml_confidence': assessment.original_confidence,
                'llm_confidence_adjustment': assessment.confidence_adjustment,
                'llm_reasoning': assessment.adjustment_reasoning,
                'llm_key_factors': assessment.key_factors,
                'llm_risk_factors': assessment.risk_factors,
                'llm_market_context_score': assessment.market_context_score,
                'llm_technical_alignment_score': assessment.technical_alignment_score,
                'llm_sentiment_alignment_score': assessment.sentiment_alignment_score,
                'llm_overall_assessment': assessment.overall_assessment,
                'llm_enhanced': True
            })
            
            # Log significant adjustments
            if abs(assessment.confidence_adjustment) > 0.1:
                direction = "INCREASED" if assessment.confidence_adjustment > 0 else "DECREASED"
                logger.info(f"🧠 {signal['symbol']}: Confidence {direction} by {abs(assessment.confidence_adjustment):.3f} - {assessment.overall_assessment}")
                logger.info(f"   Reasoning: {assessment.adjustment_reasoning[:100]}...")
            
            return enhanced_signal
            
        except Exception as e:
            logger.error(f"Error assessing signal for {signal['symbol']}: {e}")
            return signal
    
    def _create_market_context_from_signal(self, signal: Dict, portfolio: Dict) -> MarketContext:
        """Create market context from signal data and portfolio"""
        
        symbol = signal['symbol']
        positions = portfolio.get('positions', {})
        current_position = positions.get(symbol, {})
        
        # Calculate portfolio context
        total_value = portfolio.get('total_value', 0)
        position_weight = (current_position.get('value_usd', 0) / total_value * 100) if total_value > 0 else 0
        cash_percentage = (portfolio.get('cash_balance', 0) / total_value * 100) if total_value > 0 else 0
        
        # Get additional market data
        market_data = self._get_additional_market_data(symbol)
        
        return self.llm_assessor.create_market_context(
            symbol=symbol,
            signal_data=signal,
            portfolio_data={
                'position_weight': position_weight,
                'cash_percentage': cash_percentage,
                'portfolio_heat': self._calculate_portfolio_heat(portfolio),
                'correlation_risk': self._assess_correlation_risk(symbol, portfolio),
                'risk_environment': signal.get('market_selloff_severity', 'NORMAL')
            },
            market_data=market_data
        )
    
    def _get_additional_market_data(self, symbol: str) -> Dict:
        """Get additional market data not in signal"""
        try:
            # This could fetch additional data from APIs or database
            # For now, return placeholder data
            return {
                'volume_24h': 0,
                'market_cap': None,
                'volume_trend': 'STABLE',
                'momentum': 'NEUTRAL',
                'support_resistance': 'IN_RANGE',
                'volatility': 'NORMAL'
            }
        except Exception as e:
            logger.warning(f"Could not get additional market data for {symbol}: {e}")
            return {}
    
    def _calculate_portfolio_heat(self, portfolio: Dict) -> float:
        """Calculate portfolio risk heat (0.0 to 1.0)"""
        try:
            positions = portfolio.get('positions', {})
            if not positions:
                return 0.0
            
            total_value = portfolio.get('total_value', 0)
            if total_value <= 0:
                return 0.0
            
            # Calculate concentration risk
            weights = [pos.get('value_usd', 0) / total_value for pos in positions.values()]
            max_weight = max(weights) if weights else 0
            
            # Simple heat calculation: higher concentration = higher heat
            heat = min(1.0, max_weight * 5)  # Scale to 0-1
            
            return heat
            
        except Exception as e:
            logger.warning(f"Error calculating portfolio heat: {e}")
            return 0.5  # Default moderate heat
    
    def _assess_correlation_risk(self, symbol: str, portfolio: Dict) -> str:
        """Assess correlation risk for the symbol"""
        try:
            positions = portfolio.get('positions', {})
            
            # Simple correlation assessment based on crypto categories
            major_cryptos = ['BTC', 'ETH']
            altcoins = ['SOL', 'ADA', 'MATIC', 'AVAX', 'DOT', 'LINK']
            
            major_exposure = sum(1 for s in positions.keys() if s in major_cryptos)
            alt_exposure = sum(1 for s in positions.keys() if s in altcoins)
            
            # Assess based on exposure
            if symbol in major_cryptos and major_exposure >= 2:
                return 'HIGH'
            elif symbol in altcoins and alt_exposure >= 3:
                return 'HIGH'
            elif len(positions) >= 5:
                return 'MEDIUM'
            else:
                return 'LOW'
                
        except Exception as e:
            logger.warning(f"Error assessing correlation risk for {symbol}: {e}")
            return 'MEDIUM'
    
    def save_enhanced_signals_to_db(self, signals: List[Dict]) -> bool:
        """Save LLM-enhanced signals to database with additional LLM columns"""
        try:
            if not signals:
                return True
            
            # Use the parent class method but add LLM data to the signal structure
            for signal in signals:
                if signal.get('llm_enhanced'):
                    # Add LLM data to the signal for database storage
                    signal['llm_analysis'] = {
                        'adjusted_confidence': signal.get('confidence'),
                        'original_ml_confidence': signal.get('original_ml_confidence'),
                        'confidence_adjustment': signal.get('llm_confidence_adjustment'),
                        'market_context_score': signal.get('llm_market_context_score'),
                        'technical_alignment_score': signal.get('llm_technical_alignment_score'),
                        'sentiment_alignment_score': signal.get('llm_sentiment_alignment_score'),
                        'overall_assessment': signal.get('llm_overall_assessment'),
                        'key_factors': signal.get('llm_key_factors', []),
                        'risk_factors': signal.get('llm_risk_factors', [])
                    }
                    
                    signal['llm_confidence'] = signal.get('confidence')
                    signal['llm_reasoning'] = signal.get('llm_reasoning')
            
            # Call parent class method to save to database
            return self.save_signals_to_db(signals)
            
        except Exception as e:
            logger.error(f"Error saving enhanced signals to database: {e}")
            return False
    
    def _get_current_market_conditions(self):
        """Get summary of current market conditions"""
        try:
            # This is a simplified market context - in production you might 
            # fetch real-time market data
            return {
                'overall_sentiment': 'NEUTRAL',
                'volatility': 'MODERATE',
                'trend': 'CONSOLIDATING',
                'key_levels': 'Support at key levels'
            }
        except Exception as e:
            logger.warning(f"Could not fetch market conditions: {e}")
            return {
                'overall_sentiment': 'UNKNOWN',
                'volatility': 'UNKNOWN', 
                'trend': 'UNKNOWN',
                'key_levels': 'Unknown'
            }
    
    def _get_portfolio_context(self):
        """Get current portfolio context for analytics"""
        try:
            # This would typically fetch from portfolio service
            # For now return basic context
            return {
                'total_positions': 0,
                'cash_ratio': 1.0,
                'risk_exposure': 'LOW',
                'recent_performance': 'STABLE'
            }
        except Exception as e:
            logger.warning(f"Could not fetch portfolio context: {e}")
            return {
                'total_positions': 0,
                'cash_ratio': 1.0,
                'risk_exposure': 'UNKNOWN',
                'recent_performance': 'UNKNOWN'
            }

# Async wrapper for the main signal generation
async def generate_llm_enhanced_signals(symbols: List[str] = None) -> List[Dict]:
    """Standalone function to generate LLM-enhanced signals"""
    
    generator = LLMEnhancedSignalGenerator()
    return await generator.generate_signals_with_llm_assessment(symbols)

# CLI interface for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate LLM-Enhanced Trading Signals")
    parser.add_argument('--symbols', nargs='*', help='Symbols to generate signals for')
    parser.add_argument('--disable-llm', action='store_true', help='Disable LLM assessment')
    
    args = parser.parse_args()
    
    # Set environment variable if LLM is disabled
    if args.disable_llm:
        os.environ['ENABLE_LLM_ASSESSMENT'] = 'false'
    
    # Run signal generation
    async def main():
        signals = await generate_llm_enhanced_signals(args.symbols)
        
        print(f"\nGenerated {len(signals)} LLM-enhanced signals:")
        for signal in signals:
            llm_adj = signal.get('llm_confidence_adjustment', 0)
            print(f"  {signal['symbol']}: {signal['signal_type']} "
                  f"(confidence: {signal['confidence']:.3f}, "
                  f"LLM adj: {llm_adj:+.3f})")
    
    asyncio.run(main())
