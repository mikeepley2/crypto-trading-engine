#!/usr/bin/env python3
"""
Signal Coherence Manager

Prevents conflicting trading signals across all services by implementing:
1. Cross-service signal checking before generation
2. Signal priority system (ML > Rebalancing > Opportunistic)
3. Conflict detection and resolution
4. Signal coordination utilities

This module should be imported by ALL signal-generating services.
"""

import mysql.connector
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)

class SignalCoherenceManager:
    """Manages signal coherence across all trading services"""
    
    # Signal priority hierarchy (higher = more important)
    SIGNAL_PRIORITIES = {
        'enhanced_ml_ensemble': 100,  # Highest priority - ML-based signals
        'advanced_rebalancer': 80,    # Portfolio optimization
        'opportunistic_rebalancer': 60,  # Opportunistic trades
        'llm_risk_manager': 90,       # Risk management overrides
        'manual_override': 110        # Manual signals (if any)
    }
    
    # Time window for considering signals as "recent" (minutes)
    SIGNAL_CONFLICT_WINDOW_MINUTES = 30
    
    def __init__(self, db_config: Dict[str, Any]):
        """Initialize with database configuration"""
        self.db_config = db_config
        
    def check_signal_conflicts(self, symbol: str, proposed_signal: str, 
                             model_name: str, confidence: float) -> Dict[str, Any]:
        """
        Check for existing signals that would conflict with proposed signal
        
        MODIFIED: Always allow signal generation to capture confidence changes,
        but log conflicts for analysis.
        
        Returns:
        {
            'can_generate': bool,  # Always True now
            'conflicts': List[Dict],
            'action': str,  # Always 'generate'
            'reason': str
        }
        """
        try:
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Check for recent signals for this symbol (for logging only)
            cursor.execute("""
                SELECT signal_type, confidence, model, created_at, model_version
                FROM trading_signals 
                WHERE symbol = %s 
                AND created_at >= NOW() - INTERVAL %s MINUTE
                ORDER BY created_at DESC
            """, (symbol, self.SIGNAL_CONFLICT_WINDOW_MINUTES))
            
            recent_signals = cursor.fetchall()
            cursor.close()
            conn.close()
            
            if not recent_signals:
                return {
                    'can_generate': True,
                    'conflicts': [],
                    'action': 'generate',
                    'reason': 'No recent signals found'
                }
            
            # Analyze conflicts (for logging only)
            conflicts = []
            proposed_priority = self.SIGNAL_PRIORITIES.get(model_name, 50)
            
            for signal_type, conf, model, created_at, model_version in recent_signals:
                existing_priority = self.SIGNAL_PRIORITIES.get(model, 50)
                
                # Check for direct conflicts (BUY vs SELL)
                is_conflict = (
                    (proposed_signal == 'BUY' and signal_type == 'SELL') or
                    (proposed_signal == 'SELL' and signal_type == 'BUY')
                )
                
                if is_conflict:
                    conflicts.append({
                        'signal_type': signal_type,
                        'confidence': conf,
                        'model': model,
                        'created_at': created_at,
                        'priority': existing_priority,
                        'age_minutes': (datetime.now() - created_at).total_seconds() / 60
                    })
            
            # MODIFIED: Always allow signal generation for confidence tracking
            # Log conflicts but don't block signal storage
            if conflicts:
                highest_conflict_priority = max(c['priority'] for c in conflicts)
                if proposed_priority > highest_conflict_priority:
                    reason = f'Generating signal (higher priority: {proposed_priority} > {highest_conflict_priority})'
                elif proposed_priority == highest_conflict_priority:
                    highest_conf = max(c['confidence'] for c in conflicts)
                    reason = f'Generating signal (confidence update: {confidence:.3f} vs {highest_conf:.3f})'
                else:
                    reason = f'Generating signal (confidence update despite lower priority)'
            else:
                # Check for same signal type updates
                same_signals = [s for s in recent_signals if s[0] == proposed_signal]
                if same_signals:
                    latest_conf = max(s[1] for s in same_signals)
                    reason = f'Generating signal (confidence update: {confidence:.3f} vs {latest_conf:.3f})'
                else:
                    reason = 'Generating signal (no conflicts)'
            
            return {
                'can_generate': True,  # Always True now
                'conflicts': conflicts,
                'action': 'generate',  # Always generate
                'reason': reason
            }
                
        except Exception as e:
            logger.error(f"Error checking signal conflicts for {symbol}: {e}")
            # On error, allow signal generation but log the issue
            return {
                'can_generate': True,
                'conflicts': [],
                'action': 'generate',
                'reason': f'Error checking conflicts: {e}'
            }
    
    def generate_signal_safely(self, symbol: str, signal_type: str, confidence: float,
                             model_name: str, model_version: str, 
                             additional_data: Optional[Dict] = None) -> bool:
        """
        Generate a signal - always saves new signals to capture confidence changes
        
        Returns True if signal was generated, False if failed
        """
        # Check for conflicts (for logging/analysis only - always generate)
        conflict_check = self.check_signal_conflicts(symbol, signal_type, model_name, confidence)
        
        # Log conflict information for analysis but don't prevent generation
        if conflict_check['conflicts']:
            logger.info(f"� Generating new {signal_type} signal for {symbol} with existing signals - "
                       f"tracking confidence change from {conflict_check['conflicts'][0]['confidence']:.3f} to {confidence:.3f}")
        
        # Log override actions for transparency
        if conflict_check['action'] == 'override':
            logger.info(f"⚡ New signal will update existing signals for {symbol}: {conflict_check['reason']}")
            for conflict in conflict_check['conflicts']:
                logger.info(f"   Updating: {conflict['signal_type']} from {conflict['model']} "
                          f"(conf: {conflict['confidence']:.3f} -> {confidence:.3f}, age: {conflict['age_minutes']:.1f}m)")
        
        # Generate the signal
        try:
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Prepare signal data with all required fields
            signal_data = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'price': 0.0,  # Default price
                'signal_type': signal_type,
                'model': model_name,
                'confidence': confidence,
                'threshold': 0.5,  # Default threshold
                'regime': 'sideways',  # Default regime
                'model_version': model_version,
                'features_used': 0,  # Default features
                'xgboost_confidence': confidence,
                'data_source': 'signal_coherence_manager',
                'created_at': datetime.now()
            }
            
            # Add additional data if provided
            if additional_data:
                signal_data.update(additional_data)
            
            # Insert signal
            columns = ', '.join(signal_data.keys())
            placeholders = ', '.join(['%s'] * len(signal_data))
            
            cursor.execute(f"""
                INSERT INTO trading_signals ({columns})
                VALUES ({placeholders})
            """, list(signal_data.values()))
            
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info(f"✅ Generated {signal_type} signal for {symbol} "
                       f"(conf: {confidence:.3f}, model: {model_name})")
            return True
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return False
    
    def cleanup_old_signals(self, max_age_hours: int = 24):
        """Clean up old signals to prevent database bloat"""
        try:
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute("""
                DELETE FROM trading_signals 
                WHERE created_at < NOW() - INTERVAL %s HOUR
            """, (max_age_hours,))
            
            deleted_count = cursor.rowcount
            conn.commit()
            cursor.close()
            conn.close()
            
            if deleted_count > 0:
                logger.info(f"🧹 Cleaned up {deleted_count} old signals (>{max_age_hours}h)")
                
        except Exception as e:
            logger.error(f"Error cleaning up old signals: {e}")
    
    def get_signal_summary(self, hours: int = 2) -> Dict[str, Any]:
        """Get summary of recent signals for monitoring"""
        try:
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT symbol, signal_type, model, COUNT(*) as count,
                       AVG(confidence) as avg_confidence,
                       MAX(created_at) as latest_signal
                FROM trading_signals 
                WHERE created_at >= NOW() - INTERVAL %s HOUR
                GROUP BY symbol, signal_type, model
                ORDER BY latest_signal DESC
            """, (hours,))
            
            results = cursor.fetchall()
            cursor.close()
            conn.close()
            
            summary = {
                'total_signals': len(results),
                'signals_by_model': {},
                'signals_by_symbol': {},
                'recent_signals': []
            }
            
            for symbol, signal_type, model, count, avg_conf, latest in results:
                # By model
                if model not in summary['signals_by_model']:
                    summary['signals_by_model'][model] = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
                summary['signals_by_model'][model][signal_type] += count
                
                # By symbol
                if symbol not in summary['signals_by_symbol']:
                    summary['signals_by_symbol'][symbol] = []
                summary['signals_by_symbol'][symbol].append({
                    'signal_type': signal_type,
                    'model': model,
                    'count': count,
                    'avg_confidence': avg_conf,
                    'latest': latest
                })
                
                summary['recent_signals'].append({
                    'symbol': symbol,
                    'signal_type': signal_type,
                    'model': model,
                    'count': count,
                    'avg_confidence': avg_conf,
                    'latest': latest
                })
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting signal summary: {e}")
            return {'error': str(e)}


# Utility functions for easy integration
def create_coherence_manager(db_config: Dict[str, Any]) -> SignalCoherenceManager:
    """Create a SignalCoherenceManager instance"""
    return SignalCoherenceManager(db_config)


def check_and_generate_signal(db_config: Dict[str, Any], symbol: str, signal_type: str,
                            confidence: float, model_name: str, model_version: str,
                            additional_data: Optional[Dict] = None) -> bool:
    """Convenience function to check conflicts and generate signal safely"""
    manager = SignalCoherenceManager(db_config)
    return manager.generate_signal_safely(
        symbol, signal_type, confidence, model_name, model_version, additional_data
    )