"""
Enhanced Signal Analytics Tracker
Comprehensive tracking of all signal generation, LLM assessment, and final decisions
"""
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
import mysql.connector
import logging

logger = logging.getLogger(__name__)

class SignalAnalyticsTracker:
    """Tracks comprehensive signal analytics throughout the decision process"""
    
    def __init__(self, db_config: Dict[str, str]):
        self.db_config = db_config
        self.session_id = str(uuid.uuid4())
        
    def start_signal_session(self) -> str:
        """Start a new signal generation session"""
        self.session_id = str(uuid.uuid4())
        logger.info(f"🔍 Started signal analytics session: {self.session_id}")
        return self.session_id
    
    def track_base_signal(self, symbol: str, strategy_name: str, signal_type: str, 
                         confidence: float, reasoning: str = None, 
                         portfolio_context: Dict = None, market_conditions: Dict = None):
        """Track a base ML signal before LLM assessment"""
        try:
            connection = mysql.connector.connect(**self.db_config)
            cursor = connection.cursor()
            
            query = """
                INSERT INTO signal_analytics (
                    session_id, symbol, strategy_name, strategy_type, 
                    raw_signal_type, raw_confidence, raw_reasoning,
                    portfolio_context, market_conditions
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            values = (
                self.session_id, symbol, strategy_name, 'base_ml',
                signal_type, confidence, reasoning,
                json.dumps(portfolio_context) if portfolio_context else None,
                json.dumps(market_conditions) if market_conditions else None
            )
            
            cursor.execute(query, values)
            connection.commit()
            
            record_id = cursor.lastrowid
            logger.info(f"📊 Tracked base signal: {symbol} {signal_type} ({confidence:.3f}) - Strategy: {strategy_name}")
            return record_id
            
        except Exception as e:
            logger.error(f"❌ Failed to track base signal: {e}")
            return None
        finally:
            if 'connection' in locals() and connection.is_connected():
                cursor.close()
                connection.close()
    
    def track_llm_assessment(self, symbol: str, strategy_name: str, 
                           pre_score: float, post_score: float, 
                           llm_sentiment: str, llm_reasoning: str,
                           original_signal_type: str):
        """Track LLM assessment of a signal"""
        try:
            connection = mysql.connector.connect(**self.db_config)
            cursor = connection.cursor()
            
            adjustment = post_score - pre_score
            
            query = """
                INSERT INTO signal_analytics (
                    session_id, symbol, strategy_name, strategy_type,
                    raw_signal_type, raw_confidence, llm_pre_score, 
                    llm_post_score, llm_adjustment, llm_sentiment, llm_reasoning
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            values = (
                self.session_id, symbol, f"{strategy_name}_llm_assessed", 'llm_assessment',
                original_signal_type, pre_score, pre_score, post_score, 
                adjustment, llm_sentiment, llm_reasoning
            )
            
            cursor.execute(query, values)
            connection.commit()
            
            record_id = cursor.lastrowid
            logger.info(f"🧠 Tracked LLM assessment: {symbol} {pre_score:.3f}→{post_score:.3f} ({adjustment:+.3f}) - {llm_sentiment}")
            return record_id
            
        except Exception as e:
            logger.error(f"❌ Failed to track LLM assessment: {e}")
            return None
        finally:
            if 'connection' in locals() and connection.is_connected():
                cursor.close()
                connection.close()
    
    def track_final_decision(self, symbol: str, selected_strategy: str, 
                           final_signal_type: str, final_confidence: float,
                           selection_reason: str, all_candidates: List[Dict] = None):
        """Track the final signal decision and mark selected strategy"""
        try:
            connection = mysql.connector.connect(**self.db_config)
            cursor = connection.cursor()
            
            # Insert final decision record
            query = """
                INSERT INTO signal_analytics (
                    session_id, symbol, strategy_name, strategy_type,
                    raw_signal_type, raw_confidence, is_selected, selection_reason,
                    final_signal_type, final_confidence
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            values = (
                self.session_id, symbol, selected_strategy, 'final_decision',
                final_signal_type, final_confidence, True, selection_reason,
                final_signal_type, final_confidence
            )
            
            cursor.execute(query, values)
            
            # Update the selected strategy record to mark it as selected
            update_query = """
                UPDATE signal_analytics 
                SET is_selected = TRUE, selection_reason = %s,
                    final_signal_type = %s, final_confidence = %s
                WHERE session_id = %s AND symbol = %s AND strategy_name = %s
                AND strategy_type IN ('base_ml', 'llm_assessment')
                ORDER BY timestamp DESC LIMIT 1
            """
            
            cursor.execute(update_query, (
                selection_reason, final_signal_type, final_confidence,
                self.session_id, symbol, selected_strategy
            ))
            
            connection.commit()
            
            record_id = cursor.lastrowid
            logger.info(f"🎯 Tracked final decision: {symbol} {final_signal_type} ({final_confidence:.3f}) - Selected: {selected_strategy}")
            logger.info(f"   Reason: {selection_reason}")
            return record_id
            
        except Exception as e:
            logger.error(f"❌ Failed to track final decision: {e}")
            return None
        finally:
            if 'connection' in locals() and connection.is_connected():
                cursor.close()
                connection.close()
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get a summary of the current session's analytics"""
        try:
            connection = mysql.connector.connect(**self.db_config)
            cursor = connection.cursor(dictionary=True)
            
            query = """
                SELECT 
                    symbol,
                    strategy_name,
                    strategy_type,
                    raw_signal_type,
                    raw_confidence,
                    llm_pre_score,
                    llm_post_score,
                    llm_adjustment,
                    llm_sentiment,
                    is_selected,
                    final_signal_type,
                    final_confidence
                FROM signal_analytics 
                WHERE session_id = %s
                ORDER BY symbol, timestamp
            """
            
            cursor.execute(query, (self.session_id,))
            records = cursor.fetchall()
            
            # Group by symbol
            summary = {}
            for record in records:
                symbol = record['symbol']
                if symbol not in summary:
                    summary[symbol] = {
                        'strategies': [],
                        'selected_strategy': None,
                        'final_signal': None
                    }
                
                summary[symbol]['strategies'].append(record)
                
                if record['is_selected']:
                    summary[symbol]['selected_strategy'] = record['strategy_name']
                    summary[symbol]['final_signal'] = {
                        'type': record['final_signal_type'],
                        'confidence': float(record['final_confidence']) if record['final_confidence'] else None
                    }
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Failed to get session summary: {e}")
            return {}
        finally:
            if 'connection' in locals() and connection.is_connected():
                cursor.close()
                connection.close()
    
    def log_session_analytics(self):
        """Log comprehensive analytics for the session"""
        summary = self.get_session_summary()
        
        logger.info(f"📈 Session Analytics Summary ({self.session_id}):")
        logger.info("=" * 60)
        
        for symbol, data in summary.items():
            logger.info(f"🎯 {symbol}:")
            
            for strategy in data['strategies']:
                status = "✅ SELECTED" if strategy['is_selected'] else "   "
                llm_info = ""
                
                if strategy['llm_adjustment']:
                    adj = float(strategy['llm_adjustment'])
                    llm_info = f" | LLM: {strategy['llm_pre_score']:.3f}→{strategy['llm_post_score']:.3f} ({adj:+.3f})"
                
                logger.info(f"   {status} {strategy['strategy_name']} | {strategy['raw_signal_type']} ({strategy['raw_confidence']:.3f}){llm_info}")
            
            if data['final_signal']:
                logger.info(f"   🏆 FINAL: {data['final_signal']['type']} ({data['final_signal']['confidence']:.3f})")
            logger.info("")
        
        logger.info("=" * 60)
