#!/usr/bin/env python3
"""
ML Model Performance Feedback Service
Analyzes trade outcomes and provides feedback for model optimization
"""

import os
import sys
import time
import logging
import threading
import signal as unix_signal
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import mysql.connector
import pandas as pd
import numpy as np
import json
import requests
from dataclasses import dataclass
from collections import defaultdict
from fastapi import FastAPI
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [ML_FEEDBACK] %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TradeOutcome:
    """Data class for trade outcome analysis"""
    trade_id: int
    symbol: str
    action: str
    entry_price: float
    exit_price: Optional[float]
    amount_usd: float
    profit_loss: Optional[float]
    profit_loss_percentage: Optional[float]
    signal_id: Optional[str]
    model_version: Optional[str]
    confidence: float
    entry_time: datetime
    exit_time: Optional[datetime]
    duration_hours: Optional[float]
    status: str  # 'open', 'closed', 'stop_loss', 'take_profit'

@dataclass
class ModelPerformanceMetrics:
    """Model performance metrics"""
    model_version: str
    total_trades: int
    profitable_trades: int
    losing_trades: int
    win_rate: float
    avg_profit_percentage: float
    avg_loss_percentage: float
    total_return_percentage: float
    max_drawdown_percentage: float
    sharpe_ratio: float
    avg_trade_duration_hours: float
    confidence_accuracy: float
    signal_quality_score: float

class MLPerformanceFeedbackService:
    """ML Performance Feedback and Optimization Service"""
    
    def __init__(self):
        # Database configuration
        self.trades_db_config = {
            'host': os.environ.get('DATABASE_HOST', 'host.docker.internal'),
            'user': os.environ.get('DATABASE_USER', 'news_collector'),
            'password': os.environ.get('DATABASE_PASSWORD', '99Rules!'),
            'database': os.environ.get('TRADES_DATABASE', 'crypto_transactions'),
            'port': int(os.environ.get('DATABASE_PORT', 3306))
        }
        
        self.signals_db_config = {
            'host': os.environ.get('DATABASE_HOST', 'host.docker.internal'),
            'user': os.environ.get('DATABASE_USER', 'news_collector'),
            'password': os.environ.get('DATABASE_PASSWORD', '99Rules!'),
            'database': os.environ.get('SIGNALS_DATABASE', 'crypto_prices'),
            'port': int(os.environ.get('DATABASE_PORT', 3306))
        }
        
        # Service configuration
        self.analysis_interval = int(os.environ.get('ANALYSIS_INTERVAL_MINUTES', 30))
        self.min_trades_for_analysis = int(os.environ.get('MIN_TRADES_FOR_ANALYSIS', 10))
        self.lookback_days = int(os.environ.get('LOOKBACK_DAYS', 30))
        
        # ML Service URLs
        self.signal_generator_url = os.environ.get('SIGNAL_GENERATOR_URL', 'http://host.docker.internal:8025')
        
        # Thread management
        self.analysis_thread = None
        self.running = False
        
        # Performance tracking
        self.analyses_completed = 0
        self.feedback_reports_generated = 0
        self.optimization_suggestions = 0
        self.start_time = datetime.utcnow()
        
    def get_trade_outcomes(self, days_back: int = None) -> List[TradeOutcome]:
        """Get trade outcomes for analysis"""
        try:
            if days_back is None:
                days_back = self.lookback_days
            
            conn = mysql.connector.connect(**self.trades_db_config)
            cursor = conn.cursor(dictionary=True)
            
            # Get completed trades with their outcomes
            cutoff_date = datetime.utcnow() - timedelta(days=days_back)
            
            query = """
            SELECT 
                t.id as trade_id,
                t.symbol,
                t.action,
                t.price as entry_price,
                t.exit_price,
                t.amount_usd,
                t.profit_loss,
                t.profit_loss_percentage,
                t.signal_id,
                t.model_version,
                tr.confidence,
                t.created_at as entry_time,
                t.completed_at as exit_time,
                TIMESTAMPDIFF(HOUR, t.created_at, t.completed_at) as duration_hours,
                t.status
            FROM trades t
            LEFT JOIN trade_recommendations tr ON t.recommendation_id = tr.id
            WHERE t.created_at >= %s
              AND t.status IN ('completed', 'stop_loss', 'take_profit')
              AND t.profit_loss IS NOT NULL
            ORDER BY t.created_at DESC
            """
            
            cursor.execute(query, (cutoff_date,))
            trades = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            # Convert to TradeOutcome objects
            outcomes = []
            for trade in trades:
                outcome = TradeOutcome(
                    trade_id=trade['trade_id'],
                    symbol=trade['symbol'],
                    action=trade['action'],
                    entry_price=float(trade['entry_price']),
                    exit_price=float(trade['exit_price']) if trade['exit_price'] else None,
                    amount_usd=float(trade['amount_usd']),
                    profit_loss=float(trade['profit_loss']) if trade['profit_loss'] else None,
                    profit_loss_percentage=float(trade['profit_loss_percentage']) if trade['profit_loss_percentage'] else None,
                    signal_id=trade['signal_id'],
                    model_version=trade['model_version'] or 'unknown',
                    confidence=float(trade['confidence']) if trade['confidence'] else 0.5,
                    entry_time=trade['entry_time'],
                    exit_time=trade['exit_time'],
                    duration_hours=float(trade['duration_hours']) if trade['duration_hours'] else None,
                    status=trade['status']
                )
                outcomes.append(outcome)
            
            logger.info(f"📊 Retrieved {len(outcomes)} trade outcomes for analysis")
            return outcomes
            
        except Exception as e:
            logger.error(f"❌ Error getting trade outcomes: {e}")
            return []
    
    def calculate_model_performance(self, outcomes: List[TradeOutcome]) -> Dict[str, ModelPerformanceMetrics]:
        """Calculate performance metrics by model version"""
        try:
            # Group outcomes by model version
            model_outcomes = defaultdict(list)
            for outcome in outcomes:
                model_outcomes[outcome.model_version].append(outcome)
            
            performance_metrics = {}
            
            for model_version, model_trades in model_outcomes.items():
                if len(model_trades) < self.min_trades_for_analysis:
                    logger.debug(f"Skipping {model_version}: only {len(model_trades)} trades")
                    continue
                
                # Calculate basic metrics
                total_trades = len(model_trades)
                profitable_trades = sum(1 for t in model_trades if t.profit_loss and t.profit_loss > 0)
                losing_trades = sum(1 for t in model_trades if t.profit_loss and t.profit_loss < 0)
                
                win_rate = profitable_trades / total_trades if total_trades > 0 else 0
                
                # Profit/Loss metrics
                profits = [t.profit_loss_percentage for t in model_trades if t.profit_loss_percentage and t.profit_loss_percentage > 0]
                losses = [abs(t.profit_loss_percentage) for t in model_trades if t.profit_loss_percentage and t.profit_loss_percentage < 0]
                
                avg_profit_percentage = np.mean(profits) if profits else 0
                avg_loss_percentage = np.mean(losses) if losses else 0
                
                # Total return
                total_return_percentage = sum(t.profit_loss_percentage for t in model_trades if t.profit_loss_percentage)
                
                # Calculate max drawdown
                cumulative_returns = np.cumsum([t.profit_loss_percentage for t in model_trades if t.profit_loss_percentage])
                running_max = np.maximum.accumulate(cumulative_returns)
                drawdowns = cumulative_returns - running_max
                max_drawdown_percentage = abs(min(drawdowns)) if len(drawdowns) > 0 else 0
                
                # Sharpe ratio (simplified)
                returns = [t.profit_loss_percentage for t in model_trades if t.profit_loss_percentage]
                if returns and len(returns) > 1:
                    mean_return = np.mean(returns)
                    std_return = np.std(returns)
                    sharpe_ratio = mean_return / std_return if std_return > 0 else 0
                else:
                    sharpe_ratio = 0
                
                # Average trade duration
                durations = [t.duration_hours for t in model_trades if t.duration_hours]
                avg_trade_duration_hours = np.mean(durations) if durations else 0
                
                # Confidence accuracy (how well confidence predicts outcomes)
                confidence_accuracy = self.calculate_confidence_accuracy(model_trades)
                
                # Signal quality score (composite metric)
                signal_quality_score = self.calculate_signal_quality_score(
                    win_rate, avg_profit_percentage, avg_loss_percentage, confidence_accuracy
                )
                
                metrics = ModelPerformanceMetrics(
                    model_version=model_version,
                    total_trades=total_trades,
                    profitable_trades=profitable_trades,
                    losing_trades=losing_trades,
                    win_rate=win_rate,
                    avg_profit_percentage=avg_profit_percentage,
                    avg_loss_percentage=avg_loss_percentage,
                    total_return_percentage=total_return_percentage,
                    max_drawdown_percentage=max_drawdown_percentage,
                    sharpe_ratio=sharpe_ratio,
                    avg_trade_duration_hours=avg_trade_duration_hours,
                    confidence_accuracy=confidence_accuracy,
                    signal_quality_score=signal_quality_score
                )
                
                performance_metrics[model_version] = metrics
                
            logger.info(f"📈 Calculated performance for {len(performance_metrics)} model versions")
            return performance_metrics
            
        except Exception as e:
            logger.error(f"❌ Error calculating model performance: {e}")
            return {}
    
    def calculate_confidence_accuracy(self, trades: List[TradeOutcome]) -> float:
        """Calculate how well confidence scores predict profitable trades"""
        try:
            if len(trades) < 5:
                return 0.5  # Neutral score
            
            # Group trades by confidence bins
            confidence_bins = np.arange(0.5, 1.01, 0.1)
            bin_accuracies = []
            
            for i in range(len(confidence_bins) - 1):
                bin_min = confidence_bins[i]
                bin_max = confidence_bins[i + 1]
                
                bin_trades = [t for t in trades if bin_min <= t.confidence < bin_max]
                
                if len(bin_trades) >= 3:  # Minimum trades for meaningful analysis
                    profitable = sum(1 for t in bin_trades if t.profit_loss and t.profit_loss > 0)
                    accuracy = profitable / len(bin_trades)
                    bin_accuracies.append(accuracy)
            
            return np.mean(bin_accuracies) if bin_accuracies else 0.5
            
        except Exception as e:
            logger.error(f"❌ Error calculating confidence accuracy: {e}")
            return 0.5
    
    def calculate_signal_quality_score(self, win_rate: float, avg_profit: float, 
                                     avg_loss: float, confidence_accuracy: float) -> float:
        """Calculate composite signal quality score (0-100)"""
        try:
            # Weighted scoring
            win_rate_score = win_rate * 30  # 30% weight
            profit_ratio_score = min(avg_profit / max(avg_loss, 0.1), 3) * 20  # 20% weight, capped at 3:1
            return_quality_score = min(avg_profit, 10) * 2  # 20% weight, capped at 10%
            confidence_score = confidence_accuracy * 30  # 30% weight
            
            total_score = win_rate_score + profit_ratio_score + return_quality_score + confidence_score
            
            return min(total_score, 100)  # Cap at 100
            
        except Exception as e:
            logger.error(f"❌ Error calculating signal quality score: {e}")
            return 50  # Neutral score
    
    def generate_optimization_suggestions(self, performance_metrics: Dict[str, ModelPerformanceMetrics]) -> List[Dict]:
        """Generate optimization suggestions based on performance analysis"""
        suggestions = []
        
        try:
            for model_version, metrics in performance_metrics.items():
                model_suggestions = []
                
                # Win rate analysis
                if metrics.win_rate < 0.4:
                    model_suggestions.append({
                        'type': 'model_tuning',
                        'priority': 'high',
                        'issue': 'Low win rate',
                        'current_value': f"{metrics.win_rate:.2%}",
                        'recommendation': 'Increase signal confidence threshold or retrain model with more conservative parameters',
                        'target_improvement': '45%+ win rate'
                    })
                
                # Profit/Loss ratio analysis
                if metrics.avg_loss_percentage > 0 and metrics.avg_profit_percentage / metrics.avg_loss_percentage < 1.2:
                    model_suggestions.append({
                        'type': 'risk_management',
                        'priority': 'medium',
                        'issue': 'Poor risk/reward ratio',
                        'current_value': f"1:{metrics.avg_loss_percentage/metrics.avg_profit_percentage:.2f}",
                        'recommendation': 'Implement tighter stop-losses or adjust position sizing based on volatility',
                        'target_improvement': '1:1.5+ profit/loss ratio'
                    })
                
                # Confidence accuracy analysis
                if metrics.confidence_accuracy < 0.6:
                    model_suggestions.append({
                        'type': 'calibration',
                        'priority': 'medium',
                        'issue': 'Poor confidence calibration',
                        'current_value': f"{metrics.confidence_accuracy:.2%}",
                        'recommendation': 'Recalibrate confidence scores or add uncertainty quantification',
                        'target_improvement': '65%+ confidence accuracy'
                    })
                
                # Drawdown analysis
                if metrics.max_drawdown_percentage > 15:
                    model_suggestions.append({
                        'type': 'risk_management',
                        'priority': 'high',
                        'issue': 'High maximum drawdown',
                        'current_value': f"{metrics.max_drawdown_percentage:.1f}%",
                        'recommendation': 'Implement portfolio-level risk controls or reduce position sizes during volatile periods',
                        'target_improvement': '<10% maximum drawdown'
                    })
                
                # Trade duration analysis
                if metrics.avg_trade_duration_hours > 72:  # More than 3 days
                    model_suggestions.append({
                        'type': 'strategy_tuning',
                        'priority': 'low',
                        'issue': 'Long average trade duration',
                        'current_value': f"{metrics.avg_trade_duration_hours:.1f} hours",
                        'recommendation': 'Consider more aggressive take-profit targets or shorter-term signals',
                        'target_improvement': '<48 hour average duration'
                    })
                
                if model_suggestions:
                    suggestions.append({
                        'model_version': model_version,
                        'overall_score': metrics.signal_quality_score,
                        'suggestions': model_suggestions,
                        'analysis_timestamp': datetime.utcnow().isoformat()
                    })
            
            self.optimization_suggestions += len(suggestions)
            logger.info(f"💡 Generated {len(suggestions)} optimization reports")
            
            return suggestions
            
        except Exception as e:
            logger.error(f"❌ Error generating optimization suggestions: {e}")
            return []
    
    def save_performance_feedback(self, performance_metrics: Dict[str, ModelPerformanceMetrics], 
                                suggestions: List[Dict]):
        """Save performance feedback to database"""
        try:
            conn = mysql.connector.connect(**self.trades_db_config)
            cursor = conn.cursor()
            
            # Create table if not exists
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS ml_performance_feedback (
                id INT AUTO_INCREMENT PRIMARY KEY,
                model_version VARCHAR(100),
                analysis_date DATE,
                total_trades INT,
                win_rate DECIMAL(5,4),
                avg_profit_percentage DECIMAL(8,4),
                avg_loss_percentage DECIMAL(8,4),
                total_return_percentage DECIMAL(8,4),
                max_drawdown_percentage DECIMAL(8,4),
                sharpe_ratio DECIMAL(8,4),
                confidence_accuracy DECIMAL(5,4),
                signal_quality_score DECIMAL(5,2),
                optimization_suggestions JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_model_date (model_version, analysis_date)
            )
            """)
            
            # Save performance metrics
            for model_version, metrics in performance_metrics.items():
                # Find suggestions for this model
                model_suggestions = next(
                    (s['suggestions'] for s in suggestions if s['model_version'] == model_version),
                    []
                )
                
                cursor.execute("""
                INSERT INTO ml_performance_feedback 
                (model_version, analysis_date, total_trades, win_rate, avg_profit_percentage,
                 avg_loss_percentage, total_return_percentage, max_drawdown_percentage,
                 sharpe_ratio, confidence_accuracy, signal_quality_score, optimization_suggestions)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    model_version,
                    datetime.utcnow().date(),
                    metrics.total_trades,
                    metrics.win_rate,
                    metrics.avg_profit_percentage,
                    metrics.avg_loss_percentage,
                    metrics.total_return_percentage,
                    metrics.max_drawdown_percentage,
                    metrics.sharpe_ratio,
                    metrics.confidence_accuracy,
                    metrics.signal_quality_score,
                    json.dumps(model_suggestions)
                ))
            
            conn.commit()
            cursor.close()
            conn.close()
            
            self.feedback_reports_generated += len(performance_metrics)
            logger.info(f"💾 Saved performance feedback for {len(performance_metrics)} models")
            
        except Exception as e:
            logger.error(f"❌ Error saving performance feedback: {e}")
    
    def send_optimization_alerts(self, suggestions: List[Dict]):
        """Send optimization alerts to signal generator service"""
        try:
            if not suggestions:
                return
            
            # Prepare alert payload
            alert_payload = {
                'type': 'ml_performance_feedback',
                'timestamp': datetime.utcnow().isoformat(),
                'urgent_issues': [],
                'suggestions': suggestions
            }
            
            # Identify urgent issues
            for suggestion in suggestions:
                for model_suggestion in suggestion['suggestions']:
                    if model_suggestion['priority'] == 'high':
                        alert_payload['urgent_issues'].append({
                            'model_version': suggestion['model_version'],
                            'issue': model_suggestion['issue'],
                            'recommendation': model_suggestion['recommendation']
                        })
            
            # Send to signal generator
            try:
                response = requests.post(
                    f"{self.signal_generator_url}/ml_feedback",
                    json=alert_payload,
                    timeout=10
                )
                
                if response.status_code == 200:
                    logger.info(f"📤 Sent optimization alerts to signal generator")
                else:
                    logger.warning(f"⚠️ Alert send failed: {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"⚠️ Could not send alerts to signal generator: {e}")
            
        except Exception as e:
            logger.error(f"❌ Error sending optimization alerts: {e}")
    
    def performance_analysis_worker(self):
        """Worker thread for periodic performance analysis"""
        logger.info("🔄 ML Performance analysis worker started")
        
        while self.running:
            try:
                logger.info("📊 Starting ML performance analysis...")
                
                # Get trade outcomes
                outcomes = self.get_trade_outcomes()
                
                if len(outcomes) >= self.min_trades_for_analysis:
                    # Calculate performance metrics
                    performance_metrics = self.calculate_model_performance(outcomes)
                    
                    if performance_metrics:
                        # Generate optimization suggestions
                        suggestions = self.generate_optimization_suggestions(performance_metrics)
                        
                        # Save feedback to database
                        self.save_performance_feedback(performance_metrics, suggestions)
                        
                        # Send alerts for urgent issues
                        self.send_optimization_alerts(suggestions)
                        
                        self.analyses_completed += 1
                        
                        # Log summary
                        logger.info(f"✅ Analysis complete: {len(performance_metrics)} models analyzed, "
                                   f"{len(suggestions)} optimization reports generated")
                        
                        for model_version, metrics in performance_metrics.items():
                            logger.info(f"   {model_version}: {metrics.win_rate:.1%} win rate, "
                                       f"{metrics.signal_quality_score:.1f} quality score")
                    
                else:
                    logger.info(f"📊 Insufficient trades for analysis: {len(outcomes)} < {self.min_trades_for_analysis}")
                
                # Wait for next analysis
                time.sleep(self.analysis_interval * 60)  # Convert minutes to seconds
                
            except Exception as e:
                logger.error(f"❌ Error in performance analysis: {e}")
                time.sleep(60)  # Wait 1 minute on error
        
        logger.info("🛑 ML Performance analysis worker stopped")
    
    def start_feedback_service(self):
        """Start the ML feedback service"""
        logger.info("🚀 Starting ML Performance Feedback Service...")
        logger.info(f"   Analysis interval: {self.analysis_interval} minutes")
        logger.info(f"   Minimum trades for analysis: {self.min_trades_for_analysis}")
        logger.info(f"   Lookback period: {self.lookback_days} days")
        
        self.running = True
        
        # Start analysis worker thread
        self.analysis_thread = threading.Thread(
            target=self.performance_analysis_worker,
            daemon=True
        )
        self.analysis_thread.start()
        
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("🛑 Shutdown signal received")
            self.stop_feedback_service()
    
    def stop_feedback_service(self):
        """Stop the ML feedback service"""
        logger.info("🛑 Stopping ML Performance Feedback Service...")
        
        self.running = False
        
        if self.analysis_thread and self.analysis_thread.is_alive():
            self.analysis_thread.join(timeout=10)
        
        logger.info("✅ ML Performance Feedback Service stopped")
    
    def get_service_metrics(self) -> Dict:
        """Get service metrics"""
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        
        return {
            'analyses_completed': self.analyses_completed,
            'feedback_reports_generated': self.feedback_reports_generated,
            'optimization_suggestions': self.optimization_suggestions,
            'uptime_seconds': uptime,
            'running': self.running,
            'analysis_interval_minutes': self.analysis_interval,
            'timestamp': datetime.utcnow().isoformat()
        }

# FastAPI app for health checks and metrics
app = FastAPI(title="ML Performance Feedback Service", version="1.0.0")

# Global service instance
service_instance = None

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if service_instance and service_instance.running:
        return {
            "status": "healthy",
            "service": "ml_performance_feedback",
            "timestamp": datetime.utcnow().isoformat(),
            "analyses_completed": service_instance.analyses_completed,
            "running": service_instance.running
        }
    return {"status": "unhealthy", "reason": "service not running"}

@app.get("/metrics")
async def get_metrics():
    """Get service metrics"""
    if service_instance:
        return service_instance.get_service_metrics()
    return {"error": "service not initialized"}

@app.post("/ml_feedback")
async def receive_ml_feedback(feedback_data: dict):
    """Receive ML feedback from external services"""
    logger.info(f"📨 Received external ML feedback: {feedback_data.get('type', 'unknown')}")
    return {"status": "received", "timestamp": datetime.utcnow().isoformat()}

def run_api_server():
    """Run FastAPI server in separate thread"""
    uvicorn.run(app, host="0.0.0.0", port=8035, log_level="warning")

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"🛑 Received signal {signum}, shutting down...")
    raise KeyboardInterrupt()

def main():
    """Main function"""
    global service_instance
    
    # Set up signal handlers
    unix_signal.signal(unix_signal.SIGINT, signal_handler)
    unix_signal.signal(unix_signal.SIGTERM, signal_handler)
    
    # Start API server in separate thread
    api_thread = threading.Thread(target=run_api_server, daemon=True)
    api_thread.start()
    
    # Create and start ML feedback service
    service_instance = MLPerformanceFeedbackService()
    
    try:
        service_instance.start_feedback_service()
    except KeyboardInterrupt:
        logger.info("🛑 Shutdown requested")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        return 1
    finally:
        if service_instance:
            service_instance.stop_feedback_service()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())