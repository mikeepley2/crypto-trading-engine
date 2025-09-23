#!/usr/bin/env python3
"""
Test Signal Analytics Tracker
"""
from signal_analytics_tracker import SignalAnalyticsTracker

def test_analytics_tracker():
    """Test basic analytics tracking functionality"""
    
    print("🧪 Testing Signal Analytics Tracker...")
    
    # Initialize tracker
    tracker = SignalAnalyticsTracker()
    
    # Start analytics session
    session_id = tracker.start_analytics_session("TEST_STRATEGY")
    print(f"✅ Started analytics session: {session_id}")
    
    # Track base signals
    base_signals = [
        {"symbol": "BTC", "signal_type": "BUY", "confidence": 0.75, "strategy": "ml_ensemble"},
        {"symbol": "ETH", "signal_type": "SELL", "confidence": 0.65, "strategy": "momentum"}
    ]
    
    tracker.track_base_signals(session_id, base_signals)
    print(f"✅ Tracked {len(base_signals)} base signals")
    
    # Track LLM assessment for first signal
    tracker.track_llm_assessment(
        session_id=session_id,
        symbol="BTC",
        before_confidence=0.75,
        after_confidence=0.85,
        llm_reasoning="Strong bullish momentum confirmed",
        llm_sentiment="BULLISH",
        adjustment_magnitude=0.10
    )
    print("✅ Tracked LLM assessment for BTC")
    
    # Track final decision
    final_signals = [
        {"symbol": "BTC", "signal_type": "BUY", "confidence": 0.85, "selected": True},
        {"symbol": "ETH", "signal_type": "SELL", "confidence": 0.65, "selected": False}
    ]
    
    tracker.track_final_decision(session_id, final_signals, "Selected BTC due to LLM enhancement")
    print("✅ Tracked final decision")
    
    # End session
    tracker.end_analytics_session(session_id, success=True, total_signals_processed=2)
    print("✅ Ended analytics session")
    
    print("\n🎯 Analytics tracking test completed successfully!")
    return True

if __name__ == "__main__":
    try:
        test_analytics_tracker()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
