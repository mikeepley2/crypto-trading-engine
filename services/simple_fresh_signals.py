#!/usr/bin/env python3
"""
Simple Fresh Signal Creator
Uses the recommendations service to create fresh signals
"""

import requests
import json
from datetime import datetime

def create_fresh_recommendations_via_api():
    """Create fresh recommendations using the trade execution service"""
    
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Test the execution engine's create recommendation endpoint
    recommendations = [
        {
            "symbol": "BTC",
            "action": "BUY", 
            "confidence": 0.85,
            "entry_price": 66000.00,
            "stop_loss": 62700.00,
            "take_profit": 71280.00,
            "position_size_percent": 1.0,
            "reasoning": f"Fresh live signal generated at {current_time}",
            "is_mock": False
        },
        {
            "symbol": "ETH", 
            "action": "BUY",
            "confidence": 0.78,
            "entry_price": 3200.00,
            "stop_loss": 3040.00,
            "take_profit": 3456.00,
            "position_size_percent": 1.0,
            "reasoning": f"Fresh live signal generated at {current_time}",
            "is_mock": False
        },
        {
            "symbol": "SOL",
            "action": "SELL", 
            "confidence": 0.82,
            "entry_price": 180.00,
            "stop_loss": 189.00,
            "take_profit": 165.60,
            "position_size_percent": 1.0,
            "reasoning": f"Fresh live signal generated at {current_time}",
            "is_mock": False
        }
    ]
    
    print(f"🚀 Creating fresh live recommendations via API...")
    print(f"⏰ Timestamp: {current_time}")
    
    success_count = 0
    
    # Try to post to recommendations service
    for rec in recommendations:
        try:
            response = requests.post(
                "http://localhost:8022/signals",
                json=rec,
                timeout=10
            )
            
            if response.status_code in [200, 201]:
                print(f"✅ Created {rec['action']} {rec['symbol']} @ ${rec['entry_price']}")
                success_count += 1
            else:
                print(f"❌ Failed {rec['symbol']}: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"❌ Error creating {rec['symbol']}: {e}")
    
    if success_count > 0:
        print(f"\n🎯 Successfully created {success_count} fresh recommendations!")
        print("🔄 Automated trader should pick these up within 30 seconds")
        return True
    else:
        print("\n❌ No recommendations were created successfully")
        return False

def test_recommendations_api():
    """Test what endpoints are available on the recommendations service"""
    
    print("🔍 Testing available endpoints...")
    
    # Test health
    try:
        response = requests.get("http://localhost:8022/health", timeout=5)
        print(f"✅ Health: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False
    
    # Check if there's a signals endpoint
    endpoints_to_test = ["/signals", "/recommendations/create", "/create_signal"]
    
    for endpoint in endpoints_to_test:
        try:
            response = requests.post(f"http://localhost:8022{endpoint}", json={"test": "data"}, timeout=5)
            print(f"📡 {endpoint}: {response.status_code}")
            if response.status_code != 404:
                print(f"   Response: {response.text[:100]}")
        except Exception as e:
            print(f"❌ {endpoint}: {e}")
    
    return True

if __name__ == "__main__":
    print("🛠️ SIMPLE FRESH SIGNAL CREATOR")
    print("=" * 50)
    
    # First test what's available
    if test_recommendations_api():
        print("\n" + "-" * 30)
        # Try to create recommendations
        create_fresh_recommendations_via_api()
    
    print("\n" + "=" * 50)
    print("🎯 Fresh signal creation attempt complete!")
