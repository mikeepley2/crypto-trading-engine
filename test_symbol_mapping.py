#!/usr/bin/env python3

import requests
import json

def test_symbol_mapping():
    """Test the symbol mapping functionality"""
    
    # Test data for different symbols
    test_cases = [
        {"symbol": "ETH", "side": "SELL", "amount_usd": 30.0},
        {"symbol": "BTC", "side": "SELL", "amount_usd": 30.0},
        {"symbol": "ADA", "side": "SELL", "amount_usd": 30.0},
        {"symbol": "LINK", "side": "SELL", "amount_usd": 30.0},
        {"symbol": "DOT", "side": "SELL", "amount_usd": 30.0}
    ]
    
    print("=== TESTING SYMBOL MAPPING ===")
    print("Testing trade executor with updated symbol mapping...")
    print()
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test {i}: {test_case['symbol']} {test_case['side']} ${test_case['amount_usd']}")
        
        try:
            # Send request to trade executor
            response = requests.post(
                "http://localhost:8024/execute-trade",
                json=test_case,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    print(f"  ✅ SUCCESS: {result.get('message', 'Trade executed')}")
                else:
                    print(f"  ❌ FAILED: {result.get('error', 'Unknown error')}")
            else:
                print(f"  ❌ HTTP ERROR: {response.status_code} - {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"  ❌ CONNECTION ERROR: {e}")
        
        print()
    
    print("=== SYMBOL MAPPING TEST COMPLETE ===")

if __name__ == '__main__':
    test_symbol_mapping()
