#!/usr/bin/env python3
"""
Simple test script to verify containerized services are working
"""
import requests
import json

def test_service(name, url, headers=None):
    """Test if a service endpoint is responding"""
    try:
        response = requests.get(url, headers=headers, timeout=5)
        print(f"✓ {name}: HTTP {response.status_code}")
        if response.status_code == 200:
            try:
                data = response.json()
                print(f"  Response: {json.dumps(data, indent=2)[:100]}...")
            except:
                print(f"  Response: {response.text[:100]}...")
        return response.status_code == 200
    except Exception as e:
        print(f"✗ {name}: ERROR - {e}")
        return False

def main():
    print("Testing containerized trading services...")
    print("=" * 50)
    
    # Test headers
    headers = {"X-TRADING-API-KEY": "test-api-key-12345"}
    
    services = [
        ("Risk Service Health", "http://localhost:8025/health", None),
        ("Risk Service Metrics", "http://localhost:8025/metrics", None),
        ("Signals Service Health", "http://localhost:8028/health", headers),
        ("Signals Service Metrics", "http://localhost:8028/metrics", headers),
    ]
    
    results = []
    for name, url, test_headers in services:
        result = test_service(name, url, test_headers or headers)
        results.append((name, result))
    
    print("\nSummary:")
    print("=" * 50)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")

if __name__ == "__main__":
    main()
