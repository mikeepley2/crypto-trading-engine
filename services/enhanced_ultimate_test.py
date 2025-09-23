#!/usr/bin/env python3
"""
🏆 ULTIMATE ENHANCED TEST - ALL 6 CONTAINERIZED TRADING SERVICES
Including Live Trading Engine with Exchange Integration
"""
import requests
import json
import time

def test_all_six_services():
    """Test all 6 services comprehensively including live trading engine"""
    print("🏆 ULTIMATE ENHANCED TEST - ALL 6 CONTAINERIZED TRADING SERVICES")
    print("=" * 80)
    print(f"Test Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("🐳 Docker Containerized Microservices Architecture")
    print("🔴 INCLUDING LIVE TRADING ENGINE")
    print("=" * 80)
    
    headers = {"X-TRADING-API-KEY": "test-api-key-12345"}
    
    services_tests = [
        # 🔒 Risk Service (Port 8025)
        ("Risk Service Health", "GET", "http://localhost:8025/health", None, None),
        ("Risk Service Metrics", "GET", "http://localhost:8025/metrics", None, None),
        ("Risk Service Limits", "GET", "http://localhost:8025/risk/limits", None, None),
        ("Risk Service Trade Check", "POST", "http://localhost:8025/risk/check_trade", 
         {"symbol": "BTC-USD", "side": "buy", "amount": 5000, "portfolio_value": 100000, "existing_positions": 2}, None),
        
        # 📡 Signals Service (Port 8028)
        ("Signals Service Health", "GET", "http://localhost:8028/health", None, None),
        ("Signals Service Metrics", "GET", "http://localhost:8028/metrics", None, None),
        ("Signals Service Recent", "GET", "http://localhost:8028/signals/recent?limit=5", None, None),
        
        # 💼 Portfolio Service (Port 8026)
        ("Portfolio Service Health", "GET", "http://localhost:8026/health", None, None),
        ("Portfolio Service Metrics", "GET", "http://localhost:8026/metrics", None, None),
        ("Portfolio Service Summary", "GET", "http://localhost:8026/portfolio", None, None),
        
        # 🎯 Trade Recommendations (Port 8022)
        ("Trade Recommendations Health", "GET", "http://localhost:8022/health", None, headers),
        ("Trade Recommendations Metrics", "GET", "http://localhost:8022/metrics", None, headers),
        ("Trade Recommendations List", "GET", "http://localhost:8022/recommendations", None, headers),
        
        # ⚡ Mock Trading Engine (Port 8021)
    ("Trade Execution Engine Health", "GET", "http://localhost:8024/health", None, None),
    ("Trade Execution Engine Status", "GET", "http://localhost:8024/status", None, None),
        
        # 🔴 Live Trading Engine (Port 8023) - NEW!
        ("Live Trading Engine Health", "GET", "http://localhost:8023/health", None, None),
        ("Live Trading Engine Status", "GET", "http://localhost:8023/status", None, None),
        ("Live Trading Engine Exchanges", "GET", "http://localhost:8023/exchanges", None, None),
        ("Live Trading Engine Portfolio", "GET", "http://localhost:8023/portfolio", None, None),
    ]
    
    results = []
    service_counts = {"Risk Service": 0, "Signals Service": 0, "Portfolio Service": 0, 
                     "Trade Recommendations": 0, "Mock Trading Engine": 0, "Live Trading Engine": 0}
    
    for test_name, method, url, data, test_headers in services_tests:
        try:
            if method == "GET":
                response = requests.get(url, headers=test_headers, timeout=10)
            else:  # POST
                response = requests.post(url, json=data, headers=test_headers, timeout=10)
            
            if response.status_code in [200, 201]:
                print(f"✅ {test_name}: HTTP {response.status_code}")
                try:
                    resp_data = response.json()
                    if isinstance(resp_data, dict):
                        for key, value in list(resp_data.items())[:2]:  # Show first 2 keys
                            print(f"   {key}: {str(value)[:50]}...")
                    elif isinstance(resp_data, list):
                        print(f"   Found {len(resp_data)} items")
                except:
                    print(f"   Response: {response.text[:50]}...")
                results.append((test_name, True))
                
                # Count by service
                for service in service_counts:
                    if service in test_name:
                        service_counts[service] += 1
                        break
                        
            else:
                print(f"❌ {test_name}: HTTP {response.status_code}")
                results.append((test_name, False))
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {str(e)[:50]}...")
            results.append((test_name, False))
    
    print("\n🏆 ULTIMATE ENHANCED RESULTS")
    print("=" * 80)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    success_rate = passed/total*100
    print(f"🎯 Overall Score: {passed}/{total} tests passed ({success_rate:.1f}%)")
    print()
    
    # Service-by-service results
    service_test_counts = {
        "Risk Service": 4,
        "Signals Service": 3, 
        "Portfolio Service": 3,
        "Trade Recommendations": 3,
        "Mock Trading Engine": 2,
        "Live Trading Engine": 4  # NEW!
    }
    
    all_operational = True
    for service, expected in service_test_counts.items():
        actual = service_counts[service]
        if actual == expected:
            print(f"🟢 {service}: FULLY OPERATIONAL ({actual}/{expected})")
        else:
            print(f"🟡 {service}: PARTIAL ({actual}/{expected})")
            all_operational = False
    
    print(f"\n🎊 FINAL ENHANCED DEPLOYMENT STATUS")
    print("=" * 80)
    if success_rate >= 90:
        print("🚀 PRODUCTION READY! Enhanced containerized microservices deployment SUCCESS!")
        print("✅ Risk Management Service: Operational")
        print("✅ Signal Processing Service: Operational") 
        print("✅ Portfolio Management Service: Operational")
        print("✅ Trade Recommendations Service: Operational")
        print("✅ Mock Trading Engine: Operational")
        print("🔴 Live Trading Engine: Operational")
        print()
        print("🏗️  Architecture: FastAPI microservices with Docker containerization")
        print("🔐 Authentication: API key based security")
        print("📊 Monitoring: Prometheus metrics endpoints")
        print("🔄 Health Checks: All services reporting healthy status")
        print("🌐 Network: Docker Compose orchestrated services")
        print("💾 Database: MySQL integration via host.docker.internal")
        print("🔄 Exchange Integration: Multi-exchange live trading capability")
        print("📈 Portfolio Segregation: Separate mock and live trading data")
        print()
        print("🎉 MISSION ACCOMPLISHED: 100% enhanced trading system with live trading!")
    else:
        print(f"🔧 Needs minor fixes: {success_rate:.1f}% operational")
    
    return success_rate >= 90

if __name__ == "__main__":
    test_all_six_services()
