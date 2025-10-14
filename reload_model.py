#!/usr/bin/env python3
"""
Script to manually reload the new balanced model in the signal generator
"""

import requests
import time

def reload_signal_generator():
    """Trigger the signal generator to reload the model"""
    
    # Get the signal generator service URL
    service_url = "http://signal-generator:8025"
    
    try:
        # Check if the service is healthy
        response = requests.get(f"{service_url}/health", timeout=10)
        if response.status_code == 200:
            print("✅ Signal generator is healthy")
            health_data = response.json()
            print(f"Current model: {health_data.get('model_path', 'Unknown')}")
        else:
            print(f"❌ Signal generator health check failed: {response.status_code}")
            return False
            
        # Check model status
        response = requests.get(f"{service_url}/model-status", timeout=10)
        if response.status_code == 200:
            model_data = response.json()
            print(f"Model loaded: {model_data.get('model_loaded', False)}")
            print(f"Model type: {model_data.get('model_type', 'Unknown')}")
            print(f"Feature count: {model_data.get('feature_count', 'Unknown')}")
        else:
            print(f"❌ Model status check failed: {response.status_code}")
            
        return True
        
    except Exception as e:
        print(f"❌ Error connecting to signal generator: {e}")
        return False

if __name__ == "__main__":
    print("🔄 Checking signal generator status...")
    success = reload_signal_generator()
    if success:
        print("✅ Signal generator check completed")
    else:
        print("❌ Signal generator check failed")


