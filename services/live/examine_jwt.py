#!/usr/bin/env python3
"""
Extract JWT creation logic from official SDK
"""

import os
from dotenv import load_dotenv

# Load environment
load_dotenv('e:/git/aitest/.env.live')

def examine_jwt_creation():
    """Examine how the official SDK creates JWT tokens"""
    try:
        from coinbase import jwt_generator
        
        api_key = os.getenv('COINBASE_API_KEY')
        api_secret = os.getenv('COINBASE_PRIVATE_KEY')
        
        # Test REST JWT creation
        uri = "/api/v3/brokerage/accounts"
        jwt_uri = jwt_generator.format_jwt_uri("GET", uri)
        print(f"Formatted JWT URI: {jwt_uri}")
        
        jwt_token = jwt_generator.build_rest_jwt(jwt_uri, api_key, api_secret)
        print(f"Generated JWT (first 50 chars): {jwt_token[:50]}...")
        
        # Test what the format_jwt_uri function does
        print(f"Original URI: {uri}")
        print(f"Formatted URI: {jwt_uri}")
        
        return jwt_token
        
    except Exception as e:
        print(f"[!] JWT examination failed: {e}")
        return None

if __name__ == "__main__":
    examine_jwt_creation()
