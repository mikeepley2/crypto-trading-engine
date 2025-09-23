#!/usr/bin/env python3
"""
LLM Fallback Monitor Health Check
Simple script to verify the LLM fallback system can be imported and is working.
"""

import sys
import traceback

def main():
    print("🔍 LLM Fallback System Health Check")
    print("=" * 40)
    
    try:
        # Test import
        from backend.shared.llm_fallback_manager import LLMFallbackManager
        print("✅ LLM Fallback Manager import successful")
        
        # Test instantiation
        manager = LLMFallbackManager()
        print("✅ LLM Fallback Manager instantiation successful")
        
        # Test providers availability
        from backend.shared.llm_fallback_manager import LLMProvider
        available_providers = list(LLMProvider)
        print(f"✅ Available providers: {[p.value for p in available_providers]}")
        
        print("\n🎉 Health check PASSED")
        return 0
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Check that PYTHONPATH includes /app and backend/ directory exists")
        traceback.print_exc()
        return 1
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
