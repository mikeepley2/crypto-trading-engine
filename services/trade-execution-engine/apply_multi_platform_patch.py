#!/usr/bin/env python3
"""
Patch application script to add multi-platform endpoints to running trading engine
"""

import os
import sys

def apply_multi_platform_patch():
    """Apply multi-platform endpoints to the running trade execution engine"""
    
    print("🔧 Applying multi-platform patch to trade execution engine...")
    
    try:
        # Read the multi-platform endpoints code
        with open('/app/multi_platform_endpoints.py', 'r') as f:
            endpoints_code = f.read()
        
        # Read the current trade execution engine
        with open('/app/trade_execution_engine.py', 'r') as f:
            current_code = f.read()
        
        # Check if already patched
        if "get_platforms_status" in current_code:
            print("✅ Multi-platform endpoints already present")
            return True
        
        # Find the insertion point (before the main execution block)
        insertion_point = current_code.find('if __name__ == "__main__":')
        
        if insertion_point == -1:
            print("❌ Could not find insertion point in trade execution engine")
            return False
        
        # Insert the endpoints code before the main block
        patched_code = (
            current_code[:insertion_point] + 
            "\n# =========================================================================\n" +
            "# MULTI-PLATFORM TRADING ENDPOINTS\n" +
            "# =========================================================================\n\n" +
            endpoints_code + "\n\n" +
            current_code[insertion_point:]
        )
        
        # Write the patched file
        with open('/app/trade_execution_engine.py', 'w') as f:
            f.write(patched_code)
        
        print("✅ Multi-platform endpoints successfully added to trade execution engine!")
        print("📋 New endpoints available:")
        print("   GET  /platforms - Platform status")
        print("   GET  /platforms/health - Platform health check") 
        print("   GET  /platforms/portfolios - Platform portfolios")
        print("   GET  /platforms/config - Platform configuration")
        print("   POST /platforms/config - Update configuration")
        print("   POST /platforms/execute_trade - Multi-platform trading")
        print("   GET  /platforms/symbols - Available symbols")
        print("\n⚠️  Note: Restart the container to activate the new endpoints")
        
        return True
        
    except Exception as e:
        print(f"❌ Error applying multi-platform patch: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = apply_multi_platform_patch()
    sys.exit(0 if success else 1)
