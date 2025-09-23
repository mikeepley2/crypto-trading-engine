#!/usr/bin/env python3
"""
Apply multi-platform endpoints to running trading engine
This script modifies the running FastAPI application to add multi-platform support
"""

import sys
import os
import importlib.util

def apply_multi_platform_patch():
    """Apply the multi-platform patch to the running application"""
    
    print("🔧 Applying multi-platform patch to running trading engine...")
    
    try:
        # Import the running application
        sys.path.insert(0, '/app')
        
        # Import the trade execution engine module
        spec = importlib.util.spec_from_file_location("trade_execution_engine", "/app/trade_execution_engine.py")
        trade_module = importlib.util.module_from_spec(spec)
        
        # Get the FastAPI app instance from the module
        # The app should be defined as 'app' in the trade_execution_engine.py
        app = None
        if hasattr(trade_module, 'app'):
            app = trade_module.app
        else:
            # Try to find the app in the global namespace
            import trade_execution_engine
            if hasattr(trade_execution_engine, 'app'):
                app = trade_execution_engine.app
        
        if not app:
            print("❌ Could not find FastAPI app instance")
            return False
        
        # Import and apply the multi-platform endpoints
        from add_multi_platform_endpoints import add_multi_platform_endpoints
        
        success = add_multi_platform_endpoints(app)
        
        if success:
            print("✅ Multi-platform endpoints added successfully!")
            print("📋 New endpoints available:")
            print("   GET  /platforms - Platform status")
            print("   GET  /platforms/portfolios - Platform portfolios")
            print("   GET  /platforms/symbols - Available symbols")
            print("   POST /platforms/execute_trade - Multi-platform trading")
            print("   GET  /platforms/config - Platform configuration")
            print("   POST /platforms/config - Update configuration")
            print("   GET  /platforms/health - Platform health check")
            return True
        else:
            print("❌ Failed to add multi-platform endpoints")
            return False
            
    except Exception as e:
        print(f"❌ Error applying multi-platform patch: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = apply_multi_platform_patch()
    sys.exit(0 if success else 1)
