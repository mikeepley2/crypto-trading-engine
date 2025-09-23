#!/usr/bin/env python3
"""
Check current Coinbase positions using our custom API
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from coinbase_api import CoinbaseAdvancedTradeAPI
from dotenv import load_dotenv

def main():
    # Load environment variables from root directory
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '.env.live')
    env_path = os.path.abspath(env_path)
    load_dotenv(env_path)
    
    print(f"Loading environment from: {env_path}")
    print("=== Coinbase Account Analysis (Custom API) ===")
    
    # Initialize Coinbase API
    try:
        api_key = os.getenv('COINBASE_API_KEY')
        private_key = os.getenv('COINBASE_PRIVATE_KEY')
        
        print(f"API Key found: {'Yes' if api_key else 'No'}")
        print(f"Private Key found: {'Yes' if private_key else 'No'}")
        
        if not api_key or not private_key:
            print("Missing API credentials - checking .env.live file")
            return {}
            
        coinbase = CoinbaseAdvancedTradeAPI(api_key, private_key)
        
        # Get all accounts
        accounts = coinbase.get_accounts()
        print(f"Total accounts found: {len(accounts)}")
        
        # Track positions and values
        positions = {}
        total_usd_value = 0.0
        
        for account in accounts:
            currency = account.get('currency', 'Unknown')
            balance = float(account.get('available_balance', {}).get('value', 0))
            
            print(f"\nAccount: {currency}")
            print(f"  Available Balance: {balance}")
            print(f"  Account Details: {account}")
            
            if balance > 0.0001:  # Check for very small balances too
                print(f"*** POSITION FOUND: {currency}: {balance} ***")
                
                # Get current price in USD if not USD
                if currency != 'USD':
                    try:
                        # Get current market price
                        price_data = coinbase.get_current_price(f"{currency}-USD")
                        if price_data and 'amount' in price_data:
                            usd_price = float(price_data['amount'])
                            usd_value = balance * usd_price
                            positions[currency] = {
                                'balance': balance,
                                'usd_price': usd_price,
                                'usd_value': usd_value
                            }
                            total_usd_value += usd_value
                            print(f"  Price: ${usd_price:.2f}")
                            print(f"  USD Value: ${usd_value:.2f}")
                        else:
                            print(f"  Unable to get price for {currency}")
                    except Exception as e:
                        print(f"  Error getting price for {currency}: {e}")
                else:
                    positions[currency] = {
                        'balance': balance,
                        'usd_price': 1.0,
                        'usd_value': balance
                    }
                    total_usd_value += balance
        
        print(f"\n=== Portfolio Summary ===")
        print(f"Total Portfolio Value: ${total_usd_value:.2f}")
        
        if positions:
            print(f"\nPositions by value:")
            sorted_positions = sorted(positions.items(), key=lambda x: x[1]['usd_value'], reverse=True)
            for currency, data in sorted_positions:
                print(f"  {currency}: {data['balance']:.6f} = ${data['usd_value']:.2f}")
            
            # Save to file for configuration
            with open('current_positions.txt', 'w') as f:
                f.write(f"Total Portfolio Value: ${total_usd_value:.2f}\n")
                f.write("Positions:\n")
                for currency, data in sorted_positions:
                    f.write(f"{currency}: {data['balance']:.6f} = ${data['usd_value']:.2f}\n")
            
            return positions
        else:
            print("No positions found")
            return {}
            
    except Exception as e:
        print(f"Error checking positions: {e}")
        import traceback
        traceback.print_exc()
        return {}

if __name__ == "__main__":
    main()
