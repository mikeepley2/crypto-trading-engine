#!/usr/bin/env python3
"""
Check current Coinbase positions
"""

import os
from dotenv import load_dotenv

# Load environment
load_dotenv('e:/git/aitest/.env.live')

def check_current_positions():
    """Check current Coinbase positions"""
    try:
        from coinbase.rest import RESTClient
        
        client = RESTClient(
            api_key=os.getenv('COINBASE_API_KEY'),
            api_secret=os.getenv('COINBASE_PRIVATE_KEY')
        )
        
        print("=== Your Current Coinbase Positions ===")
        accounts = client.get_accounts()
        
        positions_with_value = []
        total_value = 0
        
        for account in accounts.accounts:
            try:
                balance = float(account.available_balance.value)
                if balance > 0:
                    # Get current price for non-USD assets
                    if account.currency != 'USD':
                        try:
                            product_id = f"{account.currency}-USD"
                            product = client.get_product(product_id)
                            price = float(product.price)
                            value_usd = balance * price
                        except:
                            price = 0
                            value_usd = 0
                    else:
                        price = 1.0
                        value_usd = balance
                    
                    positions_with_value.append({
                        'currency': account.currency,
                        'balance': balance,
                        'price': price,
                        'value_usd': value_usd
                    })
                    total_value += value_usd
            except:
                continue
        
        if positions_with_value:
            print(f"\nFound {len(positions_with_value)} positions:")
            for pos in positions_with_value:
                print(f"  {pos['currency']}: {pos['balance']:.6f} @ ${pos['price']:.2f} = ${pos['value_usd']:.2f}")
            print(f"\nTotal Portfolio Value: ${total_value:.2f}")
            
            # Extract tradeable symbols
            tradeable_symbols = [pos['currency'] for pos in positions_with_value if pos['currency'] != 'USD']
            print(f"\nTradeable symbols: {tradeable_symbols}")
            
            return positions_with_value, tradeable_symbols
        else:
            print("No positions with available balance found")
            return [], []
            
    except Exception as e:
        print(f"Error checking positions: {e}")
        return [], []

if __name__ == "__main__":
    check_current_positions()
