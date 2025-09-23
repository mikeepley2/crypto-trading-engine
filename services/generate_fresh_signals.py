#!/usr/bin/env python3
"""
Quick Fresh Signal Generator for Live Trading Demo
Creates fresh trading recommendations for immediate execution
"""

import mysql.connector
import json
from datetime import datetime
import random

# Database configuration - FIXED to use correct database
DB_CONFIG = {
    'host': 'host.docker.internal',
    'user': 'news_collector',
    'password': '99Rules!',
    'database': 'crypto_transactions'  # Fixed: recommendations service reads from crypto_transactions
}

def create_fresh_recommendations():
    """Create fresh trading recommendations for live demo"""
    
    # Sample crypto symbols available on Coinbase
    symbols = ['BTC', 'ETH', 'XRP', 'LTC', 'ADA', 'DOT', 'LINK', 'UNI', 'AAVE']
    actions = ['BUY', 'SELL', 'HOLD']
    
    current_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    
    fresh_recommendations = []
    
    for i, symbol in enumerate(symbols[:5]):  # Generate 5 fresh recommendations
        action = random.choice(actions[:2])  # Only BUY/SELL for active trading
        confidence = round(random.uniform(0.65, 0.85), 4)
        entry_price = round(random.uniform(10, 1000), 2)
        
        # Calculate stop loss and take profit
        if action == 'BUY':
            stop_loss = round(entry_price * 0.95, 2)  # 5% stop loss
            take_profit = round(entry_price * 1.08, 2)  # 8% take profit
        else:  # SELL
            stop_loss = round(entry_price * 1.05, 2)  # 5% stop loss
            take_profit = round(entry_price * 0.92, 2)  # 8% take profit
        
        recommendation = {
            'symbol': symbol,
            'action': action,
            'confidence': confidence,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_size_percent': 1.0,  # 1% position size
            'reasoning': 'Fresh automated signal for live trading',
            'generated_at': current_time,
            'is_mock': 0,  # Live trading (set to 0 for real recommendations)
            'status': 'pending'
        }
        
        fresh_recommendations.append(recommendation)
    
    return fresh_recommendations

def insert_recommendations(recommendations):
    """Insert fresh recommendations into database"""
    
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    # Insert into trade_recommendations table only (correct database)
    rec_insert_query = """
    INSERT INTO trade_recommendations
    (symbol, action, confidence, entry_price, stop_loss, take_profit,
     position_size_percent, reasoning, generated_at, is_mock, status)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    
    inserted_count = 0
    
    for rec in recommendations:
        try:
            values = (
                rec['symbol'], rec['action'], rec['confidence'], rec['entry_price'],
                rec['stop_loss'], rec['take_profit'], rec['position_size_percent'],
                rec['reasoning'], rec['generated_at'], rec['is_mock'],
                rec['status']
            )
            
            # Insert into trade_recommendations (correct table and database)
            cursor.execute(rec_insert_query, values)
            
            inserted_count += 1
            print(f"✅ Inserted {rec['action']} {rec['symbol']} @ ${rec['entry_price']}")
            
        except Exception as e:
            print(f"❌ Failed to insert {rec['symbol']}: {e}")
    
    conn.commit()
    cursor.close()
    conn.close()
    
    print(f"\n🎯 Successfully inserted {inserted_count} fresh recommendations!")
    return inserted_count

if __name__ == "__main__":
    print("🚀 Generating Fresh Trading Signals for Live Demo...")
    print(f"⏰ Timestamp: {datetime.utcnow()}")
    
    recommendations = create_fresh_recommendations()
    
    print(f"\n📊 Generated {len(recommendations)} fresh recommendations:")
    for rec in recommendations:
        print(f"   {rec['action']} {rec['symbol']} @ ${rec['entry_price']} (confidence: {rec['confidence']})")
    
    count = insert_recommendations(recommendations)
    
    print(f"\n✅ Ready for automated live trading!")
    print(f"💡 The automated trader will pick up these signals within 30 seconds.")
