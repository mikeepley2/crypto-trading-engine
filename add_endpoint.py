#!/usr/bin/env python3
"""
Add the missing process_recommendation endpoint to the existing trade executor
"""

import os
import mysql.connector
from fastapi import FastAPI, HTTPException

# Create a new FastAPI app instance
app = FastAPI()

def get_db_connection():
    return mysql.connector.connect(
        host=os.getenv('DB_HOST', '172.22.32.1'),
        user=os.getenv('DB_USER', 'news_collector'),
        password=os.getenv('DB_PASSWORD'),
        database='crypto_prices'
    )

@app.post('/process_recommendation/{recommendation_id}')
async def process_recommendation(recommendation_id: int):
    try:
        # Get recommendation from database
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute('SELECT * FROM trade_recommendations WHERE id = %s', (recommendation_id,))
        rec = cursor.fetchone()
        
        if not rec:
            raise HTTPException(status_code=404, detail='Recommendation not found')
        
        # Update status to EXECUTED (mock execution for now)
        cursor.execute('UPDATE trade_recommendations SET execution_status = %s, executed_at = NOW() WHERE id = %s', ('EXECUTED', recommendation_id))
        conn.commit()
        
        cursor.close()
        conn.close()
        
        return {
            'status': 'success',
            'message': f'Recommendation {recommendation_id} processed successfully',
            'recommendation': {
                'id': rec['id'],
                'symbol': rec['symbol'],
                'signal_type': rec['signal_type'],
                'amount_usd': float(rec['amount_usd']),
                'confidence': float(rec['confidence'])
            }
        }
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8025)
