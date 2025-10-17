import mysql.connector
import os
from datetime import datetime, timedelta

DB_HOST = os.getenv('DB_HOST', '172.22.32.1')
DB_USER = os.getenv('DB_USER', 'news_collector')
DB_PASSWORD = os.getenv('DB_PASSWORD', '99Rules!')
DB_NAME_PRICES = os.getenv('DB_NAME_PRICES', 'crypto_prices')

def get_db_connection():
    try:
        return mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME_PRICES
        )
    except Exception as e:
        print(f'Database connection error: {e}')
        return None

def check_llm_validation_results():
    conn = get_db_connection()
    if not conn:
        return

    cursor = conn.cursor(dictionary=True)

    print('=== RECENT LLM VALIDATION RESULTS ===')
    print(f'Time: {datetime.now()}')
    print()

    # Get recent recommendations with LLM validation details
    cursor.execute("""
        SELECT id, symbol, signal_type, confidence, execution_status, 
               llm_validation, llm_confidence, llm_reasoning, risk_assessment,
               validation_timestamp, created_at
        FROM trade_recommendations 
        WHERE created_at >= DATE_SUB(NOW(), INTERVAL 2 HOUR) 
        ORDER BY created_at DESC 
        LIMIT 15
    """)
    
    recommendations = cursor.fetchall()
    print(f'Recent recommendations (last 2 hours): {len(recommendations)}')
    print()

    for rec in recommendations:
        print(f"ID: {rec['id']}")
        print(f"  Symbol: {rec['symbol']}, Type: {rec['signal_type']}, Confidence: {rec['confidence']:.4f}")
        print(f"  Status: {rec['execution_status']}")
        print(f"  LLM Validated: {rec['llm_validation']}")
        print(f"  LLM Confidence: {rec['llm_confidence']}")
        print(f"  LLM Reasoning: {rec['llm_reasoning']}")
        print(f"  Risk Assessment: {rec['risk_assessment']}")
        print(f"  Validation Time: {rec['validation_timestamp']}")
        print(f"  Created: {rec['created_at'].strftime('%Y-%m-%d %H:%M:%S')}")
        print()

    # LLM validation statistics
    cursor.execute("""
        SELECT 
            COUNT(*) as total_recommendations,
            SUM(CASE WHEN llm_validation = 1 THEN 1 ELSE 0 END) as validated,
            SUM(CASE WHEN llm_validation = 0 THEN 1 ELSE 0 END) as rejected,
            AVG(llm_confidence) as avg_llm_confidence
        FROM trade_recommendations 
        WHERE created_at >= DATE_SUB(NOW(), INTERVAL 2 HOUR)
        AND llm_validation IS NOT NULL
    """)
    
    stats = cursor.fetchone()
    print('=== LLM VALIDATION STATISTICS (Last 2 Hours) ===')
    print(f'Total Recommendations: {stats["total_recommendations"]}')
    print(f'Validated: {stats["validated"]}')
    print(f'Rejected: {stats["rejected"]}')
    print(f'Average LLM Confidence: {stats["avg_llm_confidence"]:.3f}' if stats["avg_llm_confidence"] else 'N/A')
    print()

    cursor.close()
    conn.close()

if __name__ == '__main__':
    check_llm_validation_results()
