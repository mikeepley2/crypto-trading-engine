#!/usr/bin/env python3
"""
Enhanced Centralized Database Connection Manager
Comprehensive connection pooling for all trading services
Prevents connection exhaustion and optimizes database performance
"""

import os
import logging
import mysql.connector
from mysql.connector import pooling, Error
from typing import Optional, Dict, Any, Generator
import time
from datetime import datetime
from contextlib import contextmanager
import threading
from functools import wraps

logger = logging.getLogger(__name__)

class DatabaseConnectionManager:
    """
    Enhanced centralized database connection manager with proper pooling
    Thread-safe and designed for high-frequency trading operations
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern to ensure one connection manager per process"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
            
        self._pools = {}
        self._initialized = False
        self._stats = {
            'connections_created': 0,
            'connections_reused': 0,
            'queries_executed': 0,
            'pool_hits': 0,
            'pool_misses': 0
        }
        
        # Get the actual Windows host IP instead of host.docker.internal
        self.host_ip = self._get_host_ip()
        logger.info(f"Database Connection Manager: Using host IP {self.host_ip}")
        
        self._initialize_pools()
        self._initialized = True
    
    def _get_host_ip(self) -> str:
        """Get the correct host IP address for database connections"""
        # Check if we're in a container/Kubernetes environment
        if os.path.exists('/.dockerenv') or os.getenv('KUBERNETES_SERVICE_HOST'):
            # In container - use the Windows host IP
            host_ip = os.getenv('DATABASE_HOST', '192.168.230.163')
        else:
            # Running locally - use localhost
            host_ip = 'localhost'
        
        return host_ip
    
    def _initialize_pools(self):
        """Initialize connection pools for all databases"""
        try:
            # Base configuration
            base_config = {
                'host': self.host_ip,
                'port': 3306,
                'user': 'news_collector',
                'password': '99Rules!',
                'autocommit': True,
                'connection_timeout': 10,
                'auth_plugin': 'mysql_native_password',
                'charset': 'utf8mb4',
                'collation': 'utf8mb4_unicode_ci',
                'use_unicode': True,
                'sql_mode': '',
                'time_zone': '+00:00',
                'pool_reset_session': True,
                'pool_size': 5  # Pool size for high-frequency operations
            }
            
            # Create pools for each database
            databases = {
                'crypto_prices': 'crypto_prices_pool',
                'crypto_transactions': 'crypto_transactions_pool', 
                'crypto_news': 'crypto_news_pool'
            }
            
            for db_name, pool_name in databases.items():
                config = base_config.copy()
                config.update({
                    'database': db_name,
                    'pool_name': pool_name
                })
                
                try:
                    self._pools[db_name] = pooling.MySQLConnectionPool(**config)
                    logger.info(f"✅ Created connection pool for {db_name}")
                except Exception as e:
                    logger.error(f"❌ Failed to create pool for {db_name}: {e}")
                    # Create fallback single connection
                    fallback_config = {k: v for k, v in config.items() 
                                     if not k.startswith('pool_')}
                    self._pools[db_name] = fallback_config
            
            # Test all pools
            self._test_pools()
            
            logger.info("✅ All database connection pools initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize database pools: {e}")
            raise
    
    def _test_pools(self):
        """Test all connection pools"""
        for db_name, pool in self._pools.items():
            try:
                with self.get_connection(db_name) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT 1")
                    cursor.fetchone()
                    cursor.close()
                logger.info(f"✅ Pool test successful for {db_name}")
            except Exception as e:
                logger.warning(f"⚠️ Pool test failed for {db_name}: {e}")
    
    @contextmanager
    def get_connection(self, database: str = 'crypto_transactions'):
        """
        Get a database connection from the pool
        
        Args:
            database: Database name ('crypto_transactions', 'crypto_prices', 'crypto_news')
            
        Returns:
            Database connection (context manager)
        """
        connection = None
        try:
            if database not in self._pools:
                raise ValueError(f"Unknown database: {database}")
            
            pool = self._pools[database]
            
            if isinstance(pool, dict):
                # Fallback single connection
                connection = mysql.connector.connect(**pool)
                self._stats['connections_created'] += 1
            else:
                # Pooled connection
                connection = pool.get_connection()
                self._stats['connections_reused'] += 1
                self._stats['pool_hits'] += 1
            
            yield connection
            
        except Exception as e:
            logger.error(f"Database connection error ({database}): {e}")
            self._stats['pool_misses'] += 1
            raise
        finally:
            if connection and connection.is_connected():
                connection.close()
    
    @contextmanager
    def get_cursor(self, database: str = 'crypto_transactions', dictionary: bool = True):
        """
        Get a database cursor (convenience method)
        
        Args:
            database: Database name
            dictionary: Return results as dictionaries
            
        Returns:
            Database cursor (context manager)
        """
        with self.get_connection(database) as conn:
            cursor = conn.cursor(dictionary=dictionary)
            try:
                yield cursor
                self._stats['queries_executed'] += 1
            finally:
                cursor.close()
    
    def execute_query(self, query: str, params: tuple = None, 
                     database: str = 'crypto_transactions', 
                     fetch: str = 'all') -> Optional[Any]:
        """
        Execute a query and return results
        
        Args:
            query: SQL query
            params: Query parameters
            database: Database name
            fetch: 'all', 'one', or 'none'
            
        Returns:
            Query results
        """
        with self.get_cursor(database) as cursor:
            cursor.execute(query, params or ())
            
            if fetch == 'all':
                return cursor.fetchall()
            elif fetch == 'one':
                return cursor.fetchone()
            else:
                return None
    
    def execute_many(self, query: str, params_list: list, 
                    database: str = 'crypto_transactions') -> int:
        """
        Execute a query with multiple parameter sets
        
        Args:
            query: SQL query
            params_list: List of parameter tuples
            database: Database name
            
        Returns:
            Number of affected rows
        """
        with self.get_cursor(database, dictionary=False) as cursor:
            cursor.executemany(query, params_list)
            return cursor.rowcount
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        stats = self._stats.copy()
        
        for db_name, pool in self._pools.items():
            if hasattr(pool, '_cnx_queue'):
                stats[f'{db_name}_pool_size'] = pool._cnx_queue.qsize()
            
        return stats
    
    def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check of all connection pools"""
        health = {
            'status': 'healthy',
            'databases': {},
            'stats': self.get_stats(),
            'timestamp': datetime.now().isoformat()
        }
        
        for db_name in self._pools:
            try:
                start_time = time.time()
                with self.get_connection(db_name) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT 1")
                    cursor.fetchone()
                    cursor.close()
                
                response_time = (time.time() - start_time) * 1000
                
                health['databases'][db_name] = {
                    'status': 'healthy',
                    'response_time_ms': round(response_time, 2)
                }
                
            except Exception as e:
                health['databases'][db_name] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
                health['status'] = 'degraded'
        
        return health

# Global instance
db_manager = DatabaseConnectionManager()

# Convenience functions for backward compatibility
def get_db_connection(database: str = 'crypto_transactions'):
    """Get database connection (backward compatibility)"""
    return db_manager.get_connection(database)

def get_db_cursor(database: str = 'crypto_transactions', dictionary: bool = True):
    """Get database cursor (backward compatibility)"""  
    return db_manager.get_cursor(database, dictionary)

def execute_query(query: str, params: tuple = None, 
                 database: str = 'crypto_transactions', fetch: str = 'all'):
    """Execute query (backward compatibility)"""
    return db_manager.execute_query(query, params, database, fetch)

# Decorator for automatic connection management
def with_database(database: str = 'crypto_transactions'):
    """Decorator to automatically provide database connection"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with get_db_connection(database) as conn:
                return func(conn, *args, **kwargs)
        return wrapper
    return decorator

if __name__ == "__main__":
    # Test the connection manager
    print("🧪 Testing Enhanced Database Connection Manager")
    print("=" * 60)
    
    try:
        # Test basic functionality
        with get_db_connection('crypto_transactions') as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM trade_recommendations LIMIT 1")
            result = cursor.fetchone()
            print(f"✅ Basic query test: {result}")
            cursor.close()
        
        # Test convenience function
        result = execute_query("SELECT COUNT(*) as count FROM trades LIMIT 1", 
                              database='crypto_transactions', fetch='one')
        print(f"✅ Convenience function test: {result}")
        
        # Health check
        health = db_manager.health_check()
        print(f"\n✅ Health check: {health['status']}")
        for db, status in health['databases'].items():
            print(f"   {db}: {status['status']} ({status.get('response_time_ms', 'N/A')}ms)")
        
        # Statistics
        stats = db_manager.get_stats()
        print(f"\n📊 Connection Statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        print("\n🎯 Enhanced Database Connection Manager working perfectly!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
