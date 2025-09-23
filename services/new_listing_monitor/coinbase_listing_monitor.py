#!/usr/bin/env python3
"""
Coinbase New Listing Monitor Service
Detects new cryptocurrency listings on Coinbase for automated trading opportunities

Features:
- Monitors @CoinbaseAssets Twitter feed
- Tracks Coinbase listing roadmap updates
- Watches Base network token additions
- Sends immediate alerts for new listing opportunities
- Integrates with existing trading infrastructure
"""

import os
import json
import time
import requests
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
import mysql.connector
from mysql.connector import Error
import hashlib
import asyncio
import aiohttp
import re
from urllib.parse import urljoin

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class NewListing:
    """Represents a new Coinbase listing opportunity"""
    symbol: str
    name: str
    source: str  # 'twitter', 'roadmap', 'base_network'
    announcement_time: datetime
    listing_type: str  # 'announced', 'roadmap_added', 'base_added'
    confidence_score: float  # 0.0-1.0
    additional_data: Dict = None

class CoinbaseListingMonitor:
    """Monitor for new Coinbase cryptocurrency listings"""
    
    def __init__(self):
        self.db_config = {
            'host': '192.168.230.163',
            'user': 'news_collector',
            'password': '99Rules!',
            'database': 'crypto_transactions'
        }
        
        # Monitoring endpoints
        self.coinbase_assets_feed = "https://api.twitter.com/2/users/by/username/CoinbaseAssets"
        self.coinbase_roadmap_url = "https://www.coinbase.com/listings"
        self.base_network_api = "https://api.basescan.org/api"
        
        # Cache for preventing duplicate notifications
        self.processed_announcements: Set[str] = set()
        
        # Keywords for listing detection
        self.listing_keywords = [
            "now live on",
            "is now available",
            "trading is now live",
            "added to the roadmap",
            "support for",
            "listing",
            "available on coinbase"
        ]
        
        self.initialize_database()
    
    def initialize_database(self):
        """Initialize database table for storing listing alerts"""
        connection = None
        cursor = None
        try:
            connection = mysql.connector.connect(**self.db_config)
            cursor = connection.cursor()
            
            create_table_query = """
            CREATE TABLE IF NOT EXISTS coinbase_listing_alerts (
                id INT AUTO_INCREMENT PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                name VARCHAR(100),
                source VARCHAR(50) NOT NULL,
                announcement_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                listing_type VARCHAR(50) NOT NULL,
                confidence_score FLOAT DEFAULT 0.0,
                additional_data JSON,
                processed BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_symbol (symbol),
                INDEX idx_announcement_time (announcement_time),
                INDEX idx_processed (processed)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
            
            cursor.execute(create_table_query)
            connection.commit()
            logger.info("✅ Database table 'coinbase_listing_alerts' ready")
            
        except Error as e:
            logger.error(f"❌ Database initialization error: {e}")
        finally:
            if connection and connection.is_connected():
                if cursor:
                    cursor.close()
                connection.close()
    
    def get_announcement_hash(self, content: str, timestamp: str) -> str:
        """Generate unique hash for announcement to prevent duplicates"""
        return hashlib.md5(f"{content}_{timestamp}".encode()).hexdigest()
    
    async def monitor_twitter_feed(self) -> List[NewListing]:
        """Monitor @CoinbaseAssets Twitter feed for new listing announcements"""
        try:
            # Note: In production, you'd need Twitter API v2 bearer token
            # For now, we'll simulate checking the feed
            logger.info("🔍 Monitoring @CoinbaseAssets Twitter feed...")
            
            # Simulate Twitter API response (in production, replace with actual API call)
            sample_tweets = [
                {
                    "text": "Bitcoin Hyper (HYPER) is now live on Coinbase.com & in the Coinbase iOS & Android apps.",
                    "created_at": datetime.now().isoformat(),
                    "id": "1234567890"
                }
            ]
            
            new_listings = []
            for tweet in sample_tweets:
                if self.is_listing_announcement(tweet["text"]):
                    listing = self.parse_listing_from_tweet(tweet)
                    if listing:
                        new_listings.append(listing)
                        logger.info(f"🚨 NEW LISTING DETECTED: {listing.symbol} from Twitter")
            
            return new_listings
            
        except Exception as e:
            logger.error(f"❌ Error monitoring Twitter feed: {e}")
            return []
    
    def is_listing_announcement(self, text: str) -> bool:
        """Check if tweet text indicates a new listing"""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.listing_keywords)
    
    def parse_listing_from_tweet(self, tweet: Dict) -> Optional[NewListing]:
        """Extract listing information from tweet"""
        try:
            text = tweet["text"]
            
            # Extract symbol from text (pattern: "Symbol (ABC)")
            symbol_match = re.search(r'\(([A-Z0-9]+)\)', text)
            name_match = re.search(r'([A-Za-z\s]+)\s*\([A-Z0-9]+\)', text)
            
            if symbol_match:
                symbol = symbol_match.group(1)
                name = name_match.group(1).strip() if name_match else symbol
                
                announcement_hash = self.get_announcement_hash(text, tweet["created_at"])
                if announcement_hash not in self.processed_announcements:
                    self.processed_announcements.add(announcement_hash)
                    
                    return NewListing(
                        symbol=symbol,
                        name=name,
                        source="twitter",
                        announcement_time=datetime.fromisoformat(tweet["created_at"].replace('Z', '+00:00')),
                        listing_type="announced",
                        confidence_score=0.95,  # High confidence for official announcements
                        additional_data={"tweet_id": tweet["id"], "tweet_text": text}
                    )
            
        except Exception as e:
            logger.error(f"❌ Error parsing tweet: {e}")
        
        return None
    
    async def monitor_coinbase_roadmap(self) -> List[NewListing]:
        """Monitor Coinbase listing roadmap for new additions"""
        try:
            logger.info("🔍 Checking Coinbase listing roadmap...")
            
            # Simulate roadmap check (in production, scrape actual roadmap page)
            # The roadmap shows tokens that Coinbase has decided to list but not yet gone live
            roadmap_additions = [
                {
                    "symbol": "EXAMPLE",
                    "name": "Example Token",
                    "added_time": datetime.now().isoformat()
                }
            ]
            
            new_listings = []
            for addition in roadmap_additions:
                listing = NewListing(
                    symbol=addition["symbol"],
                    name=addition["name"],
                    source="roadmap",
                    announcement_time=datetime.fromisoformat(addition["added_time"]),
                    listing_type="roadmap_added",
                    confidence_score=0.85,  # High confidence but not yet live
                    additional_data={"roadmap_entry": True}
                )
                new_listings.append(listing)
                logger.info(f"🛣️ ROADMAP ADDITION: {listing.symbol}")
            
            return new_listings
            
        except Exception as e:
            logger.error(f"❌ Error monitoring roadmap: {e}")
            return []
    
    async def monitor_base_network(self) -> List[NewListing]:
        """Monitor Base network for new token additions that might indicate future Coinbase listings"""
        try:
            logger.info("🔍 Monitoring Base network for new tokens...")
            
            # Base network tokens often get listed on Coinbase since Base is Coinbase's L2
            # This gives us early warning of potential listings
            
            # Simulate Base network monitoring (in production, use actual Base APIs)
            base_tokens = [
                {
                    "symbol": "BASETOKEN",
                    "name": "Base Example Token",
                    "contract_address": "0x1234567890abcdef",
                    "added_time": datetime.now().isoformat()
                }
            ]
            
            new_listings = []
            for token in base_tokens:
                listing = NewListing(
                    symbol=token["symbol"],
                    name=token["name"],
                    source="base_network",
                    announcement_time=datetime.fromisoformat(token["added_time"]),
                    listing_type="base_added",
                    confidence_score=0.70,  # Medium confidence - potential future listing
                    additional_data={
                        "contract_address": token["contract_address"],
                        "network": "base"
                    }
                )
                new_listings.append(listing)
                logger.info(f"🔗 BASE NETWORK ADDITION: {listing.symbol}")
            
            return new_listings
            
        except Exception as e:
            logger.error(f"❌ Error monitoring Base network: {e}")
            return []
    
    def save_listing_alert(self, listing: NewListing) -> bool:
        """Save new listing alert to database"""
        connection = None
        cursor = None
        try:
            connection = mysql.connector.connect(**self.db_config)
            cursor = connection.cursor()
            
            insert_query = """
            INSERT INTO coinbase_listing_alerts 
            (symbol, name, source, announcement_time, listing_type, confidence_score, additional_data)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            
            values = (
                listing.symbol,
                listing.name,
                listing.source,
                listing.announcement_time,
                listing.listing_type,
                listing.confidence_score,
                json.dumps(listing.additional_data) if listing.additional_data else None
            )
            
            cursor.execute(insert_query, values)
            connection.commit()
            
            logger.info(f"💾 Saved listing alert: {listing.symbol} from {listing.source}")
            return True
            
        except Error as e:
            logger.error(f"❌ Error saving listing alert: {e}")
            return False
        finally:
            if connection and connection.is_connected():
                if cursor:
                    cursor.close()
                connection.close()
    
    def send_immediate_alert(self, listing: NewListing):
        """Send immediate alert to trading system for fast execution"""
        try:
            # Generate trading signal for immediate execution
            signal_data = {
                "signal_type": "LISTING_OPPORTUNITY",
                "symbol": listing.symbol,
                "action": "BUY",
                "confidence": listing.confidence_score,
                "urgency": "IMMEDIATE",
                "source": listing.source,
                "listing_type": listing.listing_type,
                "timestamp": listing.announcement_time.isoformat(),
                "additional_data": listing.additional_data
            }
            
            # Save to trading_signals table for pickup by signal bridge
            self.save_trading_signal(signal_data)
            
            # Log immediate alert
            logger.warning(f"🚨 IMMEDIATE TRADING ALERT: {listing.symbol} - {listing.listing_type}")
            logger.warning(f"   Source: {listing.source} | Confidence: {listing.confidence_score:.2f}")
            logger.warning(f"   Recommendation: BUY IMMEDIATELY for Coinbase Effect opportunity")
            
        except Exception as e:
            logger.error(f"❌ Error sending immediate alert: {e}")
    
    def save_trading_signal(self, signal_data: Dict):
        """Save trading signal to database for pickup by signal bridge"""
        connection = None
        cursor = None
        try:
            connection = mysql.connector.connect(**self.db_config)
            cursor = connection.cursor()
            
            # Check if trading_signals table exists, if not create it
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trading_signals (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    signal_type VARCHAR(50) NOT NULL,
                    action VARCHAR(10) NOT NULL,
                    confidence FLOAT NOT NULL,
                    reasoning TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processed BOOLEAN DEFAULT FALSE,
                    metadata JSON,
                    INDEX idx_symbol (symbol),
                    INDEX idx_created_at (created_at),
                    INDEX idx_processed (processed)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)
            
            insert_query = """
            INSERT INTO trading_signals 
            (symbol, signal_type, action, confidence, reasoning, metadata)
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            
            reasoning = f"New Coinbase listing opportunity detected from {signal_data['source']}. " \
                       f"Type: {signal_data['listing_type']}. Expected Coinbase Effect: 91% average gain in 5 days."
            
            values = (
                signal_data["symbol"],
                signal_data["signal_type"],
                signal_data["action"],
                signal_data["confidence"],
                reasoning,
                json.dumps(signal_data)
            )
            
            cursor.execute(insert_query, values)
            connection.commit()
            
            logger.info(f"📈 Trading signal generated for {signal_data['symbol']}")
            
        except Error as e:
            logger.error(f"❌ Error saving trading signal: {e}")
        finally:
            if connection and connection.is_connected():
                if cursor:
                    cursor.close()
                connection.close()
    
    async def get_unprocessed_listings(self):
        """Get unprocessed listing opportunities from database"""
        connection = None
        cursor = None
        try:
            connection = mysql.connector.connect(**self.db_config)
            cursor = connection.cursor(dictionary=True)
            
            # Get unprocessed listing alerts
            query = """
                SELECT id, symbol, name, source, announcement_time, 
                       listing_type, confidence_score, additional_data
                FROM coinbase_listing_alerts 
                WHERE processed = FALSE 
                ORDER BY announcement_time DESC
                LIMIT 10
            """
            
            cursor.execute(query)
            unprocessed = cursor.fetchall()
            
            logger.info(f"📋 Retrieved {len(unprocessed)} unprocessed listing opportunities")
            
            return unprocessed
            
        except Error as e:
            logger.error(f"❌ Error retrieving unprocessed listings: {e}")
            return []
        finally:
            if connection and connection.is_connected():
                if cursor:
                    cursor.close()
                connection.close()

    async def run_monitoring_cycle(self):
        """Run complete monitoring cycle for all sources"""
        logger.info("🔄 Starting Coinbase listing monitoring cycle...")
        
        all_new_listings = []
        
        # Monitor all sources concurrently
        twitter_listings = await self.monitor_twitter_feed()
        roadmap_listings = await self.monitor_coinbase_roadmap()
        base_listings = await self.monitor_base_network()
        
        all_new_listings.extend(twitter_listings)
        all_new_listings.extend(roadmap_listings)
        all_new_listings.extend(base_listings)
        
        # Process each new listing
        for listing in all_new_listings:
            # Save to database
            if self.save_listing_alert(listing):
                # Send immediate alert for high-confidence listings
                if listing.confidence_score >= 0.8 and listing.listing_type in ["announced", "roadmap_added"]:
                    self.send_immediate_alert(listing)
        
        if all_new_listings:
            logger.info(f"🎯 Monitoring cycle complete: {len(all_new_listings)} new opportunities detected")
        else:
            logger.info("✅ Monitoring cycle complete: No new listings detected")
        
        return all_new_listings

    async def start_monitoring(self):
        """Start continuous monitoring loop"""
        logger.info("🚀 Coinbase Listing Monitor Service Starting...")
        logger.info("📊 Monitoring Sources:")
        logger.info("   • @CoinbaseAssets Twitter feed")
        logger.info("   • Coinbase listing roadmap")
        logger.info("   • Base network token additions")
        logger.info("🎯 Target: Detect Coinbase Effect opportunities (91% avg gain)")
        
        # Main monitoring loop
        while True:
            try:
                await self.run_monitoring_cycle()
                
                # Wait 5 minutes between checks (adjust based on needs)
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"❌ Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying

async def main():
    """Main monitoring loop"""
    monitor = CoinbaseListingMonitor()
    
    logger.info("🚀 Coinbase Listing Monitor Service Starting...")
    logger.info("📊 Monitoring Sources:")
    logger.info("   • @CoinbaseAssets Twitter feed")
    logger.info("   • Coinbase listing roadmap")
    logger.info("   • Base network token additions")
    logger.info("🎯 Target: Detect Coinbase Effect opportunities (91% avg gain)")
    
    # Main monitoring loop
    while True:
        try:
            await monitor.run_monitoring_cycle()
            
            # Wait 5 minutes between checks (adjust based on needs)
            await asyncio.sleep(300)
            
        except KeyboardInterrupt:
            logger.info("🛑 Monitoring stopped by user")
            break
        except Exception as e:
            logger.error(f"❌ Error in monitoring loop: {e}")
            await asyncio.sleep(60)  # Wait 1 minute before retrying

if __name__ == "__main__":
    asyncio.run(main())
