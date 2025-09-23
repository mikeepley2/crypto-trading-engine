"""
Multi-Exchange Arbitrage System
Advanced cross-exchange price discovery, arbitrage execution, and portfolio balancing
"""

import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import statistics
from .exchange_adapters import ExchangeAdapter
from .advanced_order_types import AdvancedOrder, OrderType
from .automated_strategies import ArbitrageStrategy

class PriceSource(Enum):
    COINBASE = "coinbase"
    BINANCE = "binance"
    KRAKEN = "kraken"
    KUCOIN = "kucoin"
    GEMINI = "gemini"

@dataclass
class ExchangePrice:
    """Price data from a specific exchange"""
    exchange: PriceSource
    symbol: str
    bid: Decimal
    ask: Decimal
    last: Decimal
    volume_24h: Decimal
    timestamp: datetime
    spread: Decimal = field(init=False)
    
    def __post_init__(self):
        self.spread = self.ask - self.bid

@dataclass
class ArbitrageOpportunity:
    """Detected arbitrage opportunity"""
    symbol: str
    buy_exchange: PriceSource
    sell_exchange: PriceSource
    buy_price: Decimal
    sell_price: Decimal
    profit_amount: Decimal
    profit_percent: float
    max_quantity: Decimal
    estimated_fees: Decimal
    net_profit: Decimal
    confidence: float  # 0.0 to 1.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime = field(default_factory=lambda: datetime.utcnow() + timedelta(seconds=30))

class PriceAggregator:
    """Aggregate prices from multiple exchanges"""
    
    def __init__(self):
        self.price_cache: Dict[str, List[ExchangePrice]] = {}
        self.cache_timeout = timedelta(seconds=10)
        self.logger = logging.getLogger(__name__)
    
    async def get_exchange_prices(self, symbol: str, exchanges: List[PriceSource]) -> List[ExchangePrice]:
        """Fetch prices from multiple exchanges"""
        
        tasks = []
        for exchange in exchanges:
            tasks.append(self._fetch_exchange_price(exchange, symbol))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        prices = []
        for exchange, result in zip(exchanges, results):
            if isinstance(result, Exception):
                self.logger.error(f"Error fetching price from {exchange.value}: {result}")
            elif result:
                prices.append(result)
        
        # Cache results
        cache_key = f"{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        self.price_cache[cache_key] = prices
        
        # Cleanup old cache
        self._cleanup_cache()
        
        return prices
    
    async def _fetch_exchange_price(self, exchange: PriceSource, symbol: str) -> Optional[ExchangePrice]:
        """Fetch price from a specific exchange"""
        
        try:
            if exchange == PriceSource.COINBASE:
                return await self._fetch_coinbase_price(symbol)
            elif exchange == PriceSource.BINANCE:
                return await self._fetch_binance_price(symbol)
            elif exchange == PriceSource.KRAKEN:
                return await self._fetch_kraken_price(symbol)
            elif exchange == PriceSource.KUCOIN:
                return await self._fetch_kucoin_price(symbol)
            elif exchange == PriceSource.GEMINI:
                return await self._fetch_gemini_price(symbol)
            else:
                self.logger.warning(f"Unsupported exchange: {exchange}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error fetching {exchange.value} price for {symbol}: {e}")
            return None
    
    async def _fetch_coinbase_price(self, symbol: str) -> Optional[ExchangePrice]:
        """Fetch price from Coinbase Pro"""
        
        try:
            # Convert symbol format (e.g., BTCUSD -> BTC-USD)
            coinbase_symbol = f"{symbol[:3]}-{symbol[3:]}" if len(symbol) == 6 else symbol
            
            async with aiohttp.ClientSession() as session:
                # Get ticker data
                ticker_url = f"https://api.exchange.coinbase.com/products/{coinbase_symbol}/ticker"
                async with session.get(ticker_url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        return ExchangePrice(
                            exchange=PriceSource.COINBASE,
                            symbol=symbol,
                            bid=Decimal(str(data['bid'])),
                            ask=Decimal(str(data['ask'])),
                            last=Decimal(str(data['price'])),
                            volume_24h=Decimal(str(data['volume'])),
                            timestamp=datetime.utcnow()
                        )
        except Exception as e:
            self.logger.error(f"Coinbase price fetch error: {e}")
            return None
    
    async def _fetch_binance_price(self, symbol: str) -> Optional[ExchangePrice]:
        """Fetch price from Binance"""
        
        try:
            # Binance uses symbols like BTCUSDT
            binance_symbol = f"{symbol[:3]}USDT" if symbol.endswith('USD') else symbol
            
            async with aiohttp.ClientSession() as session:
                # Get 24hr ticker
                ticker_url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={binance_symbol}"
                async with session.get(ticker_url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Get order book for bid/ask
                        book_url = f"https://api.binance.com/api/v3/depth?symbol={binance_symbol}&limit=5"
                        async with session.get(book_url) as book_response:
                            if book_response.status == 200:
                                book_data = await book_response.json()
                                
                                return ExchangePrice(
                                    exchange=PriceSource.BINANCE,
                                    symbol=symbol,
                                    bid=Decimal(str(book_data['bids'][0][0])),
                                    ask=Decimal(str(book_data['asks'][0][0])),
                                    last=Decimal(str(data['lastPrice'])),
                                    volume_24h=Decimal(str(data['volume'])),
                                    timestamp=datetime.utcnow()
                                )
        except Exception as e:
            self.logger.error(f"Binance price fetch error: {e}")
            return None
    
    async def _fetch_kraken_price(self, symbol: str) -> Optional[ExchangePrice]:
        """Fetch price from Kraken"""
        
        try:
            # Kraken uses symbols like XBTUSD
            kraken_symbol = symbol.replace('BTC', 'XBT') if 'BTC' in symbol else symbol
            
            async with aiohttp.ClientSession() as session:
                ticker_url = f"https://api.kraken.com/0/public/Ticker?pair={kraken_symbol}"
                async with session.get(ticker_url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data['error']:
                            self.logger.error(f"Kraken API error: {data['error']}")
                            return None
                        
                        # Kraken returns nested result structure
                        pair_data = list(data['result'].values())[0]
                        
                        return ExchangePrice(
                            exchange=PriceSource.KRAKEN,
                            symbol=symbol,
                            bid=Decimal(str(pair_data['b'][0])),
                            ask=Decimal(str(pair_data['a'][0])),
                            last=Decimal(str(pair_data['c'][0])),
                            volume_24h=Decimal(str(pair_data['v'][1])),
                            timestamp=datetime.utcnow()
                        )
        except Exception as e:
            self.logger.error(f"Kraken price fetch error: {e}")
            return None
    
    async def _fetch_kucoin_price(self, symbol: str) -> Optional[ExchangePrice]:
        """Fetch price from KuCoin"""
        
        try:
            # KuCoin uses symbols like BTC-USDT
            kucoin_symbol = f"{symbol[:3]}-{symbol[3:]}" if len(symbol) == 6 else symbol
            
            async with aiohttp.ClientSession() as session:
                ticker_url = f"https://api.kucoin.com/api/v1/market/stats?symbol={kucoin_symbol}"
                async with session.get(ticker_url) as response:
                    if response.status == 200:
                        result = await response.json()
                        data = result['data']
                        
                        # Get order book
                        book_url = f"https://api.kucoin.com/api/v1/market/orderbook/level1?symbol={kucoin_symbol}"
                        async with session.get(book_url) as book_response:
                            if book_response.status == 200:
                                book_result = await book_response.json()
                                book_data = book_result['data']
                                
                                return ExchangePrice(
                                    exchange=PriceSource.KUCOIN,
                                    symbol=symbol,
                                    bid=Decimal(str(book_data['bestBid'])),
                                    ask=Decimal(str(book_data['bestAsk'])),
                                    last=Decimal(str(data['last'])),
                                    volume_24h=Decimal(str(data['vol'])),
                                    timestamp=datetime.utcnow()
                                )
        except Exception as e:
            self.logger.error(f"KuCoin price fetch error: {e}")
            return None
    
    async def _fetch_gemini_price(self, symbol: str) -> Optional[ExchangePrice]:
        """Fetch price from Gemini"""
        
        try:
            # Gemini uses symbols like btcusd (lowercase)
            gemini_symbol = symbol.lower()
            
            async with aiohttp.ClientSession() as session:
                ticker_url = f"https://api.gemini.com/v1/pubticker/{gemini_symbol}"
                async with session.get(ticker_url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        return ExchangePrice(
                            exchange=PriceSource.GEMINI,
                            symbol=symbol,
                            bid=Decimal(str(data['bid'])),
                            ask=Decimal(str(data['ask'])),
                            last=Decimal(str(data['last'])),
                            volume_24h=Decimal(str(data['volume']['BTC'])) if 'volume' in data else Decimal('0'),
                            timestamp=datetime.utcnow()
                        )
        except Exception as e:
            self.logger.error(f"Gemini price fetch error: {e}")
            return None
    
    def _cleanup_cache(self):
        """Remove expired cache entries"""
        current_time = datetime.utcnow()
        expired_keys = []
        
        for key, prices in self.price_cache.items():
            if prices and (current_time - prices[0].timestamp) > self.cache_timeout:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.price_cache[key]
    
    def calculate_consensus_price(self, prices: List[ExchangePrice]) -> Tuple[Decimal, Decimal, float]:
        """Calculate consensus price with confidence score"""
        
        if not prices:
            return Decimal('0'), Decimal('0'), 0.0
        
        # Extract last prices
        last_prices = [float(price.last) for price in prices]
        
        # Calculate weighted average (weight by volume)
        total_volume = sum(float(price.volume_24h) for price in prices)
        if total_volume > 0:
            weighted_price = sum(
                float(price.last) * float(price.volume_24h) / total_volume
                for price in prices
            )
        else:
            weighted_price = statistics.mean(last_prices)
        
        # Calculate price deviation as confidence metric
        if len(last_prices) > 1:
            price_std = statistics.stdev(last_prices)
            price_mean = statistics.mean(last_prices)
            coefficient_of_variation = price_std / price_mean if price_mean > 0 else 1.0
            confidence = max(0.0, 1.0 - coefficient_of_variation * 10)  # Scale CV to confidence
        else:
            confidence = 0.5  # Medium confidence with single source
        
        return Decimal(str(weighted_price)), Decimal(str(statistics.mean(last_prices))), confidence

class ArbitrageDetector:
    """Detect arbitrage opportunities across exchanges"""
    
    def __init__(self):
        self.min_profit_threshold = 0.3  # 0.3% minimum profit
        self.max_price_age = timedelta(seconds=30)
        self.exchange_fees = {
            PriceSource.COINBASE: 0.005,  # 0.5%
            PriceSource.BINANCE: 0.001,   # 0.1%
            PriceSource.KRAKEN: 0.0026,   # 0.26%
            PriceSource.KUCOIN: 0.001,    # 0.1%
            PriceSource.GEMINI: 0.0035    # 0.35%
        }
        self.logger = logging.getLogger(__name__)
    
    def detect_opportunities(self, prices: List[ExchangePrice]) -> List[ArbitrageOpportunity]:
        """Detect arbitrage opportunities from price data"""
        
        opportunities = []
        
        # Filter out stale prices
        current_time = datetime.utcnow()
        fresh_prices = [
            price for price in prices 
            if (current_time - price.timestamp) <= self.max_price_age
        ]
        
        if len(fresh_prices) < 2:
            return opportunities
        
        # Compare all price pairs
        for i, buy_price_data in enumerate(fresh_prices):
            for j, sell_price_data in enumerate(fresh_prices):
                if i >= j:  # Avoid duplicate comparisons
                    continue
                
                opportunity = self._calculate_arbitrage(buy_price_data, sell_price_data)
                if opportunity and opportunity.net_profit > 0:
                    opportunities.append(opportunity)
        
        # Sort by profitability
        opportunities.sort(key=lambda x: x.profit_percent, reverse=True)
        
        return opportunities
    
    def _calculate_arbitrage(self, price1: ExchangePrice, price2: ExchangePrice) -> Optional[ArbitrageOpportunity]:
        """Calculate arbitrage between two exchanges"""
        
        try:
            # Determine buy and sell exchanges
            if price1.ask < price2.bid:
                buy_exchange = price1.exchange
                sell_exchange = price2.exchange
                buy_price = price1.ask
                sell_price = price2.bid
            elif price2.ask < price1.bid:
                buy_exchange = price2.exchange
                sell_exchange = price1.exchange
                buy_price = price2.ask
                sell_price = price1.bid
            else:
                return None  # No arbitrage opportunity
            
            # Calculate gross profit
            profit_amount = sell_price - buy_price
            profit_percent = float((profit_amount / buy_price) * 100)
            
            # Calculate fees
            buy_fee_rate = self.exchange_fees.get(buy_exchange, 0.001)
            sell_fee_rate = self.exchange_fees.get(sell_exchange, 0.001)
            
            buy_fee = buy_price * Decimal(str(buy_fee_rate))
            sell_fee = sell_price * Decimal(str(sell_fee_rate))
            total_fees = buy_fee + sell_fee
            
            # Calculate net profit
            net_profit = profit_amount - total_fees
            net_profit_percent = float((net_profit / buy_price) * 100)
            
            # Check if profitable
            if net_profit_percent < self.min_profit_threshold:
                return None
            
            # Calculate maximum quantity (conservative estimate)
            # This should be enhanced with actual liquidity data
            max_quantity = min(
                price1.volume_24h * Decimal('0.001'),  # 0.1% of daily volume
                price2.volume_24h * Decimal('0.001'),
                Decimal('10')  # Maximum 10 units
            )
            
            # Calculate confidence based on spread and volume
            spread_confidence = 1.0 - min(1.0, float(max(price1.spread, price2.spread) / buy_price))
            volume_confidence = min(1.0, float(min(price1.volume_24h, price2.volume_24h)) / 1000.0)
            confidence = (spread_confidence + volume_confidence) / 2.0
            
            return ArbitrageOpportunity(
                symbol=price1.symbol,
                buy_exchange=buy_exchange,
                sell_exchange=sell_exchange,
                buy_price=buy_price,
                sell_price=sell_price,
                profit_amount=profit_amount,
                profit_percent=net_profit_percent,
                max_quantity=max_quantity,
                estimated_fees=total_fees,
                net_profit=net_profit,
                confidence=confidence
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating arbitrage: {e}")
            return None

class CrossExchangeBalancer:
    """Balance portfolio across multiple exchanges"""
    
    def __init__(self):
        self.target_allocations = {
            PriceSource.COINBASE: 0.4,  # 40%
            PriceSource.BINANCE: 0.3,   # 30%
            PriceSource.KRAKEN: 0.2,    # 20%
            PriceSource.KUCOIN: 0.1     # 10%
        }
        self.rebalance_threshold = 0.05  # 5% deviation threshold
        self.logger = logging.getLogger(__name__)
    
    def calculate_rebalancing_orders(self, 
                                   current_balances: Dict[PriceSource, Decimal],
                                   symbol: str,
                                   total_value: Decimal) -> List[AdvancedOrder]:
        """Calculate orders needed to rebalance portfolio"""
        
        orders = []
        
        try:
            # Calculate current allocations
            current_allocations = {}
            for exchange, balance in current_balances.items():
                current_allocations[exchange] = float(balance / total_value) if total_value > 0 else 0.0
            
            # Find exchanges that need rebalancing
            for exchange, target_allocation in self.target_allocations.items():
                current_allocation = current_allocations.get(exchange, 0.0)
                deviation = abs(current_allocation - target_allocation)
                
                if deviation > self.rebalance_threshold:
                    target_value = total_value * Decimal(str(target_allocation))
                    current_value = current_balances.get(exchange, Decimal('0'))
                    needed_adjustment = target_value - current_value
                    
                    if needed_adjustment > 0:
                        # Need to buy on this exchange
                        order = AdvancedOrder(
                            symbol=symbol,
                            side='buy',
                            order_type=OrderType.MARKET,
                            quantity=abs(needed_adjustment),
                            metadata={
                                'strategy': 'rebalancing',
                                'target_exchange': exchange.value,
                                'adjustment_amount': str(needed_adjustment)
                            }
                        )
                        orders.append(order)
                    else:
                        # Need to sell on this exchange
                        order = AdvancedOrder(
                            symbol=symbol,
                            side='sell',
                            order_type=OrderType.MARKET,
                            quantity=abs(needed_adjustment),
                            metadata={
                                'strategy': 'rebalancing',
                                'target_exchange': exchange.value,
                                'adjustment_amount': str(needed_adjustment)
                            }
                        )
                        orders.append(order)
            
            self.logger.info(f"Generated {len(orders)} rebalancing orders for {symbol}")
            return orders
            
        except Exception as e:
            self.logger.error(f"Error calculating rebalancing orders: {e}")
            return []

class MultiExchangeArbitrageEngine:
    """Main engine for multi-exchange arbitrage operations"""
    
    def __init__(self):
        self.price_aggregator = PriceAggregator()
        self.arbitrage_detector = ArbitrageDetector()
        self.portfolio_balancer = CrossExchangeBalancer()
        self.active_opportunities: List[ArbitrageOpportunity] = []
        self.executed_arbitrages: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)
    
    async def scan_for_opportunities(self, symbols: List[str]) -> Dict[str, List[ArbitrageOpportunity]]:
        """Scan multiple symbols for arbitrage opportunities"""
        
        all_opportunities = {}
        exchanges = [PriceSource.COINBASE, PriceSource.BINANCE, PriceSource.KRAKEN]
        
        for symbol in symbols:
            try:
                # Get prices from all exchanges
                prices = await self.price_aggregator.get_exchange_prices(symbol, exchanges)
                
                if len(prices) >= 2:
                    # Detect arbitrage opportunities
                    opportunities = self.arbitrage_detector.detect_opportunities(prices)
                    all_opportunities[symbol] = opportunities
                    
                    self.logger.info(f"Found {len(opportunities)} arbitrage opportunities for {symbol}")
                else:
                    self.logger.warning(f"Insufficient price data for {symbol}: {len(prices)} sources")
                    all_opportunities[symbol] = []
                    
            except Exception as e:
                self.logger.error(f"Error scanning {symbol} for arbitrage: {e}")
                all_opportunities[symbol] = []
        
        return all_opportunities
    
    async def execute_arbitrage_opportunity(self, opportunity: ArbitrageOpportunity) -> Dict[str, Any]:
        """Execute an arbitrage opportunity"""
        
        try:
            # Validate opportunity is still valid
            if datetime.utcnow() > opportunity.expires_at:
                return {
                    'success': False,
                    'reason': 'Opportunity expired',
                    'opportunity': opportunity
                }
            
            # Create buy and sell orders
            buy_order = AdvancedOrder(
                symbol=opportunity.symbol,
                side='buy',
                order_type=OrderType.MARKET,
                quantity=opportunity.max_quantity,
                metadata={
                    'strategy': 'arbitrage',
                    'target_exchange': opportunity.buy_exchange.value,
                    'opportunity_id': f"{opportunity.symbol}_{opportunity.timestamp.strftime('%H%M%S')}"
                }
            )
            
            sell_order = AdvancedOrder(
                symbol=opportunity.symbol,
                side='sell',
                order_type=OrderType.MARKET,
                quantity=opportunity.max_quantity,
                metadata={
                    'strategy': 'arbitrage',
                    'target_exchange': opportunity.sell_exchange.value,
                    'opportunity_id': f"{opportunity.symbol}_{opportunity.timestamp.strftime('%H%M%S')}"
                }
            )
            
            execution_result = {
                'success': True,
                'buy_order': buy_order,
                'sell_order': sell_order,
                'expected_profit': opportunity.net_profit,
                'profit_percent': opportunity.profit_percent,
                'timestamp': datetime.utcnow()
            }
            
            # Track executed arbitrage
            self.executed_arbitrages.append(execution_result)
            
            self.logger.info(f"Executed arbitrage for {opportunity.symbol}: "
                           f"{opportunity.profit_percent:.2f}% profit")
            
            return execution_result
            
        except Exception as e:
            self.logger.error(f"Error executing arbitrage opportunity: {e}")
            return {
                'success': False,
                'reason': f"Execution error: {e}",
                'opportunity': opportunity
            }
    
    async def get_market_summary(self, symbols: List[str]) -> Dict[str, Any]:
        """Get comprehensive market summary across exchanges"""
        
        summary = {
            'timestamp': datetime.utcnow(),
            'symbols': {},
            'total_opportunities': 0,
            'best_opportunity': None
        }
        
        try:
            all_opportunities = await self.scan_for_opportunities(symbols)
            
            for symbol, opportunities in all_opportunities.items():
                if opportunities:
                    best_opportunity = max(opportunities, key=lambda x: x.profit_percent)
                    
                    summary['symbols'][symbol] = {
                        'opportunity_count': len(opportunities),
                        'best_profit_percent': best_opportunity.profit_percent,
                        'best_exchange_pair': f"{best_opportunity.buy_exchange.value} -> {best_opportunity.sell_exchange.value}"
                    }
                    
                    summary['total_opportunities'] += len(opportunities)
                    
                    if (not summary['best_opportunity'] or 
                        best_opportunity.profit_percent > summary['best_opportunity'].profit_percent):
                        summary['best_opportunity'] = best_opportunity
                else:
                    summary['symbols'][symbol] = {
                        'opportunity_count': 0,
                        'best_profit_percent': 0.0,
                        'best_exchange_pair': 'None'
                    }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating market summary: {e}")
            summary['error'] = str(e)
            return summary
