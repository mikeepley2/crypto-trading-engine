"""
Automated Trading Strategies
Advanced trading strategies including signal-based execution, DCA, grid trading, and arbitrage
"""

import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import numpy as np
from .advanced_order_types import AdvancedOrder, OrderType, OrderStatus, PositionSizer, RiskManager
from .llm_analysis import MarketAnalysisService, MarketContext, AnalysisType

class StrategyType(Enum):
    SIGNAL_BASED = "signal_based"
    DCA = "dollar_cost_average"
    GRID_TRADING = "grid_trading"
    ARBITRAGE = "arbitrage"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"

class StrategyStatus(Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"

@dataclass
class TradingSignal:
    """Trading signal from ML models or other sources"""
    symbol: str
    signal_type: str  # 'buy', 'sell', 'hold'
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    price_target: Optional[Decimal] = None
    stop_loss: Optional[Decimal] = None
    time_horizon: str = "short"  # 'short', 'medium', 'long'
    source: str = "ml_model"
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None

@dataclass
class StrategyConfig:
    """Configuration for trading strategies"""
    strategy_type: StrategyType
    symbol: str
    max_position_size: Decimal
    risk_per_trade: float = 0.02
    min_signal_confidence: float = 0.6
    max_daily_trades: int = 10
    enabled: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)

class SignalBasedStrategy:
    """Execute trades based on ML signals and LLM analysis"""
    
    def __init__(self, config: StrategyConfig, llm_service: MarketAnalysisService):
        self.config = config
        self.llm_service = llm_service
        self.position_sizer = PositionSizer()
        self.risk_manager = RiskManager()
        self.daily_trades = 0
        self.last_trade_date = datetime.utcnow().date()
        self.logger = logging.getLogger(__name__)
    
    async def process_signal(self, signal: TradingSignal, market_context: MarketContext) -> Optional[AdvancedOrder]:
        """Process a trading signal and generate orders"""
        
        try:
            # Reset daily trade count if new day
            if datetime.utcnow().date() != self.last_trade_date:
                self.daily_trades = 0
                self.last_trade_date = datetime.utcnow().date()
            
            # Check daily trade limit
            if self.daily_trades >= self.config.max_daily_trades:
                self.logger.info(f"Daily trade limit reached for {self.config.symbol}")
                return None
            
            # Validate signal confidence
            if signal.confidence < self.config.min_signal_confidence:
                self.logger.info(f"Signal confidence {signal.confidence} below threshold {self.config.min_signal_confidence}")
                return None
            
            # Get LLM validation
            llm_validation = await self.llm_service.validate_trade_decision(
                symbol=signal.symbol,
                action=signal.signal_type,
                quantity=Decimal('100'),  # Placeholder
                price=market_context.current_price,
                context=market_context
            )
            
            if not llm_validation['approved']:
                self.logger.info(f"LLM rejected trade for {signal.symbol}: {llm_validation['reasoning']}")
                return None
            
            # Calculate position size
            portfolio_value = Decimal('100000')  # This should come from portfolio service
            position_size = self.position_sizer.calculate_position_size(
                portfolio_value=portfolio_value,
                entry_price=market_context.current_price,
                stop_loss_price=signal.stop_loss,
                signal_confidence=signal.confidence
            )
            
            # Create order
            order = AdvancedOrder(
                symbol=signal.symbol,
                side=signal.signal_type,
                order_type=OrderType.MARKET,
                quantity=position_size,
                stop_loss_price=signal.stop_loss,
                take_profit_price=signal.price_target,
                metadata={
                    'signal_source': signal.source,
                    'signal_confidence': signal.confidence,
                    'signal_strength': signal.strength,
                    'llm_confidence': llm_validation['confidence'],
                    'strategy': 'signal_based'
                }
            )
            
            self.daily_trades += 1
            self.logger.info(f"Generated signal-based order for {signal.symbol}: {signal.signal_type} {position_size}")
            
            return order
            
        except Exception as e:
            self.logger.error(f"Error processing signal for {signal.symbol}: {e}")
            return None

class DCAStrategy:
    """Dollar Cost Averaging strategy"""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.buy_amount = Decimal(str(config.parameters.get('buy_amount', '100')))
        self.buy_interval_hours = config.parameters.get('buy_interval_hours', 24)
        self.max_total_investment = Decimal(str(config.parameters.get('max_total_investment', '10000')))
        self.total_invested = Decimal('0')
        self.last_buy_time = None
        self.logger = logging.getLogger(__name__)
    
    async def should_execute_buy(self) -> bool:
        """Check if it's time for next DCA buy"""
        
        if self.total_invested >= self.max_total_investment:
            self.logger.info(f"DCA max investment reached for {self.config.symbol}")
            return False
        
        if not self.last_buy_time:
            return True
        
        time_since_last_buy = datetime.utcnow() - self.last_buy_time
        return time_since_last_buy >= timedelta(hours=self.buy_interval_hours)
    
    async def create_dca_order(self, current_price: Decimal) -> Optional[AdvancedOrder]:
        """Create DCA buy order"""
        
        if not await self.should_execute_buy():
            return None
        
        try:
            quantity = self.buy_amount / current_price
            
            order = AdvancedOrder(
                symbol=self.config.symbol,
                side='buy',
                order_type=OrderType.MARKET,
                quantity=quantity,
                metadata={
                    'strategy': 'dca',
                    'dca_amount': str(self.buy_amount),
                    'total_invested': str(self.total_invested)
                }
            )
            
            self.last_buy_time = datetime.utcnow()
            self.total_invested += self.buy_amount
            
            self.logger.info(f"Created DCA order for {self.config.symbol}: {quantity} at ${current_price}")
            return order
            
        except Exception as e:
            self.logger.error(f"Error creating DCA order: {e}")
            return None

class GridTradingStrategy:
    """Grid trading strategy for range-bound markets"""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.grid_levels = int(config.parameters.get('grid_levels', 10))
        self.grid_spacing_percent = float(config.parameters.get('grid_spacing_percent', 0.02))  # 2%
        self.base_order_size = Decimal(str(config.parameters.get('base_order_size', '100')))
        self.center_price = None
        self.active_orders: Dict[str, AdvancedOrder] = {}
        self.logger = logging.getLogger(__name__)
    
    async def initialize_grid(self, current_price: Decimal) -> List[AdvancedOrder]:
        """Initialize grid trading orders"""
        
        self.center_price = current_price
        orders = []
        
        try:
            # Create buy orders below current price
            for i in range(1, self.grid_levels // 2 + 1):
                buy_price = current_price * (1 - Decimal(str(self.grid_spacing_percent * i)))
                buy_order = AdvancedOrder(
                    symbol=self.config.symbol,
                    side='buy',
                    order_type=OrderType.LIMIT,
                    quantity=self.base_order_size / buy_price,
                    price=buy_price,
                    metadata={
                        'strategy': 'grid_trading',
                        'grid_level': -i,
                        'grid_price': str(buy_price)
                    }
                )
                orders.append(buy_order)
            
            # Create sell orders above current price
            for i in range(1, self.grid_levels // 2 + 1):
                sell_price = current_price * (1 + Decimal(str(self.grid_spacing_percent * i)))
                sell_order = AdvancedOrder(
                    symbol=self.config.symbol,
                    side='sell',
                    order_type=OrderType.LIMIT,
                    quantity=self.base_order_size / sell_price,
                    price=sell_price,
                    metadata={
                        'strategy': 'grid_trading',
                        'grid_level': i,
                        'grid_price': str(sell_price)
                    }
                )
                orders.append(sell_order)
            
            self.logger.info(f"Initialized grid with {len(orders)} orders for {self.config.symbol}")
            return orders
            
        except Exception as e:
            self.logger.error(f"Error initializing grid: {e}")
            return []
    
    async def on_order_filled(self, filled_order: AdvancedOrder, current_price: Decimal) -> Optional[AdvancedOrder]:
        """Handle grid order fills and create replacement orders"""
        
        try:
            grid_level = filled_order.metadata.get('grid_level', 0)
            
            # Create opposite order at the next grid level
            if filled_order.side == 'buy':
                # Buy order filled, create sell order above
                new_sell_price = current_price * (1 + Decimal(str(self.grid_spacing_percent)))
                new_order = AdvancedOrder(
                    symbol=self.config.symbol,
                    side='sell',
                    order_type=OrderType.LIMIT,
                    quantity=filled_order.quantity,
                    price=new_sell_price,
                    metadata={
                        'strategy': 'grid_trading',
                        'grid_level': grid_level + 1,
                        'parent_order': filled_order.order_id
                    }
                )
            else:
                # Sell order filled, create buy order below
                new_buy_price = current_price * (1 - Decimal(str(self.grid_spacing_percent)))
                new_order = AdvancedOrder(
                    symbol=self.config.symbol,
                    side='buy',
                    order_type=OrderType.LIMIT,
                    quantity=filled_order.quantity,
                    price=new_buy_price,
                    metadata={
                        'strategy': 'grid_trading',
                        'grid_level': grid_level - 1,
                        'parent_order': filled_order.order_id
                    }
                )
            
            self.logger.info(f"Created replacement grid order: {new_order.side} at ${new_order.price}")
            return new_order
            
        except Exception as e:
            self.logger.error(f"Error handling grid order fill: {e}")
            return None

class ArbitrageStrategy:
    """Cross-exchange arbitrage strategy"""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.min_profit_percent = float(config.parameters.get('min_profit_percent', 0.5))  # 0.5%
        self.max_trade_size = Decimal(str(config.parameters.get('max_trade_size', '1000')))
        self.exchanges = config.parameters.get('exchanges', ['coinbase', 'binance'])
        self.logger = logging.getLogger(__name__)
    
    async def find_arbitrage_opportunities(self, prices: Dict[str, Decimal]) -> List[Dict[str, Any]]:
        """Find arbitrage opportunities across exchanges"""
        
        opportunities = []
        
        try:
            # Find price differences between exchanges
            exchange_prices = [(exchange, price) for exchange, price in prices.items()]
            
            for i, (exchange1, price1) in enumerate(exchange_prices):
                for j, (exchange2, price2) in enumerate(exchange_prices[i+1:], i+1):
                    
                    # Calculate profit percentage
                    if price1 > price2:
                        profit_percent = ((price1 - price2) / price2) * 100
                        buy_exchange = exchange2
                        sell_exchange = exchange1
                        buy_price = price2
                        sell_price = price1
                    else:
                        profit_percent = ((price2 - price1) / price1) * 100
                        buy_exchange = exchange1
                        sell_exchange = exchange2
                        buy_price = price1
                        sell_price = price2
                    
                    # Check if profitable after fees (assuming 0.1% fee each side)
                    net_profit_percent = profit_percent - 0.2
                    
                    if net_profit_percent >= self.min_profit_percent:
                        opportunities.append({
                            'symbol': self.config.symbol,
                            'buy_exchange': buy_exchange,
                            'sell_exchange': sell_exchange,
                            'buy_price': buy_price,
                            'sell_price': sell_price,
                            'profit_percent': net_profit_percent,
                            'max_quantity': self.max_trade_size / buy_price
                        })
            
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Error finding arbitrage opportunities: {e}")
            return []
    
    async def execute_arbitrage(self, opportunity: Dict[str, Any]) -> Tuple[Optional[AdvancedOrder], Optional[AdvancedOrder]]:
        """Execute arbitrage trade"""
        
        try:
            quantity = min(
                opportunity['max_quantity'],
                self.max_trade_size / opportunity['buy_price']
            )
            
            # Create buy order on cheaper exchange
            buy_order = AdvancedOrder(
                symbol=opportunity['symbol'],
                side='buy',
                order_type=OrderType.MARKET,
                quantity=quantity,
                metadata={
                    'strategy': 'arbitrage',
                    'exchange': opportunity['buy_exchange'],
                    'counterpart_exchange': opportunity['sell_exchange'],
                    'expected_profit_percent': opportunity['profit_percent']
                }
            )
            
            # Create sell order on more expensive exchange
            sell_order = AdvancedOrder(
                symbol=opportunity['symbol'],
                side='sell',
                order_type=OrderType.MARKET,
                quantity=quantity,
                metadata={
                    'strategy': 'arbitrage',
                    'exchange': opportunity['sell_exchange'],
                    'counterpart_exchange': opportunity['buy_exchange'],
                    'expected_profit_percent': opportunity['profit_percent']
                }
            )
            
            self.logger.info(f"Created arbitrage orders: profit {opportunity['profit_percent']:.2f}%")
            return buy_order, sell_order
            
        except Exception as e:
            self.logger.error(f"Error executing arbitrage: {e}")
            return None, None

class StrategyManager:
    """Manage multiple trading strategies"""
    
    def __init__(self, llm_service: MarketAnalysisService):
        self.strategies: Dict[str, Any] = {}
        self.llm_service = llm_service
        self.active_strategies: Dict[str, StrategyStatus] = {}
        self.logger = logging.getLogger(__name__)
    
    def add_strategy(self, strategy_id: str, config: StrategyConfig):
        """Add a new trading strategy"""
        
        try:
            if config.strategy_type == StrategyType.SIGNAL_BASED:
                strategy = SignalBasedStrategy(config, self.llm_service)
            elif config.strategy_type == StrategyType.DCA:
                strategy = DCAStrategy(config)
            elif config.strategy_type == StrategyType.GRID_TRADING:
                strategy = GridTradingStrategy(config)
            elif config.strategy_type == StrategyType.ARBITRAGE:
                strategy = ArbitrageStrategy(config)
            else:
                raise ValueError(f"Unsupported strategy type: {config.strategy_type}")
            
            self.strategies[strategy_id] = strategy
            self.active_strategies[strategy_id] = StrategyStatus.ACTIVE if config.enabled else StrategyStatus.PAUSED
            
            self.logger.info(f"Added strategy {strategy_id}: {config.strategy_type.value}")
            
        except Exception as e:
            self.logger.error(f"Error adding strategy {strategy_id}: {e}")
            self.active_strategies[strategy_id] = StrategyStatus.ERROR
    
    async def process_market_data(self, symbol: str, market_context: MarketContext) -> List[AdvancedOrder]:
        """Process market data through all active strategies"""
        
        orders = []
        
        for strategy_id, strategy in self.strategies.items():
            if self.active_strategies.get(strategy_id) != StrategyStatus.ACTIVE:
                continue
            
            try:
                if hasattr(strategy, 'config') and strategy.config.symbol == symbol:
                    if isinstance(strategy, DCAStrategy):
                        order = await strategy.create_dca_order(market_context.current_price)
                        if order:
                            orders.append(order)
                    
                    elif isinstance(strategy, GridTradingStrategy):
                        # Grid trading handles orders differently - managed separately
                        pass
                    
                    # Add other strategy processing as needed
                    
            except Exception as e:
                self.logger.error(f"Error processing strategy {strategy_id}: {e}")
                self.active_strategies[strategy_id] = StrategyStatus.ERROR
        
        return orders
    
    async def process_signal(self, signal: TradingSignal, market_context: MarketContext) -> List[AdvancedOrder]:
        """Process trading signals through signal-based strategies"""
        
        orders = []
        
        for strategy_id, strategy in self.strategies.items():
            if (self.active_strategies.get(strategy_id) == StrategyStatus.ACTIVE and
                isinstance(strategy, SignalBasedStrategy) and
                strategy.config.symbol == signal.symbol):
                
                try:
                    order = await strategy.process_signal(signal, market_context)
                    if order:
                        orders.append(order)
                except Exception as e:
                    self.logger.error(f"Error processing signal in strategy {strategy_id}: {e}")
        
        return orders
    
    def get_strategy_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all strategies"""
        
        status = {}
        for strategy_id, strategy in self.strategies.items():
            status[strategy_id] = {
                'type': strategy.config.strategy_type.value if hasattr(strategy, 'config') else 'unknown',
                'status': self.active_strategies.get(strategy_id, StrategyStatus.ERROR).value,
                'symbol': strategy.config.symbol if hasattr(strategy, 'config') else 'unknown'
            }
        
        return status
