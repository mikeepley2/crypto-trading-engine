"""
Advanced Order Types and Risk Management
Enhanced order management with stop-loss, take-profit, and intelligent position sizing
"""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional, Dict, Any, List
from enum import Enum
import asyncio
import logging
from datetime import datetime, timedelta
import json

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"
    DCA = "dollar_cost_average"
    GRID = "grid_trading"

class OrderStatus(Enum):
    PENDING = "pending"
    ACTIVE = "active"
    FILLED = "filled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    PARTIALLY_FILLED = "partially_filled"

class RiskLevel(Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"

@dataclass
class AdvancedOrder:
    """Enhanced order with advanced features"""
    symbol: str
    side: str  # 'buy' or 'sell'
    order_type: OrderType
    quantity: Decimal
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    take_profit_price: Optional[Decimal] = None
    stop_loss_price: Optional[Decimal] = None
    trailing_amount: Optional[Decimal] = None
    trailing_percent: Optional[float] = None
    time_in_force: str = "GTC"  # Good Till Cancelled
    expires_at: Optional[datetime] = None
    parent_order_id: Optional[str] = None
    child_orders: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Risk management
    max_loss_percent: Optional[float] = None
    max_position_size: Optional[Decimal] = None
    risk_level: RiskLevel = RiskLevel.MODERATE
    
    # Status tracking
    order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: Decimal = Decimal('0')
    average_fill_price: Optional[Decimal] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

class PositionSizer:
    """Intelligent position sizing based on risk parameters"""
    
    def __init__(self, 
                 max_risk_per_trade: float = 0.02,  # 2% max risk per trade
                 max_portfolio_risk: float = 0.10,   # 10% max total portfolio risk
                 kelly_fraction: float = 0.25):      # Kelly criterion fraction
        self.max_risk_per_trade = max_risk_per_trade
        self.max_portfolio_risk = max_portfolio_risk
        self.kelly_fraction = kelly_fraction
        self.logger = logging.getLogger(__name__)
    
    def calculate_position_size(self, 
                              portfolio_value: Decimal,
                              entry_price: Decimal,
                              stop_loss_price: Optional[Decimal] = None,
                              signal_confidence: float = 0.5,
                              risk_level: RiskLevel = RiskLevel.MODERATE) -> Decimal:
        """Calculate optimal position size based on risk parameters"""
        
        try:
            # Base risk adjustment by risk level
            risk_multipliers = {
                RiskLevel.CONSERVATIVE: 0.5,
                RiskLevel.MODERATE: 1.0,
                RiskLevel.AGGRESSIVE: 1.5
            }
            
            base_risk = self.max_risk_per_trade * risk_multipliers[risk_level]
            
            # Adjust for signal confidence
            confidence_adjusted_risk = base_risk * (0.5 + signal_confidence * 0.5)
            
            # Calculate risk amount in dollars
            risk_amount = portfolio_value * Decimal(str(confidence_adjusted_risk))
            
            if stop_loss_price:
                # Calculate position size based on stop loss
                price_diff = abs(entry_price - stop_loss_price)
                position_size = risk_amount / price_diff
            else:
                # Use default 5% stop loss assumption
                default_stop_percent = Decimal('0.05')
                max_loss_per_share = entry_price * default_stop_percent
                position_size = risk_amount / max_loss_per_share
            
            # Ensure we don't exceed maximum position size
            max_position_value = portfolio_value * Decimal('0.20')  # 20% max position
            max_shares = max_position_value / entry_price
            
            final_position_size = min(position_size, max_shares)
            
            self.logger.info(f"Position sizing: risk_amount=${risk_amount}, "
                           f"position_size={final_position_size}, confidence={signal_confidence}")
            
            return final_position_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            # Conservative fallback
            return portfolio_value * Decimal('0.01') / entry_price

class RiskManager:
    """Advanced risk management and controls"""
    
    def __init__(self):
        self.daily_loss_limit = Decimal('0.05')  # 5% daily loss limit
        self.weekly_loss_limit = Decimal('0.15')  # 15% weekly loss limit
        self.max_open_positions = 10
        self.max_correlation_exposure = 0.3  # 30% max in correlated assets
        self.logger = logging.getLogger(__name__)
        
        # Track daily/weekly losses
        self.daily_pnl = Decimal('0')
        self.weekly_pnl = Decimal('0')
        self.last_reset_date = datetime.utcnow().date()
        self.last_weekly_reset = datetime.utcnow().isocalendar()[1]
    
    def check_risk_limits(self, 
                         new_order: AdvancedOrder,
                         current_portfolio_value: Decimal,
                         open_positions: List[Dict]) -> Dict[str, Any]:
        """Comprehensive risk check before order execution"""
        
        risks = {
            'approved': True,
            'warnings': [],
            'blocks': [],
            'risk_score': 0.0
        }
        
        try:
            # Reset daily/weekly tracking if needed
            self._reset_period_tracking()
            
            # Check daily loss limits
            if abs(self.daily_pnl) >= current_portfolio_value * self.daily_loss_limit:
                risks['approved'] = False
                risks['blocks'].append(f"Daily loss limit exceeded: {self.daily_pnl}")
            
            # Check weekly loss limits
            if abs(self.weekly_pnl) >= current_portfolio_value * self.weekly_loss_limit:
                risks['approved'] = False
                risks['blocks'].append(f"Weekly loss limit exceeded: {self.weekly_pnl}")
            
            # Check maximum open positions
            if len(open_positions) >= self.max_open_positions:
                risks['approved'] = False
                risks['blocks'].append(f"Maximum open positions exceeded: {len(open_positions)}")
            
            # Check position sizing
            position_value = new_order.quantity * (new_order.price or Decimal('0'))
            if position_value > current_portfolio_value * Decimal('0.25'):  # 25% max position
                risks['warnings'].append("Large position size detected")
                risks['risk_score'] += 0.3
            
            # Check correlation exposure (simplified)
            symbol_category = self._get_asset_category(new_order.symbol)
            category_exposure = self._calculate_category_exposure(open_positions, symbol_category)
            if category_exposure > self.max_correlation_exposure:
                risks['warnings'].append(f"High correlation exposure in {symbol_category}")
                risks['risk_score'] += 0.2
            
            # Calculate overall risk score
            if risks['risk_score'] > 0.7:
                risks['warnings'].append("High overall risk score")
            
            self.logger.info(f"Risk assessment for {new_order.symbol}: {risks}")
            return risks
            
        except Exception as e:
            self.logger.error(f"Error in risk assessment: {e}")
            return {
                'approved': False,
                'warnings': [],
                'blocks': [f"Risk assessment error: {e}"],
                'risk_score': 1.0
            }
    
    def _reset_period_tracking(self):
        """Reset daily/weekly tracking if period has changed"""
        current_date = datetime.utcnow().date()
        current_week = datetime.utcnow().isocalendar()[1]
        
        if current_date != self.last_reset_date:
            self.daily_pnl = Decimal('0')
            self.last_reset_date = current_date
        
        if current_week != self.last_weekly_reset:
            self.weekly_pnl = Decimal('0')
            self.last_weekly_reset = current_week
    
    def _get_asset_category(self, symbol: str) -> str:
        """Categorize asset for correlation analysis"""
        # Simplified categorization
        if symbol.upper() in ['BTC', 'BTCUSD', 'BTCUSDT']:
            return 'bitcoin'
        elif symbol.upper() in ['ETH', 'ETHUSD', 'ETHUSDT']:
            return 'ethereum'
        elif symbol.upper() in ['ADA', 'DOT', 'MATIC', 'AVAX']:
            return 'smart_contracts'
        elif symbol.upper() in ['DOGE', 'SHIB', 'PEPE']:
            return 'memecoins'
        else:
            return 'other'
    
    def _calculate_category_exposure(self, positions: List[Dict], category: str) -> float:
        """Calculate exposure to a specific asset category"""
        total_value = sum(pos.get('value', 0) for pos in positions)
        if total_value == 0:
            return 0.0
        
        category_value = sum(
            pos.get('value', 0) for pos in positions 
            if self._get_asset_category(pos.get('symbol', '')) == category
        )
        
        return category_value / total_value
    
    def update_pnl(self, pnl_change: Decimal):
        """Update daily and weekly P&L tracking"""
        self._reset_period_tracking()
        self.daily_pnl += pnl_change
        self.weekly_pnl += pnl_change

class OrderManager:
    """Advanced order management with stop-loss, take-profit, and trailing stops"""
    
    def __init__(self):
        self.active_orders: Dict[str, AdvancedOrder] = {}
        self.order_history: List[AdvancedOrder] = []
        self.position_sizer = PositionSizer()
        self.risk_manager = RiskManager()
        self.logger = logging.getLogger(__name__)
    
    async def create_bracket_order(self,
                                 symbol: str,
                                 side: str,
                                 quantity: Decimal,
                                 entry_price: Decimal,
                                 stop_loss_percent: float = 0.05,
                                 take_profit_percent: float = 0.10) -> List[AdvancedOrder]:
        """Create a bracket order with entry, stop-loss, and take-profit"""
        
        try:
            # Calculate stop loss and take profit prices
            if side.lower() == 'buy':
                stop_loss_price = entry_price * (1 - Decimal(str(stop_loss_percent)))
                take_profit_price = entry_price * (1 + Decimal(str(take_profit_percent)))
            else:  # sell
                stop_loss_price = entry_price * (1 + Decimal(str(stop_loss_percent)))
                take_profit_price = entry_price * (1 - Decimal(str(take_profit_percent)))
            
            # Create main entry order
            entry_order = AdvancedOrder(
                symbol=symbol,
                side=side,
                order_type=OrderType.LIMIT,
                quantity=quantity,
                price=entry_price,
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price
            )
            
            # Create stop loss order
            stop_order = AdvancedOrder(
                symbol=symbol,
                side='sell' if side.lower() == 'buy' else 'buy',
                order_type=OrderType.STOP_LOSS,
                quantity=quantity,
                stop_price=stop_loss_price,
                parent_order_id=entry_order.order_id
            )
            
            # Create take profit order
            tp_order = AdvancedOrder(
                symbol=symbol,
                side='sell' if side.lower() == 'buy' else 'buy',
                order_type=OrderType.TAKE_PROFIT,
                quantity=quantity,
                price=take_profit_price,
                parent_order_id=entry_order.order_id
            )
            
            # Link orders
            entry_order.child_orders = [stop_order.order_id, tp_order.order_id]
            
            orders = [entry_order, stop_order, tp_order]
            
            self.logger.info(f"Created bracket order for {symbol}: "
                           f"entry=${entry_price}, stop=${stop_loss_price}, tp=${take_profit_price}")
            
            return orders
            
        except Exception as e:
            self.logger.error(f"Error creating bracket order: {e}")
            raise
    
    async def create_trailing_stop(self,
                                 symbol: str,
                                 side: str,
                                 quantity: Decimal,
                                 trailing_percent: float = 0.05) -> AdvancedOrder:
        """Create a trailing stop order"""
        
        return AdvancedOrder(
            symbol=symbol,
            side=side,
            order_type=OrderType.TRAILING_STOP,
            quantity=quantity,
            trailing_percent=trailing_percent
        )
    
    async def update_trailing_stops(self, current_prices: Dict[str, Decimal]):
        """Update all active trailing stop orders"""
        
        for order_id, order in self.active_orders.items():
            if order.order_type == OrderType.TRAILING_STOP and order.status == OrderStatus.ACTIVE:
                try:
                    current_price = current_prices.get(order.symbol)
                    if current_price:
                        await self._update_trailing_stop(order, current_price)
                except Exception as e:
                    self.logger.error(f"Error updating trailing stop {order_id}: {e}")
    
    async def _update_trailing_stop(self, order: AdvancedOrder, current_price: Decimal):
        """Update a single trailing stop order"""
        
        if not order.trailing_percent:
            return
        
        trailing_amount = current_price * Decimal(str(order.trailing_percent))
        
        if order.side.lower() == 'sell':  # Long position
            new_stop_price = current_price - trailing_amount
            if not order.stop_price or new_stop_price > order.stop_price:
                order.stop_price = new_stop_price
                order.updated_at = datetime.utcnow()
                self.logger.info(f"Updated trailing stop for {order.symbol} to ${new_stop_price}")
        
        else:  # Short position
            new_stop_price = current_price + trailing_amount
            if not order.stop_price or new_stop_price < order.stop_price:
                order.stop_price = new_stop_price
                order.updated_at = datetime.utcnow()
                self.logger.info(f"Updated trailing stop for {order.symbol} to ${new_stop_price}")
