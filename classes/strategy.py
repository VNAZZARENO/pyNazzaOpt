# strategy.py
from classes.position import Position
from classes.instruments import Option, Stock
from classes.utils import generate_unique_uuid

import abc
import logging
logger = logging.getLogger(__name__)

class Strategy(abc.ABC):

    @abc.abstractmethod
    def run(self, portfolio, date):
        """To be implemented in the subclasses"""
        pass

class DeltaHedgeStrategy(Strategy):
    def __init__(self, pricer):
        self.pricer = pricer

    def hedge_position(self, portfolio, pos, date, fee_multiplier):
        instrument = pos.instrument
        
        if not instrument.is_priced:
            instrument.update_price(self.pricer)
        delta = instrument.get_delta()
        
        if instrument.option_type == "Call":
            sign_of_hedge = +1 if pos.get_action() == "Long" else -1
        elif instrument.option_type == "Put":
            sign_of_hedge = -1 if pos.get_action() == "Long" else +1
        else:
            raise ValueError("Must be 'Call' of 'Put'")
        
        hedge_quantity = sign_of_hedge * delta * pos.get_quantity()
        hedge_action = "Short" if hedge_quantity > 0 else "Long"
        
        stock_position = Position(
            position_id=generate_unique_uuid(),
            instrument=Stock(price=instrument.S, ticker="HEDGE"),
            quantity=abs(hedge_quantity),  
            date=date,
            action=hedge_action
        )
        
        portfolio.add_position(stock_position, fee_multiplier)
        
        logger.info(
                    f"[{date}] Hedged position {pos.get_position_id()} "
                    f"qty={pos.get_quantity()} ({pos.get_action()} {instrument.option_type}) "
                    f"with delta={delta:.3f}: "
                    f"{hedge_action.lower()}ed {abs(hedge_quantity):.3f} shares."
                )

    def run(self, portfolio, date, fee=0.01):
        positions = portfolio.get_positions_at_date(date)
        for pos in positions:
            instrument = pos.instrument
            if isinstance(instrument, Option):
                self.hedge_position(portfolio, pos, date, fee)

class GammaHedgeStrategy(Strategy):

    def __init__(self, pricer):
        super().__init__()
        self.pricer = pricer


class VegaHedgeStrategy(Strategy):
        
    def __init__(self, pricer):
        super().__init__()
        self.pricer = pricer