# position.py
from classes.utils import generate_unique_uuid
from classes.instruments import Option
import pandas as pd

class Position:
    def __init__(
        self,
        instrument,
        quantity: float,
        date: pd.Timestamp,
        action: str,
        position_id: int = None,
    ):
        self.position_id = generate_unique_uuid() if position_id is None else position_id
        self.instrument = instrument
        self.action = action
        self.date = date
        self.quantity = quantity
        
        self.is_frozen = False
        self.frozen_price = None 
        self.closed = False

        self.freeze_position()
        

    def freeze_position(self):
        if not self.is_frozen:
            self.frozen_price = self.instrument.get_price()
            if isinstance(self.instrument, Option):
                self.frozen_underlying_price = self.instrument.S
            else:
                self.frozen_underlying_price = self.instrument.get_price()
            self.is_frozen = True

    def get_value(self) -> float:
        if self.is_frozen:
            return self.quantity * self.frozen_price
        else:
            return self.quantity * self.instrument.get_price()

    def __str__(self):
        if self.is_frozen:
            if isinstance(self.instrument, Option):
                expiry_str = self.instrument.expiry_date if self.instrument.expiry_date != None else "None"
                return (
                    f"Position ID: {self.position_id}, "
                    f"Instrument: Option({self.instrument.option_type}, "
                    f"Underlying={self.instrument.underlying.ticker}, "
                    f"S={self.frozen_underlying_price:.2f}, "  
                    f"K={self.instrument.K:.2f}, "
                    f"T={self.instrument.T:.2f}, r={self.instrument.r:.2f}, "
                    f"sigma={self.instrument.sigma:.2f}, "
                    f"Action: {self.action}, "
                    f"Date: {self.date}, "
                    f"Expirty: {expiry_str}, "
                    f"Price: {self.frozen_price:.2f} (frozen), "
                    f"Quantity: {self.quantity:.2f}"
                )
            else:
                return (
                    f"Position ID: {self.position_id}, "
                    f"Instrument: {self.instrument}, "
                    f"Action: {self.action}, "
                    f"Date: {self.date}, "
                    f"Price: {self.frozen_price:.2f} (frozen), "
                    f"Quantity: {self.quantity:.2f}"
                )
        else:
            try:
                live_price = self.instrument.get_price()
                price_str  = f"{live_price:.2f}"
            except ValueError:
                price_str  = "Not Priced Yet"

            return (
                f"Position ID: {self.position_id}, "
                f"Instrument: {self.instrument}, "
                f"Action: {self.action}, "
                f"Date: {self.date}, "
                f"Price: {price_str}, "
                f"Quantity: {self.quantity:.2f}"
            )

    
    def get_position_id(self) -> int:
        return self.position_id
    
    def get_instrument(self):
        return self.instrument
    
    def get_date(self) -> pd.Timestamp:
        return self.date

    def get_price(self) -> float:
        if self.is_frozen:
            return self.frozen_price
        else:
            return self.instrument.get_price()

    def get_quantity(self) -> float:
        return self.quantity
    
    def get_action(self) -> str:
        return self.action