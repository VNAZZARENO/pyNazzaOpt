# portfolio.py
import pandas as pd
from classes.position import Position
from classes.instruments import Option, Stock, Index
from classes.pricer import Pricer

class Portfolio:
    def __init__(self, name: str, positions=None):
        self.positions_dict = {}
        self.bank_dict = {}

        self.name = name
        if isinstance(positions, list):
            for position in positions:
                self.add_position(position)
  

    def get_portfolio_name(self):
        return self.name

    def adjust_bank(self, amount, date):
        try:
            self.bank_dict[pd.Timestamp(date)] += amount
        except:
            self.bank_dict[pd.Timestamp(date)] = 0.0
            self.bank_dict[pd.Timestamp(date)] += amount
            
    def __str__(self):
        s = f"Portfolio: {self.name}\n"
        if not self.positions_dict:
            s += "  (No positions)\n"
            return s

        for date, positions in sorted(self.positions_dict.items(), key=lambda x: x[0]):
            s += f"\nDate: {date}\n"
            for pos in positions:
                s += f"  - {pos}\n"
        return s
    

    def get_portfolio_value(self, instrument_type="all"):
        """
        Get the total value of the portfolio, with robust handling of missing bank entries.
        
        Args:
            instrument_type: Filter by instrument type ('all', 'option', 'stock')
            
        Returns:
            float: Total portfolio value (positions + bank)
        """
        if not self.positions_dict:
            if self.bank_dict:
                return self.bank_dict[max(self.bank_dict.keys())]
            return 0.0
        
        most_recent_date = max(self.positions_dict.keys())
        total_value = 0.0
        
        if instrument_type == "all":
            for position in self.positions_dict[most_recent_date]:
                if isinstance(position.instrument, Option) and position.instrument.is_priced:
                    if position.get_action().lower() == 'long':
                        total_value += position.get_value()
                    elif position.get_action().lower() == 'short':
                        total_value -= position.get_value()
                elif isinstance(position.instrument, Stock):
                    if position.get_action().lower() == 'long':
                        total_value += position.get_value()
                    elif position.get_action().lower() == 'short':
                        total_value -= position.get_value()
                elif isinstance(position.instrument, Index):
                    if position.get_action().lower() == 'long':
                        total_value += position.get_value()
                    elif position.get_action().lower() == 'short':
                        total_value -= position.get_value()
                else:
                    pass
        elif instrument_type == "option":
            for position in self.positions_dict[most_recent_date]:
                if isinstance(position.instrument, Option) and position.instrument.is_priced:
                    if position.get_action().lower() == 'long':
                        total_value += position.get_value()
                    elif position.get_action().lower() == 'short':
                        total_value -= position.get_value()
                else:
                    pass
        elif instrument_type == "stock":
            for position in self.positions_dict[most_recent_date]:
                if isinstance(position.instrument, Stock):
                    if position.get_action().lower() == 'long':
                        total_value += position.get_value()
                    elif position.get_action().lower() == 'short':
                        total_value -= position.get_value()
                else:
                    pass
        
        # Get bank value for the date
        if most_recent_date in self.bank_dict:
            bank_value = self.bank_dict[most_recent_date]
        else:
            # Find the most recent date in bank_dict that is earlier than most_recent_date
            previous_dates = [d for d in self.bank_dict.keys() if d <= most_recent_date]
            if previous_dates:
                bank_value = self.bank_dict[max(previous_dates)]
            else:
                # Initialize a new bank entry
                self.bank_dict[most_recent_date] = 0.0
                bank_value = 0.0
                
        return total_value + bank_value
    

    def get_portfolio_delta(self, date: pd.Timestamp = None):
        total_delta = 0.0
        if date is None:        
            date = max(self.positions_dict.keys())
        for position in self.positions_dict[date]:
            if position.get_action().lower() == 'long':
                total_delta += position.instrument.get_delta() * position.get_quantity()
            elif position.get_action().lower() == 'short':
                total_delta -= position.instrument.get_delta() * position.get_quantity()
        return total_delta

    def get_portfolio_dates(self):
        return sorted(self.positions_dict.keys())

    def get_positions_at_date(self, date):
        return self.positions_dict[date]

    def freeze_positions_for_date(self, date: pd.Timestamp):
        if date in self.positions_dict:
            for pos in self.positions_dict[date]:
                if not pos.is_frozen:
                    pos.freeze_position()
    
    def roll_to_next_date(self, 
                          from_date: pd.Timestamp, 
                          to_date: pd.Timestamp, 

                          time_increments: float = 1, # in days by default
                            
                        #   position_K_adjustment: int = None,
                          position_r_adjustment: float = None,
                          position_sigma_adjustment: float = None,
                          
                          quantity_adjustment: float = 0.0,
                          fee: float = 0.0035,
                          base_year_nb =252 # to convert time_increments to % of year
                          ):
        
        self.freeze_positions_for_date(from_date)

        if from_date in self.positions_dict:
            for pos in self.positions_dict[from_date]:

                if isinstance(pos, Option):
                #     if position_r_adjustment:
                #         pos.instrument.r = position_r_adjustment
                #     elif position_sigma_adjustment:
                #         pos.instrument.sigma = position_sigma_adjustment
                    
                    pos.instrument.T -= time_increments / base_year_nb
                    print(f"TTM: {pos.instrument.T}")
                new_pos = Position(
                    instrument = pos.instrument,
                    quantity   = pos.quantity + quantity_adjustment,
                    date       = to_date,
                    action     = pos.action
                )
                self.add_position(new_pos, fee=fee)  

    def add_position(self, position: Position, fee: float = 0.0):
        date = pd.Timestamp(position.date)
        if date not in self.positions_dict:
            self.positions_dict[date] = []
            self.bank_dict[date] = 0.0
        self.positions_dict[date].append(position)

        if not position.instrument.is_priced:
            position.instrument.update_price(Pricer())

        cost = abs(position.get_value())
        total_fee = fee * cost
        if position.get_action().lower() == "long":
            self.adjust_bank(-1 * cost - abs(total_fee), date)
        elif position.get_action().lower() == "short":
            self.adjust_bank(cost - abs(total_fee), date)

    def remove_position(self, position_id=None, name=None, date=None):
        positions_to_remove = []
        for pos_date, positions_list in self.positions_dict.items():
            for pos in positions_list:
                if (position_id is not None and pos.get_position_id() == position_id) or \
                   (name is not None and name.lower() == str(pos).lower().split('\n')[0]) or \
                   (date is not None and date == pos.get_date()):
                    positions_to_remove.append(pos)
        
        for pos in positions_to_remove:
            self.positions_dict[pos.get_date()].remove(pos)
        
        for pos_date, positions_list in list(self.positions_dict.items()):
            if not positions_list: 
                del self.positions_dict[pos_date]
    
    def find_position_by_instrument(self, instrument, action, date):
        for pos in self.positions_dict[date]:
            if pos.instrument_type == instrument and pos.get_action() == action:
                return pos
            
    def get_positions(self):
        return [pos for _, positions_list in self.positions_dict.items() for pos in positions_list]
    
    def get_bank_value(self):
        most_recent_date = max(self.bank_dict.keys())
        return self.bank_dict[most_recent_date]
    
    def get_bank_cumsum(self):
        bank_series = pd.Series(self.bank_dict)
        bank_series = bank_series.sort_index()
        return bank_series.cumsum()

    def update_all(self, date: pd.Timestamp = None, pricer: Pricer = None):
        if pricer is None:
            pricer = Pricer()
        elif date is None:
            date = max(self.positions_dict.keys())
        elif date not in self.positions_dict:
            self.positions_dict[date] = [] 
        for pos in self.positions_dict[date]:
            if isinstance(pos.instrument, Option) and not pos.instrument.is_priced:
                pos.instrument.update_price(pricer)

    def check_option_expired(self, current_date, stock_price, symbol):
        current_date = pd.Timestamp(current_date)
        last_date = max(self.positions_dict.keys())
        
        for pos in self.get_positions_at_date(last_date):
            if isinstance(pos.instrument, Option) and not pos.closed and pos.instrument.underlying.ticker == symbol:
                if pos.instrument.expiry_date is not None and pos.instrument.expiry_date <= current_date:
                    if not pos.is_frozen:
                        pos.freeze_position()
                    payoff = pos.instrument.compute_payoff(final_underlying_price=stock_price)
                    if pos.get_action().lower() == "short":
                        settlement = - payoff * pos.get_quantity()
                    else:
                        settlement = payoff * pos.get_quantity()
                    self.adjust_bank(settlement, current_date)
                    pos.closed = True
                    print(f"Settled option {pos.instrument.option_type} position on {current_date}: Settlement = {settlement:.2f}")
                    print(f"Option K: {pos.instrument.K} / Price at T: {stock_price} / PnL: {payoff}")
    
    def close_stock_position(self, current_date, stock_price, call_pos, symbol):
        current_date = pd.Timestamp(current_date)
        last_date = max(self.positions_dict.keys())

        for pos in self.get_positions_at_date(last_date):
            print("")
            if isinstance(pos.instrument, Stock) and not pos.closed and pos.instrument.ticker == symbol:
                if pos.date < current_date:
                    entry_price = pos.get_price()  
                    if stock_price != entry_price:
                        exit_price = stock_price
                        pos.instrument._price = exit_price 
                        exit_pos = Position(
                            instrument=pos.instrument,
                            quantity=pos.get_quantity(),  
                            date=current_date,
                            action="Short"
                        )
                        # settlement = (exit_price - entry_price) * pos.get_quantity()
                        # self.adjust_bank(settlement, current_date)
                        self.add_position(exit_pos)
                        pos.closed = True
                        print(f"Closed long stock position on {current_date} at {exit_price:.2f}")
                        print(f"Entry Price: {entry_price} / Exit Price: {exit_price} PnL: {exit_price - entry_price}")
                

    def close_index_positions(self, date, index_price):
        date = pd.Timestamp(date)
        last_date = max(self.positions_dict.keys())
        for pos in self.get_positions_at_date(last_date):
            if isinstance(pos.instrument, Index) and not pos.closed:
                if pos.get_action().lower() == "short" and pos.date < date:
                    entry_price = pos.get_price()
                    exit_price = index_price
                    pos.instrument._price = exit_price
                    exit_pos = Position(
                        instrument=pos.instrument,
                        quantity=pos.get_quantity(),
                        date=date,
                        action="Long"
                    )
                    self.add_position(exit_pos)
                    pos.closed = True
                    print(f"Closed short index position on {date} at {exit_price:.2f}")
                    print(f"Entry Price: {entry_price} / Exit Price: {exit_price} PnL: {entry_price - exit_price}")

    def was_call_exercised(self, symbol, date):
        date = pd.Timestamp(date)
        last_date = max(self.positions_dict.keys())
        for pos in self.get_positions_at_date(last_date):
            if isinstance(pos.instrument, Option) and pos.instrument.option_type == "Call" and pos.instrument.underlying.ticker == symbol  and pos.date < date:
                if pos.instrument.expiry_date == date and pos.instrument.compute_payoff(pos.instrument.S) > 0:
                    return (True, pos)
        return (False, None)

    def was_put_exercised(self, symbol, date):
        date = pd.Timestamp(date)
        last_date = max(self.positions_dict.keys())
        for pos in self.get_positions_at_date(last_date):
            if isinstance(pos.instrument, Option) and pos.instrument.option_type == "Put" and pos.instrument.underlying.ticker == symbol and pos.date < date:
                if pos.instrument.expiry_date == date and pos.instrument.compute_payoff(pos.instrument.S) > 0:
                    return (True, pos)
        return (False, None)

    def has_open_position(self, symbol, date):
        date = pd.Timestamp(date)
        # last_date = max(self.positions_dict.keys())
        for pos in self.get_positions_at_date(date):
            if isinstance(pos.instrument, Stock) and pos.instrument.ticker == symbol and not pos.closed:
                return True
        return False