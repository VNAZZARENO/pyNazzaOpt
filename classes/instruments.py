# instrument.py
import numpy as np
import abc

from classes.pricer import Pricer

class InstrumentEquity(abc.ABC):
    @abc.abstractmethod
    def update_price(self, pricer: Pricer):
        """Update the instrument's price using a Pricer."""
        pass

    @abc.abstractmethod
    def get_price(self) -> float:
        """Returns the current price of the instrument."""
        pass

    @abc.abstractmethod
    def get_delta(self) -> float:
        """Returns the delta of the instrument."""
        pass

    @property
    @abc.abstractmethod
    def is_priced(self) -> bool:
        """Boolean flag indicating if this instrument is currently 'priced'."""
        pass


class InstrumentFixedIncome(abc.ABC):
    @abc.abstractmethod
    def update_price(self, pricer: Pricer):
        """Update the instrument's price using a Pricer."""
        pass

    @abc.abstractmethod
    def get_price(self) -> float:
        """Returns the current price of the instrument."""
        pass

    @abc.abstractmethod
    def get_duration(self) -> float:
        """Returns the duration of the instrument."""
        pass

    @property
    @abc.abstractmethod
    def is_priced(self) -> bool:
        """Boolean flag indicating if this instrument is currently 'priced'."""
        pass


class Stock(InstrumentEquity):
    def __init__(self, price: float, ticker: str = None):
        self._price = price
        self.ticker = ticker
        self._is_priced = True
        self._dependents = []

    @property
    def S(self) -> float:
        return self._price

    @S.setter
    def S(self, value: float):
        self._price = value
        self._notify_dependents()

    def register_dependent(self, dependent):
        self._dependents.append(dependent)

    # def remove_dependent(self, dependant):
    #     self._dependents.remove(dependant)

    def _notify_dependents(self):
        for dep in self._dependents:
            dep.is_priced = False

    def update_price(self):
        pass

    def get_price(self) -> float:
        return self._price

    def get_delta(self) -> float:
        return 1.0

    @property
    def is_priced(self) -> bool:
        return self._is_priced

    @is_priced.setter
    def is_priced(self, value: bool):
        self._is_priced = value

    def __str__(self):
        return f"Stock {self.ticker} (S={self.S:.2f})"
    

class Index(InstrumentEquity):
    def __init__(self, price: float, ticker: str = None):
        self._price = price
        self.ticker = ticker
        self._is_priced = True
        self._dependents = []

    @property
    def S(self) -> float:
        return self._price

    @S.setter
    def S(self, value: float):
        self._price = value
        self._notify_dependents()

    def register_dependent(self, dependent):
        self._dependents.append(dependent)

    # def remove_dependent(self, dependant):
    #     self._dependents.remove(dependant)

    def _notify_dependents(self):
        for dep in self._dependents:
            dep.is_priced = False

    def update_price(self):
        pass

    def get_price(self) -> float:
        return self._price

    def get_delta(self) -> float:
        return 1.0

    @property
    def is_priced(self) -> bool:
        return self._is_priced

    @is_priced.setter
    def is_priced(self, value: bool):
        self._is_priced = value

    def __str__(self):
        return f"Index {self.ticker} (Points={self.S:.2f})"


class Option(InstrumentEquity):
    def __init__(
        self,
        underlying: Stock,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "Call",
        expiry_date=None  
    ):
        self.underlying = underlying
        self.underlying.register_dependent(self)
        self._K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type
        self.expiry_date = expiry_date  

        self._is_priced = False

        self.price = None
        self.delta = None
        self.gamma = None
        self.vega = None
        self.rho = None

    @property
    def S(self):
        return self.underlying.get_price()

    @property
    def K(self):
        return self._K

    @K.setter
    def K(self, value):
        self._K = value
        self._is_priced = False

    def get_payoff(self) -> float:
        return self.compute_payoff(final_underlying_price=self.S)

    def compute_payoff(self, final_underlying_price=None) -> float:
        if final_underlying_price is None:
            final_underlying_price = self.S
        if self.option_type.lower() == "call":
            return max(final_underlying_price - self.K, 0)
        elif self.option_type.lower() == "put":
            return max(self.K - final_underlying_price, 0)
        else:
            raise ValueError("option_type must be 'Call' or 'Put'")

    def update_price(self, pricer):
        greeks_df = pricer.get_all_values(
            S=self.S,
            K=self.K,
            T=self.T,
            r=self.r,
            sigma=self.sigma,
            option_type=self.option_type,
        )
        self.price = greeks_df["price"]
        self.delta = greeks_df["delta"]
        self.gamma = greeks_df["gamma"]
        self.vega  = greeks_df["vega"]
        self.rho   = greeks_df["rho"]
        self._is_priced = True

    @property
    def is_priced(self) -> bool:
        return self._is_priced

    @is_priced.setter
    def is_priced(self, value: bool):
        self._is_priced = value

    def get_price(self) -> float:
        if not self._is_priced:
            raise ValueError("Option not priced yet. Call update_price(pricer) first.")
        return self.price

    def get_delta(self) -> float:
        if not self._is_priced:
            raise ValueError("Option not priced yet. Call update_price(pricer) first.")
        return self.delta

    def get_gamma(self) -> float:
        if not self._is_priced:
            raise ValueError("Option not priced yet. Call update_price(pricer) first.")
        return self.gamma

    def get_vega(self) -> float:
        if not self._is_priced:
            raise ValueError("Option not priced yet. Call update_price(pricer) first.")
        return self.vega

    def get_rho(self) -> float:
        if not self._is_priced:
            raise ValueError("Option not priced yet. Call update_price(pricer) first.")
        return self.rho

    def __str__(self):
        if self._is_priced:
            return (
                f"Option({self.option_type}, Underlying={self.underlying.ticker}, "
                f"S={self.S:.2f}, K={self.K:.2f}, T={self.T:.2f}, r={self.r:.2f}, "
                f"sigma={self.sigma:.2f}, price={self.price:.4f})"
            )
        else:
            return (
                f"Option({self.option_type}, Underlying={self.underlying.ticker}, "
                f"S={self.S:.2f}, K={self.K:.2f}, T={self.T:.2f}, r={self.r:.2f}, "
                f"sigma={self.sigma:.2f}, is_priced={self._is_priced})"
            )
        



from classes.instruments import Option, Stock
from classes.pricer import AmericanPricer

class AmericanOption(Option):
    """
    American Option class that extends the base Option class.
    Represents an option that can be exercised at any time before expiration.
    """
    
    def __init__(
        self,
        underlying: Stock,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "Call",
        expiry_date=None,
        steps: int = 100
    ):
        """
        Initialize an American option.
        
        Parameters:
        ----------
        underlying : Stock
            The underlying asset
        K : float
            Strike price
        T : float
            Time to expiration in years
        r : float
            Risk-free interest rate (annual, expressed as a decimal)
        sigma : float
            Volatility of the underlying asset (annual, expressed as a decimal)
        option_type : str, optional
            'Call' or 'Put', defaults to 'Call'
        expiry_date : date or Timestamp, optional
            Expiration date of the option
        steps : int, optional
            Number of steps for the binomial tree model, defaults to 100
        """
        super().__init__(
            underlying=underlying,
            K=K,
            T=T,
            r=r,
            sigma=sigma,
            option_type=option_type,
            expiry_date=expiry_date
        )
        # Create a pricer specifically for this option
        self._american_pricer = AmericanPricer(steps=steps)
        self._is_american = True
    
    def update_price(self, pricer=None):
        """
        Update the price and Greeks of the American option using the binomial tree model.
        
        Parameters:
        ----------
        pricer : Pricer, optional
            For compatibility with the base class. Ignored as AmericanOption uses its own pricer.
        """
        # Use the American pricer regardless of what's passed in
        greeks_df = self._american_pricer.get_all_values(
            S=self.S,
            K=self.K,
            T=self.T,
            r=self.r,
            sigma=self.sigma,
            option_type=self.option_type,
        )
        
        self.price = greeks_df["price"]
        self.delta = greeks_df["delta"]
        self.gamma = greeks_df["gamma"]
        self.vega = greeks_df["vega"]
        self.rho = greeks_df["rho"]
        self._is_priced = True
    
    def can_exercise_early(self) -> bool:
        """
        Returns True since American options can be exercised before expiration.
        
        Returns:
        -------
        bool
            Always returns True for American options
        """
        return True
    
    def __str__(self):
        """
        String representation of the American option.
        
        Returns:
        -------
        str
            String description of the option
        """
        base_str = super().__str__()
        # Insert "American" into the string
        return base_str.replace("Option(", "American Option(", 1)