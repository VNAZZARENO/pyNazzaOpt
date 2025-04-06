# pricer.py
import numpy as np
from scipy.stats import norm

class Pricer():
    def __init__(self):
        pass

    @staticmethod
    def compute_d1_d2(S: float, K: float, r: float, sigma: float, T: float) -> tuple:
        d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2
    
    @staticmethod
    def compute_price(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
        d1, d2 = Pricer.compute_d1_d2(S, K, r, sigma, T)
        if option_type == 'Call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        elif option_type == 'Put':
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        else:
            raise ValueError("option_type must be 'Call' or 'Put'")
        return price
    
    @staticmethod
    def compute_delta(S: float, K: float, T: float, r: float, sigma: float, option_type: str):
        d1, _ = Pricer.compute_d1_d2(S, K, r, sigma, T)
        if option_type == 'Call':
            delta = norm.cdf(d1)
        elif option_type == 'Put':
            delta = norm.cdf(d1) - 1
        else:
            raise ValueError("option_type must be 'Call' or 'Put'")
        return delta

    @staticmethod
    def compute_gamma(S: float, K: float, T: float, r: float, sigma: float):
        d1, _ = Pricer.compute_d1_d2(S, K, r, sigma, T)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        return gamma
    
    @staticmethod
    def compute_vega(S: float, K: float, T: float, r: float, sigma: float):
        d1, _ = Pricer.compute_d1_d2(S, K, r, sigma, T)
        vega = S * norm.pdf(d1) * np.sqrt(T)
        return vega
    
    @staticmethod
    def compute_rho(S: float, K: float, T: float, r: float, sigma: float, option_type: str):
        _ , d2 = Pricer.compute_d1_d2(S, K, r, sigma, T)
        if option_type == 'Call':
            rho = K * T * np.exp(-r * T) * norm.cdf(d2)
        elif option_type == 'Put':
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
        else:
            raise ValueError("option_type must be 'Call' or 'Put'")
        return rho
    
    @staticmethod
    def get_all_values(S: float, K: float, T: float, r: float, sigma: float, option_type: str):
        all_values = {
            "S": S,
            "K": K,
            "T": T, 
            "r": r,
            "sigma": sigma,
            "option_type": option_type,
            "delta": Pricer.compute_delta(S, K, T, r, sigma, option_type), 
            "gamma": Pricer.compute_gamma(S, K, T, r, sigma),
            "vega": Pricer.compute_vega(S, K, T, r, sigma),
            "rho": Pricer.compute_rho(S, K, T, r, sigma, option_type),
            "price": Pricer.compute_price(S, K, T, r, sigma, option_type)
        }
        return all_values
    



class AmericanPricer:
    """
    Implementation of Cox-Ross-Rubinstein binomial tree model for pricing American options.
    This class extends the functionality of the base Pricer class to handle American options,
    which can be exercised at any time before expiration.
    """
    
    def __init__(self, steps=100):
        """
        Initialize the American option pricer.
        
        Parameters:
        ----------
        steps : int
            Number of time steps in the binomial tree. Higher values increase accuracy but also computation time.
        """
        self.steps = steps
    
    def compute_price(self, S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
        """
        Price an American option using the binomial tree model.
        
        Parameters:
        ----------
        S : float
            Current price of the underlying asset
        K : float
            Strike price of the option
        T : float
            Time to expiration in years
        r : float
            Risk-free interest rate (annual, expressed as a decimal)
        sigma : float
            Volatility of the underlying asset (annual, expressed as a decimal)
        option_type : str
            Type of option - 'Call' or 'Put'
            
        Returns:
        -------
        float
            Price of the American option
        """
        # Tree parameters
        dt = T / self.steps
        u = np.exp(sigma * np.sqrt(dt))  # Up factor
        d = 1 / u  # Down factor
        p = (np.exp(r * dt) - d) / (u - d)  # Risk-neutral probability
        discount = np.exp(-r * dt)
        
        # Initialize price tree
        price_tree = np.zeros(self.steps + 1)
        for i in range(self.steps + 1):
            price_tree[i] = S * (u ** (self.steps - i)) * (d ** i)
        
        # Initialize option value at expiration (time step n)
        option_tree = np.zeros(self.steps + 1)
        for i in range(self.steps + 1):
            if option_type == 'Call':
                option_tree[i] = max(0, price_tree[i] - K)
            else:  # Put
                option_tree[i] = max(0, K - price_tree[i])
        
        # Backward induction through the tree
        for step in range(self.steps - 1, -1, -1):
            for i in range(step + 1):
                # Calculate underlying asset price at this node
                spot_price = S * (u ** (step - i)) * (d ** i)
                
                # Calculate option value using risk-neutral valuation
                option_tree[i] = discount * (p * option_tree[i] + (1 - p) * option_tree[i + 1])
                
                # Check for early exercise
                if option_type == 'Call':
                    exercise_value = max(0, spot_price - K)
                else:  # Put
                    exercise_value = max(0, K - spot_price)
                
                # American option value is the maximum of holding or exercising
                option_tree[i] = max(option_tree[i], exercise_value)
        
        # Return the option price at the root node
        return option_tree[0]
    
    def compute_delta(self, S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
        """
        Calculate the delta (sensitivity to underlying price) of an American option.
        
        Uses finite difference approximation.
        
        Parameters:
        ----------
        S : float
            Current price of the underlying asset
        K : float
            Strike price of the option
        T : float
            Time to expiration in years
        r : float
            Risk-free interest rate (annual, expressed as a decimal)
        sigma : float
            Volatility of the underlying asset (annual, expressed as a decimal)
        option_type : str
            Type of option - 'Call' or 'Put'
            
        Returns:
        -------
        float
            Delta of the American option
        """
        # Small price change for finite difference approximation
        dS = S * 0.01
        
        # Calculate option prices at S+dS and S-dS
        price_up = self.compute_price(S + dS, K, T, r, sigma, option_type)
        price_down = self.compute_price(S - dS, K, T, r, sigma, option_type)
        
        # Finite difference approximation of delta
        return (price_up - price_down) / (2 * dS)
    
    def compute_gamma(self, S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
        """
        Calculate the gamma (second derivative of value with respect to underlying price) of an American option.
        
        Uses finite difference approximation.
        
        Parameters:
        ----------
        S : float
            Current price of the underlying asset
        K : float
            Strike price of the option
        T : float
            Time to expiration in years
        r : float
            Risk-free interest rate (annual, expressed as a decimal)
        sigma : float
            Volatility of the underlying asset (annual, expressed as a decimal)
        option_type : str
            Type of option - 'Call' or 'Put'
            
        Returns:
        -------
        float
            Gamma of the American option
        """
        # Small price change for finite difference approximation
        dS = S * 0.01
        
        # Calculate option prices at S+dS, S, and S-dS
        price_up = self.compute_price(S + dS, K, T, r, sigma, option_type)
        price_mid = self.compute_price(S, K, T, r, sigma, option_type)
        price_down = self.compute_price(S - dS, K, T, r, sigma, option_type)
        
        # Finite difference approximation of gamma
        return (price_up - 2 * price_mid + price_down) / (dS ** 2)
    
    def compute_vega(self, S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
        """
        Calculate the vega (sensitivity to volatility) of an American option.
        
        Uses finite difference approximation.
        
        Parameters:
        ----------
        S : float
            Current price of the underlying asset
        K : float
            Strike price of the option
        T : float
            Time to expiration in years
        r : float
            Risk-free interest rate (annual, expressed as a decimal)
        sigma : float
            Volatility of the underlying asset (annual, expressed as a decimal)
        option_type : str
            Type of option - 'Call' or 'Put'
            
        Returns:
        -------
        float
            Vega of the American option
        """
        # Small volatility change for finite difference approximation
        dSigma = 0.001
        
        # Calculate option prices at sigma+dSigma and sigma-dSigma
        price_up = self.compute_price(S, K, T, r, sigma + dSigma, option_type)
        price_down = self.compute_price(S, K, T, r, sigma - dSigma, option_type)
        
        # Finite difference approximation of vega
        return (price_up - price_down) / (2 * dSigma)
    
    def compute_rho(self, S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
        """
        Calculate the rho (sensitivity to interest rate) of an American option.
        
        Uses finite difference approximation.
        
        Parameters:
        ----------
        S : float
            Current price of the underlying asset
        K : float
            Strike price of the option
        T : float
            Time to expiration in years
        r : float
            Risk-free interest rate (annual, expressed as a decimal)
        sigma : float
            Volatility of the underlying asset (annual, expressed as a decimal)
        option_type : str
            Type of option - 'Call' or 'Put'
            
        Returns:
        -------
        float
            Rho of the American option
        """
        # Small interest rate change for finite difference approximation
        dr = 0.001
        
        # Calculate option prices at r+dr and r-dr
        price_up = self.compute_price(S, K, T, r + dr, sigma, option_type)
        price_down = self.compute_price(S, K, T, r - dr, sigma, option_type)
        
        # Finite difference approximation of rho
        return (price_up - price_down) / (2 * dr)
    
    def get_all_values(self, S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> dict:
        """
        Calculate all option Greeks and the price for an American option.
        
        Parameters:
        ----------
        S : float
            Current price of the underlying asset
        K : float
            Strike price of the option
        T : float
            Time to expiration in years
        r : float
            Risk-free interest rate (annual, expressed as a decimal)
        sigma : float
            Volatility of the underlying asset (annual, expressed as a decimal)
        option_type : str
            Type of option - 'Call' or 'Put'
            
        Returns:
        -------
        dict
            Dictionary containing all option values and Greeks
        """
        price = self.compute_price(S, K, T, r, sigma, option_type)
        
        all_values = {
            "S": S,
            "K": K,
            "T": T, 
            "r": r,
            "sigma": sigma,
            "option_type": option_type,
            "price": price,
            "delta": self.compute_delta(S, K, T, r, sigma, option_type),
            "gamma": self.compute_gamma(S, K, T, r, sigma, option_type),
            "vega": self.compute_vega(S, K, T, r, sigma, option_type),
            "rho": self.compute_rho(S, K, T, r, sigma, option_type),
        }
        return all_values