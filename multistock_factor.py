import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf  # You'll need to install this if you don't have it

# Import our enhanced strategy classes
from enhanced_option_strategy import (
    OptionStrategy, MultiStockStrategy, FactorAnalysis,
    run_straddle, run_strangle, run_strangle_with_wings,
    run_strangle_with_put_hedge, run_straddle_with_put_hedge,
    compare_strategies, calculate_position_size
)

def load_sample_data():
    """
    Load sample stock, volatility, and rate data
    This is a placeholder - in your actual code, you'll use your own data
    """
    # Sample stocks to use
    stocks = ['DG.PA', 'BNP.PA', 'SAN.PA', 'MC.PA', 'OR.PA']  # Example French stocks
    
    # Download historical price data
    start_date = '2015-01-01'
    end_date = '2023-12-31'
    
    # Get stock data
    stock_data = yf.download(stocks, start=start_date, end=end_date)['Adj Close']
    
    # Create a DataFrame in the format expected by our strategy classes
    df = pd.DataFrame(index=stock_data.index)
    
    # Add stock price columns
    for ticker in stocks:
        df[f"{ticker} Equity Price"] = stock_data[ticker]
    
    # Add volatility columns (here we use a 30-day rolling volatility as a proxy)
    for ticker in stocks:
        returns = np.log(stock_data[ticker] / stock_data[ticker].shift(1))
        df[f"{ticker} Equity Vol"] = returns.rolling(window=30).std() * np.sqrt(252) * 100  # Annualized vol in percentage
    
    # Add interest rate data (using a constant as a placeholder)
    df["USGG2YR Index"] = 2.0  # 2% interest rate
    
    # Get economic factors
    factors = get_economic_factors(start_date, end_date)
    
    # Merge with our data
    df = df.join(factors)
    
    # Forward fill missing values
    df = df.fillna(method='ffill')
    
    return df, stocks

def get_economic_factors(start_date, end_date):
    """
    Get economic factors for analysis
    This is a placeholder - in your actual code, you'll use your own factor data
    """
    # Create a date range
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    
    # Create a DataFrame with random factor data
    np.random.seed(42)  # For reproducibility
    factors = pd.DataFrame(index=dates)
    
    # VIX-like volatility index
    factors['VIX'] = np.random.normal(20, 5, len(dates))
    factors['VIX'] = factors['VIX'].clip(lower=10)  # Ensure positive
    
    # Interest rates
    factors['US_10Y'] = np.random.normal(2.5, 0.5, len(dates))
    factors['EUR_10Y'] = np.random.normal(1.5, 0.5, len(dates))
    
    # Economic indicators
    factors['GDP_Growth'] = np.random.normal(2.0, 0.3, len(dates))
    factors['Inflation'] = np.random.normal(2.0, 0.5, len(dates))
    factors['Unemployment'] = np.random.normal(5.0, 0.5, len(dates))
    
    # Commodity prices
    factors['Oil_Price'] = np.random.normal(70, 10, len(dates))
    factors['Gold_Price'] = np.random.normal(1500, 100, len(dates))
    
    # Market indicators
    factors['Market_PE'] = np.random.normal(15, 2, len(dates))
    factors['Sentiment_Index'] = np.random.normal(50, 10, len(dates))
    
    # Add some trend and seasonality
    t = np.arange(len(dates))
    
    # Add trend component
    factors['VIX'] += np.sin(t/365 * 2 * np.pi) * 5  # Annual cycle
    factors['Oil_Price'] += t/len(t) * 20  # Upward trend
    
    # Add auto-correlation
    for col in factors.columns:
        factors[col] = factors[col].rolling(window=5).mean().fillna(factors[col])
    
    return factors

def run_multistock_backtest():
    """Run a backtest on multiple stocks"""
    # Load data
    print("Loading market data...")
    df, stocks = load_sample_data()
    
    # Convert tickers to format used in our data
    stock_tickers = [s.replace('.', ' ') for s in stocks]
    
    # Define backtest period
    start_date = '2018-01-01'
    end_date = '2022-12-31'
    
    # Create multi-stock strategy
    print(f"Creating multi-stock strategy for {stock_tickers}...")
    multi_strategy = MultiStockStrategy(
        df=df,
        stock_tickers=stock_tickers,
        start_date=start_date,
        end_date=end_date,
        total_capital=1000000.0  # €1M initial capital
    )
    
    # Custom position sizing function
    def custom_position_sizing(price, allocated_capital):
        # Use half of the allocated capital for options
        # Each option controls 100 shares
        option_capital = allocated_capital / 2
        notional_exposure = option_capital * 2  # Use leverage
        
        # Calculate number of options
        num_options = int(notional_exposure / (price * 100))
        
        # Ensure minimum of 1 option contract
        return max(1, num_options)
    
    # Run short strangle strategy on all stocks
    print("Running short strangle strategy on all stocks...")
    
    def strangle_strategy_func(date, expiry, stock, price, iv, rate):
        """Strangle strategy function to pass to multi-stock strategy"""
        otm_pct = 0.05
        call_strike = round(price * (1 + otm_pct))
        put_strike = round(price * (1 - otm_pct))
        
        # Calculate time to expiration
        T = (expiry - date).days / 365
        
        from classes.instruments import AmericanOption
        
        # Create the options
        call_option = AmericanOption(
            underlying=stock,
            K=call_strike,
            T=T,
            r=rate,
            sigma=iv,
            option_type="Call",
            expiry_date=expiry
        )
        
        put_option = AmericanOption(
            underlying=stock,
            K=put_strike,
            T=T,
            r=rate,
            sigma=iv,
            option_type="Put",
            expiry_date=expiry
        )
        
        # Return as a list of (option, action, quantity) tuples
        return [(call_option, "Short", 1), (put_option, "Short", 1)]
    
    # Run the strategy
    portfolio_values = multi_strategy.run(
        strategy_class=OptionStrategy,
        strategy_params={"strategy_func": strangle_strategy_func},
        position_sizing_func=custom_position_sizing
    )
    
    # Plot overall performance
    print("Plotting results...")
    multi_strategy.plot_overall_performance()
    multi_strategy.plot_individual_performances()
    multi_strategy.plot_contribution_to_return()
    
    # Print statistics
    multi_strategy.print_statistics()
    
    # Extract economic factors for analysis
    factor_columns = ['VIX', 'US_10Y', 'EUR_10Y', 'GDP_Growth', 
                     'Inflation', 'Unemployment', 'Oil_Price', 
                     'Gold_Price', 'Market_PE', 'Sentiment_Index']
    
    factors_df = df[factor_columns]
    
    # Create factor analysis object
    print("Running factor analysis...")
    fa = FactorAnalysis(multi_strategy, factors_df)
    
    # Plot correlations
    fa.plot_correlations()
    
    # Run regression to identify important factors
    fa.run_regression(factors=['VIX', 'US_10Y', 'Inflation', 'Oil_Price'])
    
    # Plot rolling correlations to see how relationships change over time
    fa.plot_rolling_correlations(window=60)  # 60-day rolling window
    
    return multi_strategy, fa

def compare_strategies_on_single_stock():
    """Compare different option strategies on a single stock"""
    # Load data
    print("Loading market data...")
    df, stocks = load_sample_data()
    
    # Use first stock for comparison
    stock_ticker = stocks[0].replace('.', ' ')
    
    # Define backtest period
    start_date = '2018-01-01'
    end_date = '2022-12-31'
    
    # Compare strategies
    print(f"Comparing strategies on {stock_ticker}...")
    results, strategy_objects = compare_strategies(
        df=df,
        stock_ticker=stock_ticker,
        start_date=start_date,
        end_date=end_date,
        initial_capital=200000.0,  # €200k initial capital
        position_sizing=1  # Default position sizing
    )
    
    # Extract economic factors for analysis
    factor_columns = ['VIX', 'US_10Y', 'EUR_10Y', 'GDP_Growth', 
                     'Inflation', 'Unemployment', 'Oil_Price', 
                     'Gold_Price', 'Market_PE', 'Sentiment_Index']
    
    factors_df = df[factor_columns]
    
    # Create strategy returns dictionary
    strategy_returns = {}
    for name, values in results.items():
        strategy_returns[name] = values.pct_change().dropna()
    
    # Create factor analysis
    from factor import StrategyFactorAnalysis
    
    factor_analysis = StrategyFactorAnalysis(strategy_returns, factors_df)
    
    # Plot factor correlations
    factor_analysis.plot_factor_correlations(top_n=8)
    
    # Build regression models for each strategy
    for strategy in strategy_returns.keys():
        print(f"\nBuilding factor model for {strategy}:")
        factor_analysis.build_factor_model(strategy=f"{strategy}_return")
    
    # Visualize optimal conditions for each strategy
    for strategy in strategy_returns.keys():
        print(f"\nOptimal conditions for {strategy}:")
        factor_analysis.visualize_optimal_conditions(strategy=f"{strategy}_return")
    
    # Build strategy selection model
    print("\nBuilding strategy selection model...")
    selection_models = factor_analysis.build_strategy_selection_model()
    
    # Plot selection factors
    factor_analysis.plot_strategy_selection_factors(selection_models)
    
    # Suggest current strategy based on most recent factor values
    current_factors = {col: df[col].iloc[-1] for col in factor_columns}
    
    print("\nSuggested strategy based on current market conditions:")
    suggested = factor_analysis.suggest_current_strategy(selection_models, current_factors)
    
    return results, strategy_objects, factor_analysis, suggested

if __name__ == "__main__":
    # Run backtest on multiple stocks
    print("=" * 80)
    print("RUNNING MULTI-STOCK OPTION STRATEGY BACKTEST")
    print("=" * 80)
    multi_strategy, factor_analysis = run_multistock_backtest()
    
    print("\n" + "=" * 80)
    print("COMPARING OPTION STRATEGIES ON SINGLE STOCK")
    print("=" * 80)
    results, strategies, strategy_fa, suggested = compare_strategies_on_single_stock()
    
    print(f"\nBased on current market conditions, the suggested strategy is: {suggested}")
    print("=" * 80)