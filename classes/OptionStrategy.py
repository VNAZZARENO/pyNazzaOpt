import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Import your classes
from classes.instruments import Stock, AmericanOption
from classes.position import Position
from classes.portfolio import Portfolio
from classes.pricer import Pricer, AmericanPricer



class OptionStrategy:
    """Base class to run option strategies"""
    
    def __init__(self, df, stock_ticker, start_date, end_date, initial_capital=100000.0):
        self.df = df
        self.stock_ticker = stock_ticker
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.initial_capital = initial_capital
        self.portfolio = Portfolio(name="Option Strategy")
        self.portfolio.adjust_bank(initial_capital, start_date)
        self.pricer = AmericanPricer(steps=100)
        
        # Track results
        self.trades = []
        self.daily_values = {self.start_date: initial_capital}
        self.trade_months = set()
        
        # Track assignments and statistics
        self.assignments = []
        self.assignment_stats = {
            'total_trades': 0,
            'assigned_trades': 0,
            'assignment_rate': 0.0,
            'avg_loss_when_assigned': 0.0,
            'max_loss_when_assigned': 0.0,
            'total_premium_collected': 0.0,
            'profit_vs_premium_ratio': 0.0,
        }
    
    def run_monthly_strategy(self, strategy_func, position_sizing=1):
        """
        Run a monthly option strategy
        
        Parameters:
        -----------
        strategy_func : callable
            Function that creates and returns option positions for a given date
        position_sizing : float
            Multiplier for position size (default 1)
        """
        # Get dates for backtest
        dates = sorted(self.df.index[(self.df.index >= self.start_date) & 
                                    (self.df.index <= self.end_date)])
        
        # Track current trade
        current_trade = None
        last_trade_date = None
        
        # Initialize to track premium collection
        total_premium_collected = 0.0
        
        for date in dates:
            try:
                # Get current market data
                price = float(self.df.loc[date, f"{self.stock_ticker} Equity Price"])
                iv = float(self.df.loc[date, f"{self.stock_ticker} Equity Vol"]) / 100
                rate = float(self.df.loc[date, "USGG2YR Index"]) / 100
                
                # Copy positions from previous date if needed
                previous_dates = [d for d in self.portfolio.positions_dict.keys() if d < date]
                if previous_dates and date not in self.portfolio.positions_dict:
                    latest_previous = max(previous_dates)
                    self.portfolio.positions_dict[date] = self.portfolio.positions_dict[latest_previous].copy()
                    
                    # Copy bank balance too
                    if date not in self.portfolio.bank_dict and latest_previous in self.portfolio.bank_dict:
                        self.portfolio.bank_dict[date] = self.portfolio.bank_dict[latest_previous]
                
                # Update positions with current market data
                if date in self.portfolio.positions_dict:
                    for pos in self.portfolio.positions_dict[date]:
                        if hasattr(pos.instrument, 'underlying'):
                            pos.instrument.underlying._price = price
                            
                        # Update option parameters
                        if hasattr(pos.instrument, 'sigma'):
                            pos.instrument.sigma = iv
                            pos.instrument.r = rate
                            if hasattr(pos.instrument, 'expiry_date') and pos.instrument.expiry_date:
                                pos.instrument.T = max((pos.instrument.expiry_date - date).days / 365, 0.001)
                
                # Check if it's time for a new monthly trade
                current_month_year = (date.year, date.month)
                if current_month_year not in self.trade_months:
                    # Calculate expiration (next month's third Friday)
                    next_month = date.month + 1 if date.month < 12 else 1
                    next_year = date.year if date.month < 12 else date.year + 1
                    expiry_date = get_third_friday(next_year, next_month)
                    
                    # Only proceed if expiry is within our test period
                    if expiry_date <= self.end_date:
                        # Create stock
                        stock = Stock(price=price, ticker=self.stock_ticker)
                        
                        # Call the strategy function to get option positions
                        options = strategy_func(date, expiry_date, stock, price, iv, rate)
                        
                        # Apply position sizing if needed
                        if position_sizing != 1:
                            sized_options = []
                            for opt, action, qty in options:
                                sized_options.append((opt, action, int(qty * position_sizing)))
                            options = sized_options
                        
                        # Create trade record
                        current_trade = {
                            'entry_date': date,
                            'expiry_date': expiry_date,
                            'entry_price': price,
                            'options': options,
                            'status': 'open',
                            'daily_values': {},
                            'assignment_status': 'not_expired'
                        }
                        
                        # Add options to portfolio and track premium
                        trade_premium = 0.0
                        for option, action, quantity in options:
                            option.update_price(self.pricer)
                            
                            # Calculate premium impact
                            # For shorts we receive premium, for longs we pay premium
                            premium_impact = option.price * quantity
                            if action == "Short":
                                trade_premium += premium_impact
                            else:
                                trade_premium -= premium_impact
                            
                            self.portfolio.add_position(Position(
                                instrument=option,
                                quantity=quantity,
                                date=date,
                                action=action
                            ))
                        
                        # Track total premium collected
                        total_premium_collected += trade_premium
                        self.assignment_stats['total_premium_collected'] += trade_premium
                        current_trade['net_premium'] = trade_premium
                        
                        # Track the trade
                        self.trades.append(current_trade)
                        self.trade_months.add(current_month_year)
                        last_trade_date = date
                        
                        print(f"  Net premium collected: ${trade_premium:.2f}")
                        
                        # Log trade
                        print(f"Entered strategy on {date}, expiry: {expiry_date}")
                        for option, action, quantity in options:
                            print(f"  {action} {quantity} {option.option_type}(s) @ {option.K}")
                
                # Check for expired options and assignments
                assignments = self.check_for_assignments(date, price)
                
                # Update portfolio values
                self.portfolio.update_all(date, self.pricer)
                
                # Get current portfolio value - adjust to show initial capital + premium collection
                # This adjusts for the issue in the Portfolio.get_portfolio_value() method
                raw_portfolio_value = self.portfolio.get_portfolio_value()
                
                # For display, show theoretical P&L (initial capital + net change)
                # This approach preserves the true P&L pattern rather than the misleading downward slope
                bank_value = self.portfolio.get_bank_value() if date in self.portfolio.bank_dict else 0
                position_values = 0
                
                if date in self.portfolio.positions_dict:
                    for pos in self.portfolio.positions_dict[date]:
                        if not pos.closed:
                            # For short positions, profit from decreasing price
                            sign = -1 if pos.action == "Short" else 1
                            if hasattr(pos.instrument, 'price'):
                                position_values += sign * pos.quantity * pos.instrument.price
                
                # Store the corrected portfolio value
                self.daily_values[date] = self.initial_capital + bank_value + position_values
                
                # Track option values for current trade
                if current_trade and current_trade['status'] == 'open':
                    # Check if this trade has expired
                    if date >= current_trade['expiry_date']:
                        current_trade['status'] = 'closed'
                        current_trade['exit_date'] = date
                        current_trade['exit_price'] = price
                        
                        print(f"Closed position on {date}, price: {price}")
                
            except Exception as e:
                print(f"Error on {date}: {e}")
                # Carry forward previous value if available
                if self.daily_values and date > min(self.daily_values.keys()):
                    last_date = max(d for d in self.daily_values.keys() if d < date)
                    self.daily_values[date] = self.daily_values[last_date]
        
        # Calculate final statistics
        self._calculate_final_statistics()
        
        return pd.Series(self.daily_values)
    
    def check_for_assignments(self, date, price):
        """Check for option assignments on a given date"""
        assignments = []
        
        # First, let the portfolio handle expired options
        expired_options = self.portfolio.check_option_expired(date, price, self.stock_ticker)
        
        # Process any assignments from expired options
        for trade in self.trades:
            if trade['status'] == 'open' and date >= trade['expiry_date']:
                trade['status'] = 'closed'
                trade['exit_date'] = date
                trade['exit_price'] = price
                
                # Check if any options were assigned (ITM at expiration)
                was_assigned = False
                assignment_pnl = 0.0
                assigned_options = []
                
                for option, action, quantity in trade['options']:
                    # Calculate option payoff at expiration
                    payoff = option.compute_payoff(final_underlying_price=price)
                    
                    # If short option is ITM, it will be assigned
                    if action == "Short" and payoff > 0:
                        was_assigned = True
                        option_pnl = -payoff * quantity
                        assignment_pnl += option_pnl
                        assigned_options.append((option, action, quantity, option_pnl))
                
                # Record assignment information
                trade['was_assigned'] = was_assigned
                trade['assignment_pnl'] = assignment_pnl
                trade['assigned_options'] = assigned_options if was_assigned else []
                
                # Update assignment statistics
                self.assignment_stats['total_trades'] += 1
                
                if was_assigned:
                    self.assignment_stats['assigned_trades'] += 1
                    
                    # Record detailed assignment info
                    self.assignments.append({
                        'date': date,
                        'price': price,
                        'trade': trade,
                        'pnl': assignment_pnl,
                        'assigned_options': assigned_options
                    })
                    
                    print(f"  Position was assigned with P&L: ${assignment_pnl:.2f}")
        
        # Update assignment statistics
        if self.assignment_stats['total_trades'] > 0:
            self.assignment_stats['assignment_rate'] = (
                self.assignment_stats['assigned_trades'] / self.assignment_stats['total_trades']
            )
        
        # Calculate average and max loss when assigned
        if self.assignments:
            assigned_pnls = [a['pnl'] for a in self.assignments]
            self.assignment_stats['avg_loss_when_assigned'] = sum(assigned_pnls) / len(assigned_pnls)
            self.assignment_stats['max_loss_when_assigned'] = min(assigned_pnls)  # Min because losses are negative
        
        # Update profit vs premium ratio
        if self.assignment_stats['total_premium_collected'] > 0:
            total_pnl = sum(a['pnl'] for a in self.assignments)
            self.assignment_stats['profit_vs_premium_ratio'] = (
                self.assignment_stats['total_premium_collected'] + total_pnl
            ) / self.assignment_stats['total_premium_collected']
        
        return assignments
    
    def _calculate_final_statistics(self):
        """Calculate and update final strategy statistics"""
        if not self.trades:
            return
        
        # Calculate win rate
        winning_trades = [t for t in self.trades if t.get('status') == 'closed' and 
                         (not t.get('was_assigned', False) or 
                          (t.get('was_assigned', False) and t.get('assignment_pnl', 0) >= 0))]
        
        self.win_rate = len(winning_trades) / len([t for t in self.trades if t.get('status') == 'closed'])
        
        # Calculate average premium collected
        self.avg_premium = self.assignment_stats['total_premium_collected'] / len(self.trades) if self.trades else 0
        
        # Calculate average trade P&L
        closed_trades = [t for t in self.trades if t.get('status') == 'closed']
        
        # Calculate P&L for each trade (premium + assignment P&L if assigned)
        trade_pnls = []
        for trade in closed_trades:
            premium = trade.get('net_premium', 0)
            assignment_pnl = trade.get('assignment_pnl', 0) if trade.get('was_assigned', False) else 0
            trade_pnls.append(premium + assignment_pnl)
        
        self.avg_pnl = sum(trade_pnls) / len(trade_pnls) if trade_pnls else 0
        
        # Calculate max drawdown
        values_series = pd.Series(self.daily_values)
        drawdown = values_series / values_series.cummax() - 1
        self.max_drawdown = drawdown.min() * 100
        
        # Calculate Sharpe ratio (annualized)
        returns = values_series.pct_change().dropna()
        self.sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 0 and returns.std() > 0 else 0
    
    def plot_results(self):
        """Plot strategy results with enhanced statistics"""
        values_series = pd.Series(self.daily_values)
        
        # Create plot with subplots
        fig, axs = plt.subplots(4, 1, figsize=(14, 16), 
                                gridspec_kw={'height_ratios': [3, 1, 1, 1]})
        
        # Portfolio value
        values_series.plot(ax=axs[0])
        axs[0].set_title(f"{self.portfolio.name} - Portfolio Value", fontsize=14)
        axs[0].set_ylabel("Portfolio Value")
        axs[0].grid(True)
        
        # Mark trade entry and exit points
        for trade in self.trades:
            if trade['entry_date'] in values_series.index:
                axs[0].scatter(trade['entry_date'], values_series[trade['entry_date']], 
                             color='green', marker='^', s=70)
            
            if 'exit_date' in trade and trade['exit_date'] in values_series.index:
                marker_color = 'red' if trade.get('was_assigned', False) else 'blue'
                axs[0].scatter(trade['exit_date'], values_series[trade['exit_date']], 
                             color=marker_color, marker='v', s=70)
                
                # Mark assignments with 'A'
                if trade.get('was_assigned', False):
                    axs[0].annotate('A', 
                                  xy=(trade['exit_date'], values_series[trade['exit_date']]),
                                  xytext=(5, 5), textcoords='offset points',
                                  color='red', fontweight='bold')
        
        # Drawdown
        drawdown = (values_series / values_series.cummax() - 1) * 100
        drawdown.plot(ax=axs[1], color='red')
        axs[1].set_title("Portfolio Drawdown")
        axs[1].set_ylabel("Drawdown (%)")
        axs[1].grid(True)
        
        # Option Premium Chart
        entry_dates = [t['entry_date'] for t in self.trades]
        entry_prices = [t['entry_price'] for t in self.trades]
        
        if entry_dates:
            # Calculate option premiums
            premiums = []
            for trade in self.trades:
                if 'net_premium' in trade:
                    premiums.append(trade['net_premium'])
                else:
                    total_premium = 0
                    for option, action, quantity in trade['options']:
                        sign = -1 if action.lower() == 'short' else 1
                        total_premium += sign * option.price * quantity
                    premiums.append(abs(total_premium))
            
            # Calculate as percentage of underlying
            premium_pcts = [premium / price * 100 for premium, price in zip(premiums, entry_prices)]
            
            sc = axs[2].scatter(entry_dates, premium_pcts, c=entry_prices, cmap='viridis', s=80)
            axs[2].set_title("Option Premium as % of Underlying")
            axs[2].set_ylabel("Premium %")
            axs[2].grid(True)
            
            # Add colorbar
            cbar = plt.colorbar(sc, ax=axs[2])
            cbar.set_label(f"{self.stock_ticker} Price")
        
        # Assignment Statistics
        if self.assignments:
            # Plot assignment rates by month
            assignment_df = pd.DataFrame(self.assignments)
            if not assignment_df.empty and 'date' in assignment_df.columns:
                assignment_df['month'] = assignment_df['date'].dt.strftime('%Y-%m')
                monthly_assignments = assignment_df.groupby('month').size()
                
                # Get all months from the backtest period
                all_months = pd.date_range(start=self.start_date, end=self.end_date, freq='MS')
                all_months_str = [d.strftime('%Y-%m') for d in all_months]
                
                # Create a complete monthly series with zeros for months without assignments
                complete_series = pd.Series(0, index=all_months_str)
                for month in monthly_assignments.index:
                    if month in complete_series.index:
                        complete_series[month] = monthly_assignments[month]
                
                axs[3].bar(complete_series.index, complete_series.values, color='red')
                axs[3].set_title("Monthly Option Assignments")
                axs[3].set_ylabel("Count")
                axs[3].set_xticklabels(complete_series.index, rotation=45)
                axs[3].grid(True, axis='y')
            else:
                # If no assignments, show assignment statistics as text
                axs[3].axis('off')
                axs[3].text(0.5, 0.5, "No assignments during the backtest period", 
                          ha='center', va='center', fontsize=12)
        else:
            # Show assignment stats in text form
            axs[3].axis('off')
            stats_text = "\n".join([
                f"Assignment Statistics:",
                f"Total Trades: {self.assignment_stats['total_trades']}",
                f"Assignment Rate: {self.assignment_stats['assignment_rate']:.2%}",
                f"Total Premium: ${self.assignment_stats['total_premium_collected']:.2f}",
                f"Profit/Premium Ratio: {self.assignment_stats['profit_vs_premium_ratio']:.2f}"
            ])
            axs[3].text(0.5, 0.5, stats_text, ha='center', va='center', fontsize=12)
        
        plt.tight_layout()
        plt.show()
    
    def print_statistics(self):
        """Print statistics about the strategy performance"""
        print(f"\n{'='*50}")
        print(f"Strategy: {self.portfolio.name}")
        print(f"{'='*50}")
        
        # Performance metrics
        start_value = list(self.daily_values.values())[0]
        end_value = list(self.daily_values.values())[-1]
        total_return = (end_value / start_value - 1) * 100
        
        print(f"Period: {self.start_date.date()} to {self.end_date.date()}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Value: ${end_value:,.2f}")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Sharpe Ratio: {self.sharpe_ratio:.2f}")
        print(f"Max Drawdown: {self.max_drawdown:.2f}%")
        
        # Trade statistics
        print(f"\nTrade Statistics:")
        print(f"Total Trades: {self.assignment_stats['total_trades']}")
        print(f"Win Rate: {self.win_rate:.2%}")
        print(f"Average Premium: ${self.avg_premium:.2f}")
        print(f"Average P&L per Trade: ${self.avg_pnl:.2f}")
        
        # Assignment statistics
        print(f"\nAssignment Statistics:")
        print(f"Total Assignments: {self.assignment_stats['assigned_trades']}")
        print(f"Assignment Rate: {self.assignment_stats['assignment_rate']:.2%}")
        if self.assignments:
            print(f"Average Loss When Assigned: ${self.assignment_stats['avg_loss_when_assigned']:.2f}")
            print(f"Maximum Loss When Assigned: ${self.assignment_stats['max_loss_when_assigned']:.2f}")
        print(f"Total Premium Collected: ${self.assignment_stats['total_premium_collected']:.2f}")
        print(f"Profit/Premium Ratio: {self.assignment_stats['profit_vs_premium_ratio']:.2f}")
        print(f"{'='*50}")

class MultiStockStrategy:
    """Class to manage a portfolio of multiple stock option strategies"""
    
    def __init__(self, df, stock_tickers, start_date, end_date, total_capital=1000000.0):
        """
        Initialize the multi-stock strategy
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with price and volatility data for all stocks
        stock_tickers : list
            List of stock ticker symbols
        start_date : str or pandas.Timestamp
            Start date for backtest
        end_date : str or pandas.Timestamp
            End date for backtest
        total_capital : float
            Total capital for the strategy
        """
        self.df = df
        self.stock_tickers = stock_tickers
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.total_capital = total_capital
        
        # Calculate capital allocation per stock (equal weighting)
        self.capital_per_stock = total_capital / len(stock_tickers)
        
        # Dictionary to store individual strategies
        self.strategies = {}
        
        # Track overall portfolio value
        self.daily_values = {}
        
        # Store portfolio weights
        self.weights = {ticker: 1.0 / len(stock_tickers) for ticker in stock_tickers}
    
    def set_weights(self, weights_dict):
        """Set custom weights for each stock in the portfolio"""
        # Validate weights
        if sum(weights_dict.values()) != 1.0:
            # Normalize weights
            total = sum(weights_dict.values())
            weights_dict = {k: v/total for k, v in weights_dict.items()}
            print(f"Weights normalized to sum to 1.0: {weights_dict}")
        
        # Update weights and capital allocation
        self.weights = weights_dict
        for ticker, weight in weights_dict.items():
            if ticker in self.stock_tickers:
                self.capital_per_stock_dict[ticker] = self.total_capital * weight
    
    def run(self, strategy_class, strategy_params=None, position_sizing_func=None):
        """
        Run a specific strategy for all stocks in the portfolio
        
        Parameters:
        -----------
        strategy_class : class
            Strategy class to run
        strategy_params : dict, optional
            Parameters for the strategy
        position_sizing_func : callable, optional
            Function to determine position sizing based on stock price
        """
        if strategy_params is None:
            strategy_params = {}
        
        # Initialize capital per stock dictionary
        self.capital_per_stock_dict = {ticker: self.capital_per_stock * self.weights.get(ticker, 1.0/len(self.stock_tickers)) 
                                      for ticker in self.stock_tickers}
        
        # Run strategy for each stock
        for ticker in self.stock_tickers:
            print(f"\nRunning strategy for {ticker}...")
            
            # Create strategy
            strategy = strategy_class(
                df=self.df, 
                stock_ticker=ticker,
                start_date=self.start_date,
                end_date=self.end_date,
                initial_capital=self.capital_per_stock_dict[ticker]
            )
            
            # Calculate position sizing based on stock price if function provided
            position_size = 1
            if position_sizing_func is not None:
                try:
                    # Get initial stock price
                    initial_price = float(self.df.loc[self.start_date, f"{ticker} Equity Price"])
                    position_size = position_sizing_func(initial_price, self.capital_per_stock_dict[ticker])
                except Exception as e:
                    print(f"Error calculating position size: {e}")
            
            # Run the strategy
            values = strategy.run_monthly_strategy(**strategy_params, position_sizing=position_size)
            
            # Store the strategy
            self.strategies[ticker] = strategy
            
            # Add to overall portfolio value
            for date, value in values.items():
                if date not in self.daily_values:
                    self.daily_values[date] = 0
                self.daily_values[date] += value
        
        # Convert to pandas Series for plotting
        return pd.Series(self.daily_values)
    
    def plot_overall_performance(self):
        """Plot overall portfolio performance"""
        values_series = pd.Series(self.daily_values)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        values_series.plot(ax=ax)
        ax.set_title("Multi-Stock Portfolio Value")
        ax.set_ylabel("Portfolio Value")
        ax.grid(True)
        plt.tight_layout()
        plt.show()
    
    def plot_individual_performances(self):
        """Plot performance of each stock strategy"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for ticker, strategy in self.strategies.items():
            pd.Series(strategy.daily_values).plot(ax=ax, label=ticker)
        
        ax.set_title("Individual Stock Strategy Performance")
        ax.set_ylabel("Strategy Value")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.show()
    
    def plot_contribution_to_return(self):
        """Plot each stock's contribution to overall return"""
        initial_values = {ticker: list(strategy.daily_values.values())[0] 
                         for ticker, strategy in self.strategies.items()}
        final_values = {ticker: list(strategy.daily_values.values())[-1] 
                       for ticker, strategy in self.strategies.items()}
        
        returns = {ticker: (final_values[ticker] / initial_values[ticker] - 1) * 100 
                  for ticker in self.strategies.keys()}
        
        # Plot returns
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, len(returns)))
        
        ax.bar(returns.keys(), returns.values(), color=colors)
        ax.set_title("Percentage Return by Stock")
        ax.set_ylabel("Return (%)")
        ax.grid(True, axis='y')
        
        # Add total return line
        total_initial = sum(initial_values.values())
        total_final = sum(final_values.values())
        total_return = (total_final / total_initial - 1) * 100
        
        ax.axhline(y=total_return, color='r', linestyle='-', label=f'Total: {total_return:.2f}%')
        ax.legend()
        
        plt.tight_layout()
        plt.show()
    
    def print_statistics(self):
        """Print overall and individual strategy statistics"""
        print(f"\n{'='*60}")
        print(f"Multi-Stock Strategy Performance")
        print(f"{'='*60}")
        
        # Overall performance
        start_value = list(self.daily_values.values())[0]
        end_value = list(self.daily_values.values())[-1]
        total_return = (end_value / start_value - 1) * 100
        
        print(f"Period: {self.start_date.date()} to {self.end_date.date()}")
        print(f"Total Capital: ${self.total_capital:,.2f}")
        print(f"Final Value: ${end_value:,.2f}")
        print(f"Total Return: {total_return:.2f}%")
        
        # Calculate portfolio metrics
        values_series = pd.Series(self.daily_values)
        returns = values_series.pct_change().dropna()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 0 and returns.std() > 0 else 0
        
        # Calculate drawdown
        drawdown = values_series / values_series.cummax() - 1
        max_drawdown = drawdown.min() * 100
        
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Max Drawdown: {max_drawdown:.2f}%")
        
        # Individual stock performance
        print(f"\nIndividual Stock Performance:")
        print(f"{'-'*60}")
        print(f"{'Stock':^10} | {'Weight':^8} | {'Return':^8} | {'Sharpe':^8} | {'Max DD':^8} | {'Assign %':^8}")
        print(f"{'-'*60}")
        
        for ticker, strategy in self.strategies.items():
            ticker_start = list(strategy.daily_values.values())[0]
            ticker_end = list(strategy.daily_values.values())[-1]
            ticker_return = (ticker_end / ticker_start - 1) * 100
            
            assign_rate = strategy.assignment_stats['assignment_rate'] * 100
            
            print(f"{ticker:^10} | {self.weights[ticker]:^8.2f} | {ticker_return:^8.2f}% | "
                  f"{strategy.sharpe_ratio:^8.2f} | {strategy.max_drawdown:^8.2f}% | {assign_rate:^8.2f}%")
        
        print(f"{'='*60}")

class FactorAnalysis:
    """Analyze impact of macroeconomic factors on strategy performance"""
    
    def __init__(self, strategy, factors_df):
        """
        Initialize factor analysis
        
        Parameters:
        -----------
        strategy : OptionStrategy or MultiStockStrategy
            The strategy to analyze
        factors_df : pandas.DataFrame
            DataFrame with macroeconomic factors (VIX, rates, etc.)
        """
        self.strategy = strategy
        self.factors_df = factors_df
        
        # Create return series from strategy
        if hasattr(strategy, 'daily_values'):
            self.returns = pd.Series(strategy.daily_values).pct_change().dropna()
        else:
            self.returns = pd.Series()
        
        # Align factor data with returns
        self.aligned_data = self._align_data()
    
    def _align_data(self):
        """Align factor data with strategy returns"""
        # Get common dates
        common_dates = set(self.returns.index).intersection(set(self.factors_df.index))
        
        # Create aligned DataFrame
        aligned_df = pd.DataFrame({'strategy_returns': self.returns})
        
        # Add factors
        for col in self.factors_df.columns:
            aligned_df[col] = pd.Series({date: self.factors_df.loc[date, col] 
                                       for date in common_dates if date in self.factors_df.index})
        
        return aligned_df.dropna()
    
    def calculate_correlations(self):
        """Calculate correlations between strategy returns and factors"""
        if self.aligned_data.empty:
            print("No aligned data available for correlation analysis")
            return pd.Series()
        
        return self.aligned_data.corr()['strategy_returns'].drop('strategy_returns')
    
    def plot_correlations(self):
        """Plot correlations between strategy returns and factors"""
        correlations = self.calculate_correlations()
        
        if correlations.empty:
            print("No correlations to plot")
            return
        
        # Sort correlations
        correlations = correlations.sort_values()
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(correlations.index, correlations.values)
        
        # Color positive and negative correlations differently
        for i, bar in enumerate(bars):
            if correlations.values[i] < 0:
                bar.set_color('red')
            else:
                bar.set_color('green')
        
        ax.set_title("Correlation of Strategy Returns with Factors")
        ax.set_xlabel("Correlation Coefficient")
        ax.grid(True, axis='x')
        
        plt.tight_layout()
        plt.show()
    
    def run_regression(self, factors=None):
        """
        Run linear regression to analyze factor impact
        
        Parameters:
        -----------
        factors : list, optional
            List of factor names to include in regression
        """
        if self.aligned_data.empty:
            print("No aligned data available for regression analysis")
            return None
        
        import statsmodels.api as sm
        
        # Select factors for regression
        if factors is None:
            factors = [col for col in self.aligned_data.columns if col != 'strategy_returns']
        
        # Prepare data
        X = self.aligned_data[factors]
        X = sm.add_constant(X)
        y = self.aligned_data['strategy_returns']
        
        # Run regression
        model = sm.OLS(y, X).fit()
        
        # Print results
        print(model.summary())
        
        return model
    
    def plot_rolling_correlations(self, window=30):
        """
        Plot rolling correlations between strategy returns and factors
        
        Parameters:
        -----------
        window : int
            Rolling window size in days
        """
        if self.aligned_data.empty or len(self.aligned_data) < window:
            print("Not enough data for rolling correlation analysis")
            return
        
        # Calculate rolling correlations
        factors = [col for col in self.aligned_data.columns if col != 'strategy_returns']
        rolling_corrs = pd.DataFrame(index=self.aligned_data.index[window-1:])
        
        for factor in factors:
            rolling_corr = self.aligned_data['strategy_returns'].rolling(window=window).corr(
                self.aligned_data[factor])
            rolling_corrs[factor] = rolling_corr
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        for factor in rolling_corrs.columns:
            rolling_corrs[factor].plot(ax=ax, label=factor)
        
        ax.set_title(f"{window}-Day Rolling Correlation with Strategy Returns")
        ax.set_ylabel("Correlation Coefficient")
        ax.legend(loc='best')
        ax.grid(True)
        
        plt.tight_layout()
        plt.show()

