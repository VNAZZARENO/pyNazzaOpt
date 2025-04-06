# data_fetcher.py
import yfinance as yf
import pandas as pd
from typing import Union, List, Dict
from datetime import timedelta

def fetch_data(
    tickers: Union[str, List[str]],
    start_date: str,
    end_date: str,
    interval: str = "1d",
    auto_adjust: bool = True
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Fetch historical data from Yahoo Finance for one or multiple tickers.

    Parameters
    ----------
    tickers : str or List[str]
        The ticker symbol(s) to download. E.g. "AAPL" or ["AAPL", "MSFT"].
    start_date : str
        Start date (YYYY-MM-DD) for fetching data.
    end_date : str
        End date (YYYY-MM-DD) for fetching data.
    interval : str, optional
        Data interval. One of {'1d', '1wk', '1mo', '1h', '5m', etc.}.
        Defaults to "1d".
    auto_adjust : bool, optional
        If True, adjust OHLC prices for splits/dividends/Corporate Actions. Defaults to True. 
    
    Returns
    -------
    pd.DataFrame or Dict[str, pd.DataFrame]
        - If a single ticker was provided, returns a single DataFrame.
        - If a list of tickers was provided, returns a dict of {ticker: DataFrame}.
          Each DataFrame has columns ['Open', 'High', 'Low', 'Close', 'Volume'] (and possibly 'Adj Close' if auto_adjust=False).
    """
    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        interval=interval,
        group_by="ticker",
        auto_adjust=auto_adjust,
        progress=False,
        threads=True
    )
    
    if isinstance(tickers, str):
        return data

    dict_data = {}
    for ticker in tickers:
        dict_data[ticker] = data[ticker].copy()

    return dict_data

def fetch_option_strike(ticker: Union[str, List[str]], date: pd.Timestamp, date_range: int = 5):
    """
    Fetches the possible strike prices for options for a given ticker and nearby expiration dates.
    If the exact date is unavailable, checks dates within the specified range, oscillating around the initial date.

    Parameters:
    - ticker: str or List[str]
        The stock ticker(s) to fetch option strikes for.
    - date: pd.Timestamp
        The target expiration date for which to fetch strike prices.
    - date_range: int (default: 5)
        The maximum number of days to search around the initial date.

    Returns:
    - dict
        A dictionary with tickers as keys and lists of possible strike prices as values.
    - None
        If no valid expiration date is found within the range.
    """
    if isinstance(ticker, str):
        ticker = [ticker]
    
    strikes = {}
    for tick in ticker:
        try:
            stock = yf.Ticker(tick)
            options_dates = stock.options  
            options_dates = [pd.Timestamp(d) for d in options_dates]

            date_offsets = [0] + [i for pair in zip(range(-1, -date_range - 1, -1), range(1, date_range + 1)) for i in pair]
            search_dates = [date + timedelta(days=offset) for offset in date_offsets]

            valid_date = next((d for d in search_dates if d in options_dates), None)
            if not valid_date:
                print(f"No valid expiration date found for {tick} within ±{date_range} days of {date}.")
                strikes[tick] = None
                continue

            expiration = valid_date.strftime('%Y-%m-%d')
            options_chain = stock.option_chain(expiration)
            strikes[tick] = sorted(set(options_chain.calls['strike']).union(options_chain.puts['strike']))
            print(f"Found valid expiration date for {tick}: {valid_date}")
        
        except Exception as e:
            print(f"Error fetching option strikes for {tick}: {e}")
            strikes[tick] = None

    return strikes


def main():
    
    df = fetch_data("AAPL", "2023-01-01", "2023-06-30")
    print("Head of AAPL data:\n", df.head())
    tickers = ["AAPL", "MSFT", "GOOGL"]
    data_dict = fetch_data(tickers, "2023-01-01", "2023-06-30")
    for t in data_dict:
        print(f"\n--- {t} ---")
        print(data_dict[t].head())
    

    print("\n Test Fetch Strike Prices")

    ticker = "AAPL"
    target_date = pd.Timestamp("2025-02-13")
    result = fetch_option_strike(ticker, target_date, date_range=10)
    print(result)

if __name__ == "__main__":
    main()