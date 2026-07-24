import os
import pandas as pd
import yfinance as yf
from datetime import datetime

class DataFetcher:
    @staticmethod
    def get_stock_data(ticker, start="2015-01-01", end=None):
        if end is None:
            end = datetime.today().strftime('%Y-%m-%d')
        os.makedirs("data", exist_ok=True)
        cache_file = f"data/{ticker}_{start}_{end}.parquet"
        if os.path.exists(cache_file):
            print(f"📂 โหลดจาก Cache: {ticker}")
            return pd.read_parquet(cache_file)
        print(f"🌐 โหลดจาก Yahoo: {ticker}")
        df = yf.download(ticker, start=start, end=end, progress=False)
        df.to_parquet(cache_file)
        return df

    @staticmethod
    def get_multiple_stocks(tickers, start="2015-01-01"):
        df_list = []
        for t in tickers:
            df = DataFetcher.get_stock_data(t, start)
            df_list.append(df['Adj Close'].rename(t))
        return pd.concat(df_list, axis=1)
