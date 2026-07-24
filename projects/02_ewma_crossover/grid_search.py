"""
Grid Search สำหรับ EWMA Crossover
"""
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.data_fetcher import DataFetcher
from strategy import generate_signals, backtest

def run_grid_search(ticker='AAPL', start='2020-01-01',
                    fast_range=range(2, 22, 2),
                    slow_range=range(20, 61, 5)):
    df_raw = DataFetcher.get_stock_data(ticker, start=start)
    results = []
    for fast in tqdm(fast_range, desc="Fast Span"):
        for slow in tqdm(slow_range, desc="Slow Span", leave=False):
            if fast >= slow:
                continue
            try:
                df_strat = generate_signals(df_raw.copy(), fast_span=fast, slow_span=slow)
                result = backtest(df_strat, initial_capital=100000, transaction_cost=0.001)
                results.append({
                    'fast_span': fast,
                    'slow_span': slow,
                    'total_return': result['total_return'],
                    'sharpe_ratio': result['sharpe_ratio'],
                    'max_drawdown': result['max_drawdown'],
                    'win_rate': result['win_rate'],
                    'final_capital': result['final_capital']
                })
            except Exception as e:
                continue
    df_results = pd.DataFrame(results)
    return df_results

def find_best_params(df_results, metric='sharpe_ratio'):
    best_idx = df_results[metric].idxmax()
    return df_results.loc[best_idx]
