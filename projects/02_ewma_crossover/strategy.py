"""
EWMA Crossover Strategy
"""
import pandas as pd
import numpy as np

def calculate_ewma(df: pd.DataFrame, column: str = 'Close', span: int = 20) -> pd.Series:
    return df[column].ewm(span=span, adjust=False).mean()

def calculate_ewma_volatility(returns: pd.Series, lambda_: float = 0.94) -> pd.Series:
    variance = returns.ewm(alpha=(1 - lambda_), adjust=False).var()
    return np.sqrt(variance)

def generate_signals(df: pd.DataFrame, fast_span: int = 10, slow_span: int = 30) -> pd.DataFrame:
    df = df.copy()
    df['EWMA_fast'] = calculate_ewma(df, span=fast_span)
    df['EWMA_slow'] = calculate_ewma(df, span=slow_span)
    df['Returns'] = df['Close'].pct_change()
    df['Volatility'] = calculate_ewma_volatility(df['Returns'])
    df['Signal'] = 0
    df.loc[df['EWMA_fast'] > df['EWMA_slow'], 'Signal'] = 1
    df.loc[df['EWMA_fast'] < df['EWMA_slow'], 'Signal'] = -1
    df['Position'] = df['Signal'].shift(1)
    df['Strategy_Return'] = df['Position'] * df['Returns']
    return df

def backtest(df: pd.DataFrame, initial_capital: float = 100000, transaction_cost: float = 0.001) -> dict:
    df = df.copy()
    df['Cumulative_Strategy'] = (1 + df['Strategy_Return']).cumprod() * initial_capital
    df['Cumulative_BuyHold'] = (1 + df['Returns']).cumprod() * initial_capital
    rolling_max = df['Cumulative_Strategy'].expanding().max()
    df['Drawdown'] = (df['Cumulative_Strategy'] - rolling_max) / rolling_max
    total_return = (df['Cumulative_Strategy'].iloc[-1] / initial_capital) - 1
    buy_hold_return = (df['Cumulative_BuyHold'].iloc[-1] / initial_capital) - 1
    excess_returns = df['Strategy_Return'] - (0.02 / 252)
    sharpe = (excess_returns.mean() / df['Strategy_Return'].std()) * np.sqrt(252) if df['Strategy_Return'].std() != 0 else 0
    max_drawdown = df['Drawdown'].min()
    winning_trades = df[df['Strategy_Return'] > 0]
    win_rate = len(winning_trades) / len(df[df['Strategy_Return'] != 0]) if len(df[df['Strategy_Return'] != 0]) > 0 else 0
    return {
        'total_return': total_return,
        'buy_hold_return': buy_hold_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'final_capital': df['Cumulative_Strategy'].iloc[-1],
        'df': df
    }
