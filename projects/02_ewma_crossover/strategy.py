"""
EWMA Crossover Strategy
ใช้ EWMA (Exponentially Weighted Moving Average) 2 เส้น
- Fast EWMA: ตอบสนองไว (period สั้น)
- Slow EWMA: ตอบสนองช้า (period ยาว)
Signal: ซื้อเมื่อ Fast > Slow, ขายเมื่อ Fast < Slow
"""

import pandas as pd
import numpy as np


def calculate_ewma(df: pd.DataFrame, column: str = 'Close', span: int = 20) -> pd.Series:
    """
    คำนวณ EWMA (Exponentially Weighted Moving Average)
    ใช้ pandas ewm() ซึ่งเทียบเท่ากับ RiskMetrics EWMA
    
    Args:
        df: DataFrame ที่มีราคา
        column: ชื่อคอลัมน์ที่ใช้คำนวณ (default: 'Close')
        span: ค่า span (ยิ่งมากยิ่ง平滑)
    
    Returns:
        pd.Series: ค่า EWMA
    """
    return df[column].ewm(span=span, adjust=False).mean()


def calculate_ewma_volatility(returns: pd.Series, lambda_: float = 0.94) -> pd.Series:
    """
    คำนวณ Volatility แบบ EWMA (RiskMetrics 1996)
    สูตร: σ²_t = λ * σ²_{t-1} + (1-λ) * r²_{t-1}
    
    Args:
        returns: ผลตอบแทนรายวัน
        lambda_: decay factor (ค่าdefault ของ RiskMetrics = 0.94)
    
    Returns:
        pd.Series: ค่า volatility (standard deviation) รายวัน
    """
    variance = returns.ewm(alpha=(1 - lambda_), adjust=False).var()
    return np.sqrt(variance)


def generate_signals(df: pd.DataFrame, fast_span: int = 10, slow_span: int = 30) -> pd.DataFrame:
    """
    สร้าง Signal จาก EWMA Crossover
    
    Args:
        df: DataFrame ที่มีราคา 'Close'
        fast_span: ค่า span ของ EWMA เร็ว (default: 10)
        slow_span: ค่า span ของ EWMA ช้า (default: 30)
    
    Returns:
        pd.DataFrame: DataFrame เดิม + คอลัมน์ signal, position
    """
    df = df.copy()
    
    # คำนวณ EWMA
    df['EWMA_fast'] = calculate_ewma(df, span=fast_span)
    df['EWMA_slow'] = calculate_ewma(df, span=slow_span)
    
    # คำนวณ Volatility แบบ EWMA
    df['Returns'] = df['Close'].pct_change()
    df['Volatility'] = calculate_ewma_volatility(df['Returns'])
    
    # สร้าง Signal: 1 = ซื้อ, -1 = ขาย, 0 = ถือ
    df['Signal'] = 0
    df.loc[df['EWMA_fast'] > df['EWMA_slow'], 'Signal'] = 1
    df.loc[df['EWMA_fast'] < df['EWMA_slow'], 'Signal'] = -1
    
    # Position: ตำแหน่งที่ถือ (shift 1 day เพื่อป้องกัน look-ahead bias)
    df['Position'] = df['Signal'].shift(1)
    
    # Strategy Returns
    df['Strategy_Return'] = df['Position'] * df['Returns']
    
    return df


def backtest(df: pd.DataFrame, initial_capital: float = 100000, 
             transaction_cost: float = 0.001) -> dict:
    """
    ทำ Backtest และคำนวณ Performance Metrics
    
    Args:
        df: DataFrame ที่มี 'Strategy_Return' และ 'Returns'
        initial_capital: เงินทุนเริ่มต้น
        transaction_cost: ค่าธรรมเนียมต่อการซื้อขาย (0.1% = 0.001)
    
    Returns:
        dict: ผลลัพธ์การ Backtest
    """
    df = df.copy()
    
    # คำนวณ Cumulative Returns
    df['Cumulative_Strategy'] = (1 + df['Strategy_Return']).cumprod() * initial_capital
    df['Cumulative_BuyHold'] = (1 + df['Returns']).cumprod() * initial_capital
    
    # คำนวณ Drawdown
    rolling_max = df['Cumulative_Strategy'].expanding().max()
    df['Drawdown'] = (df['Cumulative_Strategy'] - rolling_max) / rolling_max
    
    # คำนวณ Metrics
    total_return = (df['Cumulative_Strategy'].iloc[-1] / initial_capital) - 1
    buy_hold_return = (df['Cumulative_BuyHold'].iloc[-1] / initial_capital) - 1
    
    # Sharpe Ratio (Annualized)
    excess_returns = df['Strategy_Return'] - (0.02 / 252)  # risk-free rate 2% annual
    sharpe = (excess_returns.mean() / df['Strategy_Return'].std()) * np.sqrt(252) if df['Strategy_Return'].std() != 0 else 0
    
    # Maximum Drawdown
    max_drawdown = df['Drawdown'].min()
    
    # Win Rate
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
