"""
src/portfolio_optimizer.py
โมดูลสำหรับคำนวณ Efficient Frontier และหา Optimal Portfolio
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

def portfolio_stats(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights) * 252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    return returns, std

def negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate=0.02):
    returns, std = portfolio_stats(weights, mean_returns, cov_matrix)
    sharpe = (returns - risk_free_rate) / std if std != 0 else 0
    return -sharpe

def optimize_portfolio(mean_returns, cov_matrix, risk_free_rate=0.02):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    result = minimize(negative_sharpe, num_assets * [1. / num_assets],
                      args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

def random_portfolios(mean_returns, cov_matrix, num_portfolios=5000, risk_free_rate=0.02):
    results = []
    num_assets = len(mean_returns)
    for _ in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        returns, std = portfolio_stats(weights, mean_returns, cov_matrix)
        sharpe = (returns - risk_free_rate) / std if std != 0 else 0
        results.append([returns, std, sharpe] + list(weights))
    columns = ['Return', 'Volatility', 'Sharpe'] + [f'Weight_{i}' for i in range(num_assets)]
    return pd.DataFrame(results, columns=columns)

def get_min_variance_portfolio(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    def portfolio_volatility(weights):
        return portfolio_stats(weights, mean_returns, cov_matrix)[1]
    result = minimize(portfolio_volatility, num_assets * [1. / num_assets],
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x
