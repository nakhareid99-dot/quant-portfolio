import numpy as np
import pandas as pd
from numba import jit

@jit(nopython=True)
def simulate_gbm_numba(S0, mu, sigma, T, dt, n_steps, n_sims):
    paths = np.zeros((n_steps + 1, n_sims))
    paths[0] = S0
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)
    for t in range(1, n_steps + 1):
        paths[t] = paths[t-1] * np.exp(drift + diffusion * np.random.standard_normal(n_sims))
    return paths

def run_simulation(S0=100, mu=0.15, sigma=0.25, T=1.0, n_steps=252, n_sims=10000):
    dt = T / n_steps
    paths = simulate_gbm_numba(S0, mu, sigma, T, dt, n_steps, n_sims)
    date_index = pd.date_range(start='2024-01-01', periods=n_steps+1, freq='B')
    return pd.DataFrame(paths, index=date_index)
