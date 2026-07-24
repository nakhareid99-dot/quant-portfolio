
"""
projects/03_dashboard/app.py
Streamlit Dashboard สำหรับแสดง Sharpe Ratio และ Efficient Frontier
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
import os

# เพิ่ม Path เพื่อให้เรียก src/ ได้ (สำหรับการ Deploy)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data_fetcher import DataFetcher
from src.portfolio_optimizer import random_portfolios, optimize_portfolio, get_min_variance_portfolio, portfolio_stats

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Quant Dashboard", layout="wide")
st.title("📊 Sharpe Ratio & Efficient Frontier Dashboard")
st.markdown("---")

# Sidebar สำหรับเลือกพารามิเตอร์
with st.sidebar:
    st.header("⚙️ พารามิเตอร์")
    
    # เลือกหุ้น (ใช้หุ้น Tech ยอดนิยม)
    default_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
    tickers_input = st.text_input("ป้อนสัญลักษณ์หุ้น (คั่นด้วยเครื่องหมายจุลภาค)", 
                                  value=",".join(default_tickers))
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    
    # ช่วงเวลา
    start_date = st.date_input("วันที่เริ่มต้น", value=pd.to_datetime("2020-01-01"))
    
    # Risk-Free Rate
    risk_free = st.slider("Risk-Free Rate (Annual)", min_value=0.0, max_value=0.10, 
                          value=0.02, step=0.005, format="%.2f")
    
    # ปุ่ม Refresh
    run_btn = st.button("🔄 อัปเดตข้อมูล", use_container_width=True)

# ถ้ายังไม่กดปุ่ม ให้แสดงข้อความเริ่มต้น
if not run_btn:
    st.info("👈 ตั้งค่าพารามิเตอร์ใน Sidebar แล้วกด 'อัปเดตข้อมูล' เพื่อเริ่มต้น")
    st.stop()

# ตรวจสอบว่ามีหุ้นอย่างน้อย 2 ตัว
if len(tickers) < 2:
    st.error("กรุณาเลือกหุ้นอย่างน้อย 2 ตัว")
    st.stop()

# --- ส่วนประมวลผลหลัก ---
with st.spinner("📥 กำลังดึงข้อมูลและคำนวณ..."):
    try:
        # 1. ดึงข้อมูล
        df_prices = DataFetcher.get_multiple_stocks(tickers, start=start_date.strftime("%Y-%m-%d"))
        
        # 2. คำนวณผลตอบแทน
        returns = df_prices.pct_change().dropna()
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        
        # 3. จำลองพอร์ตสุ่ม (สำหรับ Plot Frontier)
        df_random = random_portfolios(mean_returns, cov_matrix, num_portfolios=5000, risk_free_rate=risk_free)
        
        # 4. หาพอร์ตที่เหมาะสมที่สุด
        optimal_weights = optimize_portfolio(mean_returns, cov_matrix, risk_free_rate)
        opt_return, opt_vol = portfolio_stats(optimal_weights, mean_returns, cov_matrix)
        opt_sharpe = (opt_return - risk_free) / opt_vol if opt_vol != 0 else 0
        
        # 5. หาพอร์ต Min Variance
        min_var_weights = get_min_variance_portfolio(mean_returns, cov_matrix)
        min_return, min_vol = portfolio_stats(min_var_weights, mean_returns, cov_matrix)
        
        # 6. เปรียบเทียบกับ Single Asset (ใช้ข้อมูลล่าสุด)
        last_prices = df_prices.iloc[-1]

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาด: {e}")
        st.stop()

# --- 1. แสดง Metric (ตัวเลขสำคัญ) ---
st.subheader("📈 Performance Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("💰 Max Sharpe Ratio", f"{opt_sharpe:.2f}", help="Sharpe Ratio สูงสุดที่ทำได้")
col2.metric("📊 Optimal Return (Ann.)", f"{opt_return:.2%}")
col3.metric("📉 Optimal Volatility (Ann.)", f"{opt_vol:.2%}")
col4.metric("⚖️ Min Variance Return", f"{min_return:.2%}")

# --- 2. แสดงน้ำหนักของพอร์ตที่เหมาะสมที่สุด ---
st.subheader("📋 น้ำหนักของ Optimal Portfolio (Max Sharpe)")
weights_df = pd.DataFrame({
    'Asset': tickers,
    'Weight (%)': np.round(optimal_weights * 100, 2)
})
weights_df = weights_df.sort_values('Weight (%)', ascending=False)

# แถบแสดงน้ำหนัก (Bar Chart)
st.bar_chart(weights_df.set_index('Asset'))
st.dataframe(weights_df, use_container_width=True)

# --- 3. Plot Efficient Frontier (Plotly Interactive) ---
st.subheader("🌐 Efficient Frontier")

fig = go.Figure()

# 1. พอร์ตสุ่ม (จุดสีเทา)
fig.add_trace(go.Scatter(
    x=df_random['Volatility'],
    y=df_random['Return'],
    mode='markers',
    marker=dict(color=df_random['Sharpe'], colorscale='Viridis', 
                showscale=True, size=3, colorbar=dict(title="Sharpe")),
    text=[f"Sharpe: {s:.2f}" for s in df_random['Sharpe']],
    hoverinfo='text+x+y',
    name='Random Portfolios'
))

# 2. จุด Max Sharpe (ดาวสีแดง)
fig.add_trace(go.Scatter(
    x=[opt_vol],
    y=[opt_return],
    mode='markers',
    marker=dict(color='red', size=15, symbol='star'),
    name=f'Max Sharpe (Ratio={opt_sharpe:.2f})',
    text=f"Return: {opt_return:.2%}<br>Vol: {opt_vol:.2%}",
    hoverinfo='text'
))

# 3. จุด Min Variance (สามเหลี่ยมสีเขียว)
fig.add_trace(go.Scatter(
    x=[min_vol],
    y=[min_return],
    mode='markers',
    marker=dict(color='green', size=12, symbol='triangle-up'),
    name='Min Variance',
    text=f"Return: {min_return:.2%}<br>Vol: {min_vol:.2%}",
    hoverinfo='text'
))

# 4. สินทรัพย์เดี่ยว (จุดสีน้ำเงิน)
for i, ticker in enumerate(tickers):
    fig.add_trace(go.Scatter(
        x=[np.sqrt(cov_matrix.iloc[i, i] * 252)],
        y=[mean_returns[i] * 252],
        mode='markers',
        marker=dict(color='blue', size=10),
        name=ticker
    ))

fig.update_layout(
    title="Efficient Frontier",
    xaxis_title="Annualized Volatility (Risk)",
    yaxis_title="Annualized Return",
    height=600,
    hovermode='closest'
)

st.plotly_chart(fig, use_container_width=True)

# --- 4. การปรับแต่งเพิ่มเติม (Backtest Simple) ---
st.subheader("📊 Simple Backtest (Comparison)")

# คำนวณ Cumulative Return ของ Optimal Portfolio
optimal_returns = returns.dot(optimal_weights)
cum_opt = (1 + optimal_returns).cumprod()

# เทียบกับ Buy & Hold เฉลี่ย (Equal Weight)
eq_weights = np.array([1/len(tickers)] * len(tickers))
eq_returns = returns.dot(eq_weights)
cum_eq = (1 + eq_returns).cumprod()

# Plot เส้น Cumulative
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=cum_opt.index, y=cum_opt, mode='lines', 
                          name='Optimal Portfolio', line=dict(color='gold', width=2)))
fig2.add_trace(go.Scatter(x=cum_eq.index, y=cum_eq, mode='lines', 
                          name='Equal Weight', line=dict(color='gray', width=2, dash='dash')))

fig2.update_layout(title="Equity Curve (Comparison)", 
                   xaxis_title="Date", yaxis_title="Cumulative Return")
st.plotly_chart(fig2, use_container_width=True)

st.success("✅ Dashboard โหลดเสร็จสมบูรณ์!")
