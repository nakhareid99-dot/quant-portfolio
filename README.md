# Quant Research Portfolio

คลังโปรเจคสำหรับจำลองกลยุทธ์การลงทุนเชิงปริมาณ (Quantitative Trading)

## โปรเจคในคลัง
1. **Monte Carlo GBM Simulator** - จำลองราคาสินทรัพย์และคำนวณ Value at Risk (VaR)
2. **EWMA Crossover** - กลยุทธ์ใช้ EWMA 2 เส้น พร้อม Grid Search Optimization
3. **Efficient Frontier Optimizer** - หาพอร์ตที่เหมาะสมที่สุดด้วย Modern Portfolio Theory
4. **Sharpe Ratio Dashboard** - Streamlit Dashboard แสดง Efficient Frontier และ Sharpe Ratio
5. **Mean Reversion Pairs Bot** (เร็วๆ นี้)

## การติดตั้ง
```bash
pip install -r requirements.txt
```

## การใช้งาน Dashboard
```bash
streamlit run projects/03_dashboard/app.py
```