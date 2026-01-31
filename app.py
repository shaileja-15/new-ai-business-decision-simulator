import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# -------------------------------
# PAGE CONFIG & STYLING
# -------------------------------
st.set_page_config(page_title="AI Business Decision Simulator", layout="wide")
st.title("🧠 AI Business Decision Simulator")
st.markdown("### *Simulate outcomes before you invest.*")
st.divider()

# -------------------------------
# DATA LOADING & MODEL TRAINING
# -------------------------------
# In a real scenario, this would load 'ai_business_decision_synthetic_data.csv'
# Creating a small synthetic sample here to ensure the code runs immediately.
@st.cache_data
def load_and_train():
    # [cite_start]Example Dataset Columns based on Step 5 [cite: 38]
    data = {
        "price": np.random.randint(200, 2000, 100),
        "marketing_spend": np.random.randint(5000, 50000, 100),
        "inventory_level": np.random.randint(100, 2000, 100),
        "employees": np.random.randint(5, 80, 100),
        "revenue": np.random.randint(50000, 500000, 100),
        "profit": np.random.randint(-10000, 100000, 100),
        "risk_flag": np.random.choice([0, 1], 100),
        "customer_demand": np.random.randint(100, 1500, 100)
    }
    df = pd.DataFrame(data)
    
    features = ["price", "marketing_spend", "inventory_level", "employees"]
    
    # [cite_start]Model Logic based on Step 6 [cite: 54]
    rev_model = LinearRegression().fit(df[features], df["revenue"])
    profit_model = RandomForestRegressor(n_estimators=100).fit(df[features], df["profit"])
    risk_model = LogisticRegression().fit(df[features], df["risk_flag"])
    
    return rev_model, profit_model, risk_model, df, features

rev_model, profit_model, risk_model, df, features = load_and_train()

# -------------------------------
# SIDEBAR: USER INPUTS (STEP 4)
# -------------------------------
st.sidebar.header("🕹️ Decision Control Panel")
st.sidebar.info("Adjust the sliders to simulate a business strategy.")

price = st.sidebar.slider("Product Price (₹)", 200, 2000, 800)
marketing = st.sidebar.slider("Marketing Spend (₹)", 5000, 50000, 15000)
inventory = st.sidebar.slider("Inventory Quantity", 100, 2000, 500)
employees = st.sidebar.slider("Employee Count", 5, 80, 25)
market_cond = st.sidebar.selectbox("Market Condition", ["Growing", "Stable", "Declining"])

input_data = pd.DataFrame([[price, marketing, inventory, employees]], columns=features)

# -------------------------------
# PREDICTIONS & LOGIC
# -------------------------------
pred_rev = rev_model.predict(input_data)[0]
pred_profit = profit_model.predict(input_data)[0]
risk_prob = risk_model.predict_proba(input_data)[0][1]

# [cite_start]Decision Score Logic (Step 2: Output 4) [cite: 13, 14]
# Simplified calculation: Higher profit and lower risk = higher score
base_score = (pred_profit / 100000) * 50 + (1 - risk_prob) * 50
decision_score = max(0, min(100, int(base_score)))

# -------------------------------
# DASHBOARD OUTPUTS (STEP 8)
# -------------------------------
# [cite_start]1. KPI Cards [cite: 66]
col1, col2, col3, col4 = st.columns(4)
col1.metric("Expected Revenue", f"₹{pred_rev:,.0f}")
col2.metric("Expected Profit", f"₹{pred_profit:,.0f}")
col3.metric("Decision Score", f"{decision_score}/100")
col4.metric("Risk Prob.", f"{risk_prob*100:.1f}%")

st.divider()

# [cite_start]2. Risk & Recommendation (Step 9) 
res_col1, res_col2 = st.columns(2)

with res_col1:
    st.subheader("⚠️ Risk Analysis")
    if risk_prob > 0.6:
        st.error(f"HIGH RISK ({risk_prob*100:.1f}%)")
        [cite_start]st.write("Probability of loss is significant. High overhead detected.") [cite: 9]
    elif risk_prob > 0.3:
        st.warning(f"MEDIUM RISK ({risk_prob*100:.1f}%)")
    else:
        st.success(f"LOW RISK ({risk_prob*100:.1f}%)")

with res_col2:
    st.subheader("🤖 Actionable Recommendation")
    # [cite_start]Rule-based explanations [cite: 74, 77]
    if pred_profit < 0:
        [cite_start]st.write("❌ **REDUCE:** Operating costs immediately. Current strategy leads to loss.") [cite: 18]
    if inventory > 1500 and price > 1500:
        [cite_start]st.write("⚠️ **AVOID:** High inventory with high pricing. This causes cash flow risk.") [cite: 19, 77]
    if marketing < 10000:
        [cite_start]st.write("🚀 **INCREASE:** Marketing spend to capture more market share.") [cite: 17]
    else:
        st.write("✅ **MAINTAIN:** This is a balanced strategy for current market conditions.")

st.divider()

# [cite_start]3. Visual Analysis (Step 8) [cite: 64]
st.subheader("📊 Strategy Impact Forecast")
fig, ax = plt.subplots(figsize=(10, 4))
metrics = ["Revenue", "Profit"]
values = [pred_rev, pred_profit]
ax.barh(metrics, values, color=['#1f77b4', '#2ca02c'])
for i, v in enumerate(values):
    ax.text(v, i, f" ₹{v:,.0f}", va='center')
st.pyplot(fig)

# [cite_start]4. Growth Trend (Step 2: Output 3) [cite: 10, 11]
st.subheader("📈 6-Month Revenue Forecast")
months = ["Month 1", "Month 2", "Month 3", "Month 4", "Month 5", "Month 6"]
# Simple trend simulation based on input price and marketing
trend_data = [pred_rev * (1 + (marketing/500000) * i) for i in range(6)]
st.line_chart(pd.DataFrame(trend_data, index=months, columns=["Projected Revenue"]))

# -------------------------------
# FOOTER: RECRUITER PITCH (STEP 11)
# -------------------------------
st.sidebar.divider()
st.sidebar.markdown(
    "**Recruiter Note:** *This project helps businesses simulate decisions before investing money. "
    "It predicts financial outcomes, risk, and growth using ML.*"
[cite_start]) [cite: 87]
