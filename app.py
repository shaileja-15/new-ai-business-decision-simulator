import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Business Decision Simulator", layout="wide")
st.title("📊 AI Business Decision Simulator")
[cite_start]st.markdown("#### Simulation-driven financial and risk impact analysis [cite: 87]")
st.divider()

# --- STEP 5: DATA DESIGN (Loading & Preparation) ---
@st.cache_data
def load_and_prep():
    df = pd.read_csv("ai_business_decision_synthetic_data.csv")
    # [cite_start]Mapping structured inputs to dataset columns [cite: 38-48]
    # We use 'demand' as a proxy for inventory turnover logic in this simulation
    features = ["price", "marketing_spend", "employees", "demand"]
    
    # [cite_start]Models based on Step 6 logic [cite: 53-55]
    rev_model = LinearRegression().fit(df[features], df["revenue"])
    profit_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(df[features], df["profit"])
    risk_model = LogisticRegression(max_iter=1000).fit(df[features], df["risk_flag"])
    
    return rev_model, profit_model, risk_model, df, features

rev_model, profit_model, risk_model, df, features = load_and_prep()

# --- STEP 4: STRUCTURED INPUT (Sidebar) ---
[cite_start]st.sidebar.header("🕹️ Business Manager Inputs [cite: 25-26]")
[cite_start]price = st.sidebar.slider("Product Price (₹)", 200, 2000, 800) [cite: 27]
[cite_start]mkt_spend = st.sidebar.slider("Monthly Marketing Spend (₹)", 5000, 100000, 20000) [cite: 28]
[cite_start]inventory = st.sidebar.slider("Inventory Quantity", 100, 10000, 1500) [cite: 29]
[cite_start]employees = st.sidebar.slider("Employee Count", 5, 200, 50) [cite: 30]
[cite_start]market_cond = st.sidebar.selectbox("Market Condition", df['market_condition'].unique()) [cite: 31]
[cite_start]discount = st.sidebar.slider("Discount Offered (%)", 0, 50, 10) [cite: 32]

# Prepare input for prediction (mapping inventory to demand for model compatibility)
input_df = pd.DataFrame([[price, mkt_spend, employees, inventory]], columns=features)

# --- CALCULATIONS & PREDICTIONS ---
pred_rev = rev_model.predict(input_df)[0]
pred_profit = profit_model.predict(input_df)[0]
pred_cost = pred_rev - pred_profit
risk_prob = risk_model.predict_proba(input_df)[0][1]

# [cite_start]KPI Calculations [cite: 21-23]
roi = (pred_profit / pred_cost) * 100 if pred_cost != 0 else 0
# Simplified Churn prediction based on price and discount logic
churn_rate = 5.0 + (price * 0.01) - (discount * 0.2) 

# --- STEP 2: FINAL OUTPUTS (Company Deliverables) ---

# [cite_start]🔹 Output 1: Financial Impact [cite: 3-6]
st.subheader("🔹 Output 1: Financial Impact")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Expected Revenue", f"₹{pred_rev:,.0f}")
c2.metric("Expected Profit", f"₹{pred_profit:,.0f}")
c3.metric("Operating Cost", f"₹{pred_cost:,.0f}")
c4.metric("ROI (%)", f"{roi:.1f}%")

# [cite_start]🔹 Output 2: Risk Analysis [cite: 7-9]
st.divider()
st.subheader("🔹 Output 2: Risk Analysis")
r_col1, r_col2 = st.columns(2)
with r_col1:
    risk_level = "High" if risk_prob > 0.6 else "Medium" if risk_prob > 0.3 else "Low"
    if risk_level == "High":
        st.error(f"Risk Level: {risk_level}")
    elif risk_level == "Medium":
        st.warning(f"Risk Level: {risk_level}")
    else:
        st.success(f"Risk Level: {risk_level}")
with r_col2:
    st.write(f"**Probability of Loss:** {risk_prob*100:.1f}%")

# [cite_start]🔹 Output 3: Growth Forecast [cite: 10-12]
st.divider()
st.subheader("🔹 Output 3: Growth Forecast (Next 6 Months)")
months = ["Month 1", "Month 2", "Month 3", "Month 4", "Month 5", "Month 6"]
# [cite_start]Simulate trend based on market condition [cite: 11]
growth_rate = 1.05 if market_cond == "Growing" else 0.95 if market_cond == "Recession" else 1.0
trend_data = [pred_rev * (growth_rate**i) for i in range(1, 7)]
st.line_chart(pd.DataFrame(trend_data, index=months, columns=["Revenue Trend"]))

# [cite_start]🔹 Output 4: Decision Score [cite: 13-15]
st.divider()
st.subheader("🔹 Output 4: Decision Score")
# Scoring logic: Balanced between profit and risk
score = int(np.clip((pred_profit / 500000) * 70 + (1 - risk_prob) * 30, 0, 100))
s_col1, s_col2 = st.columns(2)
s_col1.title(f"{score} / 100")
decision_label = "Better" if score > 75 else "Average" if score > 45 else "Poor"
s_col2.write(f"This is considered a **{decision_label}** decision.")

# [cite_start]🔹 Output 5: Actionable Recommendation [cite: 16-19]
st.divider()
st.subheader("🔹 Output 5: Actionable Recommendation")
rec_col1, rec_col2, rec_col3 = st.columns(3)
with rec_col1:
    st.info("**What to Increase**")
    if mkt_spend < 20000: st.write("- Marketing Budget")
    if inventory < 1000: st.write("- Inventory Stock")
with rec_col2:
    st.warning("**What to Reduce**")
    if pred_profit < 0: st.write("- Operating Costs")
    if churn_rate > 10: st.write("- Product Price")
with rec_col3:
    st.error("**What to Avoid**")
    if risk_level == "High": st.write("- High-risk market expansion")
    if discount > 30: st.write("- Excessive discounting")

# --- STEP 11: RECRUITER PITCH ---
st.sidebar.divider()
[cite_start]st.sidebar.markdown(f"**🗣️ Recruiter Pitch:**\n> {df.iloc[0]['business_type']} optimized Decision Simulator. It predicts financial outcomes and risk using ML. [cite: 87]")
