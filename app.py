import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# --- PAGE CONFIG (Step 8: Dashboard Outputs) ---
st.set_page_config(page_title="AI Business Decision Simulator", layout="wide")
st.title("📊 AI Business Decision Simulator")
[cite_start]st.markdown("### Predict financial impact, risk, and growth before investing [cite: 87]")
st.divider()

# --- STEP 5: DATA DESIGN (Synthetic Business Logic) ---
@st.cache_data
def generate_business_data():
    np.random.seed(42)
    # [cite_start]Generating 100 rows of historical business performance [cite: 37, 50]
    data = {
        "price": np.random.randint(200, 2000, 100),
        "marketing_spend": np.random.randint(5000, 50000, 100),
        "inventory_level": np.random.randint(100, 2000, 100),
        "employees": np.random.randint(5, 80, 100),
        "revenue": np.random.randint(50000, 500000, 100),
        "profit": np.random.randint(-10000, 100000, 100),
        "risk_flag": np.random.choice([0, 1], 100),
        "demand": np.random.randint(100, 1500, 100)
    }
    return pd.DataFrame(data)

df = generate_business_data()
features = ["price", "marketing_spend", "inventory_level", "employees"]

# --- STEP 6: MODEL LOGIC ---
# [cite_start]Predict revenue: Linear Regression 
rev_model = LinearRegression().fit(df[features], df["revenue"])
# [cite_start]Predict profit: Random Forest [cite: 54]
profit_model = RandomForestRegressor(n_estimators=100).fit(df[features], df["profit"])
# [cite_start]Predict risk: Logistic Regression [cite: 54]
risk_model = LogisticRegression().fit(df[features], df["risk_flag"])

# --- STEP 4: STRUCTURED INPUT (Business Manager Inputs) ---
st.sidebar.header("🧠 Enter Business Decisions")
[cite_start]price = st.sidebar.slider("Product Price", 200, 2000, 800) # [cite: 27]
[cite_start]marketing = st.sidebar.slider("Monthly Marketing Spend", 5000, 50000, 15000) # [cite: 28]
[cite_start]inventory = st.sidebar.slider("Inventory Quantity", 100, 2000, 500) # [cite: 29]
[cite_start]employees = st.sidebar.slider("Employee Count", 5, 80, 25) # [cite: 30]
[cite_start]market_cond = st.sidebar.selectbox("Market Condition", ["Growing", "Stable", "Declining"]) # [cite: 31]

input_df = pd.DataFrame([[price, marketing, inventory, employees]], columns=features)

# --- CALCULATE OUTPUTS ---
pred_rev = rev_model.predict(input_df)[0]
pred_profit = profit_model.predict(input_df)[0]
[cite_start]risk_prob = risk_model.predict_proba(input_df)[0][1] # [cite: 9]

# Output 4: Decision Score (Step 2)
# [cite_start]Calculation: Weighted average of profit performance and risk avoidance [cite: 14]
norm_profit = np.clip(pred_profit / 100000, 0, 1)
decision_score = int(((norm_profit * 0.7) + ((1 - risk_prob) * 0.3)) * 100)

# --- STEP 8: DASHBOARD VIEW ---
# [cite_start]Output 1: Financial Impact [cite: 3]
st.subheader("🔹 Output 1: Financial Impact")
f_col1, f_col2, f_col3 = st.columns(3)
[cite_start]f_col1.metric("Expected Revenue", f"₹{pred_rev:,.0f}") # [cite: 4]
[cite_start]f_col2.metric("Expected Profit", f"₹{pred_profit:,.0f}") # [cite: 5]
[cite_start]f_col3.metric("Operating Cost", f"₹{(pred_rev - pred_profit):,.0f}") # [cite: 6]

# [cite_start]Output 2 & 4: Risk and Score [cite: 7, 13]
st.divider()
o_col1, o_col2 = st.columns(2)

with o_col1:
    st.subheader("🔹 Output 2: Risk Analysis")
    risk_level = "High" if risk_prob > 0.6 else "Medium" if risk_prob > 0.3 else "Low"
    if risk_level == "High":
        [cite_start]st.error(f"⚠️ {risk_level} Risk (Loss Probability: {risk_prob*100:.1f}%)") # [cite: 8, 9]
    elif risk_level == "Medium":
        st.warning(f"🔸 {risk_level} Risk (Loss Probability: {risk_prob*100:.1f}%)")
    else:
        st.success(f"✅ {risk_level} Risk (Loss Probability: {risk_prob*100:.1f}%)")

with o_col2:
    st.subheader("🔹 Output 4: Decision Score")
    score_label = "Better" if decision_score > 70 else "Average" if decision_score > 40 else "Poor"
    st.title(f"{decision_score}/100")
    [cite_start]st.write(f"This is considered a **{score_label}** decision. [cite: 15]")

# [cite_start]Output 3: Growth Forecast (Next 6 Months) [cite: 10, 11]
st.divider()
st.subheader("🔹 Output 3: Growth Forecast")
months = ["Current", "Month 2", "Month 3", "Month 4", "Month 5", "Month 6"]
# Simulate trend based on marketing spend and market condition
growth_multiplier = 1.05 if market_cond == "Growing" else 0.95 if market_cond == "Declining" else 1.0
trend = [pred_rev * (growth_multiplier ** i) for i in range(6)]
st.line_chart(pd.DataFrame(trend, index=months, columns=["Revenue Trend"]))

# [cite_start]Output 5: Actionable Recommendation (Step 9: Explainable AI) [cite: 16, 72]
st.divider()
st.subheader("🔹 Output 5: Actionable Recommendation")
if pred_profit < 0:
    [cite_start]st.info("💡 **Recommendation:** Increase price or reduce employee count to fix negative ROI. [cite: 17, 18]")
elif risk_prob > 0.5 and inventory > 1500:
    [cite_start]st.warning("💡 **Recommendation:** Avoid overstocking. High inventory + low demand creates cash flow risk. [cite: 19, 77]")
else:
    [cite_start]st.success("💡 **Recommendation:** Strategy is balanced. Maintain current marketing spend. [cite: 17]")

# [cite_start]Footer for Recruiters [cite: 86]
st.sidebar.divider()
st.sidebar.write("**Recruiter Pitch:**")
[cite_start]st.sidebar.caption("This project helps businesses simulate decisions before investing money. It predicts financial outcomes, risk, and growth using ML. [cite: 87]")
