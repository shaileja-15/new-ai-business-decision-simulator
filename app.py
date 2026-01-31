import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# -------------------------------
# PAGE CONFIG (STEP 8: Dashboard Outputs)
# -------------------------------
st.set_page_config(page_title="AI Business Decision Simulator", layout="wide")
st.title("🧠 AI Business Decision Simulator")
st.markdown("### *Simulate outcomes before you invest.*")
st.divider()

# -------------------------------
# DATA LOADING & MODEL TRAINING (STEP 5 & 6)
# -------------------------------
@st.cache_data
def load_and_train():
    # Loading your actual CSV file
    df = pd.read_csv("ai_business_decision_synthetic_data.csv")
    
    # Using features available in your dataset
    features = ["price", "marketing_spend", "employees", "demand"]
    X = df[features]
    
    # Model Logic as per Step 6
    rev_model = LinearRegression().fit(X, df["revenue"])
    profit_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, df["profit"])
    risk_model = LogisticRegression(max_iter=1000).fit(X, df["risk_flag"])
    
    return rev_model, profit_model, risk_model, df, features

try:
    rev_model, profit_model, risk_model, df, features = load_and_train()
except Exception as e:
    st.error(f"Error loading data: {e}. Please ensure the CSV is in the same folder.")
    st.stop()

# -------------------------------
# SIDEBAR: USER INPUTS (STEP 4)
# -------------------------------
st.sidebar.header("🕹️ Decision Control Panel")
st.sidebar.info("Adjust the sliders to simulate a business strategy.")

# Updated ranges based on your dataset statistics
price = st.sidebar.slider("Product Price (₹)", 100, 3000, 800)
marketing = st.sidebar.slider("Marketing Spend (₹)", 1000, 100000, 20000)
employees = st.sidebar.slider("Employee Count", 1, 150, 25)
u_demand = st.sidebar.slider("Expected Market Demand", 100, 10000, 1500)
market_cond = st.sidebar.selectbox("Market Condition", df['market_condition'].unique())

input_data = pd.DataFrame([[price, marketing, employees, u_demand]], columns=features)

# -------------------------------
# PREDICTIONS & LOGIC (STEP 2)
# -------------------------------
pred_rev = rev_model.predict(input_data)[0]
pred_profit = profit_model.predict(input_data)[0]
risk_prob = risk_model.predict_proba(input_data)[0][1]

# Output 4: Decision Score Calculation (0-100)
# [cite_start]Higher profit and lower risk = higher score [cite: 108]
score = int(np.clip((pred_profit / df['profit'].max()) * 70 + (1 - risk_prob) * 30, 0, 100))

# -------------------------------
# DASHBOARD OUTPUTS (STEP 8)
# -------------------------------
# 🔹 Output 1: Financial Impact
st.subheader("🔹 Output 1: Financial Impact")
col1, col2, col3 = st.columns(3)
col1.metric("Expected Revenue", f"₹{pred_rev:,.0f}")
col2.metric("Expected Profit", f"₹{pred_profit:,.0f}")
col3.metric("Operating Cost", f"₹{(pred_rev - pred_profit):,.0f}")

st.divider()

# 🔹 Output 2 & 4: Risk and Score
res_col1, res_col2 = st.columns(2)

with res_col1:
    st.subheader("🔹 Output 2: Risk Analysis")
    risk_lvl = "High" if risk_prob > 0.6 else "Medium" if risk_prob > 0.3 else "Low"
    if risk_lvl == "High":
        st.error(f"⚠️ {risk_lvl} Risk (Prob: {risk_prob*100:.1f}%)")
    elif risk_lvl == "Medium":
        st.warning(f"🔸 {risk_lvl} Risk (Prob: {risk_prob*100:.1f}%)")
    else:
        st.success(f"✅ {risk_lvl} Risk (Prob: {risk_prob*100:.1f}%)")

with res_col2:
    st.subheader("🔹 Output 4: Decision Score")
    decision_cat = "Better" if score > 70 else "Average" if score > 40 else "Poor"
    st.title(f"{score}/100")
    st.write(f"This is a **{decision_cat}** decision.")

st.divider()

# 🔹 Output 5: AI Recommendation (STEP 9)
st.subheader("🤖 Output 5: Actionable Recommendation")
rec_col1, rec_col2 = st.columns(2)

with rec_col1:
    if pred_profit < 0:
        st.write("❌ **REDUCE:** Operating costs immediately. The strategy leads to a loss.")
    elif marketing > 50000:
        st.write("⚠️ **REDUCE:** High marketing spend relative to predicted profit.")
    else:
        st.write("✅ **MAINTAIN:** Current financial structure is sustainable.")

with rec_col2:
    if u_demand < 1000:
        st.write("🚀 **INCREASE:** Efforts to capture market demand. Low volume detected.")
    else:
        st.write("✅ **AVOID:** Aggressive expansion; focus on maintaining current demand levels.")

# -------------------------------
# STEP 11: RECRUITER PITCH
# -------------------------------
st.sidebar.divider()
st.sidebar.markdown(
    "**🗣️ Recruiter Pitch:**\n"
    "*This project helps businesses simulate decisions before investing money. "
    "It predicts financial outcomes, risk, and growth using machine learning.*"
)
