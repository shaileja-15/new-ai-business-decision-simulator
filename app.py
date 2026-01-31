import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="AI Business Decision Simulator", layout="wide")

st.title("📊 AI Business Decision Simulator")
st.markdown("Simulate business decisions and predict outcomes using AI")

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv("ai_business_decision_synthetic_data.csv")

# -------------------------------
# MODEL TRAINING
# -------------------------------
features = [
    "price",
    "marketing_spend",
    "employees",
    "demand"
]

X = df[features]

# Revenue Model
y_revenue = df["revenue"]
rev_model = LinearRegression()
rev_model.fit(X, y_revenue)

# Profit Model
y_profit = df["profit"]
profit_model = RandomForestRegressor(n_estimators=100, random_state=42)
profit_model.fit(X, y_profit)

# Risk Model
y_risk = df["risk_flag"]
risk_model = LogisticRegression()
risk_model.fit(X, y_risk)

# -------------------------------
# USER INPUT PANEL
# -------------------------------
st.sidebar.header("🧠 Enter Business Decisions")

price = st.sidebar.slider("Product Price", 200, 2000, 800)
marketing_spend = st.sidebar.slider("Marketing Spend", 5000, 50000, 15000)
employees = st.sidebar.slider("Number of Employees", 5, 80, 25)
demand = st.sidebar.slider("Expected Demand", 100, 1500, 600)

input_data = pd.DataFrame([[price, marketing_spend, employees, demand]],
                          columns=features)

# -------------------------------
# PREDICTIONS
# -------------------------------
predicted_revenue = rev_model.predict(input_data)[0]
predicted_profit = profit_model.predict(input_data)[0]
risk_probability = risk_model.predict_proba(input_data)[0][1]
risk_flag = 1 if risk_probability > 0.5 else 0

# -------------------------------
# OUTPUT DASHBOARD
# -------------------------------
st.subheader("📈 Prediction Results")

col1, col2, col3 = st.columns(3)

col1.metric("💰 Predicted Revenue", f"₹ {predicted_revenue:,.0f}")
col2.metric("📉 Predicted Profit", f"₹ {predicted_profit:,.0f}")
col3.metric("⚠️ Risk Level", "High Risk" if risk_flag else "Low Risk")

# -------------------------------
# RISK MESSAGE
# -------------------------------
if risk_flag:
    st.error("⚠️ This strategy is risky. Consider adjusting pricing or costs.")
else:
    st.success("✅ This strategy looks financially safe.")

# -------------------------------
# VISUAL ANALYSIS
# -------------------------------
st.subheader("📊 Decision Impact Analysis")

fig, ax = plt.subplots()
ax.bar(["Revenue", "Profit"], [predicted_revenue, predicted_profit])
st.pyplot(fig)

# -------------------------------
# AI RECOMMENDATION
# -------------------------------
st.subheader("🤖 AI Recommendation")

if predicted_profit < 0:
    st.write("🔹 Reduce operational costs or increase price.")
elif marketing_spend > 40000:
    st.write("🔹 High marketing spend detected. Optimize ROI.")
else:
    st.write("🔹 Strategy is balanced. Maintain current decisions.")
