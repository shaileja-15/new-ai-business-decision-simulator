import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="AI Business Decision Simulator", layout="wide")
st.title("📊 AI Business Decision Simulator")
st.markdown("Make informed business decisions using AI-based simulation")

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv("ai_business_decision_synthetic_data.csv")

# Encode categorical columns
df_encoded = pd.get_dummies(df, columns=["business_type", "market_condition"], drop_first=True)

features = [
    "price",
    "marketing_spend",
    "employees",
    "demand"
]

X = df_encoded[features]

# -------------------------------
# TRAIN MODELS
# -------------------------------
rev_model = LinearRegression().fit(X, df_encoded["revenue"])
profit_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, df_encoded["profit"])
risk_model = LogisticRegression().fit(X, df_encoded["risk_flag"])

# -------------------------------
# SIDEBAR INPUTS
# -------------------------------
st.sidebar.header("🧠 Business Decision Inputs")

business_type = st.sidebar.selectbox(
    "Business Type",
    ["Retail", "E-Commerce", "Manufacturing"]
)

market_condition = st.sidebar.selectbox(
    "Market Condition",
    ["Stable", "Competitive", "Recession", "Growth"]
)

price = st.sidebar.slider("Product Price (₹)", 200, 2000, 800)
marketing_spend = st.sidebar.slider("Marketing Spend (₹)", 5000, 50000, 15000)
employees = st.sidebar.slider("Number of Employees", 5, 80, 25)
demand = st.sidebar.slider("Expected Demand (Units)", 100, 1500, 600)

# -------------------------------
# START ANALYSIS BUTTON
# -------------------------------
start_analysis = st.sidebar.button("🚀 Start Analysis")

# -------------------------------
# RUN ANALYSIS ONLY AFTER BUTTON CLICK
# -------------------------------
if start_analysis:

    input_data = pd.DataFrame([[price, marketing_spend, employees, demand]],
                              columns=features)

    # Predictions
    predicted_revenue = rev_model.predict(input_data)[0]
    predicted_profit = profit_model.predict(input_data)[0]
    risk_prob = risk_model.predict_proba(input_data)[0][1]
    risk_flag = 1 if risk_prob > 0.5 else 0

    # -------------------------------
    # OUTPUT SECTION
    # -------------------------------
    st.subheader("📈 Analysis Results")

    col1, col2, col3 = st.columns(3)

    col1.metric("💰 Predicted Revenue", f"₹ {predicted_revenue:,.0f}")
    col2.metric("📉 Predicted Profit", f"₹ {predicted_profit:,.0f}")
    col3.metric("⚠️ Risk Level", "High Risk" if risk_flag else "Low Risk")

    # -------------------------------
    # RISK MESSAGE
    # -------------------------------
    if risk_flag:
        st.error("⚠️ This decision is risky. Profitability or customer stability may be impacted.")
    else:
        st.success("✅ This decision looks financially safe.")

    # -------------------------------
    # VISUALIZATION
    # -------------------------------
    st.subheader("📊 Financial Comparison")

    fig, ax = plt.subplots()
    ax.bar(["Revenue", "Profit"], [predicted_revenue, predicted_profit])
    st.pyplot(fig)

    # -------------------------------
    # AI RECOMMENDATIONS
    # -------------------------------
    st.subheader("🤖 AI Recommendation")

    if predicted_profit < 0:
        st.write("🔹 Reduce operational costs or improve pricing strategy.")
    elif marketing_spend > 40000:
        st.write("🔹 Marketing spend is high. Monitor ROI closely.")
    else:
        st.write("🔹 Strategy appears balanced. Maintain current approach.")

else:
    st.info("⬅️ Enter business decisions and click **Start Analysis** to view results.")
