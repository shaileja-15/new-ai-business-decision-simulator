import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Business Decision Simulator", layout="wide")
st.title("📊 AI Business Decision Simulator")
st.caption("Advanced decision intelligence for business leaders")

# ---------------- LOAD DATA ----------------
df = pd.read_csv("ai_business_decision_synthetic_data.csv")

features = ["price", "marketing_spend", "employees", "demand"]
X = df[features]

# ---------------- TRAIN MODELS ----------------
rev_model = LinearRegression().fit(X, df["revenue"])
profit_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, df["profit"])
risk_model = LogisticRegression().fit(X, df["risk_flag"])

# ---------------- SIDEBAR INPUTS ----------------
st.sidebar.header("🧠 Business Inputs")

business_type = st.sidebar.selectbox(
    "Business Type", ["Retail", "E-Commerce", "Manufacturing"]
)

market_condition = st.sidebar.selectbox(
    "Market Condition", ["Stable", "Competitive", "Recession", "Growth"]
)

price = st.sidebar.slider("Product Price (₹)", 200, 2000, 800)
marketing_spend = st.sidebar.slider("Marketing Spend (₹)", 5000, 50000, 15000)
employees = st.sidebar.slider("Employees", 5, 80, 25)
demand = st.sidebar.slider("Expected Demand", 100, 1500, 600)

start = st.sidebar.button("🚀 Start Analysis")

# ---------------- RUN ANALYSIS ----------------
if start:

    input_df = pd.DataFrame([[price, marketing_spend, employees, demand]], columns=features)

    revenue = rev_model.predict(input_df)[0]
    profit = profit_model.predict(input_df)[0]
    risk_prob = risk_model.predict_proba(input_df)[0][1]

    cost = revenue - profit
    margin = (profit / revenue) * 100 if revenue > 0 else 0

    # Risk Level
    if risk_prob > 0.7:
        risk_level = "High"
    elif risk_prob > 0.4:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    # Decision Score
    score = 100
    if profit < 0:
        score -= 40
    if margin < 10:
        score -= 20
    if risk_level == "High":
        score -= 30
    elif risk_level == "Medium":
        score -= 15

    score = max(0, score)

    # ---------------- KPI SECTION ----------------
    st.subheader("📌 Key Business Metrics")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Revenue", f"₹ {revenue:,.0f}")
    k2.metric("Profit", f"₹ {profit:,.0f}")
    k3.metric("Profit Margin", f"{margin:.1f}%")
    k4.metric("Decision Score", f"{score}/100")

    # ---------------- RISK STATUS ----------------
    if risk_level == "High":
        st.error("⚠️ High Risk Strategy")
    elif risk_level == "Medium":
        st.warning("⚠️ Moderate Risk Strategy")
    else:
        st.success("✅ Low Risk Strategy")

    # ---------------- VISUALS ----------------
    st.subheader("📊 Financial Breakdown")

    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots()
        ax1.bar(["Revenue", "Cost", "Profit"], [revenue, cost, profit])
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots()
        ax2.pie([cost, profit], labels=["Cost", "Profit"], autopct="%1.1f%%")
        st.pyplot(fig2)

    # ---------------- PROFIT SIMULATION ----------------
    st.subheader("📈 Profit Sensitivity Simulation")

    price_range = np.linspace(price * 0.8, price * 1.2, 10)
    sim_profit = []

    for p in price_range:
        temp = pd.DataFrame([[p, marketing_spend, employees, demand]], columns=features)
        sim_profit.append(profit_model.predict(temp)[0])

    fig3, ax3 = plt.subplots()
    ax3.plot(price_range, sim_profit)
    ax3.set_xlabel("Price")
    ax3.set_ylabel("Predicted Profit")
    st.pyplot(fig3)

    # ---------------- AI INSIGHTS ----------------
    st.subheader("🤖 AI Insights")

    insights = []
    if marketing_spend > 40000:
        insights.append("High marketing spend may reduce short-term profit.")
    if employees > 50:
        insights.append("Staffing levels are high — consider productivity optimization.")
    if margin < 10:
        insights.append("Low profit margin detected — pricing or cost review suggested.")

    if insights:
        for i in insights:
            st.write("🔹", i)
    else:
        st.write("🔹 Strategy is balanced and sustainable.")

else:
    st.info("⬅️ Enter inputs and click **Start Analysis** to view advanced results.")
