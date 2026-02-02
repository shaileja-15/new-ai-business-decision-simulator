import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# ---------------------------------
# PAGE CONFIG
# ---------------------------------
st.set_page_config(page_title="AI Business Decision Simulator", layout="wide")
st.title("📊 AI Business Decision Simulator")
st.caption("Advanced AI-powered decision analysis for businesses")

# ---------------------------------
# LOAD DATA
# ---------------------------------
df = pd.read_csv("ai_business_decision_synthetic_data.csv")

features = ["price", "marketing_spend", "employees", "demand"]
X = df[features]

# ---------------------------------
# TRAIN MODELS
# ---------------------------------
revenue_model = LinearRegression().fit(X, df["revenue"])
profit_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, df["profit"])
risk_model = LogisticRegression().fit(X, df["risk_flag"])

# ---------------------------------
# SIDEBAR INPUTS
# ---------------------------------
st.sidebar.header("🧠 Business Decision Inputs")

business_type = st.sidebar.selectbox(
    "Business Type", ["Retail", "E-Commerce", "Manufacturing"]
)

market_condition = st.sidebar.selectbox(
    "Market Condition", ["Stable", "Competitive", "Recession", "Growth"]
)

price = st.sidebar.slider("Product Price (₹)", 200, 2000, 800)
marketing_spend = st.sidebar.slider("Marketing Spend (₹)", 5000, 50000, 15000)
employees = st.sidebar.slider("Number of Employees", 5, 80, 25)
demand = st.sidebar.slider("Expected Demand (Units)", 100, 1500, 600)

start_analysis = st.sidebar.button("🚀 Start Analysis")

# ---------------------------------
# RUN ANALYSIS
# ---------------------------------
if start_analysis:

    input_df = pd.DataFrame(
        [[price, marketing_spend, employees, demand]],
        columns=features
    )

    revenue = revenue_model.predict(input_df)[0]
    profit = profit_model.predict(input_df)[0]
    risk_probability = risk_model.predict_proba(input_df)[0][1]

    cost = revenue - profit
    margin = (profit / revenue) * 100 if revenue > 0 else 0

    # ---------------------------------
    # RISK LEVEL
    # ---------------------------------
    if risk_probability > 0.7:
        risk_level = "High"
    elif risk_probability > 0.4:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    # ---------------------------------
    # DECISION SCORE (0–100)
    # ---------------------------------
    score = 100
    if profit < 0:
        score -= 40
    if margin < 10:
        score -= 20
    if risk_level == "High":
        score -= 30
    elif risk_level == "Medium":
        score -= 15

    score = max(score, 0)

    # ---------------------------------
    # KPI SECTION
    # ---------------------------------
    st.subheader("📌 Key Business Outcomes")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Revenue", f"₹ {revenue:,.0f}")
    k2.metric("Profit", f"₹ {profit:,.0f}")
    k3.metric("Profit Margin", f"{margin:.1f}%")
    k4.metric("Decision Score", f"{score}/100")

    # ---------------------------------
    # RISK MESSAGE
    # ---------------------------------
    if risk_level == "High":
        st.error("⚠️ High-risk decision: financial or customer instability detected.")
    elif risk_level == "Medium":
        st.warning("⚠️ Moderate risk: strategy needs optimization.")
    else:
        st.success("✅ Low-risk decision: strategy is stable.")

    # ============================================================
    # 📊 VISUAL 1: REVENUE vs COST vs PROFIT (Core financial view)
    # ============================================================
    st.subheader("📊 Financial Performance Overview")

    fig1, ax1 = plt.subplots()
    ax1.bar(["Revenue", "Cost", "Profit"], [revenue, cost, profit])
    ax1.set_ylabel("Amount (₹)")
    st.pyplot(fig1)

    st.caption("💡 Insight: A healthy strategy shows revenue significantly higher than cost.")

    # ============================================================
    # 📊 VISUAL 2: COST vs PROFIT / LOSS DISTRIBUTION (Safe pie)
    # ============================================================
    fig2, ax2 = plt.subplots()

    if profit >= 0:
        values = [cost, profit]
        labels = ["Cost", "Profit"]
    else:
        values = [cost, abs(profit)]
        labels = ["Cost", "Loss"]

    ax2.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
    ax2.axis("equal")
    st.pyplot(fig2)

    st.caption("💡 Insight: Higher profit share indicates better capital efficiency.")

    # ============================================================
    # 📊 VISUAL 3: PRICE vs PROFIT SENSITIVITY (What-if analysis)
    # ============================================================
    st.subheader("📈 Price Sensitivity Analysis")

    price_range = np.linspace(price * 0.8, price * 1.2, 10)
    simulated_profit = []

    for p in price_range:
        temp_df = pd.DataFrame([[p, marketing_spend, employees, demand]], columns=features)
        simulated_profit.append(profit_model.predict(temp_df)[0])

    fig3, ax3 = plt.subplots()
    ax3.plot(price_range, simulated_profit, marker="o")
    ax3.set_xlabel("Price (₹)")
    ax3.set_ylabel("Predicted Profit")
    st.pyplot(fig3)

    st.caption("💡 Insight: Identify the price point where profit peaks.")

    # ============================================================
    # 📊 VISUAL 4: DECISION QUALITY GAUGE (Simulated)
    # ============================================================
    st.subheader("🎯 Decision Quality Indicator")

    fig4, ax4 = plt.subplots()
    ax4.barh(["Decision Quality"], [score])
    ax4.set_xlim(0, 100)
    st.pyplot(fig4)

    st.caption("💡 Insight: Scores above 70 indicate strong, sustainable decisions.")

    # ---------------------------------
    # 🤖 AI DECISION COMMENTS
    # ---------------------------------
    st.subheader("🤖 AI Decision Commentary")

    if profit < 0:
        st.write("🔴 This strategy leads to losses. Reduce costs or revise pricing.")
    if marketing_spend > 40000:
        st.write("🟠 Marketing spend is high; monitor ROI carefully.")
    if employees > 50 and profit < 0:
        st.write("🟠 Overstaffing detected. Consider productivity optimization.")
    if margin > 20 and risk_level == "Low":
        st.write("🟢 Strong decision: good margin with low risk.")
    if score > 75:
        st.write("✅ Recommended strategy: financially sound and scalable.")

else:
    st.info("⬅️ Enter inputs and click **Start Analysis** to generate results.")
