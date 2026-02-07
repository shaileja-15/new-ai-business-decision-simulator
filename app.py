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
profit_model = RandomForestRegressor(n_estimators=200, random_state=42).fit(X, df["profit"])
risk_model = LogisticRegression(max_iter=1000).fit(X, df["risk_flag"])

# ---------------------------------
# SIDEBAR INPUTS
# ---------------------------------
st.sidebar.header("🧠 Business Decision Inputs")

price = st.sidebar.slider("Product Price (₹)", 200, 2000, 800)
marketing_spend = st.sidebar.slider("Marketing Spend (₹)", 5000, 50000, 15000)
employees = st.sidebar.slider("Number of Employees", 5, 80, 25)
demand = st.sidebar.slider("Expected Demand (Units)", 100, 1500, 600)

# ---------------------------------
# SESSION STATE FIX (prevents fade)
# ---------------------------------
if "run_analysis" not in st.session_state:
    st.session_state.run_analysis = False

if st.sidebar.button("🚀 Start Analysis"):
    st.session_state.run_analysis = True

# ---------------------------------
# RUN ANALYSIS
# ---------------------------------
if st.session_state.run_analysis:

    input_df = pd.DataFrame(
        [[price, marketing_spend, employees, demand]],
        columns=features
    )

    revenue = revenue_model.predict(input_df)[0]
    profit = profit_model.predict(input_df)[0]
    risk_probability = risk_model.predict_proba(input_df)[0][1]

    cost = revenue - profit
    margin = (profit / revenue) * 100 if revenue > 0 else 0

    # Risk Level
    if risk_probability > 0.7:
        risk_level = "High"
    elif risk_probability > 0.4:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    # Decision Score
    score = 100
    if profit < 0: score -= 40
    if margin < 10: score -= 20
    if risk_level == "High": score -= 30
    elif risk_level == "Medium": score -= 15
    score = max(score, 0)

    # ---------------------------------
    # KPI BLOCK
    # ---------------------------------
    st.subheader("📌 Key Business Outcomes")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Revenue", f"₹ {revenue:,.0f}")
    c2.metric("Profit", f"₹ {profit:,.0f}")
    c3.metric("Margin %", f"{margin:.1f}%")
    c4.metric("Decision Score", f"{score}/100")

    # ---------------------------------
    # ADVANCED INSIGHT: FEATURE IMPACT
    # ---------------------------------
    st.subheader("🔍 Feature Impact on Profit")

    importances = profit_model.feature_importances_
    imp_df = pd.DataFrame({
        "Feature": features,
        "Impact": importances
    }).sort_values("Impact", ascending=False)

    fig_imp, ax_imp = plt.subplots()
    ax_imp.bar(imp_df["Feature"], imp_df["Impact"])
    st.pyplot(fig_imp)

    # ---------------------------------
    # FINANCIAL BAR CHART
    # ---------------------------------
    fig1, ax1 = plt.subplots()
    ax1.bar(["Revenue", "Cost", "Profit"], [revenue, cost, profit])
    st.pyplot(fig1)

    # ---------------------------------
    # SAFE PIE
    # ---------------------------------
    fig2, ax2 = plt.subplots()
    if profit >= 0:
        ax2.pie([cost, profit], labels=["Cost","Profit"], autopct="%1.1f%%")
    else:
        ax2.pie([cost, abs(profit)], labels=["Cost","Loss"], autopct="%1.1f%%")
    ax2.axis("equal")
    st.pyplot(fig2)

    # ---------------------------------
    # WHAT-IF SCENARIO TABLE
    # ---------------------------------
    st.subheader("📈 What-If Scenario Comparison")

    scenarios = []
    for m in [0.8, 1.0, 1.2]:
        temp = pd.DataFrame([[price*m, marketing_spend, employees, demand]], columns=features)
        scenarios.append({
            "Price": price*m,
            "Predicted Profit": profit_model.predict(temp)[0]
        })

    st.dataframe(pd.DataFrame(scenarios))

    # ---------------------------------
    # RISK MESSAGE
    # ---------------------------------
    if risk_level == "High":
        st.error("⚠️ High risk strategy")
    elif risk_level == "Medium":
        st.warning("⚠️ Moderate risk strategy")
    else:
        st.success("✅ Low risk strategy")

    # ---------------------------------
    # SMART AI RECOMMENDATIONS
    # ---------------------------------
    st.subheader("🤖 AI Strategic Recommendations")

    if margin < 15:
        st.write("• Increase price or reduce operational cost")
    if marketing_spend > revenue * 0.4:
        st.write("• Marketing overspend detected — optimize ROI")
    if demand > 1000 and employees < 20:
        st.write("• Demand high — consider scaling workforce")
    if score > 75:
        st.write("• Strategy is scalable and safe to expand")

else:
    st.info("⬅️ Click Start Analysis to run model")
