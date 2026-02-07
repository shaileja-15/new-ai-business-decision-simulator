# =========================================================
# AI BUSINESS DECISION SIMULATOR — STREAMLIT APP
# Placement Project Version
# =========================================================

# -------------------------------
# Import Libraries
# -------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="AI Business Decision Simulator", layout="wide")

st.title("📊 AI Business Decision Simulator")
st.caption("ML-Based Business Risk Prediction & Decision Support")

# =========================================================
# SECTION 1 — CREATE SYNTHETIC TRAINING DATA
# (Used to train ML model inside app)
# =========================================================

@st.cache_data
def create_data():
    np.random.seed(42)

    n = 500

    data = pd.DataFrame({
        "Revenue": np.random.randint(50000, 500000, n),
        "Cost": np.random.randint(20000, 400000, n),
        "Employees": np.random.randint(5, 300, n),
        "Marketing_Spend": np.random.randint(5000, 80000, n),
        "Customer_Churn": np.random.uniform(0.05, 0.5, n)
    })

    # Business KPIs
    data["Profit"] = data["Revenue"] - data["Cost"]
    data["Profit_Margin"] = (data["Profit"] / data["Revenue"]) * 100
    data["Revenue_per_Employee"] = data["Revenue"] / data["Employees"]
    data["Marketing_Efficiency"] = data["Revenue"] / data["Marketing_Spend"]

    # Risk rule logic (explainable label)
    data["Risk_Flag"] = np.where(
        (data["Profit_Margin"] < 12) |
        (data["Customer_Churn"] > 0.30) |
        (data["Marketing_Efficiency"] < 4),
        1, 0
    )

    return data


df = create_data()

# =========================================================
# SECTION 2 — TRAIN ML MODEL
# =========================================================

features = [
    "Revenue",
    "Cost",
    "Employees",
    "Marketing_Spend",
    "Customer_Churn",
    "Profit_Margin",
    "Revenue_per_Employee",
    "Marketing_Efficiency"
]

X = df[features]
y = df["Risk_Flag"]

model = RandomForestClassifier()
model.fit(X, y)

# =========================================================
# SECTION 3 — USER INPUT PANEL
# =========================================================

st.sidebar.header("📥 Enter Business Scenario")

revenue = st.sidebar.number_input("Revenue", 10000, 1000000, 200000)
cost = st.sidebar.number_input("Cost", 5000, 900000, 120000)
employees = st.sidebar.number_input("Employees", 1, 1000, 50)
marketing = st.sidebar.number_input("Marketing Spend", 1000, 200000, 20000)
churn = st.sidebar.slider("Customer Churn Rate", 0.01, 0.60, 0.20)

# =========================================================
# SECTION 4 — START ANALYSIS BUTTON
# =========================================================

if st.sidebar.button("▶️ Start Analysis"):

    # -------------------------------
    # KPI Calculations
    # -------------------------------
    profit = revenue - cost
    margin = (profit / revenue) * 100 if revenue > 0 else 0
    rev_per_emp = revenue / employees if employees > 0 else 0
    mkt_eff = revenue / marketing if marketing > 0 else 0

    # Prepare model input
    input_data = pd.DataFrame([[
        revenue, cost, employees, marketing, churn,
        margin, rev_per_emp, mkt_eff
    ]], columns=features)

    # -------------------------------
    # ML Prediction
    # -------------------------------
    pred = model.predict(input_data)[0]
    risk_text = "HIGH RISK ⚠️" if pred == 1 else "LOW RISK ✅"

    # =====================================================
    # OUTPUT METRICS
    # =====================================================

    st.subheader("📌 Key Business Metrics")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Profit", f"{profit:,.0f}")
    c2.metric("Profit Margin %", f"{margin:.2f}")
    c3.metric("Rev per Employee", f"{rev_per_emp:,.0f}")
    c4.metric("Marketing Efficiency", f"{mkt_eff:.2f}")

    st.subheader("🧠 Model Risk Prediction")
    st.success(risk_text)

    # =====================================================
    # VISUAL 1 — Revenue vs Cost vs Profit
    # =====================================================

    st.subheader("📊 Financial Breakdown")

    fig1, ax1 = plt.subplots()
    vals = [revenue, cost, max(profit, 0)]
    labels = ["Revenue", "Cost", "Profit"]
    ax1.bar(labels, vals)

    for i, v in enumerate(vals):
        ax1.text(i, v, f"{v:,.0f}", ha="center", va="bottom")

    st.pyplot(fig1)

    # =====================================================
    # VISUAL 2 — Cost vs Profit Pie (safe values)
    # =====================================================

    st.subheader("🥧 Cost vs Profit Share")

    safe_profit = max(profit, 1)
    safe_cost = max(cost, 1)

    fig2, ax2 = plt.subplots()
    ax2.pie([safe_cost, safe_profit],
            labels=["Cost", "Profit"],
            autopct="%1.1f%%")
    st.pyplot(fig2)

    # =====================================================
    # VISUAL 3 — Marketing ROI Scatter
    # =====================================================

    st.subheader("📈 Marketing Spend vs Revenue Pattern")

    fig3, ax3 = plt.subplots()
    ax3.scatter(df["Marketing_Spend"], df["Revenue"])
    ax3.scatter([marketing], [revenue], s=200, marker="X")
    ax3.set_xlabel("Marketing Spend")
    ax3.set_ylabel("Revenue")

    st.pyplot(fig3)

    # =====================================================
    # VISUAL 4 — Feature Importance
    # =====================================================

    st.subheader("🔍 Risk Driver Importance")

    imp = pd.Series(model.feature_importances_, index=features)
    fig4, ax4 = plt.subplots()
    imp.sort_values().plot(kind="barh", ax=ax4)

    st.pyplot(fig4)

    # =====================================================
    # BUSINESS DECISION ENGINE
    # =====================================================

    st.subheader("💡 AI Decision Recommendation")

    if pred == 1:
        st.error("""
        • Risk is high based on margin, churn or efficiency  
        • Reduce operating cost  
        • Improve customer retention  
        • Optimize marketing spend  
        """)
    else:
        st.info("""
        • Business is in safe zone  
        • Scale marketing gradually  
        • Increase team capacity  
        • Invest in growth initiatives  
        """)

else:
    st.info("Enter scenario values and click **Start Analysis**")
