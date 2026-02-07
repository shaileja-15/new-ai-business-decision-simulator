# =========================================================
# AI BUSINESS DECISION SIMULATOR — FINAL VERSION
# Industry Aware + ML + Clean Visuals
# =========================================================

# -------------------------
# Import Libraries
# -------------------------
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# -------------------------
# Page Setup
# -------------------------
st.set_page_config(page_title="AI Business Decision Simulator", layout="wide")

st.title("📊 AI Business Decision Simulator")
st.caption("Industry-Aware ML Decision Support System")

# =========================================================
# DATA GENERATION (Synthetic but realistic)
# =========================================================

@st.cache_data
def create_data():

    np.random.seed(42)
    n = 800

    industry = np.random.choice(
        ["Retail", "Ecommerce", "Finance", "Manufacturing"],
        n
    )

    df = pd.DataFrame({
        "Industry": industry,
        "Revenue": np.random.randint(50000, 500000, n),
        "Cost": np.random.randint(20000, 400000, n),
        "Employees": np.random.randint(5, 300, n),
        "Marketing_Spend": np.random.randint(5000, 80000, n),
        "Customer_Churn": np.random.uniform(0.05, 0.5, n)
    })

    # -------------------------
    # Business KPI Engineering
    # -------------------------
    df["Profit"] = df["Revenue"] - df["Cost"]
    df["Profit_Margin"] = (df["Profit"] / df["Revenue"]) * 100
    df["Revenue_per_Employee"] = df["Revenue"] / df["Employees"]
    df["Marketing_Efficiency"] = df["Revenue"] / df["Marketing_Spend"]

    # Risk rule logic (explainable)
    df["Risk_Flag"] = np.where(
        (df["Profit_Margin"] < 12) |
        (df["Customer_Churn"] > 0.30) |
        (df["Marketing_Efficiency"] < 4),
        1, 0
    )

    return df


df = create_data()

# =========================================================
# SIDEBAR — USER INPUTS
# =========================================================

st.sidebar.header("📥 Scenario Inputs")

industry_choice = st.sidebar.selectbox(
    "Select Industry",
    ["Retail", "Ecommerce", "Finance", "Manufacturing"]
)

revenue = st.sidebar.number_input("Revenue", 10000, 1000000, 200000)
cost = st.sidebar.number_input("Cost", 5000, 900000, 120000)
employees = st.sidebar.number_input("Employees", 1, 1000, 50)
marketing = st.sidebar.number_input("Marketing Spend", 1000, 200000, 20000)
churn = st.sidebar.slider("Customer Churn", 0.01, 0.60, 0.20)

# Filter dataset by industry (important for realism)
filtered_df = df[df["Industry"] == industry_choice]

# =========================================================
# ML MODEL TRAINING (Industry specific)
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

X = filtered_df[features]
y = filtered_df["Risk_Flag"]

model = RandomForestClassifier()
model.fit(X, y)

# =========================================================
# START ANALYSIS BUTTON
# =========================================================

if st.sidebar.button("▶️ Start Analysis"):

    # -------------------------
    # KPI Calculation for input
    # -------------------------
    profit = revenue - cost
    margin = (profit / revenue) * 100 if revenue > 0 else 0
    rev_emp = revenue / employees if employees > 0 else 0
    mkt_eff = revenue / marketing if marketing > 0 else 0

    input_df = pd.DataFrame([[
        revenue, cost, employees, marketing, churn,
        margin, rev_emp, mkt_eff
    ]], columns=features)

    # ML prediction
    pred = model.predict(input_df)[0]

    # =====================================================
    # METRIC OUTPUTS
    # =====================================================

    st.subheader("📌 Key Business KPIs")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Profit", f"{profit:,.0f}")
    c2.metric("Profit Margin %", f"{margin:.2f}")
    c3.metric("Revenue per Employee", f"{rev_emp:,.0f}")
    c4.metric("Marketing Efficiency", f"{mkt_eff:.2f}")

    st.subheader("🧠 Risk Prediction")

    if pred == 1:
        st.error("HIGH RISK ⚠️")
    else:
        st.success("LOW RISK ✅")

    # =====================================================
    # VISUAL 1 — Financial Breakdown
    # =====================================================

    st.subheader("📊 Financial Breakdown")

    fig1, ax1 = plt.subplots()
    vals = [revenue, cost, max(profit, 0)]
    labs = ["Revenue", "Cost", "Profit"]

    ax1.bar(labs, vals)

    for i, v in enumerate(vals):
        ax1.text(i, v, f"{v:,.0f}", ha="center")

    st.pyplot(fig1)

    # =====================================================
    # VISUAL 2 — Risk Distribution (Industry)
    # =====================================================

    st.subheader("⚠️ Risk Distribution — Industry")

    risk_counts = filtered_df["Risk_Flag"].value_counts()

    fig2, ax2 = plt.subplots()
    ax2.bar(["Safe", "Risky"], risk_counts.values)

    for i, v in enumerate(risk_counts.values):
        ax2.text(i, v, str(v), ha="center")

    st.pyplot(fig2)

    # =====================================================
    # VISUAL 3 — Clean Scatter (Industry Pattern)
    # =====================================================

    st.subheader("📈 Marketing Spend vs Revenue Pattern")

    sample = filtered_df.sample(150)

    fig3, ax3 = plt.subplots()
    ax3.scatter(
        sample["Marketing_Spend"],
        sample["Revenue"],
        alpha=0.5,
        label="Industry Data"
    )

    ax3.scatter(
        [marketing],
        [revenue],
        s=250,
        marker="X",
        label="Your Scenario"
    )

    ax3.set_xlabel("Marketing Spend")
    ax3.set_ylabel("Revenue")
    ax3.set_title(industry_choice)
    ax3.legend()
    ax3.grid(True)

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
    # DECISION ENGINE
    # =====================================================

    st.subheader("💡 Decision Recommendation")

    if pred == 1:
        st.error("""
        • Profit margin below safe threshold  
        • Customer churn is high  
        • Marketing efficiency is weak  

        👉 Reduce cost and improve retention  
        👉 Optimize marketing targeting  
        """)
    else:
        st.success("""
        • Healthy margin and efficiency  
        • Risk indicators are stable  

        👉 Scale marketing gradually  
        👉 Expand operations carefully  
        """)

else:
    st.info("Enter scenario → Click Start Analysis")
