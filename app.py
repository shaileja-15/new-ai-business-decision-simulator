# =========================================================
# AI BUSINESS DECISION SIMULATOR — PPT VISUAL VERSION
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="AI Business Decision Simulator", layout="wide")

st.title("📊 AI Business Decision Simulator")
st.caption("Industry-Based ML Decision Support")

# =====================================================
# DATA
# =====================================================

@st.cache_data
def create_data():
    np.random.seed(42)
    n = 800

    industry = np.random.choice(
        ["Retail", "Ecommerce", "Finance", "Manufacturing"], n
    )

    df = pd.DataFrame({
        "Industry": industry,
        "Revenue": np.random.randint(50000, 500000, n),
        "Cost": np.random.randint(20000, 400000, n),
        "Employees": np.random.randint(5, 300, n),
        "Marketing_Spend": np.random.randint(5000, 80000, n),
        "Customer_Churn": np.random.uniform(0.05, 0.5, n)
    })

    df["Profit"] = df["Revenue"] - df["Cost"]
    df["Profit_Margin"] = (df["Profit"] / df["Revenue"]) * 100
    df["Revenue_per_Employee"] = df["Revenue"] / df["Employees"]
    df["Marketing_Efficiency"] = df["Revenue"] / df["Marketing_Spend"]

    df["Risk_Flag"] = np.where(
        (df["Profit_Margin"] < 12) |
        (df["Customer_Churn"] > 0.30) |
        (df["Marketing_Efficiency"] < 4),
        1, 0
    )

    return df


df = create_data()

# =====================================================
# SIDEBAR INPUTS
# =====================================================

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

filtered_df = df[df["Industry"] == industry_choice]

# =====================================================
# ML MODEL
# =====================================================

features = [
    "Revenue","Cost","Employees","Marketing_Spend",
    "Customer_Churn","Profit_Margin",
    "Revenue_per_Employee","Marketing_Efficiency"
]

X = filtered_df[features]
y = filtered_df["Risk_Flag"]

model = RandomForestClassifier()
model.fit(X, y)

# =====================================================
# START BUTTON
# =====================================================

if st.sidebar.button("▶️ Start Analysis"):

    profit = revenue - cost
    margin = (profit / revenue) * 100 if revenue > 0 else 0
    rev_emp = revenue / employees if employees > 0 else 0
    mkt_eff = revenue / marketing if marketing > 0 else 0

    input_df = pd.DataFrame([[revenue,cost,employees,marketing,churn,
                              margin,rev_emp,mkt_eff]], columns=features)

    pred = model.predict(input_df)[0]

# =====================================================
# KPI METRICS
# =====================================================

    st.subheader("📌 Key KPIs")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Profit", f"{profit:,.0f}")
    c2.metric("Margin %", f"{margin:.2f}")
    c3.metric("Rev per Employee", f"{rev_emp:,.0f}")
    c4.metric("Marketing Efficiency", f"{mkt_eff:.2f}")

# =====================================================
# VISUAL 1 — Revenue vs Cost
# =====================================================

    st.subheader("📊 Visual 1 — Revenue vs Cost")

    fig1, ax1 = plt.subplots()
    vals = [revenue, cost]
    labs = ["Revenue","Cost"]

    ax1.bar(labs, vals)

    for i,v in enumerate(vals):
        ax1.text(i, v, f"{v:,.0f}", ha="center")

    st.pyplot(fig1)

    st.markdown("**Insight:** Shows whether business earnings comfortably exceed expenses.")

# =====================================================
# VISUAL 2 — Profit / Loss Distribution (Industry)
# =====================================================

    st.subheader("📈 Visual 2 — Profit/Loss Distribution")

    fig2, ax2 = plt.subplots()
    ax2.hist(filtered_df["Profit"], bins=25)

    st.pyplot(fig2)

    st.markdown("**Insight:** Displays how companies in this industry are performing overall.")

# =====================================================
# VISUAL 3 — Risk Level Indicator
# =====================================================

    st.subheader("⚠️ Visual 3 — Industry Risk Levels")

    risk_counts = filtered_df["Risk_Flag"].value_counts()

    fig3, ax3 = plt.subplots()
    ax3.bar(["Safe","Risky"], risk_counts.values)

    for i,v in enumerate(risk_counts.values):
        ax3.text(i, v, str(v), ha="center")

    st.pyplot(fig3)

    st.markdown("**Insight:** Shows how many firms fall under high-risk vs safe category.")

# =====================================================
# VISUAL 4 — Employees vs Profit
# =====================================================

    st.subheader("👥 Visual 4 — Employees vs Profit")

    sample = filtered_df.sample(150)

    fig4, ax4 = plt.subplots()
    ax4.scatter(sample["Employees"], sample["Profit"], alpha=0.5)

    st.pyplot(fig4)

    st.markdown("**Insight:** Indicates how workforce size impacts profitability.")

# =====================================================
# BOLD DECISION OUTPUT
# =====================================================

    st.subheader("💡 FINAL AI DECISION")

    if pred == 1:
        st.markdown("""
# 🔴 **HIGH BUSINESS RISK DETECTED**

**Meaning:**  
Profit margin or efficiency indicators are weak.  
Customer churn or cost pressure is high.

**Action:**  
• Reduce operational cost  
• Improve customer retention  
• Optimize marketing ROI  
""")

    else:
        st.markdown("""
# 🟢 **LOW BUSINESS RISK — HEALTHY POSITION**

**Meaning:**  
Margins and efficiency are strong.  
Risk indicators are stable.

**Action:**  
• Scale operations gradually  
• Increase smart marketing  
• Invest in expansion
""")

else:
    st.info("Enter inputs → Click Start Analysis")
