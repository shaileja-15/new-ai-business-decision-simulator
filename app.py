import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Business Decision Simulator", layout="wide")

# -----------------------------
# CUSTOM UI STYLE
# -----------------------------
st.markdown("""
<style>
.main-title {
    font-size:38px;
    font-weight:800;
    color:#1f2937;
}
.insight-box {
    padding:18px;
    border-radius:12px;
    background:#eef2ff;
    font-size:18px;
    font-weight:600;
}
.decision-box {
    padding:22px;
    border-radius:14px;
    background:#dcfce7;
    font-size:22px;
    font-weight:800;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">AI Business Decision Simulator</div>', unsafe_allow_html=True)
st.caption("Company Decision Support System using Data + AI Logic")

# -----------------------------
# INDUSTRY SELECTION
# -----------------------------
industry = st.selectbox(
    "Select Industry",
    ["Retail", "E-Commerce", "Manufacturing"]
)

# Industry defaults
defaults = {
    "Retail": (50000, 220000, 160000, 80),
    "E-Commerce": (70000, 300000, 210000, 45),
    "Manufacturing": (120000, 500000, 420000, 180)
}

m_spend, revenue, cost, employees = defaults[industry]

col1, col2 = st.columns(2)

with col1:
    marketing = st.slider("Marketing Spend", 10000, 200000, m_spend)
    emp = st.slider("Number of Employees", 10, 300, employees)

with col2:
    rev = st.slider("Revenue", 50000, 800000, revenue)
    cst = st.slider("Cost", 20000, 700000, cost)

# -----------------------------
# START BUTTON
# -----------------------------
if st.button("Start Business Analysis"):

    profit = rev - cst
    profit_margin = profit / rev
    revenue_per_employee = rev / emp

    # Risk Logic
    if profit_margin < 0.1:
        risk = "High"
        risk_val = 3
    elif profit_margin < 0.25:
        risk = "Medium"
        risk_val = 2
    else:
        risk = "Low"
        risk_val = 1

    st.success("Analysis Generated")

    # -----------------------------
    # VISUAL 1 — Revenue vs Cost
    # -----------------------------
    st.subheader("Revenue vs Cost Comparison")

    fig1, ax1 = plt.subplots()
    ax1.bar(["Revenue","Cost"], [rev, cst])
    ax1.set_title("Revenue vs Cost")
    st.pyplot(fig1)

    st.markdown(
        f'<div class="insight-box">Revenue = {rev:,} | Cost = {cst:,} → Profit = {profit:,}</div>',
        unsafe_allow_html=True
    )

    # -----------------------------
    # VISUAL 2 — Profit Distribution
    # -----------------------------
    st.subheader("Profit Distribution")

    fig2, ax2 = plt.subplots()
    vals = [abs(profit), cst]
    labels = ["Profit","Cost"]
    ax2.pie(vals, labels=labels, autopct="%1.1f%%")
    st.pyplot(fig2)

    # -----------------------------
    # VISUAL 3 — Risk Indicator
    # -----------------------------
    st.subheader("Risk Level Indicator")

    fig3, ax3 = plt.subplots()
    ax3.bar(["Risk Level"], [risk_val])
    ax3.set_yticks([1,2,3])
    ax3.set_yticklabels(["Low","Medium","High"])
    st.pyplot(fig3)

    # -----------------------------
    # VISUAL 4 — Employees vs Profit
    # -----------------------------
    st.subheader("Employees vs Profit")

    fig4, ax4 = plt.subplots()
    ax4.bar(["Employees","Profit"], [emp, profit])
    ax4.set_title("Employee Count vs Profit")
    st.pyplot(fig4)

    st.markdown(
        f'<div class="insight-box">Revenue per Employee = {revenue_per_employee:,.0f}</div>',
        unsafe_allow_html=True
    )

    # -----------------------------
    # FINAL DECISION OUTPUT
    # -----------------------------
    if profit_margin > 0.25 and risk == "Low":
        decision = "INVEST & SCALE OPERATIONS"
    elif profit_margin > 0.1:
        decision = "OPTIMIZE COST & MARKETING"
    else:
        decision = "HIGH RISK — CONTROL COSTS"

    st.markdown(
        f'<div class="decision-box">Final AI Decision: {decision}</div>',
        unsafe_allow_html=True
    )
