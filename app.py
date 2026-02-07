# =========================================================
# AI BUSINESS DECISION SIMULATOR — FINAL PLACEMENT VERSION
# Clean UI + Industry Based + Explainable Decisions
# =========================================================

import streamlit as st
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="AI Business Decision Simulator", layout="wide")

# ---------------------------------------------------------
# CUSTOM UI STYLE (for beautiful cards)
# ---------------------------------------------------------
st.markdown("""
<style>
.main-title {
    font-size:40px;
    font-weight:900;
}
.insight-box {
    padding:18px;
    border-radius:12px;
    background:#eef2ff;
    font-size:17px;
    font-weight:600;
}
.decision-box {
    padding:22px;
    border-radius:14px;
    background:#dcfce7;
    font-size:22px;
    font-weight:900;
}
.warning-box {
    padding:22px;
    border-radius:14px;
    background:#fee2e2;
    font-size:22px;
    font-weight:900;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📊 AI Business Decision Simulator</div>', unsafe_allow_html=True)
st.caption("Industry-Aware Decision Support using Business KPIs")

# ---------------------------------------------------------
# INDUSTRY SCENARIO SELECTION
# ---------------------------------------------------------
industry = st.selectbox(
    "Select Industry",
    ["Retail", "E-Commerce", "Manufacturing", "Finance"]
)

# Industry-specific default ranges (realism)
defaults = {
    "Retail": (60000, 220000, 160000, 70),
    "E-Commerce": (90000, 350000, 240000, 45),
    "Manufacturing": (140000, 520000, 430000, 180),
    "Finance": (80000, 400000, 260000, 60)
}

m_def, r_def, c_def, e_def = defaults[industry]

col1, col2 = st.columns(2)

with col1:
    marketing = st.slider("Marketing Spend", 10000, 250000, m_def)
    employees = st.slider("Number of Employees", 10, 400, e_def)

with col2:
    revenue = st.slider("Revenue", 50000, 900000, r_def)
    cost = st.slider("Cost", 20000, 800000, c_def)

# ---------------------------------------------------------
# START ANALYSIS BUTTON
# ---------------------------------------------------------
if st.button("▶️ Start Business Analysis"):

    # =====================================================
    # KPI CALCULATIONS
    # =====================================================
    profit = revenue - cost
    profit_margin = profit / revenue if revenue > 0 else 0
    revenue_per_employee = revenue / employees
    marketing_efficiency = revenue / marketing
    cost_ratio = cost / revenue

    st.success("Analysis Generated Successfully")

    # =====================================================
    # KPI METRIC CARDS
    # =====================================================
    st.subheader("📌 Key Business KPIs")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Profit", f"{profit:,.0f}")
    c2.metric("Profit Margin %", f"{profit_margin*100:.2f}")
    c3.metric("Revenue / Employee", f"{revenue_per_employee:,.0f}")
    c4.metric("Marketing ROI", f"{marketing_efficiency:.2f}")

    # =====================================================
    # VISUAL 1 — Revenue vs Cost
    # =====================================================
    st.subheader("📊 Visual 1 — Revenue vs Cost")

    fig1, ax1 = plt.subplots()
    ax1.bar(["Revenue", "Cost"], [revenue, cost])
    st.pyplot(fig1)

    st.markdown(
        f'<div class="insight-box">Revenue {revenue:,} vs Cost {cost:,} → Profit {profit:,}</div>',
        unsafe_allow_html=True
    )

    # =====================================================
    # VISUAL 2 — Profit / Cost Distribution
    # =====================================================
    st.subheader("📈 Visual 2 — Profit Distribution")

    fig2, ax2 = plt.subplots()
    ax2.pie([abs(profit), cost], labels=["Profit","Cost"], autopct="%1.1f%%")
    st.pyplot(fig2)

    # =====================================================
    # VISUAL 3 — Risk Indicator
    # =====================================================
    st.subheader("⚠️ Visual 3 — Risk Level Indicator")

    if profit_margin < 0.10:
        risk = "High"
        risk_val = 3
    elif profit_margin < 0.25:
        risk = "Medium"
        risk_val = 2
    else:
        risk = "Low"
        risk_val = 1

    fig3, ax3 = plt.subplots()
    ax3.bar(["Risk Level"], [risk_val])
    ax3.set_yticks([1,2,3])
    ax3.set_yticklabels(["Low","Medium","High"])
    st.pyplot(fig3)

    # =====================================================
    # VISUAL 4 — Employees vs Profit
    # =====================================================
    st.subheader("👥 Visual 4 — Employees vs Profit")

    fig4, ax4 = plt.subplots()
    ax4.bar(["Employees","Profit"], [employees, profit])
    st.pyplot(fig4)

    # =====================================================
    # DECISION BASIS ENGINE (Explainable Suggestions)
    # =====================================================
    st.subheader("📘 Decision Basis")

    issues = []
    actions = []

    if profit_margin < 0.10:
        issues.append("Profit margin below 10%")
        actions.append("Reduce operating cost by 8–15%")

    if cost_ratio > 0.75:
        issues.append("Cost ratio above 75%")
        actions.append("Optimize vendor and fixed expenses")

    if revenue_per_employee < 4000:
        issues.append("Low employee productivity")
        actions.append("Improve productivity or automate tasks")

    if marketing_efficiency < 4:
        issues.append("Weak marketing ROI")
        actions.append("Shift to high-conversion channels")

    for i in issues:
        st.write("•", i)

    st.markdown("### ✅ Suggested Improvements")
    for a in actions:
        st.write("✔", a)

    st.info("All suggestions are KPI-threshold driven — not random.")

    # =====================================================
    # FINAL DECISION OUTPUT (BOLD)
    # =====================================================
    if profit_margin > 0.25 and risk == "Low":
        decision = "INVEST & SCALE OPERATIONS"
        st.markdown(
            f'<div class="decision-box">🟢 FINAL AI DECISION: {decision}</div>',
            unsafe_allow_html=True
        )
    elif profit_margin > 0.10:
        decision = "OPTIMIZE COST & MARKETING"
        st.markdown(
            f'<div class="insight-box">🟡 FINAL AI DECISION: {decision}</div>',
            unsafe_allow_html=True
        )
    else:
        decision = "HIGH RISK — COST CONTROL NEEDED"
        st.markdown(
            f'<div class="warning-box">🔴 FINAL AI DECISION: {decision}</div>',
            unsafe_allow_html=True
        )

else:
    st.info("Select inputs → Click Start Business Analysis")
